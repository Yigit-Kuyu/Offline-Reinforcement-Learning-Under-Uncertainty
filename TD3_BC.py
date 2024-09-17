import numpy as np
import torch
import gym
import d4rl
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define test function
def test_policy(policy, env_name, seed, mean, std, num_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	total_rewards = []
	for _ in range(num_episodes):
		state, done = eval_env.reset(), False
		episode_reward = 0.
		while not done:
			#state = (np.array(state).reshape(1,-1) - mean)/std # Normalize the state if normalize is True
			state = np.array(state).reshape(1,-1) # No normalization
			action = policy.select_action(state)
			state, reward, done, _ = eval_env.step(action)
			episode_reward += reward
		total_rewards.append(episode_reward)

		avg_reward = np.mean(total_rewards)
		std_reward = np.std(total_rewards)

		print("---------------------------------------")
		print(f"Evaluation over {num_episodes} episodes: {avg_reward:.3f} +/- {std_reward:.3f}")
		print("---------------------------------------")
	return np.mean(total_rewards), np.std(total_rewards), total_rewards


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(device),
			torch.FloatTensor(self.action[ind]).to(device),
			torch.FloatTensor(self.next_state[ind]).to(device),
			torch.FloatTensor(self.reward[ind]).to(device),
			torch.FloatTensor(self.not_done[ind]).to(device)
		)


	def convert_D4RL(self, dataset):
		self.state = dataset['observations']
		self.action = dataset['actions']
		self.next_state = dataset['next_observations']
		self.reward = dataset['rewards'].reshape(-1,1)
		self.not_done = 1. - dataset['terminals'].reshape(-1,1)
		self.size = self.state.shape[0]


	def normalize_states(self, eps = 1e-3):
		mean = self.state.mean(0,keepdims=True)
		std = self.state.std(0,keepdims=True) + eps
		self.state = (self.state - mean)/std
		self.next_state = (self.next_state - mean)/std
		return mean, std


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3_BC(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		alpha=2.5,
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.alpha = alpha

		self.total_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor loss
			pi = self.actor(state)
			Q = self.critic.Q1(state, pi)
			lmbda = self.alpha/Q.abs().mean().detach()

			actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action) 
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)


# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + seed_offset)

	rewards = []
	for _ in range(eval_episodes):
		episode_reward = 0.
		state, done = eval_env.reset(), False
		while not done:
			#state = (np.array(state).reshape(1,-1) - mean)/std # Normalize the state if normalize is True
			action = policy.select_action(state)
			state, reward, done, _ = eval_env.step(action)
			episode_reward += reward
		rewards.append(episode_reward)

	rewards = np.array(rewards)
	ret_eval = np.mean(rewards)
	std_ret = np.std(rewards)
	median_ret = np.median(rewards)

	d4rl_score = eval_env.get_normalized_score(ret_eval) * 100

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {ret_eval:.3f}, D4RL score: {d4rl_score:.3f}")
	print(f"Std: {std_ret:.3f}, Median: {median_ret:.3f}")
	print("---------------------------------------")
	return ret_eval, std_ret, median_ret


if __name__ == "__main__":
	
	# Fixed parameters
	policy = "TD3_BC"
	env_name = "halfcheetah-random-v2"
	seed = 0
	eval_freq = 10 # 1000
	max_timesteps = 10 # 1000
	expl_noise = 0.1
	batch_size = 256
	discount = 0.99
	tau = 0.005
	policy_noise = 0.2
	noise_clip = 0.5
	policy_freq = 2
	alpha = 2.5
	normalize = False

	print("---------------------------------------")
	print(f"Policy: {policy}, Env: {env_name}, Seed: {seed}")
	print("---------------------------------------")

	env = gym.make(env_name)

	# Set seeds
	env.seed(seed)
	env.action_space.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": discount,
		"tau": tau,
		# TD3
		"policy_noise": policy_noise * max_action,
		"noise_clip": noise_clip * max_action,
		"policy_freq": policy_freq,
		# TD3 + BC
		"alpha": alpha
	}

	# Initialize policy
	policy = TD3_BC(**kwargs)

	replay_buffer = ReplayBuffer(state_dim, action_dim)
	replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
	if normalize:
		mean,std = replay_buffer.normalize_states() 
	else:
		mean,std = 0,1
	
	
	# train the policy
	average_returns = []
	max_gradient_steps = int(max_timesteps * eval_freq)
	for t in range(max_gradient_steps): # For fair comparison, we run the algorithm for the same number of gradient steps
		policy.train(replay_buffer, batch_size) #   processes one batch of data per call
		
		print(f"Time steps: {t+1}")
		ret_eval, std_ret, median_ret = eval_policy(policy, env_name, seed, mean, std)
		average_returns.append(ret_eval)
		print(f"Average Return: {ret_eval}, std: {std_ret}, median: {median_ret}")

	# Calculate average, std and median of the training loop
	avg_return = np.mean(average_returns)
	std_return = np.std(average_returns)
	median_return = np.median(average_returns)

	# Write results to a text file
	with open('TD3_BC_training_results.txt', 'w') as f:
		f.write(f"Training Results for {env_name}:\n")
		f.write(f"Average Return: {avg_return:.2f}\n")
		f.write(f"Standard Deviation: {std_return:.2f}\n")
		f.write(f"Median Return: {median_return:.2f}\n")

	print("TD3_BC Training results have been written to 'TD3_BC_training_results.txt'")

	
   	# Plot training curve
	plt.figure(figsize=(10, 6))
	plt.plot(range(1, len(average_returns) + 1), average_returns, marker='o')
	plt.xlabel('Training Iterations')
	plt.ylabel('Average Return')
	plt.title('Training Curve: Average Return vs Training Iterations')
	plt.grid(True)
	plt.savefig('TD3_BC_training_curve.png')
	plt.close()

	# Save the policy
	policy.save("TD3_BC")

	# Load the policy
	loaded_policy = TD3_BC(**kwargs)
	loaded_policy.load("TD3_BC")


	# Use loaded policy for testing
	test_rewards, test_std, total_rewards = test_policy(loaded_policy, env_name, seed, mean, std)

	# Write test results to a text file
	with open('TD3_BC_test_results.txt', 'w') as f:
		f.write(f"Test Results for {env_name}:\n")
		f.write(f"Average Return: {test_rewards:.2f}\n")
		f.write(f"Standard Deviation: {test_std:.2f}\n")
		f.write(f"Total Rewards: {total_rewards}\n")

	print("TD3_BC Test results have been written to 'TD3_BC_test_results.txt'")	

	# Save test rewards
	np.save('TD3_BC_test_rewards.npy', total_rewards)
	# Load test rewards for figure
	loaded_total_rewards = np.load('TD3_BC_test_rewards.npy')
    
	# Draw total_rewards
	plt.figure(figsize=(10, 6))
	plt.plot(range(1, len(total_rewards) + 1), total_rewards, marker='o')
	plt.xlabel('Episode')
	plt.ylabel('Total Reward')
	plt.title('Total Rewards per Episode in Test')
	plt.grid(True)
	plt.savefig('TD3_BC_testing_curve.png')
	plt.show()
	plt.close()
print('stop')
	