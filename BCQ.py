import gym
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import d4rl
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define test function
def test_policy(policy, env, num_episodes=10):
	eval_env = env

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



class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, phi=0.05):
		super(Actor, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action
		self.phi = phi  # Perturbation factor


	def forward(self, state, action):
		a = F.relu(self.l1(torch.cat([state, action], 1)))
		a = F.relu(self.l2(a))
		a = self.phi * self.max_action * torch.tanh(self.l3(a)) # Perturbation applied here
		return (a + action).clamp(-self.max_action, self.max_action)


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

		self.l4 = nn.Linear(state_dim + action_dim, 400)
		self.l5 = nn.Linear(400, 300)
		self.l6 = nn.Linear(300, 1)


	def forward(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(torch.cat([state, action], 1)))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def q1(self, state, action):
		q1 = F.relu(self.l1(torch.cat([state, action], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
	def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
		super(VAE, self).__init__()
		self.e1 = nn.Linear(state_dim + action_dim, 750)
		self.e2 = nn.Linear(750, 750)

		self.mean = nn.Linear(750, latent_dim)
		self.log_std = nn.Linear(750, latent_dim)

		self.d1 = nn.Linear(state_dim + latent_dim, 750)
		self.d2 = nn.Linear(750, 750)
		self.d3 = nn.Linear(750, action_dim)

		self.max_action = max_action
		self.latent_dim = latent_dim
		self.device = device


	def forward(self, state, action):
		z = F.relu(self.e1(torch.cat([state, action], 1)))
		z = F.relu(self.e2(z))

		mean = self.mean(z)
		# Clamped for numerical stability 
		log_std = self.log_std(z).clamp(-4, 15)
		std = torch.exp(log_std)
		z = mean + std * torch.randn_like(std)
		
		u = self.decode(state, z)

		return u, mean, std


	def decode(self, state, z=None):
		# When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
		if z is None:
			z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

		a = F.relu(self.d1(torch.cat([state, z], 1)))
		a = F.relu(self.d2(a))
		return self.max_action * torch.tanh(self.d3(a))
		


class BCQ(object):
	def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005, lmbda=0.75, phi=0.05):
		latent_dim = action_dim * 2

		self.actor = Actor(state_dim, action_dim, max_action, phi).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

		self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
		self.vae_optimizer = torch.optim.Adam(self.vae.parameters()) 

		self.max_action = max_action
		self.action_dim = action_dim
		self.discount = discount
		self.tau = tau
		self.lmbda = lmbda
		self.device = device


	def select_action(self, state):		
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(self.device)
			action = self.actor(state, self.vae.decode(state))
			q1 = self.critic.q1(state, action)
			ind = q1.argmax(0)
		return action[ind].cpu().data.numpy().flatten()


	def train(self, replay_buffer, iterations, batch_size=100):

		for it in range(iterations):
			# Sample replay buffer / batch
			state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

			# Variational Auto-Encoder Training
			recon, mean, std = self.vae(state, action)
			recon_loss = F.mse_loss(recon, action)
			KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
			vae_loss = recon_loss + 0.5 * KL_loss

			self.vae_optimizer.zero_grad()
			vae_loss.backward()
			self.vae_optimizer.step()


			# Critic Training
			with torch.no_grad():
				# Duplicate next state 10 times
				next_state = torch.repeat_interleave(next_state, 10, 0)

				# Compute value of perturbed actions sampled from the VAE
				target_Q1, target_Q2 = self.critic_target(next_state, self.actor_target(next_state, self.vae.decode(next_state)))

				# Soft Clipped Double Q-learning 
				target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
				# Take max over each action sampled from the VAE
				target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)

				target_Q = reward + not_done * self.discount * target_Q

			current_Q1, current_Q2 = self.critic(state, action)
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()


			# Pertubation Model / Action Training
			sampled_actions = self.vae.decode(state)
			perturbed_actions = self.actor(state, sampled_actions)

			# Update through DPG
			actor_loss = -self.critic.q1(state, perturbed_actions).mean()
		 	 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()


			# Update Target Networks 
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

		torch.save(self.vae.state_dict(), filename + "_vae")
		torch.save(self.vae_optimizer.state_dict(), filename + "_vae_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)

		self.vae.load_state_dict(torch.load(filename + "_vae"))
		self.vae_optimizer.load_state_dict(torch.load(filename + "_vae_optimizer"))


def load_d4rl_dataset(env, replay_buffer):
    dataset = d4rl.qlearning_dataset(env)
    
    N = dataset['observations'].shape[0]
    replay_buffer.state[:N] = dataset['observations']
    replay_buffer.action[:N] = dataset['actions']
    replay_buffer.next_state[:N] = dataset['next_observations']
    replay_buffer.reward[:N] = dataset['rewards'].reshape(-1, 1)
    replay_buffer.not_done[:N] = 1. - dataset['terminals'].reshape(-1, 1)
    replay_buffer.size = N
    replay_buffer.ptr = N % replay_buffer.max_size
    print(f"Dataset size: {N}")

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = device


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		if self.size == 0:
			raise ValueError("Cannot sample from an empty buffer")
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(device),
			torch.FloatTensor(self.action[ind]).to(device),
			torch.FloatTensor(self.next_state[ind]).to(device),
			torch.FloatTensor(self.reward[ind]).to(device),
			torch.FloatTensor(self.not_done[ind]).to(device)
		)

# Trains BCQ offline
def train_BCQ(env,state_dim, action_dim, max_action,discount,tau,lmbda,phi,max_timesteps,eval_freq,batch_size):
	# Initialize policy
    policy = BCQ(state_dim, action_dim, max_action, device, discount, tau, lmbda, phi)

    # Initialize and load buffer
    replay_buffer = ReplayBuffer(state_dim, action_dim, device)
    load_d4rl_dataset(env, replay_buffer)
    average_returns = []
    training_iters = 0
    gradient_steps = 0
    max_grad_steps = int(max_timesteps*eval_freq)
    while gradient_steps < max_grad_steps:
        pol_vals = policy.train(replay_buffer, iterations=int(eval_freq), batch_size=batch_size)
        #gradient_steps += int(eval_freq) * batch_size  # Each iteration performs 'batch_size' gradient steps
        gradient_steps += int(eval_freq)
        ret_eval, std_ret, median_ret, d4rl_score = normalized_evaluate_policy(policy, env)
        average_returns.append(ret_eval)
        print(f"Gradient steps: {gradient_steps}, Average Return: {ret_eval}, std: {std_ret}, median: {median_ret}, D4RL Score: {d4rl_score}")
	
    # Calculate average, std and median of the training loop
    avg_return = np.mean(average_returns)
    std_return = np.std(average_returns)
    median_return = np.median(average_returns)
    # Write results to a text file
    with open('BCQ_training_results.txt', 'w') as f:
        f.write(f"Training Results for {env_name}:\n")
        f.write(f"Average Return: {avg_return:.2f}\n")
        f.write(f"Standard Deviation: {std_return:.2f}\n")
        f.write(f"Median Return: {median_return:.2f}\n")
        f.write(f"D4RL Score: {d4rl_score:.2f}\n")
    print("BCQ Training results have been written to 'BCQ_training_results.txt'")
    
    # Save final policy
    policy.save("BCQ")

	# Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(average_returns) + 1), average_returns, marker='o')
    plt.xlabel('Training Iterations')
    plt.ylabel('Average Return')
    plt.title('Training Curve: Average Return vs Training Iterations')
    plt.grid(True)
    plt.savefig('BCQ_training_curve.png')
    plt.close()
	

def evaluate_policy(policy, env, eval_episodes=10):
    avg_reward = 0.
    all_rewards = []
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
        all_rewards.append(avg_reward)
    avg_reward /= eval_episodes
    for j in range(eval_episodes-1, 1, -1):
        all_rewards[j] = all_rewards[j] - all_rewards[j-1]

    all_rewards = np.array(all_rewards)
    std_rewards = np.std(all_rewards)
    median_reward = np.median(all_rewards)
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward}")
    print("---------------------------------------")
    return avg_reward, std_rewards, median_reward

def normalized_evaluate_policy(policy, env, eval_episodes=10):
    avg_reward = 0.
    all_rewards = []
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
        all_rewards.append(avg_reward)
    avg_reward /= eval_episodes
    d4rl_score = env.get_normalized_score(avg_reward) * 100

    all_rewards = np.array(all_rewards)
    std_rewards = np.std(all_rewards)
    median_reward = np.median(all_rewards)
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward}, D4RL Score: {d4rl_score}")
    print("---------------------------------------")
    return avg_reward, std_rewards, median_reward, d4rl_score


if __name__ == "__main__":
	
	env_name = 'halfcheetah-random-v2'
	seed = 0
	buffer_name = "Robust"
	eval_freq = 1000 #5000
	max_timesteps = 1000#1000000
	start_timesteps = 25000
	rand_action_p = 0.3
	gaussian_std = 0.3
	batch_size = 100
	discount = 0.99
	tau = 0.005
	lmbda = 0.75
	phi = 0.05
	
	print("---------------------------------------")
	print(f"Setting: Training BCQ, Env: {env_name}, Seed: {seed}")
	print("---------------------------------------")

	env = gym.make(env_name)

	env.seed(seed)
	env.action_space.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	train_BCQ(env, state_dim, action_dim, max_action, discount, tau, lmbda, phi, max_timesteps, eval_freq, batch_size)
      
	# Load the saved policy for testing
	loaded_policy = BCQ(state_dim, action_dim, max_action, device)
	loaded_policy.load("BCQ")

	# Use loaded policy for testing
	test_rewards, test_std, total_rewards = test_policy(loaded_policy, env, num_episodes=10)
    
	# Write test results to a text file
	with open('BCQ_test_results.txt', 'w') as f:
		f.write(f"Test Results for {env_name}:\n")
		f.write(f"Average Return: {test_rewards:.2f}\n")
		f.write(f"Standard Deviation: {test_std:.2f}\n")
		f.write(f"Total Rewards: {total_rewards}\n")


	# Save test rewards
	np.save('BCQ_test_rewards.npy', total_rewards)
	# Load test rewards for figure
	loaded_total_rewards = np.load('BCQ_test_rewards.npy')
    
	# Draw total_rewards
	plt.figure(figsize=(10, 6))
	plt.plot(range(1, len(loaded_total_rewards) + 1), loaded_total_rewards, marker='o')
	plt.xlabel('Episode')
	plt.ylabel('Total Reward')
	plt.title('Total Rewards per Episode in Test')
	plt.grid(True)
	plt.savefig('BCQ_testing_curve.png')
	plt.show()
	plt.close()


	
