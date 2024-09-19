import gym
import numpy as np
import torch
import d4rl
from utils import utils
#from utils.data_sampler import Data_Sampler
#from agents.diffcps import DiffCPS as Agent
import torch
import numpy as np
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
#from agents.diffusion import Diffusion
#from agents.model import MLP
#from agents.helpers import EMA
#from agents.helpers import SinusoidalPosEmb
import math
import time
import matplotlib.pyplot as plt




hyperparameters = {
    "halfcheetah-random-v2": { # halfcheetah-medium-v2
        "lr": 3e-4,
        "lambda": 1.0,
        "max_q_backup": False,
        "reward_tune": "no",
        "eval_freq": 50,
        "num_epochs": 2000,
        "freq": 2,
        "lambda_min": 0,
        "target_kl": 0.06,
        "gn": 9.0,
    },
    "hopper-medium-v2": {
        "lr": 3e-4,
        "lambda": 1.0,
        "max_q_backup": False,
        "reward_tune": "no",
        "eval_freq": 50,
        "num_epochs": 2000,
        "lambda_min": 0,
        "target_kl": 0.05,
        "gn": 9.0,
        "freq": 2,
    },
    "walker2d-medium-v2": {
        "lr": 3e-4,
        "lambda": 1.0,
        "max_q_backup": False,
        "reward_tune": "no",
        "eval_freq": 50,
        "num_epochs": 2000,
        "lambda_min": 0,
        "target_kl": 0.03,
        "gn": 1.0,
        "freq": 2,
    },
    "halfcheetah-medium-replay-v2": {
        "lr": 3e-4,
        "lambda": 1.0,
        "max_q_backup": False,
        "reward_tune": "no",
        "eval_freq": 50,
        "num_epochs": 2000,
        "lambda_min": 0,
        "target_kl": 0.06,
        "gn": 2.0,
        "freq": 2,
    },
    "hopper-medium-replay-v2": {
        "lr": 3e-4,
        "lambda": 1.0,
        "max_q_backup": False,
        "reward_tune": "no",
        "eval_freq": 50,
        "num_epochs": 2000,
        "lambda_min": 0,
        "target_kl": 0.03,
        "gn": 4.0,
        "freq": 2,
    },
    "walker2d-medium-replay-v2": {
        "lr": 3e-4,
        "lambda": 1.0,
        "max_q_backup": False,
        "reward_tune": "no",
        "eval_freq": 50,
        "num_epochs": 2000,
        "lambda_min": 0,
        "target_kl": 0.03,
        "gn": 4.0,
        "freq": 2,
    },
    "halfcheetah-medium-expert-v2": {
        "lr": 3e-4,
        "lambda": 1.0,
        "max_q_backup": False,
        "reward_tune": "no",
        "eval_freq": 50,
        "num_epochs": 2000,
        "lambda_min": 0,
        "target_kl": 0.04,
        "gn": 7.0,
        "freq": 2,
    },
    "hopper-medium-expert-v2": {
        "lr": 3e-4,
        "lambda": 1.0,
        "max_q_backup": False,
        "reward_tune": "no",
        "eval_freq": 50,
        "num_epochs": 2000,
        "lambda_min": 0,
        "target_kl": 0.03,
        "gn": 5.0,
        "freq": 2,
    },
    "walker2d-medium-expert-v2": {
        "lr": 3e-4,
        "lambda": 1.0,
        "max_q_backup": False,
        "reward_tune": "no",
        "eval_freq": 50,
        "num_epochs": 2000,
        "lambda_min": 0,
        "target_kl": 0.04,
        "gn": 5.0,
        "freq": 2,
    },
    "antmaze-umaze-v0": {
        "lr": 3e-4,
        "lambda": 3,
        "max_q_backup": False,
        "reward_tune": "cql_antmaze",
        "eval_freq": 50,
        "num_epochs": 1000,
        "lambda_min": 0.3,
        "target_kl": 0.2,
        "gn": 2.0,
        "freq": 2,
    },
    "antmaze-umaze-diverse-v0": {
        "lr": 3e-4,
        "lambda": 3,
        "max_q_backup": True,
        "reward_tune": "cql_antmaze",
        "eval_freq": 50,
        "num_epochs": 1000,
        "lambda_min": 0.3,
        "target_kl": 0.09,
        "gn": 3.0,
        "freq": 2,
    },
    "antmaze-medium-play-v0": {
        "lr": 1e-3,
        "lambda": 1,
        "max_q_backup": True,
        "reward_tune": "cql_antmaze",
        "eval_freq": 50,
        "num_epochs": 1000,
        "lambda_min": 0.3,
        "target_kl": 0.3,
        "gn": 2.0,
        "freq": 2,
    },
    "antmaze-medium-diverse-v0": {
        "lr": 3e-4,
        "lambda": 1,
        "max_q_backup": True,
        "reward_tune": "cql_antmaze",
        "eval_freq": 50,
        "num_epochs": 1000,
        "lambda_min": 0.3,
        "target_kl": 0.2,
        "gn": 1.0,
        "freq": 2,
    },
    "antmaze-large-play-v0": {
        "lr": 3e-4,
        "lambda": 0.5,
        "max_q_backup": True,
        "reward_tune": "cql_antmaze",
        "eval_freq": 50,
        "num_epochs": 1000,
        "lambda_min": 0.3,
        "target_kl": 0.2,
        "gn": 10.0,
        "freq": 4,
    },
    "antmaze-large-diverse-v0": {
        "lr": 3e-4,
        "lambda": 0.5,
        "max_q_backup": True,
        "reward_tune": "cql_antmaze",
        "eval_freq": 50,
        "num_epochs": 1000,
        "lambda_min": 0.3,
        "target_kl": 0.2,
        "gn": 7.0,
        "freq": 4,
    },
}


###### Testing #######

def test_policy(policy, env, num_episodes=10):
    total_rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = policy.sample_action(np.array(state))
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        total_rewards.append(episode_reward)
    return np.mean(total_rewards), np.std(total_rewards), total_rewards



####### Data Sampler #######

class Data_Sampler(object):
    def __init__(self, data, device, reward_tune="no"):
        self.state = torch.from_numpy(data["observations"]).float()
        self.action = torch.from_numpy(data["actions"]).float()
        self.next_state = torch.from_numpy(data["next_observations"]).float()
        reward = torch.from_numpy(data["rewards"]).view(-1, 1).float()
        self.not_done = 1.0 - torch.from_numpy(data["terminals"]).view(-1, 1).float()

        self.size = self.state.shape[0]
        self.state_dim = self.state.shape[1]
        self.action_dim = self.action.shape[1]

        self.device = device

        if reward_tune == "normalize":
            reward = (reward - reward.mean()) / reward.std()
        elif reward_tune == "iql_antmaze":
            reward = reward - 1.0
        elif reward_tune == "iql_locomotion":
            reward = iql_normalize(reward, self.not_done)
        elif reward_tune == "cql_antmaze":
            reward = (reward - 0.5) * 4.0
        elif reward_tune == "antmaze":
            reward = (reward - 0.25) * 2.0
        self.reward = reward

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, size=(batch_size,))

        return (
            self.state[ind].to(self.device),
            self.action[ind].to(self.device),
            self.next_state[ind].to(self.device),
            self.reward[ind].to(self.device),
            self.not_done[ind].to(self.device),
        )
    
def iql_normalize(reward, not_done):
    trajs_rt = []
    episode_return = 0.0
    for i in range(len(reward)):
        episode_return += reward[i]
        if not not_done[i]:
            trajs_rt.append(episode_return)
            episode_return = 0.0
    rt_max, rt_min = torch.max(torch.tensor(trajs_rt)), torch.min(
        torch.tensor(trajs_rt)
    )
    reward /= rt_max - rt_min
    reward *= 1000.0
    return reward


###### Embedding #######

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


###### EMA #######

class EMA:
    """
    empirical moving average
    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new



###### Sampling #######

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def linear_beta_schedule(
    timesteps, beta_start=1e-4, beta_end=2e-2, dtype=torch.float32
):
    betas = np.linspace(beta_start, beta_end, timesteps)
    return torch.tensor(betas, dtype=dtype)


def vp_beta_schedule(timesteps, dtype=torch.float32):
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.0
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T**2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype=dtype)



###### Losses #######

class WeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, targ, weights=1.0):
        """
        pred, targ : tensor [ batch_size x action_dim ]
        """
        loss = self._loss(pred, targ)
        weighted_loss = (loss * weights).mean()
        return weighted_loss


class WeightedL1(WeightedLoss): # Mean Absolute Error
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss): # Mean Squared Error
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


Losses = {
    "l1": WeightedL1,
    "l2": WeightedL2,
}



###### Diffusion Model #######

class Progress:
    def __init__(
        self,
        total,
        name="Progress",
        ncol=3,
        max_length=20,
        indent=0,
        line_width=100,
        speed_update_freq=100,
    ):
        self.total = total
        self.name = name
        self.ncol = ncol
        self.max_length = max_length
        self.indent = indent
        self.line_width = line_width
        self._speed_update_freq = speed_update_freq

        self._step = 0
        self._prev_line = "\033[F"
        self._clear_line = " " * self.line_width

        self._pbar_size = self.ncol * self.max_length
        self._complete_pbar = "#" * self._pbar_size
        self._incomplete_pbar = " " * self._pbar_size

        self.lines = [""]
        self.fraction = "{} / {}".format(0, self.total)

        self.resume()

    def update(self, description, n=1):
        self._step += n
        if self._step % self._speed_update_freq == 0:
            self._time0 = time.time()
            self._step0 = self._step
        self.set_description(description)

    def resume(self):
        self._skip_lines = 1
        print("\n", end="")
        self._time0 = time.time()
        self._step0 = self._step

    def pause(self):
        self._clear()
        self._skip_lines = 1

    def set_description(self, params=[]):
        if type(params) == dict:
            params = sorted([(key, val) for key, val in params.items()])

        self._clear()

        percent, fraction = self._format_percent(self._step, self.total)
        self.fraction = fraction

       
        speed = self._format_speed(self._step)


        num_params = len(params)
        nrow = math.ceil(num_params / self.ncol)
        params_split = self._chunk(params, self.ncol)
        params_string, lines = self._format(params_split)
        self.lines = lines

        description = "{} | {}{}".format(percent, speed, params_string)
        print(description)
        self._skip_lines = nrow + 1

    def append_description(self, descr):
        self.lines.append(descr)

    def _clear(self):
        position = self._prev_line * self._skip_lines
        empty = "\n".join([self._clear_line for _ in range(self._skip_lines)])
        print(position, end="")
        print(empty)
        print(position, end="")

    def _format_percent(self, n, total):
        if total:
            percent = n / float(total)

            complete_entries = int(percent * self._pbar_size)
            incomplete_entries = self._pbar_size - complete_entries

            pbar = (
                self._complete_pbar[:complete_entries]
                + self._incomplete_pbar[:incomplete_entries]
            )
            fraction = "{} / {}".format(n, total)
            string = "{} [{}] {:3d}%".format(fraction, pbar, int(percent * 100))
        else:
            fraction = "{}".format(n)
            string = "{} iterations".format(n)
        return string, fraction

    def _format_speed(self, n):
        num_steps = n - self._step0
        t = time.time() - self._time0
        speed = num_steps / t
        string = "{:.1f} Hz".format(speed)
        if num_steps > 0:
            self._speed = string
        return string

    def _chunk(self, l, n):
        return [l[i : i + n] for i in range(0, len(l), n)]

    def _format(self, chunks):
        lines = [self._format_chunk(chunk) for chunk in chunks]
        lines.insert(0, "")
        padding = "\n" + " " * self.indent
        string = padding.join(lines)
        return string, lines

    def _format_chunk(self, chunk):
        line = " | ".join([self._format_param(param) for param in chunk])
        return line

    def _format_param(self, param):
        k, v = param
        return "{} : {}".format(k, v)[: self.max_length]

    def stamp(self):
        if self.lines != [""]:
            params = " | ".join(self.lines)
            string = "[ {} ] {}{} | {}".format(
                self.name, self.fraction, params, self._speed
            )
            self._clear()
            print(string, end="\n")
            self._skip_lines = 1
        else:
            self._clear()
            self._skip_lines = 0

    def close(self):
        self.pause()


class Silent:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, attr):
        return lambda *args: None

class Diffusion(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        model,
        max_action,
        beta_schedule="linear",
        n_timesteps=100,
        loss_type="l2",
        clip_denoised=True,
        predict_epsilon=True,
    ):
        super(Diffusion, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.model = model

        if beta_schedule == "linear":
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == "vp":
            betas = vp_beta_schedule(n_timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))

        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, x, t, s):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # @torch.no_grad()
    def p_sample_loop(self, state, shape, verbose=False, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        if return_diffusion:
            diffusion = [x]

        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)

            progress.update({"t": i})

            if return_diffusion:
                diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    # @torch.no_grad()
    def sample(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, shape, *args, **kwargs)
        return action.clamp_(-self.max_action, self.max_action)

    # # @ \torch.no_grad()
    # def log_prob(self,state):
    #     batch_size = state.shape[0]

    #     return log_probs

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t, weights)

    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)




###### Main Model #######


class MLP(nn.Module):
    """
    MLP Model
    """

    def __init__(self, state_dim, action_dim, device, t_dim=16):
        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
        )

        self.final_layer = nn.Linear(256, action_dim)

    def forward(self, x, time, state):
        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)


####### Networks #######

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

        self.q2_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class DiffCPS(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        discount,
        tau,
        max_q_backup=False,
        LA=1.0,
        beta_schedule="linear",
        n_timesteps=100,
        ema_decay=0.995,
        step_start_ema=1000,
        update_ema_every=5,
        lr=3e-4,
        lr_decay=False,
        lr_maxt=1000,
        grad_norm=1.0,
        # policy_noise=0.2,
        # noise_clip=0.1,
        policy_freq=10,
        target_kl=0.05,
        LA_max=100,
        LA_min=0,
    ):
        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)

        self.actor = Diffusion(
            state_dim=state_dim,
            action_dim=action_dim,
            model=self.model,
            max_action=max_action,
            beta_schedule=beta_schedule,
            n_timesteps=n_timesteps,
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every
        # self.policy_noise = policy_noise
        # self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.LA = torch.tensor(LA, dtype=torch.float).to(device)  # Lambda
        self.LA_min = LA_min
        self.LA_max = LA_max

        self.LA.requires_grad = True
        self.LA_optimizer = torch.optim.Adam([self.LA], lr=3e-5)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(
                self.actor_optimizer, T_max=lr_maxt, eta_min=0.0
            )
            self.critic_lr_scheduler = CosineAnnealingLR(
                self.critic_optimizer, T_max=lr_maxt, eta_min=0.0
            )
            self.lambda_lr_scheduler = CosineAnnealingLR(
                self.LA_optimizer, T_max=lr_maxt, eta_min=0.0
            )

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau

        self.target_kl = target_kl
        self.device = device
        self.max_q_backup = max_q_backup

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(self, replay_buffer, iterations, batch_size=100):
        metric = {
            "kl_loss": [],
            # "ql_loss": [],
            "actor_loss": [],
            "critic_loss": [],
            "Lambda": [],
        }
        for iteration in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(
                batch_size
            )

            """ Q Training """
            current_q1, current_q2 = self.critic(state, action)

            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                next_action_rpt = self.ema_model(next_state_rpt)

                next_action_rpt = (next_action_rpt).clamp(
                    -self.max_action, self.max_action
                )
                target_q1, target_q2 = self.critic_target(
                    next_state_rpt, next_action_rpt
                )
                target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q = torch.min(target_q1, target_q2)
            else:
                next_action = (self.ema_model(next_state)).clamp(
                    -self.max_action, self.max_action
                )
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)

            target_q = (reward + not_done * self.discount * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
                current_q2, target_q
            )

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(
                self.critic.parameters(), max_norm=self.grad_norm, norm_type=2
            )
            self.critic_optimizer.step()

            # training policy every policy_freq steps

            if self.step % self.policy_freq == 0:
                """Policy Training"""
                # print(state.shape)
                kl_loss = self.actor.loss(action, state)
                new_action = self.actor(state)

                q1_new_action, q2_new_action = self.critic(state, new_action)
                if np.random.uniform() > 0.5:
                    q_loss = -q1_new_action.mean() / q2_new_action.abs().mean().detach()
                else:
                    q_loss = -q2_new_action.mean() / q1_new_action.abs().mean().detach()
                # q_loss = - q1_new_action.mean()
                actor_loss = (
                    self.LA.clamp(self.LA_min, self.LA_max).detach() * kl_loss + q_loss
                )

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                # if self.grad_norm > 0:
                actor_grad_norms = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), max_norm=self.grad_norm, norm_type=2
                )
                self.actor_optimizer.step()

                """ Lambda loss"""

                LA_loss = (self.target_kl - kl_loss).detach() * self.LA
                self.LA_optimizer.zero_grad()
                LA_loss.backward()
                # if self.grad_norm > 0:
                LA_grad_norms = nn.utils.clip_grad_norm_(
                    self.LA, max_norm=self.grad_norm, norm_type=2
                )
                self.LA_optimizer.step()

                metric["actor_loss"].append(actor_loss.item())
                metric["kl_loss"].append(kl_loss.item())
                # metric["ql_loss"].append(q_loss.item())
                metric["critic_loss"].append(critic_loss.item())
                metric["Lambda"].append(self.LA.clamp(self.LA_min, self.LA_max).item())

            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            self.step += 1

            # Print metrics every 10 iterations
            if (iteration + 1) % 10 == 0:
                print(f"DiffCPS Iteration {iteration + 1}/{iterations}")
                print(f"Actor Loss: {metric['actor_loss'][-1]:.4f}")
                print(f"Critic Loss: {metric['critic_loss'][-1]:.4f}")
                print(f"KL Loss: {metric['kl_loss'][-1]:.4f}")
                print(f"Lambda: {metric['Lambda'][-1]:.4f}")
                print("--------------------")

        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            self.lambda_lr_scheduler.step()

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # print(state.shape)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        # print(state_rpt.shape)
        with torch.no_grad():
            action = self.actor.sample(state_rpt)
            # print(action.shape)
            q_value = self.critic_target.q_min(state_rpt, action).flatten()
            idx = torch.multinomial(F.softmax(q_value), 1)
            # print(idx.shape)
            # print(action[idx].cpu().data.numpy().flatten())
            # print(action[idx].cpu().data.numpy().flatten().shape)
            """
            Returns a tensor where each row contains num_samples indices sampled from the multinomial 
            probability distribution located in the corresponding row of tensor input.
            """
        return action[idx].cpu().data.numpy().flatten()

    def save_model(self):
            torch.save(self.actor.state_dict(), f"DiffCPS_actor.pth")
            torch.save(self.critic.state_dict(), f"DiffCPS_critic.pth")

    def load_model(self):
            self.actor.load_state_dict(torch.load(f"DiffCPS_actor.pth"))
            self.critic.load_state_dict(torch.load(f"DiffCPS_critic.pth"))



#### Early Stopping (optional) ####

class EarlyStopping(object):
    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                return True
        else:
            self.counter = 0
        return False


##### Policy Evaluation #####

def evaluate_policy(policy, env, eval_episodes=10):
    avg_reward = 0.
    all_rewards = []
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.sample_action(np.array(obs))
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
            action = policy.sample_action(np.array(obs))
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




###### Training Loop ######

def train_agent(env, state_dim, action_dim, max_action, device,reward_tune,discount,tau,max_q_backup,beta_schedule,T,LA,lr,lr_decay,gn,policy_freq,target_kl,lambda_max,lambda_min,num_steps_per_epoch,batch_size,env_name,eval_freq):
    # Load buffer
    dataset = d4rl.qlearning_dataset(env)
    data_sampler = Data_Sampler(dataset, device, reward_tune)
    

    agent = DiffCPS(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        discount=discount,
        tau=tau,
        max_q_backup=max_q_backup,
        beta_schedule=beta_schedule,
        n_timesteps=T,
        LA=LA,
        lr=lr,
        lr_decay=lr_decay,
        lr_maxt=eval_freq, # num_epochs
        grad_norm=gn,
        policy_freq=policy_freq,
        target_kl=target_kl,
        LA_max=lambda_max,
        LA_min=lambda_min,
    )

    early_stop = False
    if early_stop:
        stop_check = EarlyStopping(tolerance=1, min_delta=0.0)
    
    grad_steps = 0
    max_grad_steps = eval_freq * num_steps_per_epoch
    average_returns = []
    while (grad_steps  < max_grad_steps ) and (not early_stop):
        
        loss_metric = agent.train(
            data_sampler,
            iterations=int(eval_freq),
            batch_size=batch_size,     
        )
        grad_steps += int(eval_freq)
        curr_epoch = int(grad_steps  // int(num_steps_per_epoch))

        # Evaluation
        ret_eval, std_ret, median_ret, d4rl_score = normalized_evaluate_policy(agent, env, eval_episodes=10)
        average_returns.append(ret_eval)
          
        kl_loss = np.mean(loss_metric["kl_loss"])
        actor_loss = np.mean(loss_metric["actor_loss"])
        critic_loss = np.mean(loss_metric["critic_loss"])

        print(f"Main Loop Epoch: {curr_epoch}")
        print(f"Average Return: {ret_eval:.2f} ± {std_ret:.2f}")
        print(f"Median Return: {median_ret:.2f}")
        print(f"D4RL Score: {d4rl_score:.2f}")
        print(f"KL Loss: {kl_loss:.4f}")
        print(f"Actor Loss: {actor_loss:.4f}")
        print(f"Critic Loss: {critic_loss:.4f}")
      

    # Save the model
    agent.save_model()
    print("Policy saved successfully.")

    #Calculate average, std and median of the training loop
    avg_return = np.mean(average_returns)
    std_return = np.std(average_returns)
    median_return = np.median(average_returns)
    with open('DiffCPS_training_results.txt', 'w') as f:
        f.write(f"Training Results for {env_name}:\n")
        f.write(f"Average Return: {avg_return:.2f}\n")
        f.write(f"Standard Deviation: {std_return:.2f}\n")
        f.write(f"Median Return: {median_return:.2f}\n")
        f.write(f"D4RL Score: {d4rl_score:.2f}\n")

    print("DiffCPS Training results have been written to 'DiffCPS_training_results.txt'")

    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(average_returns) + 1), average_returns, marker='o')
    plt.xlabel('Training Iterations')
    plt.ylabel('Average Return')
    plt.title('Training Curve: Average Return vs Training Iterations')
    plt.grid(True)
    plt.savefig('DiffCPS_training_curve.png')
    plt.close()



if __name__ == "__main__":
    
    device = 0
    env_name = "halfcheetah-random-v2"
    seed = 0
    num_steps_per_epoch = 1000
    eval_freq = 1000 
    batch_size = 100
    lr_decay = False
    discount = 0.99
    tau = 0.005
    T = 5
    beta_schedule = "vp"
    lambda_max = 100
    lambda_min = 0

    device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    

    
    lr = hyperparameters[env_name]["lr"]
    LA = hyperparameters[env_name]["lambda"]
    policy_freq = hyperparameters[env_name]["freq"]
    max_q_backup = hyperparameters[env_name]["max_q_backup"]
    reward_tune = hyperparameters[env_name]["reward_tune"]
    gn = hyperparameters[env_name]["gn"]
    
    
    # default kl, comment it when you set yourself kl parameters
    lambda_min = hyperparameters[env_name]["lambda_min"]
    target_kl = hyperparameters[env_name]["target_kl"]

    env = gym.make(env_name)

    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    #thread = 8
    #torch.set_num_threads(int(thread))

    # Train the agent
    train_agent(env, state_dim, action_dim, max_action, device,reward_tune,discount,tau,max_q_backup,beta_schedule,T,LA,lr,lr_decay,gn,policy_freq,target_kl,lambda_max,lambda_min,num_steps_per_epoch,batch_size,env_name,eval_freq)
    

     # Create a new agent instance
    agent = DiffCPS(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        discount=0.99,  # You may want to use the same parameters as during training
        tau=0.005,
        max_q_backup=False,
        beta_schedule="vp",
        n_timesteps=5,
        LA=1.0,
        lr=3e-4,
        lr_decay=False,
        grad_norm=1.0,
        policy_freq=2,
        target_kl=0.05,
        LA_max=100,
        LA_min=0,
    )

     # Load the saved model
    agent.load_model()
    print("Model loaded successfully.")

    # Test the loaded policy
    avg_reward, std_reward, all_rewards = test_policy(agent, env, num_episodes=10)
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")

    with open('DiffCPS_test_results.txt', 'w') as f:
        f.write(f"Test Results for {env_name}:\n")
        f.write(f"Average Return: {avg_reward:.2f}\n")
        f.write(f"Standard Deviation: {std_reward:.2f}\n")
        f.write(f"Total Rewards: {all_rewards}\n")

    # Save test rewards
    np.save('DiffCPS_test_rewards.npy', all_rewards)
    # Load test rewards for figure
    loaded_total_rewards = np.load('DiffCPS_test_rewards.npy')

    # Plot test rewards
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loaded_total_rewards) + 1), loaded_total_rewards, marker='o')
    plt.xlabel('Test Episodes')
    plt.ylabel('Total Reward')
    plt.title('Test Rewards: Total Reward vs Test Episodes')
    plt.grid(True)
    plt.savefig('DiffCPS_test_rewards.png')
    plt.show()
    plt.close()

print("Done")
    