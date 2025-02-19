# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import torch.optim.lr_scheduler as lr_scheduler
from datetime import datetime
from sac_eval import evaluate


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 99
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    eval_episodes: int = 5
    actor_type: int = 0

    # Algorithm specific arguments
    env_id: str = "sewer-v7"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99999999999999999
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    learning_starts: int = 2000
    """timestep to start learning"""
    policy_lr: float =  3e-3 #3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-2 #1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 2  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    load: bool = False
    model_path: str = r"C:\Users\xul51\cleanrl\runs\sewer-v7__sac_func__99__12_14_00_18/sac_func_9.cleanrl_model"


def make_env(env_id, seed, idx, capture_video, run_name, train_events, train_eval_events, val_events):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, train_fct_s_times=train_events, train_fct_s_times_eval=train_eval_events, val_fct_s_times=val_events)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# LOG_STD_MAX = 2
# LOG_STD_MIN = -5
class Actor_0(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc_mean = nn.Linear(32, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(32, np.prod(env.single_action_space.shape))
        self.dropout = nn.Dropout(p=0.1)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x, bounds=None):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        if bounds is not None:
            cdfs = []
            probs = []
            bounds = torch.tensor(bounds)
            for i in range(len(bounds)):
                bound_normalized = (bounds[i] - self.action_bias) / self.action_scale
                cdfs.append(normal.cdf(torch.atanh(bound_normalized)))
            # print(cdfs)
            for i in range(len(bounds) + 1):
                if i == 0:
                    probs.append(cdfs[0][0].tolist()[0])
                elif i == len(bounds):
                    probs.append(1 - cdfs[-1][0].tolist()[0])
                else:
                    probs.append(cdfs[i][0].tolist()[0] - cdfs[i-1][0].tolist()[0])
            return action, log_prob, mean, probs
        
        return action, log_prob, mean

class Actor_1(nn.Module):
    def __init__(self, env):
        super().__init__()
        action_size = np.prod(env.single_action_space.shape)
        self.fc1_1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc1_2 = nn.Linear(256, 128)
        self.fc1_3 = nn.Linear(128, 64)
        self.fc1_4 = nn.Linear(64, 32)
        self.fc_mean_macro = nn.Linear(32, action_size - 1)
        self.fc_logstd_macro = nn.Linear(32, action_size - 1)
        self.fc2_1 = nn.Linear(action_size - 1, 16)
        self.fc_mean_micro = nn.Linear(16, 1)
        self.fc_logstd_micro = nn.Linear(16, 1)

        self.dropout = nn.Dropout(p=0.1)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1_1(x))
        x = F.relu(self.fc1_2(x))
        x = F.relu(self.fc1_3(x))
        x = F.relu(self.fc1_4(x))
        # x = self.dropout(x)
        mean_macro = self.fc_mean_macro(x)
        log_std_macro = self.fc_logstd_macro(x)
        log_std_macro = torch.tanh(log_std_macro)
        log_std_macro = LOG_STD_MIN_macro + 0.5 * (LOG_STD_MAX_macro - LOG_STD_MIN_macro) * (log_std_macro + 1)  # From SpinUp / Denis Yarats
        x_micro = F.relu(self.fc2_1(mean_macro))
        mean_micro = self.fc_mean_micro(x_micro)
        log_std_micro = self.fc_logstd_micro(x_micro)
        log_std_micro = torch.tanh(log_std_micro)
        log_std_micro = LOG_STD_MIN_micro + 0.5 * (LOG_STD_MAX_micro - LOG_STD_MIN_micro) * (log_std_micro + 1)  # From SpinUp / Denis Yarats
        if mean_macro.shape[1] > 1:
            mean = torch.cat([mean_macro[:, :1], mean_micro, mean_macro[:, 1:]], dim=1)
            log_std = torch.cat([log_std_macro[:, :1], log_std_micro, log_std_macro[:, 1:]], dim=1)
        else:
            mean = torch.cat([mean_macro, mean_micro], dim=1)
            log_std = torch.cat([log_std_macro, log_std_micro], dim=1)

        return mean, log_std

    def get_action(self, x, bounds=None):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        if bounds is not None:
            cdfs = []
            probs = []
            bounds = torch.tensor(bounds)
            for i in range(len(bounds)):
                bound_normalized = (bounds[i] - self.action_bias) / self.action_scale
                cdfs.append(normal.cdf(torch.atanh(bound_normalized)))
            # print(cdfs)
            for i in range(len(bounds) + 1):
                if i == 0:
                    probs.append(cdfs[0][0].tolist()[0])
                elif i == len(bounds):
                    probs.append(1 - cdfs[-1][0].tolist()[0])
                else:
                    probs.append(cdfs[i][0].tolist()[0] - cdfs[i-1][0].tolist()[0])
            return action, log_prob, mean, probs
        
        return action, log_prob, mean


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    Time = datetime.now().strftime("%m_%d_%H_%M")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{Time}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.env_id == "sewer-v6":
        if args.actor_type == 0:
            LOG_STD_MIN = nn.Parameter(torch.tensor([-20.0, -10, -20.0], dtype=torch.float32).to(device))
            LOG_STD_MAX = nn.Parameter(torch.tensor([-2, 0, -1], dtype=torch.float32).to(device))
        elif args.actor_type == 1:
            LOG_STD_MIN_macro = nn.Parameter(torch.tensor([-20.0, -20.0], dtype=torch.float32).to(device))
            LOG_STD_MAX_macro = nn.Parameter(torch.tensor([5.0, 5.0], dtype=torch.float32).to(device))
            LOG_STD_MIN_micro = nn.Parameter(torch.tensor([-10.0], dtype=torch.float32) .to(device))
            LOG_STD_MAX_micro = nn.Parameter(torch.tensor([5.0], dtype=torch.float32).to(device))
        train_events = [datetime(2023, 6, 4, 6, 0), datetime(2023, 6, 23, 6, 0), datetime(2023, 6, 30, 6, 0), datetime(2023, 8, 20, 6, 0), 
                        datetime(2023, 8, 22, 6, 0), datetime(2024, 4, 24, 0, 0), datetime(2024, 7, 8, 11, 0),
                        datetime(2024, 7, 24, 0, 0), datetime(2024, 8, 19, 6, 0), datetime(2024, 8, 23, 6, 0), datetime(2024, 7, 1, 6, 0), 
                        datetime(2024, 6, 2, 0, 0), datetime(2023, 6, 23, 6, 0), datetime(2024, 6, 24, 6, 0), datetime(2024, 3, 31, 3, 0)]
        train_eval_events = [datetime(2023, 6, 4, 6, 0), datetime(2023, 6, 30, 6, 0), datetime(2023, 8, 20, 6, 0), datetime(2024, 3, 31, 3, 0),
                             datetime(2024, 6, 2, 0, 0), datetime(2024, 6, 24, 6, 0), datetime(2024, 7, 1, 6, 0), datetime(2024, 7, 8, 11, 0)]
        val_events = [datetime(2023, 8, 10, 6, 0), datetime(2024, 4, 18, 0, 0), datetime(2024, 6, 23, 0, 0), datetime(2024, 7, 10, 0, 0),
                      datetime(2024, 6, 28, 6, 0), datetime(2023, 5, 22, 12, 0), datetime(2023, 8, 16, 0, 0),]
        # train_events = [datetime(2024, 2, 10, 0, 0), datetime(2024, 3, 26, 0, 0), datetime(2024, 5, 22, 0, 0), datetime(2024, 6, 26, 0, 0), 
        #                 datetime(2024, 8, 21, 0, 0), datetime(2023, 9, 7, 0, 0), datetime(2023, 11, 30, 0, 0), datetime(2023, 12, 17, 0, 0)]
        # val_events = [datetime(2024, 1, 29, 0, 0), datetime(2024, 4, 28, 0, 0), datetime(2024, 7, 19, 0, 0), datetime(2023, 10, 6, 0, 0)]
    elif args.env_id == "sewer-v7":
        if args.actor_type == 0:
            LOG_STD_MIN = nn.Parameter(torch.tensor([-20.0, -20.0], dtype=torch.float32).to(device))
            LOG_STD_MAX = nn.Parameter(torch.tensor([5, 5], dtype=torch.float32).to(device))
        elif args.actor_type == 1:
            LOG_STD_MIN_macro = nn.Parameter(torch.tensor([-20.0], dtype=torch.float32).to(device))
            LOG_STD_MAX_macro = nn.Parameter(torch.tensor([5.0], dtype=torch.float32).to(device))
            LOG_STD_MIN_micro = nn.Parameter(torch.tensor([-10.0], dtype=torch.float32).to(device))
            LOG_STD_MAX_micro = nn.Parameter(torch.tensor([5.0], dtype=torch.float32).to(device))
        # train_events = [datetime(2024, 2, 10, 0, 0), datetime(2024, 3, 26, 0, 0), datetime(2024, 5, 22, 0, 0), datetime(2024, 6, 26, 0, 0), 
        #                 datetime(2024, 8, 21, 0, 0), datetime(2023, 9, 7, 0, 0), datetime(2023, 11, 30, 0, 0), datetime(2023, 12, 17, 0, 0)]
        # val_events = [datetime(2024, 1, 29, 0, 0), datetime(2024, 4, 28, 0, 0), datetime(2024, 7, 19, 0, 0), datetime(2023, 10, 6, 0, 0)]
        # train_events = [datetime(2024, 2, 10, 12, 0), datetime(2024, 5, 22, 12, 0), datetime(2024, 6, 26, 12, 0), 
        #                 datetime(2024, 7, 29, 12, 0), datetime(2024, 8, 21, 12, 0), datetime(2023, 9, 7, 12, 0), datetime(2023, 10, 19, 12, 0), datetime(2023, 11, 30, 12, 0), datetime(2023, 12, 17, 12, 0),
        #                 datetime(2023, 1, 11, 12, 0), datetime(2023, 2, 26, 12, 0), datetime(2023, 3, 13, 12, 0), datetime(2023, 3, 24, 12, 0),
        #                 datetime(2023, 4, 13, 12, 0), datetime(2023, 4, 17, 12, 0), datetime(2023, 5, 1, 12, 0), datetime(2023, 5, 24, 12, 0), datetime(2023, 6, 17, 12, 0),
        #                 datetime(2023, 7, 27, 12, 0), datetime(2023, 8, 12, 12, 0), datetime(2023, 9, 5, 12, 0), datetime(2024, 1, 15, 12, 0),
        #                 datetime(2024, 1, 27, 12, 0), datetime(2024, 2, 5, 12, 0), datetime(2024, 3, 6, 12, 0), datetime(2024, 3, 12, 12, 0), datetime(2024, 4, 30, 12, 0), 
        #                 datetime(2024, 5, 26, 12, 0), datetime(2024, 6, 1, 12, 0),  datetime(2024, 6, 14, 12, 0), datetime(2024, 7, 7, 12, 0), datetime(2024, 8, 11, 12, 0),
        #                 datetime(2024, 8, 27, 12, 0)]   
        # val_events = [datetime(2024, 1, 29, 12, 0), datetime(2024, 4, 28, 12, 0), datetime(2024, 7, 19, 12, 0), datetime(2023, 10, 6, 12, 0), datetime(2023, 2, 24, 12, 0),
        #               datetime(2023, 6, 10, 12, 0), datetime(2023, 1, 25, 12, 0), datetime(2023, 7, 23, 12, 0), datetime(2024, 4, 14, 12, 0), datetime(2024, 5, 14, 12, 0), 
        #               datetime(2023, 8, 27, 12, 0), datetime(2024, 2, 29, 12, 0), datetime(2024, 3, 26, 12, 0)]
        # train_events = [datetime(2023, 1, 11, 12, 0), datetime(2023, 1, 25, 12, 0), datetime(2023, 2, 26, 12, 0), datetime(2023, 3, 13, 12, 0), datetime(2023, 4, 13, 12, 0), datetime(2023, 4, 17, 12, 0), 
        #                 datetime(2023, 5, 24, 12, 0), datetime(2023, 6, 3, 12, 0), datetime(2023, 7, 23, 12, 0), datetime(2023, 7, 27, 12, 0), datetime(2023, 8, 29, 12, 0), datetime(2023, 9, 5, 12, 0),
        #                 datetime(2023, 10, 6, 12, 0), datetime(2023, 10, 19, 12, 0), datetime(2023, 12, 17, 12, 0), datetime(2024, 1, 15, 12, 0), datetime(2024, 1, 29, 12, 0), datetime(2024, 2, 5, 12, 0), 
        #                 datetime(2024, 2, 29, 12, 0), datetime(2024, 3, 6, 12, 0), datetime(2024, 3, 26, 12, 0), datetime(2024, 4, 14, 12, 0), datetime(2024, 4, 30, 12, 0), datetime(2024, 5, 14, 12, 0), 
        #                 datetime(2024, 5, 26, 12, 0), datetime(2024, 6, 3, 12, 0), datetime(2024, 6, 26, 12, 0), datetime(2024, 7, 6, 12, 0), datetime(2024, 7, 29, 12, 0), datetime(2024, 8, 11, 12, 0)]
        # train_eval_events = [datetime(2023, 1, 25, 12, 0), datetime(2023, 5, 24, 12, 0), datetime(2023, 7, 23, 12, 0), datetime(2023, 12, 17, 12, 0), datetime(2024, 3, 6, 12, 0), datetime(2024, 4, 30, 12, 0), datetime(2024, 6, 3, 12, 0)]
        # val_events = [datetime(2023, 2, 24, 12, 0), datetime(2023, 3, 24, 12, 0), datetime(2023, 5, 1, 12, 0), datetime(2023, 6, 17, 12, 0), datetime(2023, 8, 12, 12, 0), datetime(2023, 9, 7, 12, 0), 
        #                     datetime(2023, 11, 29, 12, 0), datetime(2024, 1, 27, 12, 0), datetime(2024, 2, 10, 12, 0), datetime(2024, 3, 12, 12, 0), datetime(2024, 4, 28, 12, 0), datetime(2024, 5, 22, 12, 0), 
        #                     datetime(2024, 6, 14, 12, 0), datetime(2024, 7, 19, 12, 0), datetime(2024, 8, 21, 12, 0), datetime(2024, 8, 27, 12, 0)]
        # test_events = [datetime(2023, 1, 3, 12, 0), datetime(2023, 1, 27, 12, 0), datetime(2023, 2, 25, 12, 0), datetime(2023, 5, 6, 12, 0), datetime(2023, 7, 28, 12, 0), datetime(2023, 10, 31, 12, 0), datetime(2023, 12, 21, 12, 0),
        #             datetime(2023, 11, 15, 12, 0), datetime(2024, 1, 8, 12, 0), datetime(2024, 3, 10, 12, 0), datetime(2024, 4, 16, 12, 0), datetime(2024, 5, 20, 12, 0), datetime(2024, 6, 30, 12, 0), datetime(2024, 7, 27, 12, 0), datetime(2024, 8, 17, 12, 0)]
        train_events = [datetime(2023, 1, 11, 12, 0), datetime(2023, 1, 25, 12, 0), datetime(2023, 2, 26, 12, 0), datetime(2023, 3, 13, 12, 0), datetime(2023, 4, 13, 12, 0), datetime(2023, 4, 17, 12, 0), 
                        datetime(2023, 5, 24, 12, 0), datetime(2023, 6, 3, 12, 0), datetime(2023, 7, 23, 12, 0), datetime(2023, 7, 27, 12, 0), datetime(2023, 8, 29, 12, 0), datetime(2023, 9, 15, 12, 0),
                        datetime(2023, 10, 6, 12, 0), datetime(2023, 10, 19, 12, 0), datetime(2023, 12, 17, 12, 0), datetime(2024, 1, 16, 12, 0), datetime(2024, 1, 29, 12, 0), datetime(2024, 2, 5, 12, 0), 
                        datetime(2024, 2, 29, 12, 0), datetime(2024, 3, 9, 12, 0), datetime(2024, 3, 26, 12, 0), datetime(2024, 4, 14, 12, 0), datetime(2024, 4, 30, 12, 0), datetime(2024, 5, 14, 12, 0), 
                        datetime(2024, 5, 26, 12, 0), datetime(2024, 6, 3, 12, 0), datetime(2024, 6, 26, 12, 0), datetime(2024, 7, 6, 12, 0), datetime(2024, 7, 29, 12, 0), datetime(2024, 8, 11, 12, 0)]
        # train_events = [datetime(2023, 5, 24, 12, 0), datetime(2023, 7, 27, 12, 0), datetime(2023, 8, 29, 12, 0), datetime(2023, 9, 15, 12, 0), datetime(2024, 5, 14, 12, 0), datetime(2024, 6, 3, 12, 0),
        #                 datetime(2024, 6, 26, 12, 0), datetime(2024, 7, 29, 12, 0)]
        train_eval_events = [datetime(2023, 1, 25, 12, 0), datetime(2023, 5, 24, 12, 0), datetime(2023, 7, 23, 12, 0), datetime(2023, 12, 17, 12, 0), datetime(2024, 3, 9, 12, 0), datetime(2024, 4, 30, 12, 0), datetime(2024, 6, 3, 12, 0)]
        val_events = [datetime(2023, 2, 24, 12, 0), datetime(2023, 3, 24, 12, 0), datetime(2023, 5, 1, 12, 0), datetime(2023, 6, 17, 12, 0), datetime(2023, 8, 12, 12, 0), datetime(2023, 9, 7, 12, 0), 
                        datetime(2023, 11, 29, 12, 0), datetime(2024, 1, 27, 12, 0), datetime(2024, 2, 10, 12, 0), datetime(2024, 3, 12, 12, 0), datetime(2024, 4, 28, 12, 0), datetime(2024, 5, 22, 12, 0), 
                        datetime(2024, 6, 14, 12, 0), datetime(2024, 7, 19, 12, 0), datetime(2024, 8, 27, 12, 0)]
        test_events = [datetime(2023, 1, 3, 12, 0), datetime(2023, 1, 27, 12, 0), datetime(2023, 2, 25, 12, 0), datetime(2023, 5, 6, 12, 0), datetime(2023, 7, 28, 12, 0), datetime(2023, 10, 31, 12, 0), datetime(2023, 12, 21, 12, 0),
                datetime(2023, 11, 15, 12, 0), datetime(2024, 1, 8, 12, 0), datetime(2024, 3, 10, 12, 0), datetime(2024, 4, 16, 12, 0), datetime(2024, 5, 20, 12, 0), datetime(2024, 6, 30, 12, 0), datetime(2024, 7, 27, 12, 0), datetime(2024, 8, 17, 12, 0)]
        # train_events = [datetime(2023, 1, 11, 12, 0),datetime(2023, 1, 25, 12, 0)]
        # train_eval_events = [datetime(2023, 1, 25, 12, 0)]
        # val_events = [datetime(2023, 2, 24, 12, 0)]




    # env setup
    # train_events = [datetime(2023, 6, 30, 6, 0)]
    # val_events = [datetime(2023, 8, 10, 6, 0)]
    # wet
    # train_events = [datetime(2023, 6, 30, 6, 0), datetime(2024, 7, 8, 11, 0), datetime(2024, 7, 10, 0, 0)]
    # val_events = [datetime(2023, 8, 10, 6, 0), datetime(2024, 4, 18, 0, 0)]
    # dry
    # train_events = [datetime(2024, 4, 28, 0, 0), datetime(2024, 6, 26, 0, 0), datetime(2024, 7, 19, 0, 0)]
    # val_events = [datetime(2024, 2, 10, 0, 0), datetime(2024, 8, 21, 0, 0)]
    events = train_eval_events + val_events
    metric_type = ['level_metric', 'pump_metric', 'orifice_metric', 'overflow_metric', 'electricity_metric', 'total_metric']
    arrays = [
        [e.strftime("%y%m%d_%H%M") for e in events for _ in range(len(metric_type))],
        metric_type * len(events)
    ]
    columns = pd.MultiIndex.from_arrays(arrays, names=('DateTime', 'metric'))
    metric_df = pd.DataFrame(columns=columns)
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name, train_events, train_eval_events, val_events)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    if args.actor_type == 0:
        actor = Actor_0(envs).to(device)
    elif args.actor_type == 1:
        actor = Actor_1(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    if args.load:
        actor_params, qf1_params, qf2_params = torch.load(args.model_path, map_location=device)
        actor.load_state_dict(actor_params)
        qf1.load_state_dict(qf1_params)
        qf2.load_state_dict(qf2_params)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    # q_optimizer = optim.Adagrad(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    # actor_optimizer = optim.Adagrad(list(actor.parameters()), lr=args.policy_lr)
    # q_optimizer = optim.Adadelta(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    # actor_optimizer = optim.Adadelta(list(actor.parameters()), lr=args.policy_lr)
    # actor_optimizer = optim.RMSprop(list(actor.parameters()), lr=args.policy_lr)
    q_scheduler = lr_scheduler.CosineAnnealingLR(q_optimizer, T_max=20000, eta_min=1e-5)
    actor_scheduler = lr_scheduler.CosineAnnealingLR(actor_optimizer, T_max=20000, eta_min=1e-5)
    



    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        # log_alpha = torch.tensor([torch.log(torch.tensor(3000.0))], requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    reward_log = deque(maxlen=10)
    global_episode = 0
    dir_path = f'C:/Users/xul51/cleanrl/envs/log_plot/{args.env_id}_{args.exp_name}_{Time}'
    os.makedirs(dir_path, exist_ok=True)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)



        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"episode_id={global_episode}, global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                reward_log.append(info["episode"]["r"])
                global_episode += 1
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            # print(data.next_observations.shape)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()
            q_scheduler.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    actor_scheduler.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # if global_step % 301 == 299:
            # if global_step % 451 == 449:
            if global_step % (151*int(len(train_events)*2/3)) == (151*int(len(train_events)*2/3) - 1):
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", global_step / (time.time() - start_time))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
            
            # if global_step % 1510 == 149:
            # # if global_step % 187 == 185:
            #     print(f'previous 10 episodic reward mean:{np.mean(reward_log)}')
            #     metric = envs.call('metric_calculation')
            #     print(f'episode {int((global_step+1)/151)} metric result: {metric}')
            #     envs.call('plot', dir_path, f'test_{int((global_step+1)/151)}')

            # if args.save_model and (global_episode % 20 == 0 or global_episode == 10) and "final_info" in infos:
            if args.save_model and global_episode % 3 == 0 and "final_info" in infos:
                model_path = f"runs/{run_name}/{args.exp_name}_{global_episode}.cleanrl_model"
                torch.save((actor.state_dict(), qf1.state_dict(), qf2.state_dict()), model_path)
                print(f"model saved to {model_path}")
                envs.call('evaluate_mode', 'on')
                if args.actor_type == 0:
                    episodic_returns, metrics = evaluate(
                                                env_id=args.env_id,
                                                model_path=model_path,
                                                envs=envs,
                                                eval_episodes=1,
                                                save_path=dir_path,
                                                global_episode=global_episode,
                                                Model=(Actor_0, SoftQNetwork, SoftQNetwork),
                                                num_of_events=len(events),
                                                device=device,
                                                )
                elif args.actor_type == 1:
                    episodic_returns, metrics = evaluate(
                                                env_id=args.env_id,
                                                model_path=model_path,
                                                envs=envs,
                                                eval_episodes=1,
                                                save_path=dir_path,
                                                global_episode=global_episode,
                                                Model=(Actor_1, SoftQNetwork, SoftQNetwork),
                                                num_of_events=len(events),
                                                device=device,
                                                )
                for i, e in enumerate(events):
                    for j, n in enumerate(metric_type):
                        metric_df.loc[global_episode, (e.strftime("%y%m%d_%H%M"), n)] = metrics[0][i][j]
                metric_df.to_csv(os.path.join(dir_path, 'metrics.csv'))
                print(f'episode {global_episode} evaluate reward {episodic_returns}, metric result {np.round(metrics[0], 2)}')
                envs.call('evaluate_mode', 'off')

                for idx, episodic_return in enumerate(episodic_returns):
                    writer.add_scalar("eval/episodic_return", episodic_return, idx)
            


    envs.close()
    writer.close()
