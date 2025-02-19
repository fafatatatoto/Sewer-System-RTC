import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
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
# [39, 118, 325, 641, 715, 864, 31, 930, 645, 575]
rs = 575
random.seed(99)
np.random.seed(rs)
torch.manual_seed(99)
torch.backends.cudnn.deterministic = True

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

env_id = "sewer-v8"
actor_type = 0
device = torch.device("cuda" if torch.cuda.is_available() and True else "cpu")
if env_id == "sewer-v7":
    if actor_type == 0:
        LOG_STD_MIN = nn.Parameter(torch.tensor([-20.0, -10, -20.0], dtype=torch.float32).to(device))
        LOG_STD_MAX = nn.Parameter(torch.tensor([-2, 0, -1], dtype=torch.float32).to(device))
    elif actor_type == 1:
        LOG_STD_MIN_macro = nn.Parameter(torch.tensor([-20.0, -20.0], dtype=torch.float32).to(device))
        LOG_STD_MAX_macro = nn.Parameter(torch.tensor([5.0, 5.0], dtype=torch.float32).to(device))
        LOG_STD_MIN_micro = nn.Parameter(torch.tensor([-10.0], dtype=torch.float32) .to(device))
    #     LOG_STD_MAX_micro = nn.Parameter(torch.tensor([5.0], dtype=torch.float32).to(device))
    # train_eval_events = [datetime(2023, 6, 4, 6, 0), datetime(2023, 6, 23, 6, 0), datetime(2023, 6, 30, 6, 0), datetime(2023, 8, 20, 6, 0), datetime(2024, 3, 31, 3, 0),
    #                          datetime(2024, 6, 2, 0, 0), datetime(2024, 6, 24, 6, 0), datetime(2024, 7, 1, 6, 0), datetime(2024, 7, 8, 11, 0)]
    # train_eval_events = [datetime(2023, 8, 22, 6, 0), datetime(2024, 4, 24, 0, 0),
    #                     datetime(2024, 7, 24, 0, 0), datetime(2024, 8, 19, 6, 0), datetime(2024, 8, 23, 6, 0)]
    train_events = [datetime(2023, 6, 4, 6, 0), datetime(2023, 6, 23, 6, 0), datetime(2023, 6, 30, 6, 0), datetime(2023, 8, 20, 6, 0), 
                        datetime(2023, 8, 22, 6, 0), datetime(2024, 4, 24, 0, 0), datetime(2024, 7, 8, 11, 0),
                        datetime(2024, 7, 24, 0, 0), datetime(2024, 8, 19, 6, 0), datetime(2024, 8, 23, 6, 0), datetime(2024, 7, 1, 6, 0), 
                        datetime(2024, 6, 2, 0, 0), datetime(2024, 6, 24, 6, 0), datetime(2024, 3, 31, 3, 0)]
    val_events = [datetime(2023, 8, 10, 6, 0), datetime(2024, 4, 18, 0, 0), datetime(2024, 6, 23, 0, 0), datetime(2024, 7, 10, 0, 0),
                      datetime(2024, 6, 28, 6, 0), datetime(2023, 5, 22, 12, 0), datetime(2023, 8, 16, 0, 0)]
    test_events = [datetime(2023, 6, 10, 9, 0), datetime(2023, 5, 30, 22), datetime(2023, 6, 8, 18), datetime(2023, 9, 2, 0), datetime(2024, 5, 12, 12, 0), 
               datetime(2024, 6, 16, 6, 0), datetime(2024, 7, 2, 3, 0)]
    # train_events = [datetime(2023, 6, 4, 6, 0), datetime(2023, 6, 30, 6, 0), datetime(2023, 8, 10, 6, 0), datetime(2023, 8, 16, 0, 0), datetime(2024, 7, 10, 0, 0),datetime(2024, 6, 16, 6, 0), datetime(2024, 7, 2, 3, 0)]

elif env_id == "sewer-v8":
    if actor_type == 0:
        LOG_STD_MIN = nn.Parameter(torch.tensor([-20.0, -20.0], dtype=torch.float32).to(device))
        LOG_STD_MAX = nn.Parameter(torch.tensor([5, 5], dtype=torch.float32).to(device))
    elif actor_type == 1:
        LOG_STD_MIN_macro = nn.Parameter(torch.tensor([-20.0], dtype=torch.float32).to(device))
        LOG_STD_MAX_macro = nn.Parameter(torch.tensor([5.0], dtype=torch.float32).to(device))
        LOG_STD_MIN_micro = nn.Parameter(torch.tensor([-10.0], dtype=torch.float32).to(device))
        LOG_STD_MAX_micro = nn.Parameter(torch.tensor([5.0], dtype=torch.float32).to(device))
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

# events = [datetime(2024, 8, 27, 12, 0), datetime(2023, 10, 19, 12, 0), datetime(2023, 6, 17, 12, 0)]
events = train_events + val_events +test_events

if __name__ == "__main__":
    from sac_eval_151 import evaluate
    # from sac_eval import evaluate
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 99, 0, False, '', [datetime(2023, 1, 11, 12, 0)], [], events)])
    model_name = 'sac_func'
    time_name = '11_04_18_06'
    env_name = 'sewer-v7'
    episode = 45
    # time_name = '12_23_20_48'
    # env_name = 'sewer-v8'
    # episode = 24
    model_path = rf"C:\Users\309\cleanrl\runs\{env_name}__{model_name}__99__{time_name}\sac_func_{episode}.cleanrl_model"
    # model_path = r"C:\Users\309\Downloads\sac_func_3.cleanrl_model"
    # save_path = r"C:\Users\309\Downloads\sac_func_3"
    save_path = r"C:\Users\309\Downloads\sac_func_45"
    # save_path = rf"C:/Users/309/cleanrl/envs/log_plot/{model_name}_{time_name}/test"
    # save_path = rf"C:/Users/309/cleanrl/envs/log_plot/{model_name}_{time_name}/imperfect/{rs}"
    os.makedirs(save_path, exist_ok=True)
    if actor_type == 0:
        envs.call('evaluate_mode', 'on')
        episodic_returns, metrics = evaluate(
                                        env_id=env_id,
                                        model_path=model_path,
                                        envs=envs,
                                        eval_episodes=1,
                                        save_path=save_path,
                                        global_episode=1,
                                        Model=(Actor_0, SoftQNetwork, SoftQNetwork),
                                        num_of_events=len(events),
                                        device=device,
                                        )
        print(f'evaluate reward {episodic_returns}, metric result {np.round(metrics[0], 2)}')
        envs.call('evaluate_mode', 'off')
    else:
        pass
