from typing import Callable

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import random

def evaluate(
    env_id: str,
    model_path: str,
    envs, 
    eval_episodes: int,
    Model: nn.Module,
    save_path: str,
    global_episode: int,
    num_of_events: int,
    device: torch.device = torch.device("cpu"),
    
):
    # envs = gym.vector.SyncVectorEnv([make_env(env_id, 99, 0, capture_video, run_name)])
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    torch.backends.cudnn.deterministic = True
    actor = Model[0](envs).to(device)
    qf1 = Model[1](envs).to(device)
    qf2 = Model[2](envs).to(device)
    actor_params, qf1_params, qf2_params = torch.load(model_path, map_location=device)
    actor.load_state_dict(actor_params)
    actor.eval()
    qf1.load_state_dict(qf1_params)
    qf1.eval()
    qf2.load_state_dict(qf2_params)
    qf2.eval()
    # note: qf is not used in this script

    obs, _ = envs.reset()
    episodic_returns = []
    metric_list = []
    step = 0
    episode = 0
    prob_list = []
    if env_id == 'sewer-v6':
        bounds = [0.2, 0.4, 0.6, 0.8]
    elif env_id == 'sewer-v7':
        bounds = [1/3, 2/3]
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            actions, _, _, probs = actor.get_action(torch.Tensor(obs).to(device), bounds=bounds)
            actions = actions.detach().cpu().numpy()
            # actions = actions.tolist()
            # actions.append(probs)
        # print(type(actions), actions)
        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs
        prob_list.append(probs)

        if step % (151*num_of_events) == (151*num_of_events - 1):
        # if step % 451 == 449:
        # if step % 151 == 149:
        # if step % 751 == 749:
            metric = envs.call('metric_calculation')
            envs.call('plot', save_path, f'test_{global_episode}_{episode}', prob_list)
            episode += 1


        step += 1

    return episodic_returns, metric


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    from cleanrl.ddpg_continuous_action import Actor, QNetwork, make_env

    model_path = hf_hub_download(
        repo_id="cleanrl/HalfCheetah-v4-ddpg_continuous_action-seed1", filename="ddpg_continuous_action.cleanrl_model"
    )
    evaluate(
        model_path,
        make_env,
        "HalfCheetah-v4",
        eval_episodes=10,
        run_name=f"eval",
        Model=(Actor, QNetwork),
        device="cpu",
        capture_video=False,
    )

