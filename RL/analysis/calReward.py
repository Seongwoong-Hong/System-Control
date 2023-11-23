import os
import numpy as np
from scipy import io
from common.util import make_env
from algos.torch.ppo import PPO
from common.analyzer import exec_policy


def reward_fn(obs, acts, human_obs, human_acts=None):
    obs_diff = obs - human_obs
    rews = np.exp(-100/np.linalg.norm(human_obs[:, 0]) * obs_diff[:, 0] ** 2) \
        + 0.2*np.exp(-10/np.linalg.norm(human_obs[:, 1]) * obs_diff[:, 1] ** 2) \
        - 0.1*acts[:, 0]**2 + 0.1
    return [rews]


if __name__ == "__main__":
    env_type = "IP"
    env_id = f"{env_type}_custom"
    subj = "sub04"
    trial = 11
    isPseudo = True
    use_norm = True
    policy_num = 1
    tmp_num = 13
    name_tail = "_DeepMimic_actionSkip_ptb3/PD1000200"

    if isPseudo:
        env_type = "Pseudo" + env_type
    subpath = os.path.join("../..", "demos", env_type, subj, subj)
    states = [None for _ in range(35)]
    humanData = io.loadmat(subpath + f"i{trial}.mat")
    bsp = humanData['bsp']
    states[trial - 1] = humanData['state']
    if use_norm:
        env_type += "_norm"
        model_dir = os.path.join("..", "scripts", "tmp", "log", f"{env_type}", "ppo" + name_tail, f"policies_{policy_num}")
        norm_pkl_path = model_dir + f"/normalization_{tmp_num}.pkl"
    else:
        model_dir = os.path.join("..", "scripts", "tmp", "log", f"{env_type}", "ppo" + name_tail, f"policies_{policy_num}")
        norm_pkl_path = False

    env = make_env(f"{env_id}-v0", bsp=bsp, humanStates=states, use_norm=norm_pkl_path)

    agent = PPO.load(model_dir + f"/agent_{tmp_num}")

    obs, acts, nrews, _ = exec_policy(env, agent, render="rgb_array", deterministic=True, repeat_num=1)
    rews = reward_fn(env.unnormalize_obs(obs[0][:-1]), acts[0] / 200, states[trial - 1])
    human_rews = reward_fn(states[trial - 1], acts[0] / 200, states[trial - 1])
    print(np.sum(rews[0]), np.sum(human_rews[0]))
