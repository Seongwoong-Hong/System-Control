import os
import numpy as np
from scipy import io

from algos.torch.ppo import PPO
from common.util import make_env
from common.analyzer import exec_policy


def reward_fn(obs, acts, human_obs, human_acts=None):
    obs_diff = (obs - human_obs) / np.abs(human_obs).max(axis=0)
    rews = (np.exp(np.sum(-20*obs_diff[:, :1]**2, axis=1)) + 0.2*np.exp(np.sum(-2*obs_diff[:, 1:]**2, axis=1)) + 0.1).squeeze()
    # rews[0] -= 0.5e-5 / ((np.clip(np.abs(0.0 - acts[0]), 0.0, 0.1) - 0.1) ** 2 + 1e-5)
    # rews[1:] -= 0.5e-5 / ((np.clip(np.abs(acts[:-1] - acts[1:]), 0.0, 0.1) - 0.1) ** 2 + 1e-5)
    rews -= 1e-5/((np.abs(acts[:]).squeeze() - 1)**2 + 1e-5)
    return [rews]


if __name__ == "__main__":
    env_type = "IP"
    env_id = f"{env_type}_MimicHuman"
    subj = "sub04"
    trial = 1
    isPseudo = False
    use_norm = True
    policy_num = 1
    tmp_num = 12
    PDgain = np.array([1000, 200])
    name_tail = f"_DeepMimic_ptb1to4/PD1000200_ankLim"

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

    env = make_env(f"{env_id}-v0", bsp=bsp, humanStates=states, use_norm=norm_pkl_path, PDgain=PDgain)

    agent = PPO.load(model_dir + f"/agent_{tmp_num}")

    obs, acts, nrews, _, tqs = exec_policy(env, agent, render="None", deterministic=True, repeat_num=1)
    if use_norm:
        obs = [env.unnormalize_obs(obs[0])]
    rews = reward_fn(obs[0][:-1], tqs[0] / 100, states[trial - 1])
    human_rews = reward_fn(states[trial - 1], tqs[0] / 100, states[trial - 1])
    print(np.sum(rews[0]), np.sum(human_rews[0]))
