import os
import numpy as np
from scipy import io
from common.util import make_env
from algos.torch.ppo import PPO
from common.analyzer import exec_policy


def reward_fn(obs, acts, human_obs, human_acts=None):
    obs_diff = obs - human_obs
    rews = (np.exp(np.sum(-200 * obs_diff[:, :2] ** 2)) + 0.2*np.exp(np.sum(-1 * obs_diff[:, 2:] ** 2)) - 0.1*np.sum(acts**2) + 0.1).squeeze()
    rews -= 1/((np.abs(acts[:, 0]).squeeze() - 1)**2 + 1e-6)
    return [rews]


if __name__ == "__main__":
    env_type = "IDP"
    env_id = f"{env_type}_custom"
    subj = "sub04"
    trial = 15
    isPseudo = True
    use_norm = True
    policy_num = 1
    tmp_num = 3
    PDgain = np.array([1000, 100])
    name_tail = f"_DeepMimic_actionSkip_ptb3to3/PD{PDgain[0]}{PDgain[1]}_ankLim"

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
    rews = reward_fn(env.unnormalize_obs(obs[0][:-1]), acts[0] / 100, states[trial - 1])
    human_rews = reward_fn(states[trial - 1], acts[0] / 100, states[trial - 1])
    print(np.sum(rews[0]), np.sum(human_rews[0]))
