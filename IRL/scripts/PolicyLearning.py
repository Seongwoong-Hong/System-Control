import os
import pickle
import torch as th
import numpy as np

from stable_baselines3.common import callbacks
from stable_baselines3.common.vec_env import VecNormalize
from imitation.policies import serialize
from imitation.data.rollout import make_sample_until, flatten_trajectories
from matplotlib import pyplot as plt

from algos.torch.MaxEntIRL import RewardNet
from algos.torch.sac import SAC
from common.callbacks import VideoCallback
from common.util import make_env
from common.wrappers import ActionNormalizeRewardWrapper
from common.rollouts import generate_trajectories_without_shuffle
from scipy import io


def learning_specific_one():
    env_type = "HPC"
    algo_type = "MaxEntIRL"
    name = "sq_sac_linear_ppoagent_mm_reset"
    env_id = f"{env_type}_custom"

    n = 21
    load_dir = os.path.abspath(os.path.join("..", "tmp", "log", env_id, algo_type, name, "model", f"{n:03d}"))

    stats_path = None
    if os.path.isfile(load_dir + "/normalization.pkl"):
        stats_path = load_dir + "/normalization.pkl"

    with open(load_dir + "/reward_net.pkl", "rb") as f:
        reward_net = pickle.load(f).double()

    env = make_env(f"{env_id}-v1", subpath="../demos/HPC/sub01/sub01", num_envs=1,
                   wrapper=ActionNormalizeRewardWrapper, wrapper_kwrags={'rwfn': reward_net.eval()},
                   use_norm=stats_path)

    device = "cpu"
    log_dir = os.path.join(load_dir, "add_rew_learning")

    algo_used = "sac"
    algo = def_policy(algo_used, env, device=device, log_dir=log_dir, verbose=1)
    from algos.torch.sac import SAC
    # prev_algo = PPO.load(os.path.abspath(os.path.join(load_dir, "..", f"{n - 1:03d}", "agent")))
    # algo.policy.load_from_vector(prev_algo.policy.parameters_to_vector())
    os.makedirs(log_dir + f"/{algo_used}_policies_{n:03d}", exist_ok=True)
    save_policy_callback = serialize.SavePolicyCallback(log_dir + f"/{algo_used}_policies_{n:03d}", None)
    save_policy_callback = callbacks.EveryNTimesteps(int(3e5), save_policy_callback)
    for i in range(15):
        algo.learn(total_timesteps=int(3e4), tb_log_name="extra", callback=save_policy_callback, reset_num_timesteps=False)
        algo.save(log_dir + f"/{algo_used}_policies_{n:03d}/{algo_used}{i}")
    print(f"saved as {algo_used}_policies_{n}/{algo_used}")


def learning_whole_iter():
    env_type = "IDP"
    algo_type = "MaxEntIRL"
    name = "sq_lqr_ppo"
    env_id = f"{env_type}_custom"
    device = "cuda:1"

    pltqs = []
    for i in range(10):
        file = os.path.abspath("../demos/HPC/sub01/sub01" + f"i{i + 1}.mat")
        pltqs += [io.loadmat(file)['pltq']]

    n = 0
    filename = os.path.abspath(f"../tmp/log/{env_id}/{algo_type}/{name}/model/{n:03d}/reward_net.pkl")
    while os.path.isfile(filename):

        with open(filename, "rb") as f:
            reward_net = pickle.load(f).double()
        env = make_env(f"{env_id}-v1", use_vec_env=False, pltqs=pltqs,
                       wrapper=ActionNormalizeRewardWrapper, wrapper_kwrags={'rwfn': reward_net.eval()})

        proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        log_dir = os.path.join(proj_path, "tmp", "log", env_id, algo_type, name, "add_rew_learning")
        algo = def_policy("sac", env, device=device, log_dir=log_dir, verbose=1)
        os.makedirs(log_dir + f"/policies_{n}", exist_ok=False)
        video_recorder = VideoCallback(make_env(f"{env_id}-v0", use_vec_env=False, pltqs=pltqs),
                                       n_eval_episodes=5,
                                       render_freq=int(1e5))
        save_policy_callback = serialize.SavePolicyCallback(log_dir + f"/policies_{n}", None)
        save_policy_callback = callbacks.EveryNTimesteps(int(1e5), save_policy_callback)
        callback_list = callbacks.CallbackList([video_recorder, save_policy_callback])
        algo.learn(total_timesteps=int(0.25e6), tb_log_name="extra", callback=callback_list)
        algo.save(log_dir + f"/policies_{n}/{algo_type}0")
        print(f"saved as policies_{n}/{algo_type}0")
        n += 1
        filename = os.path.abspath(f"../tmp/log/{env_id}/{algo_type}/{name}/model/{n:03d}/reward_net.pkl")


def reward_learning():
    vec_env = make_env("HPC_pybullet-v0", subpath="../demos/HPC/sub01/sub01",
                       use_vec_env=True, num_envs=1, wrapper="ActionWrapper")
    load_dir = "../tmp/log/HPC_pybullet/MaxEntIRL/cnn_sub01_reset/model/000"
    reward_net = RewardNet(inp=8, arch=[8, 8], feature_fn=feature_fn,
                           use_action_as_inp=True, device='cpu', alpha=0.1).double()
    agent = SAC.load(load_dir + "/agent")
    reward_net.train()
    losses = []
    for _ in range(200):
        sample_until = make_sample_until(n_timesteps=None, n_episodes=35)
        trajectories = generate_trajectories_without_shuffle(agent, vec_env, sample_until, deterministic_policy=False)
        agent_trans = flatten_trajectories(trajectories)
        expert_dir = "../../demos/HPC/sub01.pkl"
        with open(expert_dir, "rb") as f:
            expert_trajs = pickle.load(f)
        expt_trans = flatten_trajectories(expert_trajs)
        acts = vec_env.env_method('action', agent_trans.acts)[0]
        agent_input = th.from_numpy(np.concatenate([agent_trans.obs, acts], axis=1)).double()
        expt_input = th.from_numpy(np.concatenate([expt_trans.obs, expt_trans.acts], axis=1)).double()
        agent_rewards = reward_net(agent_input)
        expt_rewards = reward_net(expt_input)
        loss = (agent_rewards.sum() - expt_rewards.sum()) / 35
        reward_net.optimizer.zero_grad()
        loss.backward()
        print(loss.item())
        losses += [loss.item()]
        reward_net.optimizer.step()
    plt.plot(losses)
    plt.show()


if __name__ == "__main__":
    def feature_fn(x):
        # return x
        return x.square()
        # return th.cat([x, x.square()], dim=1)
    learning_specific_one()
