import pickle
import torch as th
import numpy as np

from imitation.data.rollout import flatten_trajectories, make_sample_until
from algos.torch.MaxEntIRL import RewardNet
from algos.torch.sac import SAC
from common.rollouts import generate_trajectories_without_shuffle
from common.util import make_env


if __name__ == "__main__":
    def feature_fn(x):
        return x

    vec_env = make_env("HPC_pybullet-v0", subpath="../../IRL/demos/HPC/sub01/sub01", use_vec_env=True, num_envs=1,
                       wrapper="ActionWrapper")
    load_dir = "../../IRL/tmp/log/HPC_pybullet/MaxEntIRL/cnn_sub01_reset/model/000"
    reward_net = RewardNet(inp=8, arch=[8, 8], feature_fn=feature_fn, use_action_as_inp=True, device='cpu', alpha=0.1)
    agent = SAC.load(load_dir + "/agent")
    for _ in range(200):
        sample_until = make_sample_until(n_timesteps=None, n_episodes=35)
        trajectories = generate_trajectories_without_shuffle(agent, vec_env, sample_until, deterministic_policy=False)
        agent_trans = flatten_trajectories(trajectories)
        expert_dir = "../../IRL/demos/HPC/sub01.pkl"
        with open(expert_dir, "rb") as f:
            expert_trajs = pickle.load(f)
        expt_trans = flatten_trajectories(expert_trajs)
        acts = vec_env.action(agent_trans.acts)
        agent_input = th.from_numpy(np.concatenate([agent_trans.obs, acts], axis=1)).double()
        expt_input = th.from_numpy(np.concatenate([expt_trans.obs, expt_trans.acts], axis=1)).double()
        agent_rewards = reward_net(agent_input)
        expt_rewards = reward_net(expt_input)
        loss = (agent_rewards - expt_rewards) / 35
        print(loss.item())
        reward_net.optimizer.zero_grad()
        loss.backward()
        reward_net.optimizer.step()