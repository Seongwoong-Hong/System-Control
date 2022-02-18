import os
import time
import pickle
import numpy as np
import torch as th

from matplotlib import pyplot as plt
from imitation.data.rollout import flatten_trajectories, make_sample_until
from scipy import io

from algos.torch.ppo import PPO
from algos.torch.sac import SAC
from algos.tabular.viter import FiniteSoftQiter, SoftQiter
from common.util import make_env, CPU_Unpickler
from common.rollouts import DiscEnvTrajectories
from common.wrappers import *

irl_path = os.path.abspath("..")


def compare_obs(subj="sub01", actuation=1, learned_trial=1):
    rendering = False
    plotting = True

    def feature_fn(x):
        # return x
        return x ** 2
        # return th.cat([x, x ** 2], dim=1)

    env_type = "DiscretizedHuman"
    name = f"{env_type}"
    expt = f"17171719_quadcost/{subj}_{actuation}"
    bsp = io.loadmat(os.path.join(irl_path, "demos", "HPC", subj, subj + "i1.mat"))['bsp']
    with open(f"{irl_path}/demos/{env_type}/{expt}.pkl", "rb") as f:
        expert_trajs = pickle.load(f)
    # expert_trajs = []
    # for lt in range((learned_trial - 1) * 5, learned_trial * 5):
    #     for t in range(3):
    #         traj = DiscEnvTrajectories()
    #         mat = io.loadmat(f"{irl_path}/demos/HPC/{subj}_half/{subj}i{lt + 1}_{t}.mat")
    #         traj.obs = -mat['state'][:, :4]
    #         traj.acts = -mat['tq']
    #         expert_trajs.append(traj)
    init_states = []
    for traj in expert_trajs:
        init_states.append(traj.obs[0])

    load_dir = f"{irl_path}/tmp/log/{env_type}/MaxEntIRL/sq_handnorm_finite_{expt}_{learned_trial}/model"
    with open(load_dir + "/reward_net.pkl", "rb") as f:
        reward_fn = CPU_Unpickler(f).load().to('cpu')
    reward_fn.feature_fn = feature_fn
    # env = make_env(f"{name}-v2", num_envs=1, wrapper=RewardWrapper, wrapper_kwrags={'rwfn': reward_fn.eval()})
    env = make_env(f"{name}-v2", N=[17, 17, 17, 19], NT=[11, 11], bsp=bsp,
                   wrapper=RewardInputNormalizeWrapper, wrapper_kwrags={'rwfn': reward_fn.eval()})
    d_env = make_env(env, num_envs=1, wrapper=DiscretizeWrapper)

    agent = FiniteSoftQiter(d_env, gamma=1., alpha=0.01, device='cpu', verbose=False)
    agent.learn(0)

    eval_env = make_env(f"{name}-v0", N=[17, 17, 17, 19], NT=[11, 11], bsp=bsp, init_states=init_states,
                        wrapper=RewardInputNormalizeWrapper, wrapper_kwrags={'rwfn': reward_fn.eval()})
    # eval_env = make_env(f"{name}-v0", num_envs=1, init_states=init_states)

    agent_trajs = []
    for init_state in init_states:
        traj = DiscEnvTrajectories()
        obs, acts, rews = agent.predict(init_state, deterministic=False)
        traj.obs = obs
        traj.acts = acts
        traj.rews = rews
        agent_trajs.append(traj)

    # expt_obs = flatten_trajectories(expert_trajs).obs
    # expt_acts = flatten_trajectories(expert_trajs).acts
    # agent_obs = flatten_trajectories(agent_trajs).obs
    # agent_acts = flatten_trajectories(agent_trajs).acts
    #
    # print(f"Mean obs difference ({subj}_{actuation}): {np.abs(expt_obs - agent_obs).mean(axis=0)}")
    # print(f"Mean acts difference ({subj}_{actuation}): {np.abs(expt_acts - agent_acts).mean(axis=0)}")

    if plotting:
        x_value = range(1, 51)
        obs_fig = plt.figure(figsize=[27, 18], dpi=100.0)
        acts_fig = plt.figure(figsize=[27, 9], dpi=100.0)
        for ob_idx in range(4):
            ax = obs_fig.add_subplot(2, 2, ob_idx + 1)
            for traj_idx in range(len(expert_trajs)):
                ax.plot(x_value, agent_trajs[traj_idx].obs[:-1, ob_idx], color='k')
                ax.plot(x_value, expert_trajs[traj_idx].obs[:-1, ob_idx], color='b')
            ax.legend(['agent', 'expert'], fontsize=28)
            ax.tick_params(axis='both', which='major', labelsize=24)
        for act_idx in range(2):
            ax = acts_fig.add_subplot(1, 2, act_idx + 1)
            for traj_idx in range(len(expert_trajs)):
                ax.plot(x_value, agent_trajs[traj_idx].acts[:, act_idx], color='k')
                ax.plot(x_value, expert_trajs[traj_idx].acts[:, act_idx], color='b')
            ax.legend(['agent', 'expert'], fontsize=28)
            ax.tick_params(axis='both', which='major', labelsize=24)
        obs_fig.tight_layout()
        acts_fig.tight_layout()
        plt.show()

    if rendering:
        for i in range(len(expert_trajs)):
            done = False
            obs = eval_env.reset()
            while not done:
                act, _ = agent.predict(obs, deterministic=False)
                obs, rew, done, _ = eval_env.step(act)
                eval_env.render()
                time.sleep(eval_env.get_attr("dt")[0])
        eval_env.close()


def feature():
    from imitation.data.rollout import flatten_trajectories, make_sample_until, generate_trajectories
    from common.wrappers import ActionWrapper
    env_type = "HPC"
    env_id = f"{env_type}_custom"
    subj = "sub01"
    name = f"extcnn_{subj}_reset_weightnorm"
    i = 25
    print(name)
    proj_path = os.path.join("..", "tmp", "log", env_id, "BC", name)
    with open(f"../demos/{env_type}/{subj}.pkl", "rb") as f:
        expert_trajs = pickle.load(f)
    expt_trans = flatten_trajectories(expert_trajs)
    test_len = len(expert_trajs)
    subpath = os.path.join("..", "demos", env_type, subj)
    wrapper = ActionWrapper if env_type == "HPC" else None
    venv = make_env(f"{env_id}-v0", num_envs=1, wrapper=wrapper, subpath=subpath + f"/{subj}")
    expt_input = th.from_numpy(np.concatenate([expt_trans.obs, expt_trans.acts], axis=1))
    sample_until = make_sample_until(n_timesteps=None, n_episodes=test_len)
    agent = SAC.load(os.path.join(proj_path, "model", f"{i:03d}", "agent"))
    agent_trajs = generate_trajectories(agent, venv, sample_until=sample_until, deterministic_policy=False)
    agent_trans = flatten_trajectories(agent_trajs)
    agent_input = th.from_numpy(np.concatenate([agent_trans.obs, agent_trans.acts], axis=1))
    with open(os.path.join(proj_path, "model", f"{i:03d}", "reward_net.pkl"), "rb") as f:
        reward_fn = pickle.load(f).double()
    print("env")


def compare_handtune_result_and_irl_result(subj="sub06", actuation=1, learned_trial=1):
    def feature_fn(x):
        return x ** 2

    plotting = True

    env_type = "DiscretizedHuman"
    name = f"{env_type}"
    expt = f"19171717_quadcost/{subj}_{actuation}"
    bsp = io.loadmat(os.path.join(irl_path, "demos", "HPC", subj, subj + "i1.mat"))['bsp']
    with open(f"{irl_path}/demos/{env_type}/{expt}.pkl", "rb") as f:
        expert_trajs = pickle.load(f)
    init_states = []
    for traj in expert_trajs:
        init_states.append(traj.obs[0])

    load_dir = f"{irl_path}/tmp/log/{env_type}/MaxEntIRL/sq_{expt}_finite_normalize_{learned_trial}/model"
    with open(load_dir + "/reward_net.pkl", "rb") as f:
        reward_fn = CPU_Unpickler(f).load()
    reward_fn.feature_fn = feature_fn

    irl_env = make_env(f"{name}-v2", num_envs=1, N=[19, 17, 17, 17], bsp=bsp,
                       wrapper=RewardInputNormalizeWrapper, wrapper_kwrags={'rwfn': reward_fn.eval()})
    irl_agent = FiniteSoftQiter(irl_env, gamma=1, alpha=0.01, device='cpu', verbose=False)
    irl_agent.learn(0)

    hand_env = make_env(f"{name}-v2", num_envs=1, N=[19, 17, 17, 17], bsp=bsp)
    hand_agent = FiniteSoftQiter(hand_env, gamma=1, alpha=0.0001, device='cpu', verbose=False)
    hand_agent.learn(0)

    irl_obs, irl_acts = [], []
    hand_obs, hand_acts = [], []
    for init_state in init_states:
        i_ob, i_act, i_rew = irl_agent.predict(init_state, deterministic=False)
        h_ob, h_act, h_rew = hand_agent.predict(init_state, deterministic=False)
        irl_obs.append(i_ob)
        irl_acts.append(i_act)
        hand_obs.append(h_ob)
        hand_acts.append(h_act)

    irl_reward_mat = irl_env.env_method("get_reward_mat")[0]
    hand_reward_mat = hand_env.env_method("get_reward_mat")[0]
    print(f"Mean reward difference: {np.abs(irl_reward_mat - hand_reward_mat).mean()}")
    print(f"Mean & std of reward: {irl_reward_mat.mean()}, {irl_reward_mat.std()}")
    print(f"Mean obs difference ({subj}_{actuation}): {np.abs(np.vstack(irl_obs) - np.vstack(hand_obs)).mean()}")
    print(f"Mean acts difference ({subj}_{actuation}): {np.abs(np.vstack(irl_acts) - np.vstack(hand_acts)).mean()}")

    if plotting:
        x_value = range(1, 51)
        obs_fig = plt.figure(figsize=[18, 12], dpi=150.0)
        acts_fig = plt.figure(figsize=[18, 6], dpi=150.0)
        for ob_idx in range(4):
            ax = obs_fig.add_subplot(2, 2, ob_idx + 1)
            for traj_idx in range(len(expert_trajs)):
                ax.plot(x_value, hand_obs[traj_idx][:, ob_idx], color='k')
                ax.plot(x_value, irl_obs[traj_idx][:, ob_idx], color='b')
        for act_idx in range(2):
            ax = acts_fig.add_subplot(1, 2, act_idx + 1)
            for traj_idx in range(len(expert_trajs)):
                ax.plot(x_value, hand_acts[traj_idx][:, act_idx], color='k')
                ax.plot(x_value, irl_acts[traj_idx][:, act_idx], color='b')
        obs_fig.tight_layout()
        acts_fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    for subj in [f"sub{i:02d}" for i in [6]]:
        for actuation in range(1, 2):
            for learn_trial in range(1, 3):
                compare_obs(subj, actuation, learn_trial)