import os
import time
import pickle
import torch as th

from matplotlib import pyplot as plt
from matplotlib import gridspec
from imitation.data.rollout import flatten_trajectories, make_sample_until, types
from scipy import io, signal

from algos.tabular.viter import FiniteSoftQiter
from common.util import make_env, CPU_Unpickler
from common.rollouts import generate_trajectories_without_shuffle
from common.wrappers import *
from IRL.src import *

irl_path = os.path.abspath("..")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def draw_trajs(subj="sub01", actuation=1, learned_trial=1):
    def feature_fn(x):
        t, dt, u = th.split(x, 2, 1)
        prev_u = th.cat([th.zeros(1, 2), u], dim=0)
        u_diff = u - prev_u[:-1]
        return th.cat([t, dt, u, u_diff], dim=-1)
        # return x ** 2
        # return th.cat([x ** 2, x ** 4], dim=-1)

    env_type = "HPC"
    name = f"{env_type}_custom"
    expt = f"full/{subj}_{actuation}"
    bsp = io.loadmat(os.path.join(irl_path, "demos", "HPC", subj, subj + "i1.mat"))['bsp']
    with open(f"{irl_path}/demos/{env_type}/{expt}.pkl", "rb") as f:
        expert_trajs = pickle.load(f)
    init_states = []
    pltqs = []
    for traj in expert_trajs:
        pltqs.append(traj.pltq)
        init_states.append(traj.obs[0])

    load_dir = f"{irl_path}/tmp/log/{name}/MaxEntIRL/xx_001alpha_{expt}_{learned_trial}/model/150"
    with open(load_dir + "/reward_net.pkl", "rb") as f:
        reward_fn = CPU_Unpickler(f).load().to('cpu')
    reward_fn.feature_fn = feature_fn
    # reward_fn.len_act_w = 1
    venv = make_env(f"{name}-v2", num_envs=1, bsp=bsp, init_states=init_states, pltqs=pltqs,
                    wrapper=RewardWrapper, wrapper_kwargs={'rwfn': reward_fn.eval()})

    agent = IDPDiffLQRPolicy(venv, gamma=1., alpha=0.001, device='cpu', verbose=False)
    agent.learn(0)
    print(reward_fn.reward_layer.weight.square())
    eval_venv = make_env(f"{name}-v0", num_envs=1, bsp=bsp, init_states=init_states, pltqs=pltqs)

    # for finite src
    agent_trajs = []
    for init_state in init_states:
        # traj = DiscEnvTrajectories()
        obs, acts, rews = agent.predict(init_state, deterministic=True)
        data_dict = {'obs': obs, 'acts': acts, 'infos': None}
        traj = types.Trajectory(**data_dict)
        agent_trajs.append(traj)

    # for infinite src
    # sample_until = make_sample_until(n_timesteps=None, n_episodes=len(init_states))
    # agent_trajs = generate_trajectories_without_shuffle(agent, eval_venv, sample_until, deterministic_policy=True)

    expt_obs = flatten_trajectories(expert_trajs).obs
    expt_acts = flatten_trajectories(expert_trajs).acts
    agent_obs = flatten_trajectories(agent_trajs).obs
    agent_acts = flatten_trajectories(agent_trajs).acts

    print(f"Mean obs difference ({subj}_{actuation}): {np.abs(expt_obs - agent_obs).mean(axis=0)}")
    print(f"Mean acts difference ({subj}_{actuation}): {np.abs(expt_acts - agent_acts).mean(axis=0)}")

    if plotting:
        t = np.arange(0, 360) * venv.get_attr("dt")
        for ob_idx in range(4):
            ax = fig.add_subplot(inner_grid[ob_idx])
            for traj_idx in range(len(expert_trajs)):
                ax.plot(t, expert_trajs[traj_idx].obs[:-1, ob_idx], color='b')
                ax.plot(t, agent_trajs[traj_idx].obs[:-1, ob_idx], color='k')
            if ob_idx == 3:
                ax.legend(['human', 'controller'], fontsize=11)
            ax.tick_params(axis='both', which='major', labelsize=11)
        for act_idx in range(2):
            ax = fig.add_subplot(inner_grid[act_idx + 4])
            for traj_idx in range(len(expert_trajs)):
                ax.plot(t, expert_trajs[traj_idx].acts[:, act_idx] * 300, color='b')
                ax.plot(t, agent_trajs[traj_idx].acts[:, act_idx] * 300, color='k')
            ax.tick_params(axis='both', which='major', labelsize=11)
    # fig.axes[1].set_ylim([-0.02, 0.03])
    # fig.axes[2].set_ylim([-0.1, 0.2])
    # fig.axes[3].set_ylim([-40, 40])
    if rendering:
        for i in range(len(expert_trajs)):
            done = False
            obs = eval_venv.reset()
            while not done:
                act, _ = agent.predict(obs, deterministic=False)
                obs, rew, done, _ = eval_venv.step(act)
                eval_venv.render()
                time.sleep(eval_venv.get_attr("dt")[0])
        eval_venv.close()


def compare_x1_vs_x2(subj, actuation, learned_trial):
    def feature_fn(x):
        return x ** 2

    env_type = "HPC"
    name = f"{env_type}_custom"
    expt = f"uncropped/{subj}_{actuation}"
    bsp = io.loadmat(os.path.join(irl_path, "demos", "HPC", subj, subj + "i1.mat"))['bsp']

    load_dir = f"{irl_path}/tmp/log/{name}/MaxEntIRL/sq_001alpha_{expt}_{learned_trial}"
    with open(f"{load_dir}/{subj}_{actuation}.pkl", "rb") as f:
        expert_trajs = pickle.load(f)

    pltqs = []
    init_states = []
    for traj in expert_trajs:
        init_states.append(traj.obs[0])
        pltqs.append(traj.pltq)

    with open(load_dir + "/model/reward_net.pkl", "rb") as f:
        reward_fn = CPU_Unpickler(f).load()
    reward_fn.feature_fn = feature_fn
    reward_fn.len_act_w = 2

    venv = make_env(f"{name}-v0", bsp=bsp, init_states=init_states, pltqs=pltqs, num_envs=1,
                    wrapper=RewardWrapper, wrapper_kwargs={'rwfn': reward_fn.eval()})
    agent = IDPLQRPolicy(venv, gamma=1, alpha=0.001)

    sample_until = make_sample_until(n_timesteps=None, n_episodes=len(expert_trajs))
    agent_trajs = generate_trajectories_without_shuffle(
        agent, venv, sample_until, deterministic_policy=True,
    )

    if plotting:
        ax1 = fig.add_subplot(inner_grid[0])
        ax2 = fig.add_subplot(inner_grid[1])
        for traj in expert_trajs:
            ax1.plot(traj.obs[:, 0], traj.obs[:, 1])
        for traj in agent_trajs:
            ax2.plot(traj.obs[:, 0], traj.obs[:, 1])


if __name__ == "__main__":
    rendering = False
    plotting = True
    save_figs = False
    figs = []
    for subj in [f"sub{i:02d}" for i in [5]]:
        fig = plt.figure(figsize=[7.35, 7.25])
        outer_grid = gridspec.GridSpec(1, 1, wspace=0.3, hspace=0.2)
        outer_grid_idx = 0
        for actuation in range(3, 4):
            for learn_trial in range(1, 2):
                ax_outer = fig.add_subplot(outer_grid[outer_grid_idx])
                # ax_outer.set_title(f"{subj}_{actuation}_{learn_trial}", fontsize=24)
                ax_outer.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
                ax_outer.axis('off')
                inner_grid = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=outer_grid[outer_grid_idx], wspace=0.2, hspace=0.2)
                draw_trajs(subj, actuation, learn_trial)
                outer_grid_idx += 1
        fig.subplots_adjust(left=0.08, bottom=0.05, right=0.99, top=0.98)
        if save_figs:
            fig_name = f"figures/HPC_custom/MaxEntIRL/sq_005alpha_afpert/sub05_trajs_1_1.png"
            os.makedirs(os.path.dirname(fig_name), exist_ok=True)
            fig.savefig(fig_name)
        figs.append(fig)
    if plotting:
        plt.show()
