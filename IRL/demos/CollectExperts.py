import os
import pickle
from scipy import io, signal

from imitation.data import rollout, types

from gym_envs.envs import DataBasedDiscretizationInfo, FaissDiscretizationInfo, UncorrDiscretizationInfo
from common.util import make_env
from common.wrappers import *
from common.rollouts import generate_trajectories_without_shuffle, TrajectoryWithPltqs
from IRL.src import IDPLQRPolicy


def main():
    # env_op = 0.1
    env_type = "HPC"
    name = f"{env_type}_custom"
    demo_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    subpath = os.path.join(demo_path, "HPC", subj, subj)
    with open(f"{demo_path}/HPC/full/{subj}_{actuation}.pkl", "rb") as f:
        expt_trajs = pickle.load(f)
    init_states = []
    pltqs = []
    for traj in expt_trajs:
        init_states += [traj.obs[0]]
        pltqs.append(traj.pltq)
    bsp = io.loadmat(subpath + f"i1.mat")['bsp']
    venv = make_env(f"{name}-v0", init_states=init_states, pltqs=pltqs, bsp=bsp, num_envs=25)

    ExpertPolicy = IDPLQRPolicy(venv, gamma=1, alpha=0.02, device='cpu')
    # trajectories = []
    # for _ in range(1500):
    #     init_state = venv.reset()
    #     obs, acts, rews = ExpertPolicy.predict(init_state, deterministic=False)
    #     data_dict = {'obs': obs, 'acts': acts, 'rews': rews.flatten(), 'infos': None}
    #     traj = types.TrajectoryWithRew(**data_dict)
    #     trajectories.append(traj)
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=125)
    gen_trajs = generate_trajectories_without_shuffle(ExpertPolicy, venv, sample_until, deterministic_policy=True)
    trajectories = []
    for idx, traj in enumerate(gen_trajs):
        trajectories.append(TrajectoryWithPltqs(obs=traj.obs, acts=traj.acts, pltq=pltqs[idx // 25], infos=traj.infos))
    save_name = f"{env_type}/quadcost_lqr/{subj}_{actuation}.pkl"
    types.save(save_name, trajectories)
    print(f"Expert Trajectories are saved in the {save_name}")


if __name__ == "__main__":
    for subj in [f"sub{i:02d}" for i in [1]]:
        for actuation in range(1, 7):
            main()
