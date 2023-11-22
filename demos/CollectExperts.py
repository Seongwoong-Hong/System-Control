import os
import pickle
from scipy import io, signal

from imitation.data import rollout, types

from gym_envs.envs import DataBasedDiscretizationInfo, FaissDiscretizationInfo, UncorrDiscretizationInfo
from common.util import make_env
from common.wrappers import *
from common.rollouts import generate_trajectories_without_shuffle, TrajectoryWithPltqs
from RL.src import IPLQRPolicy


def main():
    # env_op = 0.1
    env_type = "IP"
    env_id = f"{env_type}_custom"
    subpath = (Path(env_type) / subj / subj)
    states = [np.nan for _ in range(35)]
    for i in range(6, 11):
        humanData = io.loadmat(str(subpath) + f"i{i}.mat")
        bsp = humanData['bsp']
        states[i - 1] = humanData['state']
    env = make_env(f"{env_id}-v2", bsp=bsp, humanStates=states)

    ExpertPolicy = IPLQRPolicy(env, gamma=1, alpha=0.02, device='cpu')
    # trajectories = []
    # for _ in range(1500):
    #     init_state = venv.reset()
    #     obs, acts, rews = ExpertPolicy.predict(init_state, deterministic=False)
    #     data_dict = {'obs': obs, 'acts': acts, 'rews': rews.flatten(), 'infos': None}
    #     traj = types.TrajectoryWithRew(**data_dict)
    #     trajectories.append(traj)
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=10)
    gen_trajs = generate_trajectories_without_shuffle(ExpertPolicy, env, sample_until, deterministic_policy=True)
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
