import gym_envs
from imitation.data import types
from common.rollouts import generate_trajectories_from_data
from stable_baselines3.common.vec_env import DummyVecEnv
from scipy import io

if __name__ == '__main__':
    env_type = "HPC"
    n_steps, n_episodes = 600, 5
    env_id = f"{env_type}_custom-v0"
    env = DummyVecEnv([lambda: gym_envs.make(env_id, n_steps=n_steps)])
    trajectories = []
    subi = 1
    sub = f"sub{subi:02d}"
    for i in range(10):
        file = "HPC/" + sub + "/" + sub + f"i{i+1}.mat"
        data = {'state': io.loadmat(file)['state'],
                'T': io.loadmat(file)['tq'],
                'pltq': io.loadmat(file)['pltq'],
                'bsp': io.loadmat(file)['bsp'],
                }
        trajectories += generate_trajectories_from_data(data, env)
    save_name = f"{env_type}/{sub}_1&2.pkl"
    types.save(save_name, trajectories)
    print("Expert Trajectories are saved")
