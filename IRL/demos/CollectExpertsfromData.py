from imitation.data import types
from common.rollouts import generate_trajectories_from_data
from common.util import make_env
from scipy import io
import numpy as np

if __name__ == '__main__':
    env_type = "HPC"
    env_id = f"{env_type}_custom-v0"
    env = make_env(env_id)
    act_coeff = env.model.actuator_gear[0, 0]
    trajectories = []
    subi = 1
    sub = f"sub{subi:02d}"
    for i in range(35):
        file = f"HPC/{sub}/{sub}i{i+1}.mat"
        state = io.loadmat(file)['state']
        state[0, 4:] = np.array([0, 0])
        state[1:, 4:] = (state[1:, 2:4] - state[:-1, 2:4]) / env.dt
        # state[:, 4:] = state[:, 4:]
        data = {'state': state,
                'T': io.loadmat(file)['tq'] / act_coeff,
                'pltq': io.loadmat(file)['pltq'] / act_coeff,
                'bsp': io.loadmat(file)['bsp'],
                }
        trajectories += generate_trajectories_from_data(data, env)
    save_name = f"{env_type}/{sub}_acc.pkl"
    types.save(save_name, trajectories)
    print("Expert Trajectories are saved")
