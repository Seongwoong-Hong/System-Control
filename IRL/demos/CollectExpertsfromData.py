from imitation.data import types
from common.rollouts import generate_trajectories_from_data
from common.util import make_env
from scipy import io
import numpy as np

if __name__ == '__main__':
    env_type = "DiscretizedHuman"
    env_id = f"{env_type}-v2"
    env = make_env(env_id, N=[11, 17, 17, 17])
    act_coeff = 1
    # act_coeff = env.model.actuator_gear[0, 0]
    for subi in [1, 4, 5, 6, 7, 9, 10]:
        # subi = 3
        for actuation in [1, 2, 3, 4, 5, 6, 7]:
            # actuation = 3
            sub = f"sub{subi:02d}"
            for trial in range(1, 6):
                trajectories = []
                for part in range(6):
                    file = f"HPC/{sub}_cropped/{sub}i{5 * (actuation - 1) + trial}_{part}.mat"
                    state = io.loadmat(file)['state'][:, :4]
                    state = env.get_obs_from_idx(env.get_idx_from_obs(state))
                    T = env.get_act_from_idx(env.get_idx_from_act(io.loadmat(file)['tq']))
                    data = {'state': state,
                            'T': T / act_coeff,
                            'pltq': io.loadmat(file)['pltq'] / act_coeff,
                            'bsp': io.loadmat(file)['bsp'],
                            }
                    trajectories += generate_trajectories_from_data(data, env)
                save_name = f"{env_type}/11171717/{sub}_{actuation}_{trial}.pkl"
                types.save(save_name, trajectories)
                print(f"Expert Trajectory {save_name} is saved")
