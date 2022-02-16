from imitation.data import types
from common.rollouts import generate_trajectories_from_data
from common.util import make_env
from scipy import io
import numpy as np

if __name__ == '__main__':
    env_type = "DiscretizedHuman"
    env_id = f"{env_type}-v2"
    env = make_env(env_id, N=[17, 17, 17, 19], NT=[11, 11])
    act_coeff = 1
    # act_coeff = env.model.actuator_gear[0, 0]
    for subi in [1, 2, 4, 5, 6, 7, 9, 10]:
        # subi = 3
        sub = f"sub{subi:02d}"
        for actuation in range(1, 7):
            trajectories = []
            # actuation = 3
            for exp_trial in range(1, 6):
                for part in range(3):
                    file = f"HPC/{sub}_half/{sub}i{5 * (actuation - 1) + exp_trial}_{part}.mat"
                    state = env.get_obs_from_idx(env.get_idx_from_obs(-io.loadmat(file)['state'][:, :4]))
                    T = env.get_acts_from_idx(env.get_idx_from_acts(-io.loadmat(file)['tq'] / act_coeff))
                    data = {'state': state,
                            'T': T,
                            'pltq': -io.loadmat(file)['pltq'] / act_coeff,
                            'bsp': io.loadmat(file)['bsp'],
                            }
                    trajectories += generate_trajectories_from_data(data, env)
            save_name = f"{env_type}/17171719_log1017/{sub}_{actuation}.pkl"
            types.save(save_name, trajectories)
            print(f"Expert Trajectory {save_name} is saved")
