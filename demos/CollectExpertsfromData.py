from scipy import io
from imitation.data import types
from common.sb3.rollouts import TrajectoryWithPltqs
from common.sb3.util import make_env

if __name__ == '__main__':
    env_type = "IDP"
    env_id = f"{env_type}_custom-v2"
    act_coeff = 100
    env = make_env(f"{env_id}")
    # act_coeff = env.model.actuator_gear[0, 0]
    for subi in range(1, 11):
        # subi = 3
        sub = f"sub{subi:02d}"
        for actuation in range(1, 8):
            trajectories = []
            for exp_trial in range(1, 6):
                # for part in range(0):
                file = f"IDP/{sub}_full/{sub}i{5 * (actuation - 1) + exp_trial}.mat"
                states = io.loadmat(file)['state'][None, ...]
                Ts = io.loadmat(file)['tq'][None, ...]
                for idx in range(len(states)):
                    state = states[idx, :, :2].copy(order='C')
                    T = Ts[idx, :].copy(order='C') / act_coeff
                    pltq = io.loadmat(file)['pltq'] / act_coeff
                    trajectories += [TrajectoryWithPltqs(obs=state, acts=T, infos=None, pltq=pltq)]
            save_name = f"{env_type}/full/{sub}_{actuation}.pkl"
            types.save(save_name)
            print(f"Expert Trajectory {save_name} is saved")
