from scipy import io
from imitation.data import types
from gym_envs.envs import FaissDiscretizationInfo
from common.rollouts import TrajectoryWithPltqs
from common.util import make_env

if __name__ == '__main__':
    env_type = "IP_HPC"
    env_id = f"{env_type}-v2"
    act_coeff = 300
    env = make_env(f"{env_id}")
    # act_coeff = env.model.actuator_gear[0, 0]
    for subi in range(5, 6):
        # subi = 3
        sub = f"sub{subi:02d}"
        for actuation in range(4, 5):
            trajectories = []
            for exp_trial in range(1, 6):
                # for part in range(0):
                file = f"HPC/{sub}_full/{sub}i{5 * (actuation - 1) + exp_trial}.mat"
                states = io.loadmat(file)['state'][None, ...]
                Ts = io.loadmat(file)['tq'][None, ...]
                import matplotlib.pyplot as plt
                fig = plt.figure()
                for i in range(6):
                    fig.add_subplot(3, 2, i+1)
                for i in range(4):
                    fig.axes[i].plot(states[0, :, i])
                for j in range(2):
                    fig.axes[j+4].plot(Ts[0, :, j])
                fig.tight_layout()
                fig.show()
                for idx in range(len(states)):
                    state = states[idx, :, :2].copy(order='C')
                    T = Ts[idx, :].copy(order='C') / act_coeff
                    pltq = io.loadmat(file)['pltq'] / act_coeff
                    trajectories += [TrajectoryWithPltqs(obs=state, acts=T, infos=None, pltq=pltq)]
            save_name = f"{env_type}/full/{sub}_{actuation}.pkl"
            types.save(save_name, trajectories)
            print(f"Expert Trajectory {save_name} is saved")
