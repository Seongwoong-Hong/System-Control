import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3.common.logger import Figure
from imitation.util import logger


class LogFigureCallback:
    def __init__(self, cycle):
        self.cycle = cycle
        self.calling = 0

    def log_figure(self, network, itr):
        if itr % self.cycle == 0:
            fig = plt.figure()
            t = np.arange(0, 360) / 120
            for i in range(4):
                ax = fig.add_subplot(3, 2, i+1)
                for traj in network.agent_trajectories:
                    ax.plot(t, traj.obs[:-1, i], 'k')
                for traj in network.expert_trajectories:
                    ax.plot(t, traj.obs[:-1, i], 'b')
            for i in range(2):
                ax = fig.add_subplot(3, 2, i+5)
                for traj in network.agent_trajectories:
                    ax.plot(t, traj.acts[:, i], 'k')
                for traj in network.expert_trajectories:
                    ax.plot(t, traj.acts[:, i], 'b')
            with logger.accumulate_means("figure"):
                logger.record(
                    "trajectory",
                    Figure(fig, close=True),
                    exclude=("stdout", "log", "json", "csv")
                )
            logger.dump(self.calling)
            self.calling += 1
            plt.close()
