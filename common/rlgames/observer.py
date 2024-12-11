import torch
from matplotlib import pyplot as plt


class PlayerObserver:
    def before_init(self, base_name, config, experiment_name):
        pass

    def before_run(self):
        pass

    def before_play(self):
        pass

    def after_play(self):
        pass

    def after_run(self):
        pass

    def after_init(self, algo):
        pass

    def process_infos(self, infos, done_indices):
        pass

    def after_steps(self):
        pass

    def after_print_stats(self, frame, epoch_num, total_time):
        pass


class MultiObserverPlayer(PlayerObserver):
    """Meta-observer that allows the user to add several observers."""

    def __init__(self, observers_):
        super().__init__()
        self.observers = observers_

    def _call_multi(self, method, *args_, **kwargs_):
        for o in self.observers:
            getattr(o, method)(*args_, **kwargs_)

    def before_init(self, base_name, config, experiment_name):
        self._call_multi('before_init', base_name, config, experiment_name)

    def before_run(self):
        self._call_multi('before_run')

    def before_play(self):
        self._call_multi('before_play')

    def after_play(self):
        self._call_multi('after_play')

    def after_run(self):
        self._call_multi('after_run')

    def after_init(self, algo):
        self._call_multi('after_init', algo)

    def process_infos(self, infos, done_indices):
        self._call_multi('process_infos', infos, done_indices)

    def after_steps(self):
        self._call_multi('after_steps')

    def after_clear_stats(self):
        self._call_multi('after_clear_stats')

    def after_print_stats(self, frame, epoch_num, total_time):
        self._call_multi('after_print_stats', frame, epoch_num, total_time)


class DrawTimeTrajObserver(PlayerObserver):
    """Allows us to log stats from the env along with the algorithm running stats. """
    def __init__(self):
        super().__init__()
        self.algo = None
        self.obs = None
        self.acts = None
        self.rews = None
        self.dones = None
        self.infos = {}
        self.fig = None

    def before_run(self):
        if self.fig is None:
            raise AttributeError("figure 가 정의되지 않았습니다.")
        pass

    def before_play(self):
        obs = torch.concat([self.algo.env.dof_pos, self.algo.env.dof_vel], dim=-1)
        self.obs = obs[None, ...].clone().cpu()
        if hasattr(self.algo.env, "actions"):
            acts = torch.clamp(self.algo.env.actions, min=-1.0, max=1.0)
        else:
            acts = self.algo.get_action(obs, self.algo.is_deterministic)
        self.acts = acts[None, ...].clone().cpu()
        self.rews = self.algo.env.rew_buf[None, ...].clone().cpu()
        for k, v in self.algo.env.extras.items():
            self.infos[k] = v[None, ...].clone().cpu()

    def after_play(self):
        for i in range(self.obs.shape[-1]):
            self.fig.axes[i].plot(self.obs[1:, 0, i])
        for i in range(self.acts.shape[-1]):
            self.fig.axes[i + self.obs.shape[-1]].plot(self.acts[1:, 0, i])
        self.fig.axes[-1].plot(self.rews[1:])

    def after_run(self):
        self.fig.tight_layout()
        if self.show_fig:
            self.fig.show()

    def after_init(self, algo):
        self.algo = algo
        self.show_fig = self.algo.player_config.get("show_fig", False)

    def after_steps(self):
        obs = torch.concat([self.algo.env.dof_pos, self.algo.env.dof_vel], dim=-1)
        self.obs = torch.concat([self.obs, obs[None, ...].clone().cpu()], dim=0)
        if hasattr(self.algo.env, "actions"):
            acts = torch.clamp(self.algo.env.actions, min=-1.0, max=1.0)
        else:
            acts = self.algo.get_action(obs, self.algo.is_deterministic)
        self.acts = torch.concat([self.acts, acts[None, ...].clone().cpu()], dim=0)
        self.rews = torch.concat([self.rews, self.algo.env.rew_buf[None, ...].clone().cpu()], dim=0)
        for k, v in self.algo.env.extras.items():
            if k == 'time_outs':
                continue
            self.infos[k] = torch.concat([self.infos[k], v[None, ...].clone().cpu()], dim=0)