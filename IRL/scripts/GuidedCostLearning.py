import datetime
import os
import pickle
import shutil
import torch as th
from scipy import io, signal

from imitation.util import logger

from common.util import make_env
from common.callbacks import SaveCallback
from common.wrappers import *
from algos.torch.MaxEntIRL import GuidedCostLearning, ContMaxEntIRL
from algos.torch.sac import MlpPolicy, SAC
from algos.torch.OptCont import DiscreteLQRPolicy


class IDPLQRPolicy(DiscreteLQRPolicy):
    def _build_env(self):
        I1, I2 = self.env.envs[0].model.body_inertia[1:, 0]
        l1 = self.env.envs[0].model.body_pos[2, 2]
        lc1, lc2 = self.env.envs[0].model.body_ipos[1:, 2]
        m1, m2 = self.env.envs[0].model.body_mass[1:]
        g = 9.81
        M = np.array([[I1 + m1*lc1**2 + I2 + m2*l1**2 + 2*m2*l1*lc2 + m2*lc2**2, I2 + m2*l1*lc2 + m2*lc2**2],
                      [I2 + m2*l1*lc2 + m2*lc2**2, I2 + m2*lc2**2]])
        C = np.array([[m1*lc1*g + m2*l1*g + m2*g*lc2, m2*g*lc2],
                      [m2*g*lc2, m2*g*lc2]])
        # self.A, self.B = np.zeros([4, 4]), np.zeros([4, 2])
        # self.A[:2, 2:] = np.eye(2, 2)
        self.A[2:, :2] = np.linalg.inv(M) @ C
        self.A[2:, :2] = np.array([])
        self.B[2:, :] = np.linalg.inv(M) @ np.eye(2, 2)
        # self.B[2:, :] = np.array([[1, -7.7], [0, 24.5]])
        self.A, self.B, _, _, dt = signal.cont2discrete((self.A, self.B, np.array([1, 1, 1, 1]), 0), self.env.envs[0].dt)
        # self.max_t = self.env.get_attr("spec")[0].max_episode_steps // 2
        self.max_t = 400
        self.Q = np.diag([1.9139, 1.5872182, 2.0639979, 1.9540204])
        self.R = np.diag([.01, .01])
        self.gear = 100


if __name__ == "__main__":
    env_type = "2DTarget"
    algo_type = "ContMaxEntIRL"
    device = "cuda:1"
    subj, actu = "sub06", 1
    expt = f"sac_1"
    name = f"{env_type}"
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    subpath = os.path.join(proj_path, "demos", env_type, subj, subj)

    # Load data
    expert_dir = os.path.join(proj_path, "demos", env_type, f"{expt}.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    expt_traj_num = len(expert_trajs)
    # bsp = io.loadmat(subpath + f"i1.mat")['bsp']
    init_states = []
    for traj in expert_trajs:
        init_states += [traj.obs[0]]

    env = make_env(f"{name}-v0", init_states=init_states)
    eval_env = make_env(f"{name}-v0", init_states=init_states)

    # Setup log directories
    log_dir = os.path.join(proj_path, "tmp", "log", name, algo_type)
    log_dir += f"/sq_{expt}_sac_noreset"
    os.makedirs(log_dir, exist_ok=False)
    shutil.copy(os.path.abspath(__file__), log_dir)
    shutil.copy(expert_dir, log_dir)

    def feature_fn(x):
        # return x
        return x ** 2
        # return th.cat([x, x**2], dim=1)

    model_dir = os.path.join(log_dir, "model")
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # Setup callbacks
    save_net_callback = SaveCallback(cycle=1, dirpath=model_dir)

    # Setup Logger
    logger.configure(log_dir, format_strs=["stdout", "tensorboard"])

    # Setup Learner
    agent = SAC(
        MlpPolicy,
        env=env,
        gamma=0.99,
        ent_coef='auto',
        target_entropy=-0.1,
        tau=0.01,
        buffer_size=int(1e4),
        learning_starts=2000,
        train_freq=1,
        gradient_steps=1,
        device=device,
        verbose=1,
        policy_kwargs={'net_arch': [32, 32]}
    )

    learner = ContMaxEntIRL(
        env,
        eval_env=eval_env,
        feature_fn=feature_fn,
        agent=agent,
        expert_trajectories=expert_trajs,
        use_action_as_input=True,
        rew_arch=[],
        device=device,
        env_kwargs={'vec_normalizer': None, 'reward_wrapper': RewardWrapper},
        rew_kwargs={'type': 'ann', 'scale': 1,
                    'optim_kwargs': {'weight_decay': 0.01, 'lr': 1e-2, 'betas': (0.9, 0.999)}
                    },
    )

    # Run Learning
    learner.learn(
        total_iter=300,
        agent_learning_steps=3e4,
        n_episodes=expt_traj_num,
        max_agent_iter=1,
        min_agent_iter=1,
        max_gradient_steps=1,
        min_gradient_steps=1,
        early_stop=True,
    )

    # Save the result of learning
    reward_path = model_dir + "/reward_net.pkl"
    with open(reward_path + ".tmp", "wb") as f:
        pickle.dump(learner.reward_net, f)
    os.replace(reward_path + ".tmp", reward_path)
    learner.agent.save(model_dir + "/agent")
    if learner.agent.get_vec_normalize_env():
        learner.wrap_env.save(model_dir + "/normalization.pkl")
    now = datetime.datetime.now()
    print(f"Endtime: {now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}")
