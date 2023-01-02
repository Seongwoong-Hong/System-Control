import os
import json
import pickle
import shutil
import datetime
import torch as th

from imitation.util import logger
from scipy import io

from common.util import make_env
from common.callbacks import SaveCallback
from common.wrappers import *
from algos.torch.MaxEntIRL import ContMaxEntIRL
from IRL.src import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def main(
        log_dir,
        env,
        eval_env,
        feature_fn,
        expert_trajs,
        use_action_as_input,
        rew_arch,
        env_kwargs,
        rew_kwargs,
        agent_cls,
        agent_kwargs,
        device,
        callback_fns,
):
    # Setup Learner
    agent = agent_cls(**agent_kwargs, device=device)
    learner = ContMaxEntIRL(
        env,
        eval_env=eval_env,
        feature_fn=feature_fn,
        agent=agent,
        expert_trajectories=expert_trajs,
        use_action_as_input=use_action_as_input,
        rew_arch=rew_arch,
        device=device,
        env_kwargs=env_kwargs,
        rew_kwargs=rew_kwargs,
    )

    # Run Learning
    learner.learn(
        total_iter=500,
        agent_learning_steps=0,
        n_episodes=len(expert_trajs),
        max_agent_iter=1,
        min_agent_iter=1,
        max_gradient_steps=1,
        min_gradient_steps=1,
        callback_fns=callback_fns,
        early_stop=True,
    )

    # Save the result of learning
    reward_path = log_dir + "/model/reward_net"
    learner.reward_net.eval().save(reward_path)
    # learner.agent.save(model_dir + "/agent")
    if learner.agent.get_vec_normalize_env():
        learner.wrap_env.save(model_dir + "/normalization.pkl")
    now = datetime.datetime.now()
    print(f"Endtime: {now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}")
    del learner, agent
    if 'cuda' in device:
        with th.cuda.device(device):
            th.cuda.empty_cache()


if __name__ == "__main__":
    env_type = "HPC"
    algo_type = "MaxEntIRL"
    device = "cpu"
    name = f"{env_type}_custom"

    script_args = []
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(f"{proj_path}/demos/bound_info.json", "r") as f:
        bound_info = json.load(f)

    def feature_fn(x):
        t, dt, u = th.split(x, 2, 1)
        prev_u = th.cat([th.zeros(1, 2), u], dim=0)
        u_diff = u - prev_u[:-1]
        return th.cat([t, dt, u, u_diff], dim=-1)
        # return th.cat([x, x ** 2], dim=-1)
    # [1, 5, 6, 7, 9, 10]
    for subj in [f"sub{i:02d}" for i in [5]]:
        for trial in range(1, 5):
            for actu in range(3, 6):
                expt = f"full/{subj}_{actu}"
                subpath = os.path.join(proj_path, "demos", "HPC", subj + "_full", subj)

                # Load data
                expert_path = os.path.join(proj_path, "demos", env_type, f"{expt}.pkl")
                with open(expert_path, "rb") as f:
                    expert_trajs = pickle.load(f)
                bsp = io.loadmat(subpath + f"i1.mat")['bsp']
                pltqs = []
                init_states = []
                for traj in expert_trajs:
                    pltqs += [traj.pltq]
                    init_states += [traj.obs[0]]

                # Define environments
                env = make_env(f"{name}-v2", bsp=bsp, init_states=init_states, pltqs=pltqs)
                eval_env = make_env(f"{name}-v0", bsp=bsp, init_states=init_states, pltqs=pltqs)

                # Setup log directories
                log_dir = os.path.join(proj_path, "tmp", "log", name, algo_type)
                log_dir += f"/xx_001alpha_{expt}_{trial}"
                os.makedirs(log_dir, exist_ok=False)
                shutil.copy(os.path.abspath(__file__), log_dir)
                shutil.copy(expert_path, log_dir)
                model_dir = os.path.join(log_dir, "model")
                if not os.path.isdir(model_dir):
                    os.mkdir(model_dir)

                save_net_callback = SaveCallback(cycle=50, dirpath=model_dir)

                kwargs = {
                    'log_dir': log_dir,
                    'env': env,
                    'eval_env': eval_env,
                    'feature_fn': feature_fn,
                    'expert_trajs': expert_trajs,
                    'use_action_as_input': True,
                    'rew_arch': [],
                    'env_kwargs': {'vec_normalizer': None, 'num_envs': 1, 'reward_wrapper': RewardWrapper},
                    'rew_kwargs': {'type': 'xx',
                                   'optim_kwargs': {'weight_decay': 0.0, 'lr': 3e-2, 'betas': (0.9, 0.999)},
                                   # 'lr_scheduler_cls': th.optim.lr_scheduler.StepLR,
                                   # 'lr_scheduler_kwargs': {'step_size': 1, 'gamma': 0.95}
                                   },
                    'agent_cls': IDPDiffLQRPolicy,
                    'agent_kwargs': {'env': env, 'gamma': 1, 'alpha': 0.001},
                    'device': device,
                    'callback_fns': [save_net_callback.rew_save],
                }
                script_args.append(kwargs)

    for kwargs in script_args:
        logger.configure(kwargs['log_dir'], format_strs=["stdout", "tensorboard"])
        main(**kwargs)
