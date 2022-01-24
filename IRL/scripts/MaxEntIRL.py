import datetime
import os
import pickle
import shutil
import torch as th

from imitation.util import logger
from scipy import io

from common.util import make_env
from common.callbacks import SaveCallback
from common.wrappers import *
from algos.torch.MaxEntIRL import MaxEntIRL
from IRL.scripts.project_policies import def_policy


def main(subj, actu, trial):
    env_type = "DiscretizedHuman"
    algo_type = "MaxEntIRL"
    device = "cuda:1"
    name = f"{env_type}"
    expt = f"19171717_done/{subj}_{actu}"
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    subpath = os.path.join(proj_path, "demos", "HPC", subj, subj)

    # Load data
    expert_dir = os.path.join(proj_path, "demos", env_type, f"{expt}.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    expt_traj_num = len(expert_trajs)
    bsp = io.loadmat(subpath + f"i1.mat")['bsp']
    init_states = []
    for traj in expert_trajs:
        init_states += [traj.obs[0]]

    # Define environments
    env = make_env(f"{name}-v2", subpath=subpath, N=[19, 17, 17, 17], NT=[11, 11], bsp=bsp)
    eval_env = make_env(f"{name}-v0", subpath=subpath, N=[19, 17, 17, 17], NT=[11, 11], init_states=init_states,
                        bsp=bsp)
    # env = make_env(f"{name}-v2")
    # eval_env = make_env(f"{name}-v0", init_states=init_states)

    # Setup log directories
    log_dir = os.path.join(proj_path, "tmp", "log", name, algo_type)
    log_dir += f"/sq_normalize_{expt}_{trial}"
    os.makedirs(log_dir, exist_ok=False)
    shutil.copy(os.path.abspath(__file__), log_dir)
    shutil.copy(expert_dir, log_dir)
    shutil.copy(proj_path + "/scripts/project_policies.py", log_dir)
    model_dir = os.path.join(log_dir, "model")
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # Define feature function
    def feature_fn(x):
        # if len(x.shape) == 1:
        #     x = x.reshape(1, -1)
        # ft = th.zeros([x.shape[0], map_size], dtype=th.float32)
        # for i, row in enumerate(x):
        #     idx = int(row.item())
        #     ft[i, idx] = 1
        # return ft
        # return x
        return x ** 2
        # x1, x2, x3, x4 = th.split(x, 1, dim=1)
        # return th.cat((x, x1*x2, x3*x4, x1*x3, x2*x4, x1*x4, x2*x3, x**2, x**3), dim=1)
        # return th.cat([x, x ** 2], dim=1)

    # Setup callbacks
    save_net_callback = SaveCallback(cycle=10, dirpath=model_dir)

    # Setup Logger
    logger.configure(log_dir, format_strs=["stdout", "tensorboard"])

    # Setup Learner
    agent = def_policy("softqiter", env, device=device, verbose=1)
    learner = MaxEntIRL(
        env,
        eval_env=eval_env,
        feature_fn=feature_fn,
        agent=agent,
        expert_trajectories=expert_trajs,
        use_action_as_input=True,
        rew_arch=[],
        device=device,
        env_kwargs={'vec_normalizer': None, 'num_envs': 1, 'reward_wrapper': RewardInputNormalizeWrapper},
        rew_kwargs={'type': 'ann', 'scale': 1, 'norm_coeff': 0.0, 'lr': 1e-2},
    )

    # Run Learning
    learner.learn(
        total_iter=120,
        agent_learning_steps=2000,
        n_episodes=expt_traj_num,
        max_agent_iter=1,
        min_agent_iter=1,
        max_gradient_steps=1,
        min_gradient_steps=1,
        callback=None,
        early_stop=True,
    )

    # Save the result of learning
    reward_path = model_dir + "/reward_net"
    learner.reward_net.save(reward_path)
    # learner.agent.save(model_dir + "/agent")
    if learner.agent.get_vec_normalize_env():
        learner.wrap_env.save(model_dir + "/normalization.pkl")
    now = datetime.datetime.now()
    print(f"Endtime: {now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}")
    del learner, agent
    with th.cuda.device(device):
        th.cuda.empty_cache()


if __name__ == "__main__":
    for trial in [1, 2, 3]:
        for subj in [f"sub{i:02d}" for i in [1]]:
            for actu in range(1, 7):
                main(subj, actu, trial)
