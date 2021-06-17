import datetime
import os
import pickle
import shutil
import torch as th

from imitation.data import rollout
from imitation.util import logger
from scipy import io
from stable_baselines3.common.vec_env import VecNormalize

from common.util import make_env
from common.callbacks import SaveCallback
from algos.torch.MaxEntIRL import GuidedCostLearning
from IRL.scripts.project_policies import def_policy


if __name__ == "__main__":
    env_type = "IDP"
    algo_type = "GCL"
    device = "cpu"
    name = "IDP_pybullet"
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    subpath = os.path.join(proj_path, "demos", env_type, "sub01", "sub01")
    pltqs = []
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        file = "../demos/HPC/sub01/sub01" + f"i{i + 1}.mat"
        pltqs += [io.loadmat(file)['pltq']]
    env = make_env(f"{name}-v1", use_vec_env=False, num_envs=8, pltqs=pltqs)

    # Load data
    expert_dir = os.path.join(proj_path, "demos", env_type, "lqr_ppo.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    expt_traj_num = len(expert_trajs)
    transitions = rollout.flatten_trajectories(expert_trajs)

    # Setup log directories
    log_dir = os.path.join(proj_path, "tmp", "log", env_type, algo_type)
    log_dir += "/cnn_lqr_ppo"
    os.makedirs(log_dir, exist_ok=False)
    shutil.copy(os.path.abspath(__file__), log_dir)
    shutil.copy(expert_dir, log_dir)
    shutil.copy(proj_path + "/scripts/project_policies.py", log_dir)

    def feature_fn(x):
        return x

    model_dir = os.path.join(log_dir, "model")
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # Setup callbacks
    save_net_callback = SaveCallback(cycle=1, dirpath=model_dir)

    # Setup Logger
    logger.configure(log_dir, format_strs=["stdout", "tensorboard"])

    # Setup Learner
    agent = def_policy("sac", env, device=device, verbose=1)
    learner = GuidedCostLearning(
        env,
        feature_fn=feature_fn,
        agent=agent,
        agent_learning_steps_per_one_loop=int(3e4),
        expert_transitions=transitions,
        use_action_as_input=True,
        rew_lr=1e-3,
        rew_arch=[8, 8],
        device=device,
        env_kwargs={},
        rew_kwargs={'type': 'cnn'},
    )

    # Run Learning
    learner.learn(
        total_iter=50,
        gradient_steps=50,
        n_episodes=expt_traj_num,
        max_agent_iter=5,
        callback=save_net_callback.net_save,
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
