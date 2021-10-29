import datetime
import os
import pickle
import shutil
import torch as th

from imitation.data import rollout
from imitation.util import logger
from stable_baselines3.common.vec_env import VecNormalize

from common.util import make_env
from common.callbacks import SaveCallback
from common.wrappers import RewardWrapper
from algos.torch.MaxEntIRL import GuidedCostLearning
from IRL.scripts.project_policies import def_policy


if __name__ == "__main__":
    env_type = "2DTarget"
    algo_type = "GCL"
    device = "cpu"
    sub = "sac"
    name = f"{env_type}"
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    subpath = os.path.join(proj_path, "demos", env_type, sub, sub)
    env = make_env(f"{name}-v2", use_vec_env=False, subpath=subpath)
    eval_env = make_env(f"{name}-v2", use_vec_env=False, subpath=subpath)

    # Load data
    expert_dir = os.path.join(proj_path, "demos", env_type, f"{sub}.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    expt_traj_num = len(expert_trajs)

    # Setup log directories
    log_dir = os.path.join(proj_path, "tmp", "log", name, algo_type)
    log_dir += f"/ext_{sub}_linear_reset_0.2_2"
    os.makedirs(log_dir, exist_ok=False)
    shutil.copy(os.path.abspath(__file__), log_dir)
    shutil.copy(expert_dir, log_dir)
    shutil.copy(proj_path + "/scripts/project_policies.py", log_dir)

    def feature_fn(x):
        # return x
        return th.cat([x, x**2], dim=1)

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
        eval_env=eval_env,
        feature_fn=feature_fn,
        agent=agent,
        expert_trajectories=expert_trajs,
        use_action_as_input=False,
        rew_arch=[],
        device=device,
        env_kwargs={'vec_normalizer': None, 'reward_wrapper': RewardWrapper},
        rew_kwargs={'type': 'ann', 'scale': 1, 'alpha': 0.1},
    )

    # Run Learning
    learner.learn(
        total_iter=50,
        agent_learning_steps=1e4,
        n_episodes=expt_traj_num,
        max_agent_iter=25,
        min_agent_iter=8,
        max_gradient_steps=500,
        min_gradient_steps=500,
        callback=save_net_callback.net_save,
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
