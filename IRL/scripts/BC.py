import os
import shutil
import pickle
import datetime
import torch as th

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.util import logger
from stable_baselines3.common.vec_env import VecNormalize

from algos.torch.MaxEntIRL import MaxEntIRL
from common.callbacks import SaveCallback
from common.util import make_env
from common.wrappers import ObsConcatWrapper
from IRL.scripts.project_policies import def_policy

if __name__ == "__main__":
    env_type = "HPC"
    algo_type = "BC"
    device = "cpu"
    name = f"{env_type}_pybullet"
    policy_type = "sac"
    expt = "sub01"
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    subpath = os.path.join(proj_path, "demos", env_type, "sub01", "sub01")
    env = make_env(f"{name}-v1", subpath=subpath)
    eval_env = make_env(f"{name}-v0", subpath=subpath)

    # Load data
    expert_dir = os.path.join(proj_path, "demos", env_type, f"{expt}.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    expt_traj_num = len(expert_trajs)
    transitions = rollout.flatten_trajectories(expert_trajs)

    # Setup log directories
    log_dir = os.path.join(proj_path, "tmp", "log", name, algo_type)
    log_dir += f"/cnn_{expt}_mm_reset_0.1"
    os.makedirs(log_dir, exist_ok=False)
    shutil.copy(os.path.abspath(__file__), log_dir)
    shutil.copy(expert_dir, log_dir)
    shutil.copy(proj_path + "/scripts/project_policies.py", log_dir)

    model_dir = os.path.join(log_dir, "model")
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # Setup Logger
    logger.configure(log_dir, format_strs=["stdout", "tensorboard"])

    def feature_fn(x):
        return x
        # return x.square()
        # return th.cat([x, x.square()], dim=1)

    policy_kwargs = None
    if policy_type == "ppo":
        from algos.torch.ppo import MlpPolicy
        policy_kwargs = {'log_std_range': [None, 1.8],
                         'net_arch': [{'pi': [32, 32], 'vf': [32, 32]}]
                         }
    elif policy_type == "sac":
        from algos.torch.sac import MlpPolicy
        policy_kwargs = {'net_arch': {'pi': [32, 32], 'qf': [32, 32]},
                         'optimizer_kwargs': {'betas': (0.9, 0.999)}
                         }

    bc.BC.DEFAULT_BATCH_SIZE = 256
    learner = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy_class=MlpPolicy,
        policy_kwargs=policy_kwargs,
        expert_data=transitions,
        device=device,
        ent_weight=5e-2,
        l2_weight=1e-2,
    )

    with logger.accumulate_means("BC"):
        learner.train(n_epochs=150)

    learner.save_policy(model_dir + "/policy")

    agent = def_policy(policy_type, env, device=device, verbose=1)
    agent.policy = learner.policy

    del learner

    save_net_callback = SaveCallback(cycle=1, dirpath=model_dir)
    learner = MaxEntIRL(
        env,
        eval_env=eval_env,
        feature_fn=feature_fn,
        agent=agent,
        expert_transitions=transitions,
        use_action_as_input=True,
        rew_arch=[4, 4, 4, 4, 4, 4],
        device=device,
        env_kwargs={'vec_normalizer': None},
        rew_kwargs={'type': 'cnn', 'scale': 1, 'alpha': 0.1},
    )

    # Run Learning
    learner.learn(
        total_iter=50,
        agent_learning_steps=1e4,
        n_episodes=expt_traj_num,
        max_agent_iter=20,
        min_agent_iter=5,
        max_gradient_steps=300,
        min_gradient_steps=30,
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
