import os
import pickle
import shutil
import torch as th

from imitation.data import rollout
from imitation.util import logger

from common.util import make_env, create_path
from common.callbacks import SaveCallback
from algo.torch.MaxEntIRL import MaxEntIRL


if __name__ == "__main__":
    env_type = "IDP"
    algo_type = "MaxEntIRL"
    device = "cpu"
    name = "IDP_custom"
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    subpath = os.path.join(proj_path, "demos", env_type, "sub01", "sub01")
    env = make_env(f"{name}-v2", use_vec_env=False, num_envs=8, n_steps=600, subpath=subpath)

    # Load data
    expert_dir = os.path.join(proj_path, "demos", env_type, "lqr.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    transitions = rollout.flatten_trajectories(expert_trajs)

    # Setup log directories
    log_dir = os.path.join(proj_path, "tmp", "log", env_type, algo_type, name)
    log_dir += "_lqr_square_feature"
    assert not os.path.isdir(log_dir), "The log directory already exists"
    create_path(log_dir)
    print(f"All Tensorboards and logging are being written inside {log_dir}/.")
    shutil.copy(os.path.abspath(__file__), log_dir)

    model_dir = os.path.join(log_dir, "model")
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # Setup callbacks
    save_net_callback = SaveCallback(cycle=1, dirpath=model_dir)

    # Setup Logger
    logger.configure(log_dir, format_strs=["stdout", "tensorboard"])

    # Setup Learner
    learning = MaxEntIRL(
        env,
        agent_learning_steps_per_one_loop=3e4,
        expert_transitions=transitions,
        rew_lr=1e-3,
        rew_arch=[],
        device=device,
        sac_kwargs={'verbose': 1},
        rew_kwargs={'feature_fn': lambda x: th.square(x)},
    )

    # Run Learning
    losses = learning.learn(
        total_iter=50,
        gradient_steps=500,
        n_episodes=8,
        max_sac_iter=5,
        callback=save_net_callback.net_save,
    )

    # Save the result of learning
    reward_path = model_dir + "/reward_net.pkl"
    with open(reward_path + ".tmp", "wb") as f:
        pickle.dump(learning.reward_net, f)
    os.replace(reward_path + ".tmp", reward_path)
    learning.agent.save(model_dir + "/agent")
