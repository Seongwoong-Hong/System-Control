import os
import pickle
import shutil

from imitation.data import rollout
from imitation.util import logger
from imitation.policies import serialize
from stable_baselines3.common import callbacks

from common.util import make_env, create_path
from common.callbacks import SaveRewardCallback
from algo.torch.MaxEntIRL import MaxEntIRL

if __name__ == "__main__":
    env_type = "IDP"
    algo_type = "MaxEntIRL"
    device = "cpu"
    name = "IDP_custom"
    env = make_env(f"{name}-v2", use_vec_env=False, num_envs=8, n_steps=600, sub="sub01")
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Load data
    expert_dir = os.path.join(proj_path, "demos", env_type, "ppoeasy.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    transitions = rollout.flatten_trajectories(expert_trajs)

    # Setup log directories
    log_dir = os.path.join(proj_path, "tmp", "log", env_type, algo_type, name)
    log_dir += "3"
    if not os.path.isdir(log_dir):
        create_path(log_dir)
    else:
        print("The log directory already exists")
        raise SystemExit
    print(f"All Tensorboards and logging are being written inside {log_dir}/.")
    shutil.copy(os.path.abspath(__file__), log_dir)

    model_dir = os.path.join(log_dir, "model")
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # Setup callbacks
    save_policy_callback = serialize.SavePolicyCallback(model_dir, None)
    save_policy_callback = callbacks.EveryNTimesteps(int(5e4), save_policy_callback)
    save_reward_callback = SaveRewardCallback(cycle=5, dirpath=model_dir)

    # Setup Logger
    logger.configure(log_dir, format_strs=["stdout", "tensorboard"])

    # Setup Learner
    learning = MaxEntIRL(env,
                         agent_learning_steps_per_one_loop=5e4,
                         expert_transitions=transitions,
                         rew_lr=1e-3,
                         rew_arch=[8, 8],
                         device=device,
                         sac_kwargs={'verbose': 1}
                         )

    # Run Learning
    losses = learning.learn(total_iter=10,
                            gradient_steps=500,
                            n_episodes=8,
                            max_sac_iter=20,
                            rew_callback=save_reward_callback.save,
                            agent_callback=save_policy_callback,
                            )

    # Save the result of learning
    reward_path = model_dir + "/reward_net.pkl"
    with open(reward_path + ".tmp", "wb") as f:
        pickle.dump(learning.reward_net, f)
    os.replace(reward_path + ".tmp", reward_path)
    learning.agent.save(model_dir + "/agent")

