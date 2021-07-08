import os
import shutil
import pickle
import datetime
from scipy import io

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.util import logger

from algos.torch.ppo import MlpPolicy
from algos.torch.MaxEntIRL import MaxEntIRL
from common.callbacks import SaveCallback
from common.util import make_env
from IRL.scripts.project_policies import def_policy

if __name__ == "__main__":
    env_type = "IDP"
    algo_type = "BC"
    device = "cpu"
    name = "IDP_pybullet"
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    subpath = os.path.join(proj_path, "demos", env_type, "sub01", "sub01")
    pltqs = []
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        file = f"{proj_path}/demos/HPC/sub01/sub01" + f"i{i + 1}.mat"
        pltqs += [io.loadmat(file)['pltq']]
    env = make_env(f"{name}-v1", use_vec_env=False, num_envs=1, pltqs=pltqs)

    # Load data
    expert_dir = os.path.join(proj_path, "demos", env_type, "lqr_ppo.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    expt_traj_num = len(expert_trajs)
    transitions = rollout.flatten_trajectories(expert_trajs)

    # Setup log directories
    log_dir = os.path.join(proj_path, "tmp", "log", name, algo_type)
    log_dir += "/no_lqr_ppo_noreset_sqlast"
    os.makedirs(log_dir, exist_ok=False)
    shutil.copy(os.path.abspath(__file__), log_dir)
    shutil.copy(expert_dir, log_dir)
    shutil.copy(proj_path + "/scripts/project_policies.py", log_dir)

    model_dir = os.path.join(log_dir, "model")
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # Setup Logger
    logger.configure(log_dir, format_strs=["stdout", "tensorboard"])

    # bc.BC.DEFAULT_BATCH_SIZE = 128
    learner = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy_class=MlpPolicy,
        policy_kwargs={'log_std_range': [None, 1.8],
                       'net_arch': [{'pi': [32, 32], 'vf': [32, 32]}],
                       },
        expert_data=transitions,
        device=device,
        ent_weight=1e-4,
        l2_weight=1e-3,
    )

    learner.train(n_epochs=300)

    learner.save_policy(model_dir + "/policy")

    agent = def_policy("ppo", env, device=device, verbose=1)
    agent.policy = learner.policy

    def feature_fn(x):
        return x

    save_net_callback = SaveCallback(cycle=1, dirpath=model_dir)

    learner = MaxEntIRL(
        env,
        feature_fn=feature_fn,
        agent=agent,
        expert_transitions=transitions,
        use_action_as_input=True,
        rew_arch=[8, 8],
        device=device,
        env_kwargs={'vec_normalizer': None},
        rew_kwargs={'type': 'ann', 'scale': 1},
    )

    # Run Learning
    learner.learn(
        total_iter=50,
        agent_learning_steps=4e4,
        gradient_steps=25,
        n_episodes=expt_traj_num,
        max_agent_iter=40,
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