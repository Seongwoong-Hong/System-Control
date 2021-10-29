import os
import pickle
import pytest
from scipy import io

from imitation.data import rollout
from imitation.util import logger
from imitation.algorithms import bc
from matplotlib import pyplot as plt

from common.util import make_env
from algos.torch.MaxEntIRL import MaxEntIRL
from IRL.scripts.project_policies import def_policy


@pytest.fixture
def irl_path():
    return os.path.abspath("../../IRL")


@pytest.fixture
def pltqs(irl_path):
    pltqs = []
    for i in range(35):
        file = f"{irl_path}/demos/HPC/sub01/sub01" + f"i{i + 1}.mat"
        pltqs += [io.loadmat(file)['pltq']]
    return pltqs


@pytest.fixture
def expert(irl_path):
    expert_dir = os.path.join(irl_path, "demos", "2DWorld", "sac.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    return rollout.flatten_trajectories(expert_trajs)


@pytest.fixture
def env(irl_path):
    subpath = os.path.join(irl_path, "demos", "HPC", "sub01", "sub01")
    return make_env("2DWorld-v0", use_vec_env=False, num_envs=1, subpath=subpath)


def test_learn(env, expert, policy_type="sac"):
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

    # Setup Logger
    logger.configure("tmp/log/BC", format_strs=["stdout", "tensorboard"])
    bc.BC.DEFAULT_BATCH_SIZE = 128
    learner = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy_class=MlpPolicy,
        policy_kwargs=policy_kwargs,
        expert_data=expert,
        device='cpu',
        ent_weight=5e-2,
        l2_weight=1e-2,
    )
    with logger.accumulate_means("./BC"):
        learner.train(n_epochs=200)

    learner.save_policy("policy")
    return learner


def test_irl(env, expert):
    policy_type = "sac"
    learner = test_learn(env, expert, policy_type=policy_type)
    agent = def_policy(policy_type, env, device="cpu", verbose=1)
    agent.policy = learner.policy

    def feature_fn(x):
        return x

    learner = MaxEntIRL(
        env,
        feature_fn=feature_fn,
        agent=agent,
        expert_trajectories=expert,
        use_action_as_input=True,
        rew_arch=[8, 8, 8, 8],
        device="cpu",
        env_kwargs={'vec_normalizer': None},
        rew_kwargs={'type': 'ann', 'scale': 1},
    )

    # Run Learning
    learner.learn(
        total_iter=50,
        agent_learning_steps=1e4,
        gradient_steps=150,
        n_episodes=10,
        max_agent_iter=1,
        callback=None,
        early_stop=False
    )

    # Save the result of learning
    reward_path = "reward_net.pkl"
    with open(reward_path + ".tmp", "wb") as f:
        pickle.dump(learner.reward_net, f)
    os.replace(reward_path + ".tmp", reward_path)
    learner.agent.save("agent")


def test_policy(env, expert):
    policy = bc.reconstruct_policy("policy")
    learned_acts = []
    for obs in expert.obs[:100, :]:
        act, _ = policy.predict(obs, deterministic=True)
        learned_acts.append(act)
    plt.plot(learned_acts)
    plt.show()
    plt.plot(expert.acts[:100, :])
    plt.show()

