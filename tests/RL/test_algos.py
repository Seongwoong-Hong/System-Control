import os
import pickle
from algos.torch.ppo import PPO, MlpPolicy
from common.util import make_env, CPU_Unpickler
from common.wrappers import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def test_finite_algo(idpdiffpolicy, hpc_env):
    env = hpc_env
    agent = idpdiffpolicy(env, gamma=1, alpha=0.002)
    ob = env.reset()
    obs, acts, rws = agent.predict(ob, deterministic=True)
    assert len(obs) == len(acts) + 1
    print(obs[-1])


def test_get_gains_wrt_init_states(idpilqrpolicy, hpc_with_rwrap_env):
    env = hpc_with_rwrap_env
    agent = idpilqrpolicy(env, gamma=1, alpha=0.002)
    ks, kks = [], []
    fig = plt.figure()
    for _ in range(10):
        ob = env.reset()
        obs, _, _ = agent.predict(ob, deterministic=True)
        for i in range(4):
            ax = fig.add_subplot(2, 2, i+1)
            ax.plot(obs[:, i])
        ks.append(agent.ks)
        kks.append(agent.kks)
    plt.show()
    env.close()


def test_1d(rl_path):
    name = "1DTarget_disc"
    env_id = f"{name}"
    map_size = 50
    env = make_env(f"{env_id}-v2", map_size=map_size, num_envs=1)
    model_dir = os.path.join(rl_path, name, "tmp", "log", env_id, "softqiter", "policies_1")
    with open(model_dir + "/agent.pkl", "rb") as f:
        algo = pickle.load(f)
    # algo = PPO.load(model_dir + "/agent")
    plt.imshow(algo.policy.policy_table, cmap=cm.rainbow)
    plt.show()
    print('end')


def test_total_reward(ip_env, proj_path):
    env = ip_env

    ob = env.reset()
    done = False
    acts = []
    obs = []
    rewards = []
    while not done:
        # act, _ = algo.predict(ob, deterministic=False)
        # ob, reward, done, info = env.step(-np.array([[1000, 200, 300, 50], [200, 200, 50, 50]]) @ ob.T/100)
        ob, reward, done, info = env.step(-np.array([800, 300]) @ ob.T/100)
        rewards.append(reward)
        acts.append(info['acts'])
        obs.append(ob)
    plt.plot(np.array(acts)[:, 0])
    # plt.plot(env.env.ptb_acc)
    plt.show()
