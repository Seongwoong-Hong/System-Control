from algos.torch.ppo import PPO, MlpPolicy
from common.wrappers import *
import matplotlib.pyplot as plt


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


def test_reward_norm(ip_env, proj_path, ip_env_norm):
    env = ip_env
    norm_env = ip_env_norm

    prev_ob = env.reset()
    norm_env.reset()

    ob, reward, _, _ = env.step(-np.array([800, 300]) @ prev_ob.T/200)
    norm_ob, norm_reward, _, _ = norm_env.step([-np.array([800, 300]) @ prev_ob.T/200])
    normd_reward = norm_env.normalize_reward(reward)
    print(norm_reward[0], normd_reward)
    print(norm_env.unnormalize_obs(norm_ob)[0], ob)


def test_ppo_algo(idp_env2):
    env = idp_env2
    algo = PPO(
        MlpPolicy,
        env=env,
        n_steps=4096,
        batch_size=1024,
        learning_rate=0.0003,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        vf_coef=0.5,
        ent_coef=0.001,
        device='cpu',
        policy_kwargs={'net_arch': [dict(pi=[64, 64], vf=[64, 64])]},
        verbose=1,
    )
    algo.learn(total_timesteps=int(1e6))
