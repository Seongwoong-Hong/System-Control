from common.util import make_env


def test_n_steps():
    env = make_env("IP_custom-v1")
    for _ in range(10):
        obs = env.reset()
        print(obs)
        done = False
        t = 0
        while not done:
            act = env.action_space.sample()
            obs, rew, done, _ = env.step(act)
            t += 1
        print(t)
    env.close()
    assert hasattr(env, "_max_episode_steps")


def test_n_steps_hpc():
    env = make_env("HPC_custom-v1", subpath="../../IRL/demos/HPC/sub01/sub01")
    for _ in range(10):
        obs = env.reset()
        print(obs)
        done = False
        t = 0
        while not done:
            act = env.action_space.sample()
            obs, rew, done, _ = env.step(act)
            t += 1
        print(t)
    env.close()


def test_pltq():
    from scipy import io
    pltqs = []
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        file = f"../../demos/HPC/sub01/sub01i{i + 1}.mat"
        pltqs += [io.loadmat(file)['pltq']]
    env = make_env("HPC_custom-v2", pltqs=pltqs)
    for pltq in pltqs:
        env.set_pltq(pltq)
        obs = env.reset()
        print(obs)
        done = False
        t = 0
        while not done:
            act = env.action_space.sample()
            obs, rew, done, _ = env.step(act)
            t += 1
        print(t)
    env.close()
