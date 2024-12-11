import copy
import pickle


def test_env_init(idp_env):
    env = idp_env
    env.close()


def test_pickle(ip_env2_vec):
    env = ip_env2_vec
    print(env.get_attr("cost_ratio"))
    pickled_env = pickle.dumps(env)
    unpickled_env = pickle.loads(pickled_env)
    print(unpickled_env.get_attr("cost_ratio"))


def test_deepcopy(ip_env):
    env = ip_env
    # getattr(ip_env_norm, "abc")
    copied = copy.deepcopy(env)
    copied2 = copy.deepcopy(copied)
    print(copied.class_attributes)


def test_vecenv_state(ip_env_vec):
    env = ip_env_vec
    obs = env.reset()
    print(obs)


def test_env_time_idx(idp_env):
    env = idp_env
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step(env.action_space.sample())
    env.close()


def test_env_acc(idp_env_vec):
    env = idp_env_vec
    env.reset()
    assert (env.get_attr("ptb_acc")[0] != 0).sum() != 0
    assert (len(env.get_attr("ptb_acc")[0]) == 360)


def test_det_env_diff(idp_env, idp_det_env):
    from matplotlib import pyplot as plt
    for _ in range(10):
        idp_env.reset()
        idp_det_env.reset()
        plt.plot(idp_det_env.ptb_acc)
        plt.plot(idp_env.ptb_acc)
        plt.show()
