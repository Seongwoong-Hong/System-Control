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
