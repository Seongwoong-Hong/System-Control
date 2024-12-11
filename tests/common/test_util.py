from common.sb3.util import *
from common.wrappers import *


def test_write_analyzed_result():
    def ana_fnc():
        Q = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])
        R = 1e-6 * np.array([[1, 0],
                             [0, 1]])
        param = 0
        for _ in range(7):
            obs = env.reset()
            done = False
            while not done:
                act, _ = policy.predict(obs, deterministic=True)
                param += obs[:4] @ Q @ obs[:4] + act @ R @ act
                obs, _, done, _ = env.step(act)
        return {'cost': param}

    env = make_env("IDP_custom-v0", use_vec_env=False, n_steps=600)
    ana_dir = os.path.join("..", "tmp", "log", "IDP", "ppo", "AIRL_Single_test")
    policy = PPO.load(ana_dir + "/0/model/gen.zip")
    write_analyzed_result(ana_fnc, ana_dir, iter_name=0)


def test_making_env():
    env = make_env(f"DiscretizedHuman-v2", num_envs=10, wrapper=DiscretizeWrapper, N=[17, 17, 17, 19], NT=[11, 11])
    print(env.reset())