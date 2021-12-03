from common.util import make_env
from RL.project_policies import def_policy


def test_ppo_log():
    env = make_env("IDP_custom-v1", use_vec_env=False, num_envs=8)
    algo = def_policy("sac", env, log_dir="tmp/log/rl")
    algo.learn(total_timesteps=1e5)


def test_learn_ppo():
    env = make_env("IDP_pybullet-v2")
    algo = def_policy("ppo", env, device='cpu', log_dir=None, verbose=1)
    algo.learn(total_timesteps=1e5)


def test_iter_predict():
    env = make_env("2DTarget_disc-v2", use_vec_env=True, num_envs=1)
    algo = def_policy("viter", env, device='cpu')
    algo.learn(10000)
    obs = []
    ob = env.reset()
    done = False
    obs.append(ob)
    while not done:
        act, _ = algo.predict(ob, deterministic=False)
        ob, r, done, _ = env.step(act)
        obs.append(ob)
    print('end')
