import pytest

from IRL.scripts.project_policies import def_policy
from common.verification import verify_policy
from common.util import make_env


def test_env_order():
    env = make_env("HPC_custom-v1", subpath="../../IRL/demos/HPC/sub01/sub01")
    exp = def_policy("HPC", env)
    for _ in range(40):
        a_list, o_list, _, = verify_policy(env, exp, "None")
    print(env.order)
    assert env.order == 5


def test_env_traj_len():
    env = make_env("HPC_custom-v1", subpath="../../IRL/demos/HPC/sub01/sub01")
    exp = def_policy("HPC", env)
    a_list, o_list, _, = verify_policy(env, exp, "None")
    assert len(a_list) == env.n_steps


def test_pybullet_envs():
    import gym, time
    from common.util import make_env
    env = make_env("HPC_pybullet-v0", subpath="../../IRL/demos/HPC/sub01/sub01")
    env.render(mode="human")
    env.reset()
    env.set_state([0.25, 0.25], [0.0, 0.0])
    env.camera_adjust()
    done = False
    while not done:
        act = env.action_space.sample()
        ob, rew, done, info = env.step(act)
        time.sleep(0.01)
    assert isinstance(env, gym.Env)


def test_hpc_obs_reset():
    from scipy import io
    init_states = []
    for i in range(35):
        file = f"../../IRL/demos/HPC/sub01/sub01i{i+1}.mat"
        init_states += [io.loadmat(file)['state'][0, :4]]
    env = make_env("HPC_pybullet-v0", subpath="../../IRL/demos/HPC/sub01/sub01")
    for i in range(35):
        env.render(mode="None")
        init = env.reset()
        assert (init[:4] == init_states[i]).all()
        done = False
        while not done:
            act = env.action_space.sample()
            ob, rew, done, info = env.step(act)
