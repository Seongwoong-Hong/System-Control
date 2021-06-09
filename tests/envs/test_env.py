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
