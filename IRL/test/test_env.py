from IRL.project_policies import def_policy
from common.verification import verify_policy


def test_env(env):
    exp = def_policy("HPC", env)
    for _ in range(40):
        a_list, o_list, _, = verify_policy(env, exp, "rgb_array")
    print(env.order)
    assert env.order == 5
