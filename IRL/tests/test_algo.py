import os
from algo.torch.ppo import PPO
from IRL.project_policies import def_policy
from common.verification import verify_policy
from matplotlib import pyplot as plt


def test_hpc_algo(env):
    algo = def_policy("HPC", env)
    a_list, o_list, _ = verify_policy(env, algo)


def test_hpcdiv_algo(tenv):
    algo = def_policy("HPCDiv", tenv)
    for _ in range(35):
        a_list, o_list, _ = verify_policy(tenv, algo)


def test_learned_policy(env):
    name = "IDP/ppo/AIRL_hype_tune/" + "111"
    model_dir = os.path.join("..", "tmp", "log", name, "model")
    algo = PPO.load(model_dir + "/gen.zip")
    a_list, o_list, _ = verify_policy(env, algo)