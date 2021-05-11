import os

from common.util import make_env
from common.verification import verify_policy
from algo.torch.ppo import PPO


def test_mujoco_envs_learned_policy():
    env_name = "Hopper"
    env = make_env(f"{env_name}-v2", use_vec_env=False)
    name = f"{env_name}/ppo"
    model_dir = os.path.join("..", "mujoco_envs", "tmp", "log", name)
    algo = PPO.load(model_dir + "/policies_1/000002000000/model.pkl")
    a_list, o_list, _ = verify_policy(env, algo)


def test_IDP_learned_policy():
    env_name = "IDP_custom"
    env = make_env(f"{env_name}-v2", use_vec_env=False)
    name = f"{env_name}/ppo"
    model_dir = os.path.join("..", "IDP", "tmp", "log", name)
    algo = PPO.load(model_dir + "/policies_1/000000500000/model.pkl")
    a_list, o_list, _ = verify_policy(env, algo)