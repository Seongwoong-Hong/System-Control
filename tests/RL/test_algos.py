import os

from algos.torch.ppo import PPO
from algos.torch.sac import SAC
from common.util import make_env
from common.verification import verify_policy


def test_mujoco_envs_learned_policy():
    env_name = "Hopper"
    env = make_env(f"{env_name}-v2", use_vec_env=False)
    name = f"{env_name}/ppo"
    model_dir = os.path.join("..", "..", "RL", "mujoco_envs", "tmp", "log", name)
    algo = PPO.load(model_dir + "/policies_1/000002000000/model.pkl")
    a_list, o_list, _ = verify_policy(env, algo)


def test_rl_learned_policy():
    env_type = "IDP"
    env = make_env(f"{env_type}_custom-v0", subpath="../../IRL/demos/HPC/sub01/sub01")
    name = f"{env_type}_custom/ppo"
    model_dir = os.path.join("..", "..", "RL", env_type, "tmp", "log", name, "policies_3")
    algo = PPO.load(model_dir + f"/{int(2.5e6):012d}/model.pkl")
    a_list, o_list, _ = verify_policy(env, algo)

