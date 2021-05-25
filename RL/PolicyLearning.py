import os
import pickle

from stable_baselines3.common import callbacks
from stable_baselines3.common.vec_env import VecNormalize
from imitation.policies import serialize

from common.callbacks import VideoCallback
from common.util import create_path, make_env
from common.wrappers import RewardWrapper
from RL.project_policies import def_policy

from mujoco_py import GlfwContext


def wrapped_env(environment, reward_wrap=None, norm_wrap=False):
    renv = environment
    if reward_wrap:
        renv = RewardWrapper(renv, reward_wrap)
    if norm_wrap:
        renv = VecNormalize(renv, norm_obs=True, norm_reward=False)
    return renv


if __name__ == "__main__":
    env_type = "IDP"
    algo_type = "ppo"
    name = "IDP_custom"
    device = "cpu"
    env_id = f"{name}-v2"
    env = make_env(env_id, use_vec_env=True, num_envs=8, n_steps=600)
    name += "_abs_ppo"
    # with open(f"../IRL/tmp/log/{env_type}/MaxEntIRL/{name}/model/reward_net.pkl", "rb") as f:
    #     reward_net = pickle.load(f).double()
    # env = wrapped_env(env, reward_wrap=reward_net.eval(), norm_wrap=False)
    current_path = os.path.dirname(__file__)
    log_dir = os.path.join(current_path, env_type, "tmp", "log", name, algo_type)
    create_path(dirname=log_dir)
    GlfwContext(offscreen=True)
    algo = def_policy(algo_type, env, device=device, log_dir=log_dir, verbose=1)
    n = 1
    while os.path.isdir(log_dir + f"/extra_{n}"):
        n += 1
    create_path(log_dir + f"/policies_{n}")
    video_recorder = VideoCallback(make_env(env_id, use_vec_env=False, n_steps=600),
                                   n_eval_episodes=5,
                                   render_freq=int(5e5))
    # save_policy_callback = serialize.SavePolicyCallback(log_dir + f"/policies_{n}", None)
    # save_policy_callback = callbacks.EveryNTimesteps(int(5e5), save_policy_callback)
    # callback_list = callbacks.CallbackList([video_recorder, save_policy_callback])
    algo.learn(total_timesteps=int(1e6), tb_log_name="extra")
    algo.save(log_dir+f"/policies_{n}/{algo_type}0")
    print(f"saved as policies_{n}/{algo_type}0")
