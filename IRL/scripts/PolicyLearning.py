import os
import pickle
import torch as th

from stable_baselines3.common import callbacks
from imitation.policies import serialize

from IRL.scripts.project_policies import def_policy
from common.callbacks import VideoCallback
from common.util import make_env
from common.wrappers import RewardWrapper
from mujoco_py import GlfwContext
from scipy import io


def learning_specific_one():
    env_type = "IDP"
    algo_type = "MaxEntIRL"
    name = "extcnn_lqr_ppo_deep"
    env_id = f"{env_type}_custom"

    pltqs = []
    for i in range(10):
        file = "../demos/HPC/sub01/sub01" + f"i{i + 1}.mat"
        pltqs += [io.loadmat(file)['pltq']]

    env = make_env(f"{env_id}-v1", use_vec_env=False, pltqs=pltqs)

    n = 10
    with open(f"../tmp/log/{env_id}/{algo_type}/{name}/model/{n:03d}/reward_net.pkl", "rb") as f:
        reward_net = pickle.load(f).double()
    env = RewardWrapper(env, reward_net.eval())

    device = "cuda:1"
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    log_dir = os.path.join(proj_path, "tmp", "log", env_id, algo_type, name, "add_rew_learning")
    GlfwContext(offscreen=True)
    algo = def_policy("ppo", env, device=device, log_dir=log_dir, verbose=1)
    os.makedirs(log_dir + f"/ppo_policies_{n}", exist_ok=False)
    video_recorder = VideoCallback(make_env(f"{env_id}-v0", use_vec_env=False, pltqs=pltqs),
                                   n_eval_episodes=5,
                                   render_freq=int(1e6))
    save_policy_callback = serialize.SavePolicyCallback(log_dir + f"/policies_{n}", None)
    save_policy_callback = callbacks.EveryNTimesteps(int(1e6), save_policy_callback)
    callback_list = callbacks.CallbackList([video_recorder, save_policy_callback])
    algo.learn(total_timesteps=int(1e7), tb_log_name="extra", callback=callback_list)
    algo.save(log_dir + f"/policies_{n}/{algo_type}0")
    print(f"saved as policies_{n}/{algo_type}0")


def learning_whole_iter():
    env_type = "IDP"
    algo_type = "MaxEntIRL"
    name = "sq_lqr_ppo"
    env_id = f"{env_type}_custom"
    device = "cuda:1"

    pltqs = []
    for i in range(10):
        file = os.path.abspath("../demos/HPC/sub01/sub01" + f"i{i + 1}.mat")
        pltqs += [io.loadmat(file)['pltq']]

    n = 1
    filename = os.path.abspath(f"../tmp/log/{env_id}/{algo_type}/{name}/model/{n:03d}/reward_net.pkl")
    while os.path.isfile(filename):
        env = make_env(f"{env_id}-v1", use_vec_env=False, pltqs=pltqs)

        with open(filename, "rb") as f:
            reward_net = pickle.load(f).double()
        env = RewardWrapper(env, reward_net.eval())

        proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        log_dir = os.path.join(proj_path, "tmp", "log", env_id, algo_type, name, "add_rew_learning")
        GlfwContext(offscreen=True)
        algo = def_policy("sac", env, device=device, log_dir=log_dir, verbose=1)
        os.makedirs(log_dir + f"/policies_{n}", exist_ok=False)
        video_recorder = VideoCallback(make_env(f"{env_id}-v0", use_vec_env=False, pltqs=pltqs),
                                       n_eval_episodes=5,
                                       render_freq=int(1e5))
        save_policy_callback = serialize.SavePolicyCallback(log_dir + f"/policies_{n}", None)
        save_policy_callback = callbacks.EveryNTimesteps(int(1e5), save_policy_callback)
        callback_list = callbacks.CallbackList([video_recorder, save_policy_callback])
        algo.learn(total_timesteps=int(0.25e6), tb_log_name="extra", callback=callback_list)
        algo.save(log_dir + f"/policies_{n}/{algo_type}0")
        print(f"saved as policies_{n}/{algo_type}0")
        n += 1
        filename = os.path.abspath(f"../tmp/log/{env_id}/{algo_type}/{name}/model/{n:03d}/reward_net.pkl")


if __name__ == "__main__":
    def feature_fn(x):
        return th.cat([x, x.square()], dim=1)
    learning_specific_one()
