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


if __name__ == "__main__":
    env_type = "IDP"
    algo_type = "MaxEntIRL"
    name = f"no_lqr_ppo_ent"
    env_id = f"{env_type}_custom-v0"
    n_steps = 600

    pltqs = []
    for i in range(10):
        file = "../demos/HPC/sub01/sub01" + f"i{i + 1}.mat"
        pltqs += [io.loadmat(file)['pltq']]

    env = make_env(env_id, use_vec_env=False, num_envs=8, n_steps=600, pltqs=pltqs)

    def feature_fn(x):
        return x

    with open(f"../tmp/log/{env_type}/{algo_type}/{name}/model/020/reward_net.pkl", "rb") as f:
        reward_net = pickle.load(f).double()
    env = RewardWrapper(env, reward_net.eval())

    device = "cuda:3"
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    log_dir = os.path.join(proj_path, "tmp", "log", env_type, algo_type, name, "add_rew_learning")
    GlfwContext(offscreen=True)
    algo = def_policy("sac", env, device=device, log_dir=log_dir, verbose=1)
    n = 1
    while os.path.isdir(log_dir + f"/extra_{n}"):
        n += 1
    os.makedirs(log_dir + f"/policies_{n}", exist_ok=False)
    video_recorder = VideoCallback(make_env(env_id, use_vec_env=False, n_steps=n_steps, pltqs=pltqs),
                                   n_eval_episodes=5,
                                   render_freq=100000)
    save_policy_callback = serialize.SavePolicyCallback(log_dir + f"/policies_{n}", None)
    save_policy_callback = callbacks.EveryNTimesteps(100000, save_policy_callback)
    callback_list = callbacks.CallbackList([video_recorder, save_policy_callback])
    for i in range(10):
        algo.learn(total_timesteps=500000, tb_log_name="extra", callback=callback_list)
    algo.save(log_dir + f"/policies_{n}/{algo_type}0")
    print(f"saved as policies_{n}/{algo_type}0")
