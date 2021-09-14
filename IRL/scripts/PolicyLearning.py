import os
import pickle
import torch as th

from stable_baselines3.common import callbacks
from stable_baselines3.common.vec_env import VecNormalize
from imitation.policies import serialize

from IRL.scripts.project_policies import def_policy
from common.callbacks import VideoCallback
from common.util import make_env
from common.wrappers import ActionRewardWrapper
from scipy import io


def learning_specific_one():
    env_type = "HPC"
    algo_type = "BC"
    name = "ext_sub01_linear_noreset"
    env_id = f"{env_type}_custom"

    n = 7
    load_dir = os.path.abspath(os.path.join("..", "tmp", "log", env_id, algo_type, name, "model", f"{n:03d}"))

    stats_path = None
    if os.path.isfile(load_dir + "/normalization.pkl"):
        stats_path = load_dir + "/normalization.pkl"

    with open(load_dir + "/reward_net.pkl", "rb") as f:
        reward_net = pickle.load(f).double()

    env = make_env(f"{env_id}-v1", subpath="../demos/HPC/sub01/sub01", num_envs=1,
                   wrapper=ActionRewardWrapper, wrapper_kwrags={'rwfn': reward_net.eval()}, use_norm=stats_path)

    device = "cuda:3"
    log_dir = os.path.join(load_dir, "add_rew_learning")

    algo_used = "sac"
    algo = def_policy(algo_used, env, device=device, log_dir=log_dir, verbose=1)
    from algos.torch.sac import SAC
    prev_algo = SAC.load(os.path.abspath(os.path.join(load_dir, "..", f"{n - 1:03d}", "agent")))
    algo.policy.load_from_vector(prev_algo.policy.parameters_to_vector())
    os.makedirs(log_dir + f"/{algo_used}_policies_{n:03d}", exist_ok=True)
    save_policy_callback = serialize.SavePolicyCallback(log_dir + f"/{algo_used}_policies_{n:03d}", None)
    save_policy_callback = callbacks.EveryNTimesteps(int(3e5), save_policy_callback)
    for i in range(15):
        algo.learn(total_timesteps=int(1e4), tb_log_name="extra", callback=save_policy_callback, reset_num_timesteps=False)
        algo.save(log_dir + f"/{algo_used}_policies_{n:03d}/{algo_used}{i}")
    print(f"saved as {algo_used}_policies_{n}/{algo_used}")


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

    n = 0
    filename = os.path.abspath(f"../tmp/log/{env_id}/{algo_type}/{name}/model/{n:03d}/reward_net.pkl")
    while os.path.isfile(filename):

        with open(filename, "rb") as f:
            reward_net = pickle.load(f).double()
        env = make_env(f"{env_id}-v1", use_vec_env=False, pltqs=pltqs,
                       wrapper=ActionRewardWrapper, wrapper_kwrags={'rwfn': reward_net.eval()})

        proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        log_dir = os.path.join(proj_path, "tmp", "log", env_id, algo_type, name, "add_rew_learning")
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
