import os

from stable_baselines3.common import callbacks
from imitation.policies import serialize

from common.callbacks import VideoCallback
from common.util import make_env
from RL.project_policies import def_policy


if __name__ == "__main__":
    env_type = "IDP"
    algo_type = "sac"
    name = "IDP_classic"
    device = "cpu"
    env_id = f"{name}-v0"
    env = make_env(env_id, use_vec_env=False, num_envs=1, use_norm=True)
    name += "_1"
    current_path = os.path.dirname(__file__)
    log_dir = os.path.join(current_path, env_type, "tmp", "log", name, algo_type)
    os.makedirs(log_dir, exist_ok=True)
    algo = def_policy(algo_type, env, device=device, log_dir=log_dir, verbose=1)
    n = 1
    while os.path.isdir(log_dir + f"/extra_{n}"):
        n += 1
    os.makedirs(log_dir + f"/policies_{n}", exist_ok=False)
    video_recorder = VideoCallback(make_env(env_id, use_vec_env=False, n_steps=600),
                                   n_eval_episodes=5,
                                   render_freq=int(2.5e5))
    save_policy_callback = serialize.SavePolicyCallback(log_dir + f"/policies_{n}", None)
    save_policy_callback = callbacks.EveryNTimesteps(int(2.5e5), save_policy_callback)
    callback_list = callbacks.CallbackList([video_recorder, save_policy_callback])
    algo.learn(total_timesteps=int(1e5), tb_log_name="extra", callback=callback_list)
    algo.save(log_dir+f"/policies_{n}/{algo_type}0")
    env.save(log_dir+"/policies_{n}/normalization")
    print(f"saved as policies_{n}/{algo_type}0")
