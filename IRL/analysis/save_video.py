import os

from IRL.scripts.project_policies import def_policy
from algos.torch.ppo import PPO
from common.verification import verify_policy, video_record
from common.util import make_env

if __name__ == "__main__":
    env_type = "IDP"
    algo_type = "ppo"
    name = "IDP_custom"
    env = make_env(f"{name}-v2", use_vec_env=False, sub="sub01", n_steps=600)
    expt_env = make_env(f"{name}-v2", use_vec_env=False, sub="sub01", n_steps=600)
    expt = PPO.load(f"../../RL/{env_type}/tmp/log/{name}/ppo/policies_1/ppo0")
    # expt = def_policy(env_type, env)
    name += "_easy"
    model_dir = os.path.join("..", "tmp", "log", env_type, algo_type, name, "49", "model")
    algo = PPO.load(model_dir + "/gen.zip")

    dt = env.dt
    _, _, imgs = verify_policy(env, algo, 'rgb_array', repeat_num=1)
    _, _, expt_imgs = verify_policy(expt_env, expt, 'rgb_array', repeat_num=1)
    if imgs[0] is not None:
        video_record(imgs, f"videos/{env_type}/{algo_type}/{name}_agent.avi", dt)
        video_record(expt_imgs, f"videos/{env_type}/{algo_type}/{name}_expert.avi", dt)
        print(f"{name} videos are saved")
    else:
        print("No images for saving")
    env.close()
    expt_env.close()
