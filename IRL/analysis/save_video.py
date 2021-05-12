import os

from IRL.scripts.project_policies import def_policy
from algo.torch.ppo import PPO
from common.verification import verify_policy, video_record
from common.util import make_env

if __name__ == "__main__":
    env_type = "IDP"
    algo_type = "ppo"
    name = "{}/{}/AIRL_hype_tune/111".format(env_type, algo_type)
    model_dir = os.path.join("tmp", "log", name, "model")
    # costfn = torch.load(model_dir + f"/costfn{num}.pt")
    algo = PPO.load(model_dir + f"/gen.zip")
    # algo = PPO.load(model_dir + "/extra_ppo")
    env = make_env("{}_custom-v0".format(env_type), use_vec_env=False, sub="sub01", n_steps=600)
    expt_env = make_env("{}_custom-v0".format(env_type), use_vec_env=False, sub="sub01", n_steps=600)
    # env = make_env("HalfCheetah-v2", use_vec_env=False)
    # env = CostWrapper(gym_envs.make("{}_custom-v1".format(env_type), n_steps=200), costfn)
    expt = def_policy(env_type, env)
    # expt = PPO.load("../RL/mujoco_envs/tmp/log/HalfCheetah/ppo/ppo0.zip")
    dt = env.dt

    _, _, imgs = verify_policy(env, algo, 'rgb_array', repeat_num=3)
    _, _, expt_imgs = verify_policy(expt_env, expt, 'rgb_array', repeat_num=3)
    if imgs[0] is not None:
        video_record(imgs, f"videos/{name}_agent.avi", dt)
        video_record(expt_imgs, f"videos/{name}_expert.avi", dt)
    env.close()
    expt_env.close()
