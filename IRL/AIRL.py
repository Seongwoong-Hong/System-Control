import os
import shutil
import pickle
import pathlib
import gym_envs
import datetime
import time

from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms import adversarial
from imitation.data import rollout
from imitation.util import logger

from IRL.project_policies import def_policy
from common.callbacks import VideoCallback
from mujoco_py import GlfwContext
from scipy import io


if __name__ == "__main__":
    env_type = "IDP"
    algo_type = "ppo"
    env_id = "{}_custom-v0".format(env_type)
    n_steps = 600
    device = "cpu"
    current_path = os.path.dirname(__file__)
    GlfwContext(offscreen=True)
    sub = "sub01"
    expert_dir = os.path.join(current_path, "demos", env_type, "expert" + ".pkl")
    pltqs = []

    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)

    transitions = rollout.flatten_trajectories(expert_trajs)

    # for i in range(35):
    #     file = os.path.join(current_path, "demos", env_type, sub, sub + "i%d.mat" % (i + 1))
    #     pltqs += [io.loadmat(file)['pltq']]

    env = DummyVecEnv([lambda: gym_envs.make(env_id, n_steps=n_steps) for _ in range(10)])

    algo = def_policy(algo_type, env, device=device, log_dir=None, verbose=1)

    now = datetime.datetime.now()
    current_path = os.path.dirname(__file__)
    log_dir = os.path.join(current_path, "tmp", "log", env_type, algo_type, "AIRL")
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    name = "/%s-%s-%s-%s-%s-%s" % (now.year, now.month, now.day, now.hour, now.minute, now.second)
    # name = "/14"
    log_dir += name
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    print(f"All Tensorboards and logging are being written inside {log_dir}/.")

    model_dir = os.path.join(log_dir, "model")
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    # Copy used file to logging folder
    shutil.copy(os.path.abspath(current_path + "/../gym_envs/envs/{}_custom_exp.py".format(env_type)), model_dir)
    shutil.copy(os.path.abspath(__file__), model_dir)
    shutil.copy(os.path.abspath(current_path + "/project_policies.py"), model_dir)

    logger.configure(log_dir, format_strs=["stdout", "log", "csv", "tensorboard"])

    video_recorder = VideoCallback(gym_envs.make(env_id, n_steps=n_steps),
                                   n_eval_episodes=5,
                                   render_freq=10)

    airl_trainer = adversarial.AIRL(
        env,
        expert_data=transitions,
        expert_batch_size=8,
        gen_algo=algo,
        discrim_kwargs={"entropy_weight": 0.4},
        disc_opt_kwargs={"lr": 1e-3}
    )
    airl_trainer.train(total_timesteps=6000000)

    disc_path = model_dir + "/discrim.pkl"
    p = pathlib.Path(disc_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(disc_path + ".tmp", "wb") as f:
        pickle.dump(airl_trainer.discrim, f)
    os.replace(disc_path + ".tmp", disc_path)

    gen_path = model_dir + "/gen"
    airl_trainer.gen_algo.save(gen_path)
