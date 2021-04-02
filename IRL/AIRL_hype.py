import os
import shutil
import pickle
import pathlib
import gym_envs
import datetime
import time
import random
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms import adversarial
from imitation.data import rollout
from imitation.util import logger

from algo.torch.ppo import PPO, MlpPolicy
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
    for i in range(200):
        gamma = np.random.choice(np.linspace(0.95, 0.995, num=10))
        gae_lambda = np.random.choice(np.linspace(0.94, 0.96, num=5))
        entropy_weight = np.random.choice(np.linspace(0.1, 0.5, num=5))
        lr = np.log10(np.random.choice(np.logspace(3e-5, 1e-2, num=6)))

        env = DummyVecEnv([lambda: gym_envs.make(env_id, n_steps=n_steps) for _ in range(10)])

        now = datetime.datetime.now()
        current_path = os.path.dirname(__file__)
        log_dir = os.path.join(current_path, "tmp", "log", env_type, algo_type, "AIRL_hype_tune2")
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        name = "/" + str(i)
        log_dir += name
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        print(f"All Tensorboards and logging are being written inside {log_dir}/.")

        model_dir = os.path.join(log_dir, "model")
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        # Copy used file to logging folder
        shutil.copy(os.path.abspath(__file__), model_dir)
        f = open(model_dir+"/hyper_parameters.txt", 'w')
        f.write("gamma: {}\n".format(gamma))
        f.write("gae_lambda: {}\n".format(gae_lambda))
        f.write("entropy_weight: {}\n".format(entropy_weight))
        f.write("lr: {}".format(lr))
        f.close()
        logger.configure(log_dir, format_strs=["stdout", "log", "csv", "tensorboard"])

        algo = PPO(MlpPolicy,
                   env=env,
                   n_steps=1200,
                   batch_size=200,
                   gamma=gamma,
                   gae_lambda=gae_lambda,
                   ent_coef=0.0,
                   n_epochs=4,
                   ent_schedule=1.0,
                   clip_range=0.2,
                   verbose=1,
                   device=device,
                   tensorboard_log=None,
                   policy_kwargs={'log_std_range': [-5, 5],
                                  'net_arch': [{'pi': [128, 128], 'vf': [128, 128]}]},
                   )

        airl_trainer = adversarial.AIRL(
            env,
            expert_data=transitions,
            expert_batch_size=8,
            gen_algo=algo,
            discrim_kwargs={"entropy_weight": entropy_weight},
            disc_opt_kwargs={"lr": lr}
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
