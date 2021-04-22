import os
import shutil
import pickle
import pathlib
import gym_envs
import datetime
import random
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms import adversarial
from imitation.data import rollout
from imitation.util import logger

from algo.torch.ppo import PPO, MlpPolicy
from scipy import io


if __name__ == "__main__":
    env_type = "HPC"
    algo_type = "ppo"
    env_id = "{}_custom-v1".format(env_type)
    n_steps = 600
    device = "cpu"
    current_path = os.path.dirname(__file__)
    sub = "sub01"
    # expert_dir = os.path.join(current_path, "demos", env_type, sub + ".pkl")
    expert_dir = os.path.join(current_path, "demos", env_type, "lqrDivTest.pkl")
    pltqs = []

    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)

    for i in range(35):
        file = os.path.join(current_path, "demos", env_type, sub, sub + "i%d.mat" % (i + 1))
        pltqs += [io.loadmat(file)['pltq']]

    env = DummyVecEnv([lambda: gym_envs.make(env_id, n_steps=n_steps, pltqs=pltqs) for _ in range(10)])

    transitions = rollout.flatten_trajectories(expert_trajs)
    for i in range(200):
        b_size = random.sample([8, 16, 32], 1)[0]
        n_epochs = random.sample([2, 4, 6, 10], 1)[0]
        entropy_weight = np.random.choice(np.linspace(0.1, 0.5, num=5))
        lr = np.log10(np.random.choice(np.logspace(3e-5, 1e-2, num=6)))

        now = datetime.datetime.now()
        current_path = os.path.dirname(__file__)
        log_dir = os.path.join(current_path, "tmp", "log", env_type, algo_type, "AIRL_div_test")
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
        f = open(model_dir + "/hyper_parameters.txt", 'w')
        f.write("expert_batch_size: {}\n".format(b_size))
        f.write("n_epochs: {}\n".format(n_epochs))
        f.write("entropy_weight: {}\n".format(entropy_weight))
        f.write("lr: {}\n".format(lr))
        f.close()
        logger.configure(log_dir, format_strs=["stdout", "log", "csv", "tensorboard"])

        algo = PPO(MlpPolicy,
                   env=env,
                   n_steps=1200,
                   batch_size=200,
                   gamma=0.975,
                   gae_lambda=0.95,
                   ent_coef=0.0,
                   n_epochs=n_epochs,
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
            expert_batch_size=b_size,
            gen_algo=algo,
            discrim_kwargs={"entropy_weight": entropy_weight},
            disc_opt_kwargs={"lr": lr}
        )
        airl_trainer.train(total_timesteps=3000000)

        disc_path = model_dir + "/discrim.pkl"
        p = pathlib.Path(disc_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(disc_path + ".tmp", "wb") as f:
            pickle.dump(airl_trainer.discrim, f)
        os.replace(disc_path + ".tmp", disc_path)

        gen_path = model_dir + "/gen"
        airl_trainer.gen_algo.save(gen_path)
