import os
import shutil
import pickle
import random
import numpy as np

from imitation.algorithms import adversarial
from imitation.data import rollout
from imitation.util import logger
from stable_baselines3.common.vec_env import VecNormalize

from common.util import make_env, create_path
from algo.torch.ppo import PPO, MlpPolicy


if __name__ == "__main__":
    env_type = "IP"
    algo_type = "AIRL"
    device = "cpu"
    name = "IP_custom"
    env = make_env(f"{name}-v2", use_vec_env=True, num_envs=8, n_steps=600, sub="sub01")
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    log_dir = os.path.join(proj_path, "tmp", "log", env_type, algo_type, name + "_normal")
    create_path(log_dir)
    shutil.copy(os.path.abspath(__file__), log_dir)

    expert_dir = os.path.join(proj_path, "demos", env_type, "expert.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    transitions = rollout.flatten_trajectories(expert_trajs)

    for i in range(50):
        # b_size = random.sample([8, 16, 32], 1)[0]
        b_size = 1024
        total_timesteps = int(random.sample([1.024e6, 2*1.024e6, 3*1.024e6], 1)[0])
        # total_timesteps = int(3*1.024e6)
        disc_lr = np.random.choice(np.linspace(1e-5, 1e-3, num=6))
        gen_lr = np.random.choice(np.linspace(1e-5, 1e-3, num=6))
        n_epochs = 10

        # n_epochs = random.sample([2, 5, 8, 10], 1)[0]

        name = log_dir + f"/{i}"
        if not os.path.isdir(name):
            os.mkdir(name)
        else:
            print("The log directory already exists")
            raise SystemExit
        print(f"All tensorboard outputs and logging are being written inside {name}/.")

        model_dir = os.path.join(name, "model")
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        # Copy used file to logging folder
        f = open(model_dir + "/hyper_parameters.txt", 'w')
        f.write(f"expert_batch_size: {b_size}\n")
        f.write(f"total_timesteps: {total_timesteps}\n")
        f.write(f"gen_lr: {gen_lr}\n")
        f.write(f"disc_lr: {disc_lr}\n")
        f.write(f"n_epochs: {n_epochs}\n")
        f.close()
        logger.configure(name, format_strs=["stdout", "tensorboard"])

        algo = PPO(MlpPolicy,
                   env=env,
                   learning_rate=gen_lr,
                   n_steps=256,
                   batch_size=64,
                   gamma=0.99,
                   gae_lambda=0.95,
                   ent_coef=0.0,
                   n_epochs=10,
                   ent_schedule=1.0,
                   clip_range=0.2,
                   verbose=1,
                   device=device,
                   tensorboard_log=None,
                   policy_kwargs={'log_std_range': [-5, 5],
                                  'net_arch': [{'pi': [32, 32], 'vf': [32, 32]}],
                                  },
                   )

        airl_trainer = adversarial.AIRL(
            env,
            expert_data=transitions,
            expert_batch_size=b_size,
            gen_algo=algo,
            n_disc_updates_per_round=4,
            discrim_kwargs={"entropy_weight": 0.1},
            disc_opt_kwargs={"lr": disc_lr}
        )
        airl_trainer.train(total_timesteps=total_timesteps)

        disc_path = model_dir + "/discrim.pkl"
        with open(disc_path + ".tmp", "wb") as f:
            pickle.dump(airl_trainer.discrim, f)
        os.replace(disc_path + ".tmp", disc_path)
        airl_trainer.gen_algo.save(model_dir + "/gen")
        if isinstance(env, VecNormalize):
            env.save(model_dir + "/vec_normalize.pkl")
