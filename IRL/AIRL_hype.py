import os
import shutil
import pickle
import random
import numpy as np

from imitation.algorithms import adversarial
from imitation.data import rollout
from imitation.util import logger
from common.util import make_env

from algo.torch.ppo import PPO, MlpPolicy


if __name__ == "__main__":
    env_type = "IDP"
    algo_type = "ppo"
    device = "cpu"
    env = make_env(f"{env_type}_custom-v1", num_envs=10, n_steps=600, sub="sub01")

    current_path = os.path.dirname(__file__)
    log_dir = os.path.join(current_path, "tmp", "log", env_type, algo_type, "AIRL_test")
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    # expert_dir = os.path.join(current_path, "demos", env_type, sub + ".pkl")
    expert_dir = os.path.join(current_path, "demos", env_type, "expert.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    transitions = rollout.flatten_trajectories(expert_trajs)

    for i in range(200):
        b_size = random.sample([8, 16, 32], 1)[0]
        total_timesteps = int(random.sample([1.2e6, 3e6, 6e6, 12e6, 24e6], 1)[0])
        disc_lr = np.log10(np.random.choice(np.logspace(3e-5, 1e-2, num=6)))
        gen_lr = np.log10(np.random.choice(np.logspace(3e-5, 1e-2, num=6)))

        name = log_dir + "/" + str(i)
        if not os.path.isdir(name):
            os.mkdir(name)
        else:
            print("The log directory already exists")
            raise SystemExit
        print(f"All Tensorboards and logging are being written inside {name}/.")

        model_dir = os.path.join(name, "model")
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        # Copy used file to logging folder
        shutil.copy(os.path.abspath(__file__), model_dir)
        f = open(model_dir + "/hyper_parameters.txt", 'w')
        f.write(f"expert_batch_size: {b_size}\n")
        f.write(f"total_timesteps: {total_timesteps}\n")
        f.write(f"gen_lr: {gen_lr}\n")
        f.write(f"disc_lr: {disc_lr}\n")
        f.close()
        logger.configure(name, format_strs=["stdout", "log", "csv", "tensorboard"])

        algo = PPO(MlpPolicy,
                   env=env,
                   learning_rate=gen_lr,
                   n_steps=1200,
                   batch_size=200,
                   gamma=0.975,
                   gae_lambda=0.95,
                   ent_coef=0.0,
                   n_epochs=10,
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
            discrim_kwargs={"entropy_weight": 0.1},
            disc_opt_kwargs={"lr": disc_lr}
        )
        airl_trainer.train(total_timesteps=total_timesteps)

        disc_path = model_dir + "/discrim.pkl"
        with open(disc_path + ".tmp", "wb") as f:
            pickle.dump(airl_trainer.discrim, f)
        os.replace(disc_path + ".tmp", disc_path)

        gen_path = model_dir + "/gen"
        airl_trainer.gen_algo.save(gen_path)
