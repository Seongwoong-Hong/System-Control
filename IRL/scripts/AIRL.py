import os
import shutil
import pickle
import datetime

from imitation.algorithms import adversarial
from imitation.data import rollout
from imitation.util import logger

from IRL.scripts.project_policies import def_policy
from common.util import make_env

if __name__ == "__main__":
    env_type = "mujoco_envs"
    algo_type = "ppo"
    device = "cpu"
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # env = make_env(f"{env_type}_custom-v0", sub="sub01", num_envs=10)
    env = make_env("HalfCheetah-v2", num_envs=8)

    # expert_dir = os.path.join(current_path, "demos", env_type, sub + ".pkl")
    expert_dir = os.path.join(proj_path, "demos", env_type, "halfcheetah.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    transitions = rollout.flatten_trajectories(expert_trajs)

    algo = def_policy(algo_type, env, device=device, log_dir=None, verbose=1)

    now = datetime.datetime.now()
    log_dir = os.path.join(proj_path, "tmp", "log", env_type, algo_type, "HalfCheetah")
    name = "/default"
    # name = "/%s-%s-%s-%s-%s-%s" % (now.year, now.month, now.day, now.hour, now.minute, now.second)
    log_dir += name
    print(f"All Tensorboards and logging are being written inside {log_dir}/.")

    model_dir = os.path.join(log_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    # Copy used file to logging folder
    # shutil.copy(os.path.abspath(proj_path + "/../gym_envs/envs/{}_custom.py".format(env_type)), model_dir)
    shutil.copy(os.path.abspath(__file__), model_dir)
    # shutil.copy(os.path.abspath(proj_path + "/project_policies.py"), model_dir)

    logger.configure(log_dir, format_strs=["stdout", "log", "csv", "tensorboard"])

    airl_trainer = adversarial.AIRL(
        env,
        expert_data=transitions,
        expert_batch_size=1024,
        gen_algo=algo,
        n_disc_updates_per_round=4,
        discrim_kwargs={"entropy_weight": 0.1},
        disc_opt_kwargs={"lr": 1e-4}
    )
    airl_trainer.train(total_timesteps=int(2e6))

    disc_path = model_dir + "/discrim.pkl"
    with open(disc_path + ".tmp", "wb") as f:
        pickle.dump(airl_trainer.discrim, f)
    os.replace(disc_path + ".tmp", disc_path)

    gen_path = model_dir + "/gen"
    airl_trainer.gen_algo.save(gen_path)
