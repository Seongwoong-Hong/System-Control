import os
import pickle
import pathlib
import gym_envs
import datetime

from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms import adversarial
from imitation.data import rollout
from imitation.util import logger

from IRL.project_policies import def_policy
from mujoco_py import GlfwContext
from scipy import io


if __name__ == "__main__":
    env_type = "HPC"
    algo_type = "ppo"
    num = 2
    env_id = "{}_custom-v0".format(env_type)
    n_steps = 600
    device = "cpu"
    current_path = os.path.dirname(__file__)
    GlfwContext(offscreen=True)
    sub = "sub01"
    expert_dir = os.path.join(current_path, "demos", env_type, sub + ".pkl")
    pltqs = []

    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
        learner_trajs = []
    transitions = rollout.flatten_trajectories(expert_trajs)

    for i in range(35):
        file = os.path.join(current_path, "demos", env_type, sub, sub + "i%d.mat" % (i + 1))
        pltqs += [io.loadmat(file)['pltq']]

    env = DummyVecEnv([lambda: gym_envs.make(env_id, n_steps=n_steps, pltqs=pltqs) for i in range(10)])

    algo = def_policy(algo_type, env, device=device, log_dir=None, verbose=1)

    now = datetime.datetime.now()
    current_path = os.path.dirname(__file__)
    log_dir = os.path.join(current_path, "tmp", "log", env_type, algo_type, "AIRL")
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    name = "/%s-%s-%s-%s-%s-%s" % (now.year, now.month, now.day, now.hour, now.minute, now.second)
    log_dir += name
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    print(f"All Tensorboards and logging are being written inside {log_dir}/.")

    logger.configure(log_dir, format_strs=["stdout", "log", "csv", "tensorboard"])
    airl_trainer = adversarial.AIRL(
        env,
        expert_data=transitions,
        expert_batch_size=15,
        gen_algo=algo,
    )
    airl_trainer.train(total_timesteps=10240000)

    path = log_dir + "result.pkl"
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(airl_trainer, f)
    os.replace(path + ".tmp", path)
