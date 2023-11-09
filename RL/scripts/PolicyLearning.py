import os
import shutil
from scipy import io

# from algos.torch.sac import SAC, MlpPolicy
from algos.torch.ppo import PPO, MlpPolicy
from common.util import make_env
from common.wrappers import ActionWrapper, DiscretizeWrapper


if __name__ == "__main__":
    env_type = "IP"
    algo_type = "ppo"
    name = f"{env_type}_custom"
    device = "cpu"
    env_id = f"{name}-v2"
    subj = "sub05"
    irl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "IRL"))
    subpath = os.path.join(irl_dir, "demos", "HPC", subj, subj)
    bsp = io.loadmat(subpath + f"i1.mat")['bsp']
    env = make_env(env_id)
    # env = make_env(env_id, num_envs=1, N=[19, 19, 19, 19], NT=[11, 11], bsp=bsp, wrapper=ActionWrapper)
    # env = make_env(env_id, map_size=1)
    name += f"_{subj}"
    current_path = os.path.dirname(__file__)
    log_dir = os.path.join(current_path, env_type, "tmp", "log", name, algo_type)
    os.makedirs(log_dir, exist_ok=True)
    # algo = FiniteSoftQiter(env, gamma=1, alpha=0.001, device=device)
    algo = PPO(
        MlpPolicy,
        env=env,
        n_steps=2048*5,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.9,
        vf_coef=0.5,
        ent_coef=0.0,
        tensorboard_log=log_dir,
        device=device,
        verbose=1,
    )
    n = 1
    while os.path.isdir(log_dir + f"/policies_{n}"):
        n += 1
    os.makedirs(log_dir + f"/policies_{n}", exist_ok=False)
    shutil.copy(os.path.abspath(__file__), log_dir + f"/policies_{n}")
    for i in range(10):
        algo.learn(total_timesteps=int(1e6), tb_log_name=f"extra_{n}", reset_num_timesteps=False)
        algo.save(log_dir + f"/policies_{n}/agent_{i+1}")
    # if algo.get_vec_normalize_env():
    #     algo.env.save(log_dir + f"/policies_{n}/normalization.pkl")
    print(f"saved as policies_{n}/agent.pkl")
