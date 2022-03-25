import os
import shutil
from scipy import io

from algos.tabular.viter import SoftQiter, FiniteSoftQiter
from algos.torch.sac import SAC, MlpPolicy
# from stable_baselines3.sac import SAC, MlpPolicy
from common.util import make_env


if __name__ == "__main__":
    env_type = "2DTarget"
    algo_type = "sac"
    name = f"{env_type}"
    device = "cpu"
    env_id = f"{name}-v2"
    subj = "sub06"
    irl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "IRL"))
    subpath = os.path.join(irl_dir, "demos", "HPC", subj, subj)
    bsp = io.loadmat(subpath + f"i1.mat")['bsp']
    # env = make_env(env_id, num_envs=1, N=[17, 17, 17, 19], NT=[11, 11], bsp=bsp)
    env = make_env(env_id, map_size=1)
    # name += f"_{subj}"
    current_path = os.path.dirname(__file__)
    log_dir = os.path.join(current_path, env_type, "tmp", "log", name, algo_type)
    os.makedirs(log_dir, exist_ok=True)
    # algo = FiniteSoftQiter(env, gamma=1, alpha=0.
    # 001, device=device)
    algo = SAC(
        MlpPolicy,
        env=env,
        gamma=0.99,
        ent_coef=0.00001,
        tau=0.01,
        buffer_size=int(1e5),
        learning_starts=10000,
        train_freq=1,
        gradient_steps=1,
        tensorboard_log=log_dir,
        device=device,
        verbose=1,
        policy_kwargs={'net_arch': [32, 32]}
    )
    n = 1
    while os.path.isfile(log_dir + f"/policies_{n}/agent.zip"):
        n += 1
    os.makedirs(log_dir + f"/policies_{n}", exist_ok=False)
    shutil.copy(os.path.abspath(__file__), log_dir + f"/policies_{n}")
    algo.learn(total_timesteps=int(1.5e5), tb_log_name=f"extra_{n}", reset_num_timesteps=False)
    algo.save(log_dir + f"/policies_{n}/agent")
    if algo.get_vec_normalize_env():
        algo.env.save(log_dir + f"/policies_{n}/normalization.pkl")
    print(f"saved as policies_{n}/agent.pkl")
