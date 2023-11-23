import shutil
import numpy as np
from pathlib import Path

from scipy import io

from algos.torch.ppo import PPO, MlpPolicy
from common.util import make_env


if __name__ == "__main__":
    # 환경 설정
    env_type = "IDP"
    algo_type = "ppo"
    env_id = f"{env_type}_custom"
    device = "cpu"
    subj = "sub04"
    isPseudo = True
    use_norm = True
    PDgain = np.array([1000, 10])
    stptb = 1
    edptb = 6
    ankle_max = 100
    name_tail = f"_DeepMimic_actionSkip_ptb{stptb}to{edptb}/PD{PDgain[0]}{PDgain[1]}_ankLim"

    if isPseudo:
        env_type = "Pseudo" + env_type
    proj_dir = Path(__file__).parent.parent.parent
    subpath = (proj_dir / "demos" / env_type / subj / subj)
    states = [None for _ in range(35)]
    for i in range(5*(stptb - 1) + 1, 5*edptb + 1):
        humanData = io.loadmat(str(subpath) + f"i{i}.mat")
        bsp = humanData['bsp']
        states[i - 1] = humanData['state']
    env = make_env(f"{env_id}-v2", num_envs=8, bsp=bsp, humanStates=states, use_norm=use_norm, PDgain=PDgain, ankle_max=ankle_max)
    if use_norm:
        env_type += "_norm"
    log_dir = (Path(__file__).parent / "tmp" / "log" / env_type / (algo_type + name_tail))
    log_dir.mkdir(parents=True, exist_ok=True)
    algo = PPO(
        MlpPolicy,
        env=env,
        n_steps=512,
        batch_size=1024,
        learning_rate=0.0003,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        vf_coef=0.5,
        ent_coef=0.003,
        tensorboard_log=str(log_dir),
        device=device,
        policy_kwargs={'net_arch': [dict(pi=[16, 16], vf=[32, 32])]},
        verbose=1,
    )
    n = 1
    while (log_dir / f"policies_{n}").is_dir():
        n += 1
    (log_dir / f"policies_{n}").mkdir(parents=True, exist_ok=False)
    shutil.copy(str(Path(__file__)), str(log_dir / f"policies_{n}"))
    for i in range(15):
        algo.learn(total_timesteps=int(1e6), tb_log_name=f"extra_{n}", reset_num_timesteps=False)
        algo.save(str(log_dir / f"policies_{n}/agent_{i + 1}"))
        if use_norm:
            algo.env.save(str(log_dir / f"policies_{n}/normalization_{i + 1}.pkl"))
    print(f"Policy saved in policies_{n}")
