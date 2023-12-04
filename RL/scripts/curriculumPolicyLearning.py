import shutil
import numpy as np
from pathlib import Path
from scipy import io

from algos.torch.ppo import PPO
from common.util import make_env


if __name__ == "__main__":
    # 환경 설정
    env_type = "IP"
    algo_type = "ppo"
    env_id = f"{env_type}_MimicHuman"
    device = "cpu"
    subj = "sub04"
    isPseudo = False
    use_norm = True
    policy_num = 2
    tmp_num = 10
    base_curri_order = None
    curri_order = 1
    env_kwargs = {
        'PDgain': np.array([1000, 200]),
        'ankle_max': 100,
        # 'w': 0.0,
    }
    stptb = 1
    edptb = 4
    name_tail = f"_DeepMimic_ptb{stptb}to{edptb}/PD1000200_ankLim"
    except_trials = [13, 16]

    if isPseudo:
        env_type = "Pseudo" + env_type
    proj_dir = Path(__file__).parent.parent.parent
    subpath = (proj_dir / "demos" / env_type / subj / subj)
    states = [None for _ in range(35)]
    for i in range(5*(stptb - 1) + 1, 5*edptb + 1):
        humanData = io.loadmat(str(subpath) + f"i{i}.mat")
        bsp = humanData['bsp']
        states[i - 1] = humanData['state']
    for trial in except_trials:
        states[trial - 1] = None
    if use_norm:
        env_type += "_norm"

    log_dir = (Path(__file__).parent / "tmp" / "log" / env_type / (algo_type + name_tail) / f"policies_{policy_num}")
    prev_agent_dir = log_dir
    if base_curri_order is not None:
        prev_agent_dir = prev_agent_dir / f"curriculum_{base_curri_order}"
    if use_norm:
        use_norm = str((prev_agent_dir / f"normalization_{tmp_num}.pkl"))
    env = make_env(f"{env_id}-v2", num_envs=8, bsp=bsp, humanStates=states, use_norm=use_norm, **env_kwargs)

    algo = PPO.load(str(prev_agent_dir / f"agent_{tmp_num}"))

    log_dir = (log_dir / f"curriculum_{curri_order}")
    log_dir.mkdir(parents=True, exist_ok=False)
    algo.init_kwargs['tensorboard_log'] = str(log_dir)
    algo.reset_std(env)

    shutil.copy(str(Path(__file__)), str(log_dir))
    for i in range(15):
        algo.learn(total_timesteps=int(1e6), tb_log_name=f"extra", reset_num_timesteps=False)
        algo.save(str(log_dir / f"agent_{i + 1}"))
        if use_norm:
            algo.env.save(str(log_dir / f"normalization_{i + 1}.pkl"))
    print(f"Policy saved in curriculum_{curri_order}")
