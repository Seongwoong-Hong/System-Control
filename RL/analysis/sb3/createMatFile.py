from pathlib import Path

import numpy as np
from scipy import io

from common.analyzer import exec_policy
from common.path_config import LOG_DIR, MAIN_DIR
from common.sb3.util import load_result

if __name__ == "__main__":

    target = "delay_passive/limLevel_50"
    load_kwargs = {
        "algo_type": "ppo",
        "env_type": "IDP",
        "env_id": "MinEffort",
        "trials": list(range(1, 36)),
        "policy_num": 3,
        "tmp_num": 80,
        "log_dir": LOG_DIR / "erectCost" / "constNotEarly0823",
        "name_tail": f"_MinEffort_ptb1to7/{target}/const1.0_vel0.1_atm100",
        "env_kwargs": {"ankle_torque_max": 120},
    }
    save_video = None

    env, agent, loaded_result = load_result(**load_kwargs)
    bsp = loaded_result['bsp']
    obs, _, _, _, ifs = exec_policy(env, agent, render=None, deterministic=True, repeat_num=len(load_kwargs['trials']))
    tqs = ifs['torque']

    bsps = []
    pltdds = [[] for _ in range(35)]
    pltqs = [[] for _ in range(35)]
    for trial in load_kwargs['trials']:
        humanData = io.loadmat(str(MAIN_DIR / "demos" / "IDP" / "sub10" / f"sub10i{trial}.mat"))
        bsps += [bsp]
        pltdd = env.get_attr("ptb_acc", 0)[0][:360]
        pltdds[env.get_attr("ptb_idx", 0)[0]] = pltdd
        m1 = 2 * sum(bsp[2:4, 0])
        m2 = bsp[6, 0]
        l1c = (bsp[2, 0] * bsp[2, 2] + bsp[3, 0] * (bsp[2, 1] + bsp[3, 2])) / sum(bsp[2:4, 0])
        l1 = sum(bsp[2:4, 1])
        l2c = bsp[6, 2]
        pltq1 = (m1*l1c*np.cos(obs[trial-1][:-1, 0]) + m2*l1*np.cos(obs[trial-1][:-1, 0]) + m2*l2c*np.cos(obs[trial-1][:-1, 1]))*pltdd
        pltq2 = m2*l2c*np.cos(obs[trial-1][:-1, 1])*pltdd
        pltqs[env.get_attr("ptb_idx", 0)[0]] = np.array([pltq1, pltq2]).T
        env.reset()

    tg_dir = Path(f"../MATLAB/{load_kwargs['env_id']}/{target}_lean3/sub10")
    tg_dir.mkdir(parents=True, exist_ok=True)
    for i, (ob, tq, bsp, pltdd, pltq) in enumerate(zip(obs, tqs, bsps, pltdds, pltqs)):
        data = {
            'bsp': bsp,
            'pltdd': pltdd,
            'pltq': pltq,
            'state': ob[:-1, :],
            'tq': tq,
        }
        io.savemat(str(tg_dir / f"sub10i{i+1}.mat"), data)
