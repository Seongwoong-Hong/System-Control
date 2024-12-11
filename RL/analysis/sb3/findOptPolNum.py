import json
import shutil

import numpy as np

from common.analyzer import exec_policy
from common.path_config import LOG_DIR
from common.sb3.util import load_result, str2bool

if __name__ == "__main__":
    load_kwargs = {
        "algo_type": "ppo",
        "env_type": "IDP",
        "env_id": "MinEffort",
        "policy_num": 1,
        "log_dir": LOG_DIR / "modified_env",
        "trials": list(range(1, 36)),
        "device": 'cpu',
    }
    opt_dict = {}
    for ntail in ["limLevel_50"]:
        load_kwargs["name_tail"] = f"_MinEffort_ptb1to7/delay_passive/{ntail}"
        opt_rew, opt_pol_num = 0.0, 0
        for policy_num in range(8, 9):
            load_kwargs['policy_num'] = policy_num
            max_rew, opt_policy_num, opt_mean_len = 0.0, 0, 0
            for tmp_num in range(4, 16):
                load_kwargs['tmp_num'] = tmp_num
                env, agent, loaded_result = load_result(**load_kwargs)
                if env is None:
                    break
                _, _, rews, _, _ = exec_policy(env, agent, render=None, repeat_num=len(load_kwargs['trials']))
                mean_rew = np.mean([rew.sum() for rew in rews])
                mean_len = np.mean([len(rew) for rew in rews])
                if max_rew < mean_rew:
                    max_rew = mean_rew
                    opt_mean_len = mean_len
                    opt_policy_num = tmp_num
            if env is not None:
                print(f"for {load_kwargs['name_tail']}/policies_{policy_num}")
                print(f"opt@{opt_policy_num}, mean reward: {max_rew}, mean episode length: {opt_mean_len}")
                tg_agent = loaded_result['save_dir'] + f"/agent_{opt_policy_num}.zip"
                shutil.copy(tg_agent, loaded_result['save_dir'] + f"/agent_opt.zip")
                if str2bool(loaded_result['config']['use_norm']):
                    shutil.copy(loaded_result['save_dir'] + f"/normalization_{opt_policy_num}.pkl", loaded_result['save_dir'] + f"/normalization_opt.pkl")
                if opt_rew < max_rew:
                    opt_rew = max_rew
                    opt_pol_num = policy_num
        opt_dict[ntail] = f"optimal @{opt_pol_num}, max reward: {opt_rew}"
    print(opt_dict)
    json_opt_dict = json.dumps(opt_dict, indent=4)
    with open(str(load_kwargs['log_dir'] / "opt_policy_result.json"), 'w') as f:
        f.write(json_opt_dict)
