import numpy as np
from matplotlib import pyplot as plt

from common.analyzer import exec_policy
from common.sb3.util import load_result


if __name__ == "__main__":
    load_kwargs = {
        "env_type": "IDP",
        "env_id": "MinEffort",
        "subj": "sub04",
        "trials": list(range(1, 36)),
        "use_norm": True,
        "isPseudo": False,
        "policy_num": 1,
        "tmp_num": 30,
        "name_tail": "_MinEffort_ptb1to7/direcTq_5vs5_softLim",
        "curri_order": None,
        "env_kwargs": {},
        "except_trials": [13, 16],
    }
    save_video = None

    env, agent, loaded_result = load_result(**load_kwargs)

    render = "rgb_array"
    if save_video is None:
        render = None
    obs, acts, _, imgs, tqs = exec_policy(env, agent, render=render, deterministic=True, repeat_num=len(load_kwargs['trials']))
    if load_kwargs['use_norm']:
        norm_obs = []
        for ob in obs:
            norm_obs.append(env.unnormalize_obs(ob))
        del obs
        obs = norm_obs

    agt_Rs = []
    for agt_ob in obs:
        agt_Rs.append(np.corrcoef(agt_ob[40:80, 0], agt_ob[40:80, 1])[0, 1])
    hum_Rs = []
    for hum_ob in loaded_result['states']:
        if hum_ob is not None:
            hum_Rs.append(np.corrcoef(hum_ob[40:80, 0], hum_ob[40:80, 1])[0, 1])

    plt.plot(agt_Rs)
    plt.plot(hum_Rs)
    plt.tight_layout()
    plt.show()
