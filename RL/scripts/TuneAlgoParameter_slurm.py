import os
import re

import numpy as np

import sys

import json
from pathlib import Path
from argparse import ArgumentParser

import random

from RL.scripts.PolicyLearning import define_algo
from common.path_config import LOG_DIR

os.environ["MKL_THREADING_LAYER"] = "GNU"
sys.path.append(str(Path(__file__).parent.joinpath('..')))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', type=str, help='json string for sampled train configuration')
    parser.add_argument('--job_dir', type=str, help='slurm job directory')
    parser.add_argument('--log_dir', type=str, help='train data directory')
    parser.add_argument('--report', type=str, default=None, help='report file name')
    parser.add_argument('--time_limit', type=int, default=None, help='bound on train time')
    parser.add_argument('--use_neptune', action='store_true', default=False, help='use neptune logger')
    parser.add_argument('--creation_time', type=str, default=None, help='used as neptune tag')
    parser.add_argument('--tagging', nargs='+', type=str, default=None, help='used as neptune tag')
    args = parser.parse_args()

    algo_kwargs = json.loads(args.config)
    algo_kwargs['policy_kwargs'] = {
        'net_arch': [{"pi": [64, 64], "vf": [64, 64]}],
        'log_std_range': [-10, None]
    },
    job_dir = Path(args.job_dir)
    log_dir = Path(args.log_dir)

    job_num = int(re.findall(r'\d+', job_dir.name)[0])
    np.random.seed(job_num)
    random.seed(job_num)

    if not job_dir.exists():
        job_dir.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "env_type": "IDP",
        "algo_type": "ppo",
        "env_id": "MinEffort",
        "device": "cuda:3",
        "subj": "sub04",
        "use_norm": "True",
        "isPseudo": "False",
        "log_dir": LOG_DIR / "TuneParam",
        "env_kwargs": {
            'use_seg_ang': False,
            'ptb_act_time': 1/3,
            'ankle_max': 100,
            'delay': True,
            'cost_ratio': 0.5,
            'state_ratio': 0.5,
            'stiffness': 500,
            'damping': 50,
            'limLevel': 0.0,  # 0(soft) ~ 1(hard)
            'torque_rate_limit': False,
        },
        'algo_kwargs': algo_kwargs,
        "stptb": 1,
        "edptb": 7,
        "totaltimesteps": 10,  # millions
    }

    define_algo(input_argu_dict=config_dict, **config_dict)
