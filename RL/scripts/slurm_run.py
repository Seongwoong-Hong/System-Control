import json
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from datetime import datetime
import time
from functools import partial
from scipy import io

# from ray import tune

# from algos.torch.sac import SAC, MlpPolicy
# from algos.torch.ppo import PPO, MlpPolicy
# from common.util import make_env
# from common.wrappers import ActionWrapper

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('config', type=str, help='json string for sampled train configuration')
    parser.add_argument('--job_dir', type=str, help='slurm job directory')
    args = parser.parse_args()
    #
    # config = json.loads(args.config)
    # job_dir = Path(args.job_dir)
    # log_dir = Path(args.log_dir)
    #
    # if not job_dir.exists():
    #     job_dir.mkdir(parents=True, exist_ok=True)
    time.sleep(10)

