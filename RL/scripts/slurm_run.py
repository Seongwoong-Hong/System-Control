import json
from argparse import ArgumentParser
from pathlib import Path

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('config', type=str, help='json string for sampled train configuration')
    parser.add_argument('--job_dir', type=str, help='slurm job directory')

    args = parser.parse_args()
    config_dict = json.loads(args.config)
    name_tail = f"_MinEffort_direcTq_ptb{config_dict['stptb']}to{config_dict['edptb']}/5vs5_softLim/stateCost_8vs2"
    config_dict["name_tail"] = name_tail
    # job_dir = Path(args.job_dir)
    # log_dir = Path(args.log_dir)
    #
    # if not job_dir.exists():
    #     job_dir.mkdir(parents=True, exist_ok=True)
    # main(**config_dict)
    print("correct?")



