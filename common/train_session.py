import json
import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from common.path_config import MAIN_DIR
from common.cluster_trainer import ClusterTrainer

logging.basicConfig(
    format='[%(name)s] %(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO,
    datefmt='%y/%m/%d %H:%M:%S',
)

cur_dir = Path(__file__).parent
creation_time = datetime.now().strftime("%y/%m/%d-%H:%M:%S")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('name', type=str, help='session folder containing train data')
    parser.add_argument('target', type=str, choices=['TuneAlgoParameter'], help='target experiment type')
    parser.add_argument('--main_dir', type=str, default=MAIN_DIR, help='logged session data directory')
    parser.add_argument('--cluster_config', type=str, default='slurm_cluster.json', help='slurm configuration')
    parser.add_argument('--train_session', type=str, default='train', help='train session name')
    parser.add_argument('--tagging', nargs='+', type=str, default=None, help='used as neptune tag')
    args = parser.parse_args()

    # python common/train_session.py RL TuneAlgoParameter

    logger = logging.getLogger('session')
    logger.info(f'train starts!, session name is {args.name}, main_dir is {args.main_dir}')

    # load configurations
    target = args.target
    train_py = MAIN_DIR / 'RL' / 'scripts' / f'{target}.py'
    train_config = json.loads((MAIN_DIR / 'common' / 'config' / f'tune_config.json').read_text())
    cluster_config = json.loads((MAIN_DIR / 'common' / 'config' / args.cluster_config).read_text())

    # directory setup
    main_dir = Path(args.main_dir)
    session_dir = main_dir / args.name
    assert session_dir.exists(), f"session doesn't exist {session_dir}"

    # create and run trainer
    py_args = ""
    if train_config.get("time_limit"):
        py_args += f" --time_limit {train_config['time_limit']}"

    # add tagging
    if hasattr(args.tagging, '__iter__'):
        py_args += " --tagging " + " ".join(args.tagging)

    trainer = ClusterTrainer(cluster_config, main_dir=main_dir, train_session=args.train_session)
    trainer.run(
        train_py=train_py,
        name=args.name,
        num_samples=train_config['num_samples'],
        config=train_config,
        py_args=py_args,
    )
