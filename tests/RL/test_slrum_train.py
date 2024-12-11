import json
from pathlib import Path

import pytest
import socket

from common.cluster_trainer import ClusterTrainer


train_config = json.loads(Path("../../common/config/tune_config.json").read_text())
cluster_config = json.loads(Path("../../common/config/slurm_cluster.json").read_text())
train_py = Path("../../RL/scripts/slurm_run.py")


@pytest.mark.skipif(condition='lab7003' != socket.gethostname(), reason="only lab7003 is allowed")
def test_estimator_cluster_train(tmp_path):
    (tmp_path / 'test').mkdir(parents=True, exist_ok=True)
    train_config['num_samples'] = 1
    cluster_config['sbatch_kwargs']['partition'] = "debug_hsw"

    trainer = ClusterTrainer(cluster_config, output_dir=tmp_path)
    trainer.run(
        train_py=train_py,
        name='test',
        num_samples=train_config['num_samples'],
        config=train_config,
    )

