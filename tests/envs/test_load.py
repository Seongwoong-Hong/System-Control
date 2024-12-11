import os
import pytest
from common.sb3.util import make_env
from scipy import io


@pytest.fixture
def demo_dir():
    return os.path.abspath(os.path.join("..", "..", "IRL", "demos"))


def test_bsp(demo_dir):
    file = os.path.join(demo_dir, "HPC", "sub01", "sub01i1")
    data = {'state': io.loadmat(file)['state'],
            'T': io.loadmat(file)['tq'],
            'pltq': io.loadmat(file)['pltq'],
            'bsp': io.loadmat(file)['bsp'],
            }
    env = make_env("HPC_custom-v1", use_vec_env=False, subpath=demo_dir + '/HPC/sub01/sub01', bsp=data['bsp'])
    print(data['bsp'])


def test_make_env(demo_dir):
    file = os.path.join(demo_dir, "HPC", "sub02", "sub02i2")
    env = make_env("HPC_pybullet-v1", use_vec_env=False, subpath=demo_dir + '/HPC/sub02/sub02')
    print(io.loadmat(file)['bsp'])


def test_load_error(IDPbsp, IDPhumanStates):
    for _ in range(100):
        make_env("IDP_MinEffort-v2", num_envs=8, bsp=IDPbsp, humanStates=IDPhumanStates)
