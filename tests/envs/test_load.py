import os
from common.util import make_env
from scipy import io


def test_bsp():
    file = os.path.abspath(os.path.join("..", "..", "IRL", "demos", "HPC", "sub01", "sub01i1"))
    data = {'state': io.loadmat(file)['state'],
            'T': io.loadmat(file)['tq'],
            'pltq': io.loadmat(file)['pltq'],
            'bsp': io.loadmat(file)['bsp'],
            }
    env = make_env("HPC_custom-v1", use_vec_env=False, subpath='../../IRL/demos/HPC/sub01/sub01', bsp=data['bsp'])
    print(data['bsp'])


