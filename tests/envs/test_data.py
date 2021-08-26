import os
import pickle
import numpy as np

from IRL.scripts.project_policies import def_policy
from common.util import make_env

from imitation.data import rollout
from matplotlib import pyplot as plt
from scipy import io


def test_hpc_data():
    subi = 1
    sub = f"sub{subi:02d}"
    file = "../../IRL/demos/HPC/" + sub + "/" + sub + f"i{0 + 1}.mat"
    data = {'state': io.loadmat(file)['state'],
            'T': io.loadmat(file)['tq'],
            'pltq': io.loadmat(file)['pltq'],
            'bsp': io.loadmat(file)['bsp'],
            }
    plt.plot(data['state'][:, :2])
    plt.show()
    pltqs = [data['pltq']]
    env = make_env("HPC_custom-v0", use_vec_env=False, n_steps=600, pltqs=pltqs)
    algo = def_policy("HPC", env)
    obs_list = []
    obs = env.reset()
    obs_list.append(obs)
    done = False
    while not done:
        act, _ = algo.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(act)
        obs_list.append(obs)
    plt.plot(np.array(obs_list)[:, :2])
    plt.show()
    print(data['state'][0, :4])


def test_pkl_data():
    expert_dir = os.path.join("../../IRL", "demos", "HPC", "sub01_4.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    transitions = rollout.flatten_trajectories(expert_trajs)
    plt.plot(transitions.obs)
    plt.show()
