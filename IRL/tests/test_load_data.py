import gym_envs
import torch
from algo.torch.GCL.modules import CostNet
from common.wrappers import CostWrapper
from common.rollouts import generate_trajectories_from_data
from stable_baselines3.common.vec_env import DummyVecEnv
from scipy import io

sub = "sub%02d" % 1
file = "../demos/HPC/" + sub + "/" + sub + "i%d.mat" % 1
data = {'state': io.loadmat(file)['state'],
        'T': io.loadmat(file)['tq'],
        'pltq': io.loadmat(file)['pltq'],
        'bsp': io.loadmat(file)['bsp'],
        }

env_type = "HPC"
device = 'cpu'
n_steps, n_episodes = 300, 5
steps_for_learn = 1536000
env_id = "{}_custom-v0".format(env_type)
env = gym_envs.make(env_id, n_steps=n_steps, bsp=data['bsp'], pltq=data['pltq'])
num_obs = env.observation_space.shape[0]
num_act = env.action_space.shape[0]

costfn = CostNet(arch=[num_obs, 2*num_obs],
                 act_fcn=torch.nn.ReLU,
                 device=device,
                 num_expert=15,
                 num_samp=n_episodes,
                 lr=3e-4,
                 decay_coeff=0.0,
                 num_act=num_act
                 ).double().to(device)

env = DummyVecEnv([lambda: CostWrapper(env, costfn) for i in range(5)])
traj = generate_trajectories_from_data(data, env)
