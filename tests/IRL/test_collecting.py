import pickle
from common.sb3.util import make_env
from imitation.data import rollout
from scipy import io


def test_collect_hpc():
    env = make_env("HPC_custom-v1", use_vec_env=True, num_envs=1, subpath="../../IRL/demos/HPC/sub01/sub01")
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=10)
    ExpertPolicy = def_policy("HPC", env, noise_lv=0.25)
    trajectories = rollout.generate_trajectories(ExpertPolicy, env, sample_until, deterministic_policy=False)
    print(trajectories[0].obs[-1, :])
    assert len(trajectories[0]) == 600


def test_collecting_from_data():
    env = make_env("DiscretizedHuman-v2", N=[17, 17, 17, 19], NT=[11, 11])
    file = "../../demos/HPC/sub06_half/sub06i1_0.mat"
    init_state = env.get_obs_from_idx(env.get_idx_from_obs(-io.loadmat(file)['state'][0][:4]))
    print(init_state)
    init_action = env.get_acts_from_idx(env.get_idx_from_acts(-io.loadmat(file)['tq'][0]))
    print(init_action)
