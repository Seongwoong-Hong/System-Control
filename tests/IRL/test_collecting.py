from common.util import make_env
from imitation.data import rollout
from IRL.scripts.project_policies import def_policy


def test_collect_hpc():
    env = make_env("HPC_custom-v1", use_vec_env=True, num_envs=1, subpath="../../IRL/demos/HPC/sub01/sub01")
    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=10)
    ExpertPolicy = def_policy("HPC", env, noise_lv=0.25)
    trajectories = rollout.generate_trajectories(ExpertPolicy, env, sample_until, deterministic_policy=False)
    print(trajectories[0].obs[-1, :])
    assert len(trajectories[0]) == 600
