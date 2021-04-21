import pytest

from IRL.project_policies import def_policy
from common.verification import verify_policy


@pytest.fixture()
def algo(env):
    algo = def_policy("ppo", env, device='cpu', log_dir=None, verbose=1)
    return algo


# def test_airl(env, algo):
#     airl_trainer = adversarial.AIRL(
#         env,
#         expert_data=transitions,
#         expert_batch_size=8,
#         gen_algo=algo,
#         discrim_kwargs={"entropy_weight": 0.1},
#         disc_opt_kwargs={"lr": 0.0060120000000000035}
#     )
#     airl_trainer.train(total_timesteps=2400000)
