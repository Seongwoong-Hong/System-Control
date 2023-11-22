from algos.torch.ppo import PPO
from algos.torch.ppo import MlpPolicy as PPOMlpPolicy
from algos.torch.sac import SAC
from algos.torch.sac import MlpPolicy as SACMlpPolicy
from common.util import make_env


def test_iter_predict():
    env = make_env("2DTarget-v2", map_size=1)
    algo = SAC(
        SACMlpPolicy,
        env=env,
        gamma=0.99,
        ent_coef='auto',
        tau=0.01,
        buffer_size=int(5e4),
        learning_starts=10000,
        train_freq=1,
        gradient_steps=1,
        device='cpu',
        verbose=1,
        policy_kwargs={'net_arch': [32, 32]}
    )
    algo.learn(10000)


def test_ppo_algo(ip_env):
    env = ip_env
    algo = PPO(
        PPOMlpPolicy,
        env=env,
        n_steps=4096,
        batch_size=1024,
        learning_rate=0.0003,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        vf_coef=0.5,
        ent_coef=0.001,
        device='cpu',
        policy_kwargs={'net_arch': [dict(pi=[16, 16], vf=[32, 32])]},
        verbose=1,
    )
    algo.learn(total_timesteps=int(1e6))
