from algos.torch.ppo import PPO
from algos.torch.sac import SAC


def def_policy(algo_type, env, device='cpu', log_dir=None, verbose=0, **kwargs):
    if algo_type == "ppo":
        from algos.torch.ppo import MlpPolicy
        n_steps = 250
        batch_size = 256
        if hasattr(env, "num_envs"):
            n_steps = int(n_steps / int(env.num_envs))
        return PPO(MlpPolicy,
                   env=env,
                   n_steps=n_steps,
                   batch_size=batch_size,
                   gamma=0.99,
                   gae_lambda=0.95,
                   learning_rate=3e-4,
                   ent_coef=0.0,
                   n_epochs=10,
                   ent_schedule=1.0,
                   clip_range=0.2,
                   verbose=verbose,
                   device=device,
                   tensorboard_log=log_dir,
                   policy_kwargs={'log_std_range': [None, 1.8],
                                  'net_arch': [{'pi': [32, 32], 'vf': [32, 32]}],
                                  },
                   **kwargs,
                   )

    elif algo_type == "sac":
        from algos.torch.sac import MlpPolicy
        return SAC(MlpPolicy,
                   env=env,
                   batch_size=256,
                   learning_starts=100,
                   train_freq=1,
                   gradient_steps=1,
                   gamma=0.99,
                   ent_coef='auto',
                   verbose=verbose,
                   device=device,
                   tensorboard_log=log_dir,
                   policy_kwargs={'net_arch': {'pi': [32, 32], 'qf': [32, 32]},
                                  'optimizer_kwargs': {'betas': (0.9, 0.99)}},
                   **kwargs,
                   )
    else:
        raise NameError("Not implemented policy name")
