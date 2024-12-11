from datetime import datetime

from RL.scripts.PolicyLearning import define_algo, TimeTrajCallback
from common.path_config import LOG_DIR


if __name__ == "__main__":
    # 환경 설정
    config = {
        "env_type": "IDP",
        "algo_type": "ppo",
        "env_id": "MinEffort",
        "device": "cuda",
        "subj": "sub10",
        "use_norm": "True",
        "isPseudo": "False",
        "log_dir": LOG_DIR / "test",
        "tags": [datetime.now().strftime('%Y-%m-%d-%H-%M-%S')],
        "env_kwargs": {
            'use_seg_ang': False,
            'ptb_act_time': 1/3,
            'ankle_limit': 'soft',
            'ankle_torque_max': 120,
            'delay': True,
            'delayed_time': 0.12,
            'const_ratio': 1.0,
            'tq_ratio': 0.5,
            'tqcost_ratio': 1.0,
            'ank_ratio': 0.5,
            'vel_ratio': 0.01,
            'limLevel': 0.9,
            'torque_rate_limit': False,
            'ptb_range': [0.03 + 0.015*i for i in range(10)],
        },
        "algo_kwargs": {
            "n_steps": 8192,
            "batch_size": 2048,
            "learning_rate": 1e-4,
            "n_epochs": 5,
            "gamma": 0.999,
            "gae_lambda": 0.99,
            "vf_coef": 0.5,
            "ent_coef": 3e-4,
            "policy_kwargs": {
                'net_arch': [{"pi": [128, 128], "vf": [128, 128]}],
                'log_std_range': [-10, None]
            },
        },
        "verbose": 0,
        "stptb": 1,
        "edptb": 7,
        "totaltimesteps": 60,  # millions
        "num_envs": 32,
    }

    #########################################################################
    # !!!!! You Need to Change this Name before you start the LEARNING!!!!! #
    #########################################################################
    name_tail = f"_{config['env_id']}_ptb{config['stptb']}to{config['edptb']}"
    if config['env_kwargs']['delay']:
        name_tail += "/delay_passive"
        config['env_kwargs']['stiffness'] = [300, 70]
        config['env_kwargs']['damping'] = [30, 10]
    else:
        name_tail += "/no_delay"
    if config['env_kwargs']['ankle_limit'] == 'hard':
        name_tail += "/limHard"
    elif config['env_kwargs']['ankle_limit'] == 'soft':
        name_tail += f"/limLevel_{round(config['env_kwargs']['limLevel'] * 100)}"
    else:
        name_tail += "/limSatu"
    config["name_tail"] = name_tail + f"/const{config['env_kwargs']['const_ratio']}_vel{config['env_kwargs']['vel_ratio']}"
    algo = define_algo(input_argu_dict=config, **config)

    callback = TimeTrajCallback([t * 5 + 1 for t in range(config['stptb'] - 1, config['edptb'])], **config)
    for i in range(config['totaltimesteps']):
        algo.learn(total_timesteps=int(1e6), tb_log_name=f"tensorboard_log", reset_num_timesteps=False, callback=callback)
        algo.save(algo.tensorboard_log + f"/agent_{i + 1}")
        if config['use_norm']:
            algo.env.save(algo.tensorboard_log + f"/normalization_{i + 1}.pkl")
    print(f"Policy saved in {algo.tensorboard_log}")
