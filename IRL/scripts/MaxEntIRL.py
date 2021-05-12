import os
import pickle
import shutil

from imitation.data import rollout

from common.util import make_env, create_path
from algo.torch.MaxEntIRL import MaxEntIRL

if __name__ == "__main__":
    env_type = "IP"
    algo_type = "MaxEntIRL"
    device = "cuda:0"
    name = "IP_custom"
    env = make_env(f"{name}-v2", use_vec_env=False, num_envs=8, n_steps=600, sub="sub01")

    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    log_dir = os.path.join(proj_path, "tmp", "log", env_type, algo_type, name)
    create_path(log_dir)
    shutil.copy(os.path.abspath(__file__), log_dir)

    expert_dir = os.path.join(proj_path, "demos", env_type, "expert.pkl")
    with open(expert_dir, "rb") as f:
        expert_trajs = pickle.load(f)
    transitions = rollout.flatten_trajectories(expert_trajs)

    name = log_dir + f"/0"
    if not os.path.isdir(name):
        os.mkdir(name)
    else:
        print("The log directory already exists")
        raise SystemExit
    print(f"All Tensorboards and logging are being written inside {name}/.")

    model_dir = os.path.join(name, "model")
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    learning = MaxEntIRL(env,
                         agent_learning_steps=5e4,
                         expert_transitions=transitions,
                         rew_lr=1e-5,
                         rew_arch=[3, 8, 8],
                         device=device,
                         sac_kwargs={'verbose': 1}
                         )
    losses = learning.learn(total_iter=100, gradient_steps=10, n_episodes=8)

    reward_path = model_dir + "/reward_net.pkl"
    with open(reward_path + ".tmp", "wb") as f:
        pickle.dump(learning.reward_net, f)
    os.replace(reward_path + ".tmp", reward_path)
    agent_path = model_dir + "/agent"
    learning.agent.save(agent_path)
    f = open(model_dir + "/losses.txt", 'w')
    for i, loss in enumerate(losses):
        f.write(f"{i}: {loss}\n")
    f.close()
