import os
import gym, gym_envs, imageio
import numpy as np
from stable_baselines import PPO2, DDPG
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from RL.algo.torch.ppo import PPO
from matplotlib import pyplot as plt
from matplotlib import animation

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        action *= 0.125 * (self.action_space.high - self.action_space.low)
        return np.clip(action, self.action_space.low, self.action_space.high)

a = "tf/"
name = "IP_ctl/" + a + "ppo_ctl_Comp"
log_dir = "tmp/" + name + ".zip"
stats_dir = "tmp/" + name + ".pkl"
env_name = "CartPoleContTest-v0"
stats_path = os.path.join(stats_dir)

# Load the agent
if a == "tf/":
    model = PPO2.load(log_dir)
    env = NormalizedActions(gym.make(id=env_name, max_ep=1000))
    # env = DummyVecEnv([lambda: NormalizedActions(gym.make(id=env_name, max_ep=1000))])
    # env = VecNormalize.load(stats_path, env)
elif a == "torch/":
    model = PPO.load(log_dir)
    env = NormalizedActions(gym.make(id=env_name, max_ep=1000))

else:
    raise ValueError("Check typo")

# Load the saved statistics
# env = DummyVecEnv([lambda: gym.make(env_name)])
#  do not update them at test time
env.training = False
# reward normalization is not needed at test time
env.norm_reward = False


max_step = 1000
obs_result = np.zeros((4, max_step))
act_result = np.zeros(max_step)
cost_result = np.zeros(max_step)
rew_result = np.zeros(max_step)
# coeff_result = []


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

frames = []
for _ in range(1):
    step = 0
    obs = env.reset()
    env.set_state(np.array([0, 0, 0, 1.2]))
    obs = env.__getattr__('state')
    # env.env_method('set_state', np.array([0.1]), np.array([0.05]))
    # obs = env.normalize_obs(env.env_method('_get_obs'))
    # obs = np.concatenate((env.get_attr('data')[0].qpos, env.get_attr('data')[0].qvel, env.get_attr('data')[0].qfrc_constraint))
    done = False
    while (not done) and (step < max_step):
        obs_result[:, step] = env.__getattr__('state')
        frames.append(env.render("rgb_array"))
        act, _ = model.predict(obs, deterministic=True)
        obs, cost, done, info = env.step(act)
        action = info['action']
        # coeff_result.append(act.squeeze().tolist())
        act_result[step] = action
        rew_result[step] = np.exp(cost)
        cost_result[step] = cost
        step += 1

env.close()
# save_frames_as_gif(frames)
# imageio.mimsave("anim.gif", [np.array(frames) for i, img in enumerate(frames) if i%2 == 0], fps=29)
print(np.sum(cost_result))
plt.plot(cost_result)
plt.show()

plt.plot(act_result)
plt.show()