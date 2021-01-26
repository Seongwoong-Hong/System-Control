import gym
import imageio
import numpy as np

from gym_envs.envs.cartpolecont import CartPoleContEnv


class CartPoleContTestEnv(gym.Env):
    def __init__(self, max_ep, high=np.pi, low=0):
        self.env = CartPoleContEnv(max_ep, high=high, low=low)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    def step(self, action):
        state, rew, done, info = self.env.step(action)
        cost = 0.5 * (state.T @ self.env.Q @ state + action * self.env.R * action)
        return state, cost, done, info

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        return self.env.render(mode)

    def set_state(self, state):
        self.env.set_state(state)

    def saving_gif(self, name="anim.gif", frames=None):
        if frames is None:
            raise ValueError("There are no frame inputs")
        print("Saving input frames as gif...")
        fps = int(1/self.tau)
        imageio.mimsave(name, [np.array(frame) for i, frame in enumerate(frames) if i % 2 == 0], fps=fps)

    def close(self):
        self.env.close()
