import time
import numpy as np

from copy import deepcopy


def verify_policy(environment, policy, render=True):
    action_list = np.zeros((1,) + environment.action_space.shape)
    obs = environment.reset()
    ob_list = deepcopy(obs.reshape(1, -1))
    done = False
    if render:
        environment.render()
    while not done:
        act, _ = policy.predict(obs, deterministic=True)
        obs, rew, done, info = environment.step(act)
        if render:
            environment.render()
        action_list = np.append(action_list, act.reshape(1, -1), 0)
        ob_list = np.append(ob_list, obs.reshape(1, -1), 0)
        time.sleep(environment.dt)
    return action_list, ob_list
