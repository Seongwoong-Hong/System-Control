import os, torch, gym, gym_envs, time, cv2
import numpy as np
from algo.torch.ppo import PPO
from algo.torch.OptCont.policies import LQRPolicy
from matplotlib import pyplot as plt
from mujoco_py import GlfwContext

class ExpertPolicy(LQRPolicy):
    def _build_env(self):
        m, g, h, I = 5.0, 9.81, 0.5, 1.667
        self.Q = np.array([[1, 0], [0, 1]])
        self.R = 0.001
        self.A = np.array([[0, m*g*h/I], [1, 0]])
        self.B = np.array([[1/I], [0]])
        return self.A, self.B, self.Q, self.R

GlfwContext(offscreen=True)
model_dir = os.path.join("..", "tmp", "model")
# rwfn = torch.load(model_dir + "/bar1_rwfn.pt")
algo = PPO.load(model_dir + "/bar7_ppo.zip")
env = gym.make("IP_custom-v2", n_steps=100)
exp = ExpertPolicy(env)
dt = env.dt
q = np.array([0.12, 0.06])
imgs1, imgs2 = [], []

for _ in range(1):
    rew_list = []
    cost1_list = []
    cost2_list = []
    obs = env.reset()
    env.set_state(np.array([q[0]]), np.array([q[1]]))
    imgs1.append(env.render("rgb_array"))
    done = False
    while not done:
        act, _ = exp.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(act)
        cost = obs @ exp.Q @ obs.T + act*act*exp.R
        imgs1.append(env.render("rgb_array"))
        rew_list.append(rew.item())
        cost1_list.append(cost.item())
        time.sleep(dt)
    env.reset()
    done = False
    env.set_state(np.array([q[0]]), np.array([q[1]]))
    imgs2.append(env.render("rgb_array"))
    while not done:
        act, _ = algo.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(act)
        cost = obs @ exp.Q @ obs.T + act*act*exp.R
        imgs2.append(env.render("rgb_array"))
        rew_list.append(rew.item())
        cost2_list.append(cost.item())
        time.sleep(dt)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(cost1_list)
    ax2.plot(cost2_list)
    plt.show()
    print(sum(cost1_list), sum(cost2_list))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    width, height, _ = imgs1[0].shape
    writer1 = cv2.VideoWriter("expert.avi", fourcc, 1/dt, (width, height))
    writer2 = cv2.VideoWriter("agent.avi", fourcc, 1/dt, (width, height))
    for img1 in imgs1:
        img = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        writer1.write(img)
    for img2 in imgs2:
        img = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        writer2.write(img)