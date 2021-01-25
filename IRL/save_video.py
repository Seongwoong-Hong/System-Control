import os, torch, gym, gym_envs, time, cv2
import numpy as np
from algo.torch.ppo import PPO
from algo.torch.OptCont.policies import LQRPolicy
from common.wrappers import CostWrapper
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

def VideoRecorde(imgs, filename, dt):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    width, height, _ = imgs[0].shape
    writer = cv2.VideoWriter(filename, fourcc, 1 / dt, (width, height))
    for img1 in imgs:
        img = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        writer.write(img)

GlfwContext(offscreen=True)
name = "2021-1-24-0-23-35"
num = 4
model_dir = os.path.join("tmp", "log", name, "model")
costfn = torch.load(model_dir + "/costfn{}.pt".format(num))
# algo = PPO.load(model_dir + "/ppo{}.zip".format(num))
algo = PPO.load(model_dir + "/extra_ppo.zip")
env = CostWrapper(gym.make("IP_custom-v1", n_steps=100), costfn)
exp = ExpertPolicy(env)
dt = env.dt
qs = np.array([[0.20, 0.06],
               [-0.20, -0.06],
               [0.25, -0.1],
               [-0.25, 0.1]])
imgs1, imgs2 = [], []

for q in qs:
    rew1_list = []
    rew2_list = []
    cost1_list = []
    cost2_list = []
    obs = env.reset()
    env.set_state(np.array([q[0]]), np.array([q[1]]))
    imgs1.append(env.render("rgb_array"))
    done = False
    while not done:
        act, _ = exp.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(act)
        cost = obs @ exp.Q @ obs.T + act*act*exp.R*exp.gear**2
        imgs1.append(env.render("rgb_array"))
        rew1_list.append(act.item())
        cost1_list.append(cost.item())
        time.sleep(dt)
    print(obs)
    env.reset()
    done = False
    env.set_state(np.array([q[0]]), np.array([q[1]]))
    imgs2.append(env.render("rgb_array"))
    while not done:
        act, _ = algo.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(act)
        cost = obs @ exp.Q @ obs.T + act*act*exp.R*exp.gear**2
        imgs2.append(env.render("rgb_array"))
        rew2_list.append(act.item())
        cost2_list.append(cost.item())
        time.sleep(dt)
    print(obs)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(cost1_list)
    ax2.plot(cost2_list)
    plt.show()
    # print(sum(rew1_list), sum(rew2_list))

VideoRecorde(imgs1, "videos/{}_expert.avi".format(name), dt)
VideoRecorde(imgs2, "videos/{}_agent.avi".format(name), dt)