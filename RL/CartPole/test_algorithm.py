import torch, gym
import numpy as np
from REINFORCE import policy_model, NormalizedWrapper

if __name__ == "__main__":
    env = NormalizedWrapper(gym.make("CartPoleCont-v0"))
    n = 1000
    pi = policy_model(env)
    pi.load_state_dict(torch.load("./model_parameters.pt"))
    pi.eval()

    ##### check #####
    x = env.reset().reshape(4, 1)
    pi.eval()
    env.render('human')
    for _ in range(n):
        a, _, _ = pi.forward(torch.Tensor(x).squeeze())
        ob, _, done, _ = env.step(np.array([a.item()]))
        x = ob.reshape(4, 1)
        env.render('human')
        if done:
            break
    env.close()