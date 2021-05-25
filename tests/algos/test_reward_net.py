from algos.torch.MaxEntIRL import RewardNet
import pickle
import dill
import torch as th


def test_pickling():
    reward_net = RewardNet(inp=4, arch=[], feature_fn=lambda x: th.square(x)).double()
    with open("test.pkl", "wb") as f:
        dill.dump(reward_net, f)


def test_loading():
    with open("test.pkl", "rb") as f:
        rew = dill.load(f)