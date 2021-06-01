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


def test_feature_fn():
    def feature_fn(x):
        return th.square(x)
    reward_net = RewardNet(inp=4, arch=[], feature_fn=feature_fn).double().eval()
    inp = th.tensor([1, 2, 3, 4], dtype=th.float64)
    print(reward_net(inp).item())
