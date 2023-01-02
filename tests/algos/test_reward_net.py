from algos.torch.MaxEntIRL.reward_net import *


def test_saving_rewardnet():
    def feature_fn(x):
        return x

    reward_net = RewardNet(inp=8, arch=[], feature_fn=feature_fn, use_action_as_inp=True, device='cuda:1')
    log_dir = "tmp/log/reward_net.pkl"
    reward_net.save(log_dir)


def test_feature_grad():
    def feature_fn(x):
        return x
    reward_net = QuadraticRewardNet(inp=6, arch=[8, 8], feature_fn=feature_fn, use_action_as_inp=True)
    inp = th.nn.Parameter(th.FloatTensor([1, 2, 3, 4]))
    feature = reward_net.feature_layers.eval()(inp)
    deriv_mat = []
    for y in feature:
        y.backward()
        deriv_mat.append(feature.grad)