import os
from scipy import io

from algos.tabular.viter import FiniteSoftQiter
from common.wrappers import RewardInputNormalizeWrapper, DiscretizeWrapper
from common.analyzer import video_record
from common.util import make_env, CPU_Unpickler

if __name__ == "__main__":
    def feature_fn(x):
        return x ** 2
    env_type = "DiscretizedHuman"
    subj = "sub06"
    name = f"sq_handnorm_17171719_quadcost_finite_many/{subj}_1_1"
    irl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "IRL"))
    subpath = os.path.join(irl_dir, "demos", "HPC", subj, subj)
    bsp = io.loadmat(subpath + f"i1.mat")['bsp']
    rwfn_dir = irl_dir + f"/tmp/log/DiscretizedHuman/MaxEntIRL/{name}/model"
    with open(rwfn_dir + "/reward_net.pkl", "rb") as f:
        rwfn = CPU_Unpickler(f).load()
    rwfn.feature_fn = feature_fn
    env = make_env(f"{env_type}-v2", num_envs=1, N=[17, 17, 17, 19], NT=[11, 11], bsp=bsp,
                   wrapper=RewardInputNormalizeWrapper, wrapper_kwrags={'rwfn': rwfn})
    eval_env = make_env(f"{env_type}-v0", N=[17, 17, 17, 19], NT=[11, 11], bsp=bsp, wrapper=DiscretizeWrapper)
    agent = FiniteSoftQiter(env=env, gamma=1, alpha=0.001, device='cpu', verbose=True)
    agent.learn(0)
    agent.set_env(eval_env)
    ob = eval_env.reset()
    imgs = [eval_env.render(mode="rgb_array")]
    for _ in range(10):
        for t in range(50):
            obs_idx = eval_env.get_idx_from_obs(ob)
            act_idx = agent.policy.choice_act(agent.policy.policy_table[t].T[obs_idx])
            act = eval_env.get_acts_from_idx(act_idx)
            ob, r, _, _ = eval_env.step(act[0])
            imgs.append(eval_env.render(mode="rgb_array"))
        ob = eval_env.reset()
    video_record(imgs, f"videos/{name}.mp4", eval_env.dt)
