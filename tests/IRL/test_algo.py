import os
import pickle
from algos.torch.ppo import PPO
from algos.torch.sac import SAC
from common.util import make_env, CPU_Unpickler
from common.analyzer import verify_policy
from common.wrappers import *


def test_2D():
    name = "2DTarget"
    env = make_env(f"{name}-v2")
    model_dir = os.path.join("../..", "tmp", "log", name, "GCL", "sq_sac_1_sac", "model")
    algo = SAC.load(model_dir + f"/agent")
    trajs = []
    for i in range(20):
        st = env.reset()
        done = False
        sts, rs = [], []
        while not done:
            action, _ = algo.predict(st)
            st, r, done, _ = env.step(action)
            sts.append(st)
            rs.append(r)
        trajs.append(np.append(np.array(sts), np.array(rs).reshape(-1, 1), axis=1))
    env.draw(trajs)


def test_1D(irl_path):
    name = "1DTarget"
    env_id = f"{name}_disc"
    env = make_env(f"{env_id}-v0")
    model_dir = os.path.join(irl_path, "tmp", "log", env_id, "MaxEntIRL", "ext_ppo_disc_linear_ppoagent_svm_reset",
                             "model")
    algo = PPO.load(model_dir + "/008/agent")
    a_list, o_list, _ = verify_policy(env, algo, render="None", repeat_num=9, deterministic=False)
    print('end')
