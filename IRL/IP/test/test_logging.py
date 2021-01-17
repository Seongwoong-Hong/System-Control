import os, shutil

if __name__ == "__main__":
    name = "test"
    log_dir = os.path.join(os.path.abspath("../tmp/log"), name)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    model_dir = os.path.join(os.path.dirname(__file__), "tmp", "model")
    expert_dir = os.path.join(os.path.dirname(__file__), "demos", "expert_bar_100.pkl")
    shutil.copy(os.path.abspath("../../../common/modules.py"), log_dir)
    shutil.copy(os.path.abspath("../../../gym_envs/envs/IP_custom_PD.py"), log_dir)