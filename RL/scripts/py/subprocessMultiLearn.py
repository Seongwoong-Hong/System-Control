import itertools
import random
import subprocess
import time

if __name__ == '__main__':
    # Python 스크립트 이름
    PYTHON_SCRIPT = "IsaacgymLearning.py"
    NUM_RUNS = 8
    NUM_THREADS = 8
    DEFAULT = ["test=False", "pipeline=gpu", "rl_device=cuda", "sim_device=cuda"]
    num_envs = [2048]
    gamma = [0.999]
    tau = [0.99]
    kl_threshold = [0.002]
    horizon_length = [64]
    minibatch_size = [8192]
    lim_level = [0.0, 0.5]
    tqrate_ratio = [0.0, 0.1]

    all_combinations = list(itertools.product(num_envs, gamma, tau, kl_threshold, horizon_length, minibatch_size, lim_level, tqrate_ratio))

    # 조합에서 NUM_RUNS만큼 랜덤 샘플링
    # selected_combinations = random.sample(all_combinations, NUM_RUNS)
    # selected_combinations = random.choices(all_combinations, k=NUM_RUNS)
    selected_combinations = 2 * all_combinations

    for pidx in range(NUM_RUNS // NUM_THREADS):
        # 병렬로 실행된 프로세스를 저장할 리스트
        ARGS = selected_combinations[pidx * NUM_THREADS:(pidx + 1) * NUM_THREADS]
        processes = []
        # 병렬 실행
        for i in range(1, NUM_THREADS + 1):
            ARG = [
                f"num_envs={ARGS[i-1][0]}",
                f"gamma={ARGS[i-1][1]}",
                f"tau={ARGS[i-1][2]}",
                f"kl_threshold={ARGS[i-1][3]}",
                f"horizon_length={ARGS[i-1][4]}",
                f"minibatch_size={ARGS[i-1][5]}",
                f"lim_level={ARGS[i-1][6]}",
                f"tqrate_ratio={ARGS[i-1][7]}",
                f"experiment_name=stiff_ana/tqr{ARGS[i-1][7]}",
                "stiff_ank=300"
            ]
            process = subprocess.Popen(["python", PYTHON_SCRIPT] + DEFAULT + ARG)
            processes.append(process)
            time.sleep(2)

        # 모든 프로세스가 완료될 때까지 대기
        for i, process in enumerate(processes, start=1):
            process.wait()
