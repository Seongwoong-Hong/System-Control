import itertools
import json
import random
import subprocess
import time

from common.util import str2bool

if __name__ == '__main__':
    with open(f"multi_learn_config.json", 'r') as f:
        config = json.load(f)
    # Python 스크립트 이름
    PYTHON_SCRIPT = config["PYTHON_SCRIPT"]
    NUM_RUNS = config["NUM_RUNS"]
    NUM_THREADS = config["NUM_THREADS"]
    DEFAULT = ["test=False", "pipeline=gpu", "rl_device=cuda", "sim_device=cuda"]

    comb_keys = [*config["learn_config"].keys()]
    all_combinations = list(itertools.product(*config["learn_config"].values()))

    # 조합에서 NUM_RUNS만큼 랜덤 샘플링
    sampling = str2bool(config["sampling"]) if "sampling" in config else True
    if sampling:
        selected_combinations = random.choices(all_combinations, k=NUM_RUNS)
    else:
        selected_combinations = ((NUM_RUNS - 1) // len(all_combinations) + 1) * all_combinations

    for pidx in range(NUM_RUNS // NUM_THREADS):
        # 병렬로 실행된 프로세스를 저장할 리스트
        ARGS = selected_combinations[pidx * NUM_THREADS:(pidx + 1) * NUM_THREADS]
        processes = []
        # 병렬 실행
        for i in range(1, NUM_THREADS + 1):
            ARG = [f"{comb_keys[j]}={ARGS[i-1][j]}" for j in range(len(comb_keys))]
            EXPNAME = f"experiment_name={config['EXP_NAME']}/{comb_keys[0]}{ARGS[i - 1][0]}"
            for namei in range(1, config['num_names_add']):
                EXPNAME += f"_{comb_keys[namei]}{ARGS[i - 1][namei]}"
            ARG += [EXPNAME]
            process = subprocess.Popen(["python", PYTHON_SCRIPT] + DEFAULT + ARG)
            processes.append(process)
            time.sleep(2)

        # 모든 프로세스가 완료될 때까지 대기
        for i, process in enumerate(processes, start=1):
            process.wait()
