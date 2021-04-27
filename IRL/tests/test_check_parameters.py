import os


def test_load_logging():
    hype = {}
    dir_name = os.path.join("..", "tmp", "log", "HPC", "ppo", "AIRL_test", "10", "model")
    f = open(dir_name + "/hyper_parameters.txt", "r")
    for line in f.readlines():
        name, value = line[:line.find(":")], float(line[line.find(":")+1:-1])
        hype[name] = value
    print(hype)


def test_multiple_loading():
    dir_name = os.path.join("..", "tmp", "log", "HPC", "ppo", "AIRL_test")
    num = 0
    result_dict = {}
    while True:
        file = dir_name + f"/{num}/model/hyper_parameters.txt"
        if not os.path.isfile(file):
            break
        f = open(file, "r")
        for line in f.readlines():
            name, value = line[:line.find(":")], float(line[line.find(":")+1:-1])
            if name in result_dict:
                result_dict[name].append([value, num])
            else:
                result_dict[name] = [[value, num]]
        num += 1
    return result_dict


def test_draw_scatter():
    from matplotlib import pyplot as plt
    import numpy as np
    result_dict = test_multiple_loading()
    for key, value in result_dict.items():
        fig = plt.figure()
        ax = fig.subplots()
        array = np.array(value)
        ax.set_title(key)
        ax.scatter(array[:, 0], array[:, 1])
        plt.show()
