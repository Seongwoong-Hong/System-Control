from pathlib import Path

from scipy import io
from matplotlib import pyplot as plt


if __name__ == "__main__":
    env_type = "IP"
    subj = "sub04"
    trials = range(1, 21)
    isPseudo = False
    except_trials = [13, 16]

    if isPseudo:
        env_type = "Pseudo" + env_type

    fig = plt.figure(figsize=[6.4, 9.6])
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    for trial in trials:
        if trial not in except_trials:
            humanData = io.loadmat(str(Path(env_type) / subj / f"{subj}i{trial}.mat"))
            state = humanData['state']
            tq = humanData['tq']
            print(state[:,0].argmin())
            ax1.plot(state[:, 0])
            ax2.plot(tq[:, 0])
    plt.show()
