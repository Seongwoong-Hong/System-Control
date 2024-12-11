import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    pmax, nmax = .12, .05
    magst = np.linspace(-nmax, pmax, 100)
    posmagst = np.clip(magst, a_max=pmax, a_min=0)
    negmagst = np.clip(magst, a_min=-nmax, a_max=0)
    fig1 = plt.figure(figsize=[6.4, 7.2])
    ax1 = fig1.add_subplot()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    div = np.logspace(-5, -2, 3)
    for limLevel in [0.0, 0.5, 1.0]:
        i = 10 ** (limLevel * ((-5) - (-2)) + (-2))
        # ax1.plot(magst, -i / ((posmagst/pmax - 1)**2 + i), color='b')
        # ax1.plot(magst, -i / ((negmagst/nmax + 1)**2 + i), color='r')
        # ax1.plot(magst, 2*i - i *(1 / ((posmagst/pmax - 1)**2 + i) + 1 / ((negmagst/nmax + 1) ** 2 + i)), color=[50 / 255, 50 / 255, 50 / 255])
        ax1.plot(magst, -2*i + i *(1 / ((posmagst/pmax - 1)**2 + i) + 1 / ((negmagst/nmax + 1) ** 2 + i)), color=[50 / 255, 50 / 255, 50 / 255])

    ax1.axhline(y=0, xmin=0, xmax=1)
    ax1.axhline(y=-1, xmin=0, xmax=1)
    ax1.set_xlim([-0.05, 0.12])
    fig1.tight_layout()
    fig1.show()
