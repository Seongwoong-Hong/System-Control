import numpy as np
import torch
from matplotlib import pyplot as plt

if __name__ == "__main__":
    pmax, nmax = 1.0, 1.0
    magst = np.linspace(-nmax, pmax, 100)
    const_var = torch.from_numpy(magst).float()

    poscop = torch.clamp(const_var, min=0., max=pmax)
    negcop = torch.clamp(const_var, min=-nmax, max=0.)

    fig1 = plt.figure(figsize=[6.4, 7.2])
    ax1 = fig1.add_subplot()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    div = np.logspace(-5, -2, 3)
    for limLevel in [0.0, 0.25, 0.5]:
        i = 10 ** (limLevel * ((-5) - (-2)) + (-2))
        r_penalty = i*(-2 / (i + 1) + (1 / ((poscop / pmax - 1) ** 2 + i) + 1 / ((negcop / nmax + 1) ** 2 + i)))
        # ax1.plot(magst, -i / ((posmagst/pmax - 1)**2 + i), color='b')
        # ax1.plot(magst, -i / ((negmagst/nmax + 1)**2 + i), color='r')
        # ax1.plot(magst, 2*i - i *(1 / ((posmagst/pmax - 1)**2 + i) + 1 / ((negmagst/nmax + 1) ** 2 + i)), color=[50 / 255, 50 / 255, 50 / 255])
        ax1.plot(magst, r_penalty, color=[50 / 255, 50 / 255, 50 / 255])

    ax1.axhline(y=0, xmin=0, xmax=1)
    ax1.axhline(y=1, xmin=0, xmax=1)
    ax1.axvline(x=-0.8, ymin=0, ymax=1)
    # ax1.set_ylim([-0.05, 3.1])
    # ax1.set_xlim([-0.05, 0.12])
    fig1.tight_layout()
    fig1.show()
