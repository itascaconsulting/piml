import numpy as np
import pandas as pd
import matplotlib; matplotlib.rcParams["savefig.directory"] = "."
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 18})


def show_actual_predicted(actual, predicted):
    assert len(actual.shape)==1
    assert len(predicted.shape)==1

    fig = plt.figure(constrained_layout=True, figsize=(15, 10))
    gs = fig.add_gridspec(3, 5) # vert the horiz
    ax1 = fig.add_subplot(gs[0:2, 0:2]) # scatter dim
    ax2 = fig.add_subplot(gs[0:2, 2:4]) # scatter percent
    ax3 = fig.add_subplot(gs[0:2, 4]) # target hist
    ax4 = fig.add_subplot(gs[2, :2]) # dimensional error hist
    ax5 = fig.add_subplot(gs[2, 2:4]) # percent error hist
    #ax6 = fig.add_subplot(gs[2, 4]) # text input


    error = actual-predicted
    data_bc_error = (error) / actual
    data_bc_error = data_bc_error * 100

    points = np.array([error, actual]).T
    dist = np.log10(np.mean(cKDTree(points).query(points, k=100)[0], axis=1))
    order = np.argsort(dist)[::-1]
    dist = dist[order]
    p_error = np.array(error)[order]
    p_actual = np.array(actual)[order]
    ax1.scatter(p_error, p_actual, c=dist, cmap=plt.cm.get_cmap('jet').reversed(), s=2)
    ax1.set_ylabel("Actual []")
    ax1.set_xlabel("Error []")
    ax1.axvline(0, linestyle='--', color='k', lw=1.5)

    points = np.array([data_bc_error, actual]).T
    dist = np.log10(np.mean(cKDTree(points).query(points, k=100)[0], axis=1))
    order = np.argsort(dist)[::-1]
    dist = dist[order]
    p_error = np.array(data_bc_error)[order]
    p_actual = np.array(actual)[order]
    ax2.scatter(p_error, p_actual, c=dist, cmap=plt.cm.get_cmap('jet').reversed(), s=2)
    ax2.set_ylabel("Actual []")
    ax2.set_xlabel("Error %")
    ax2.axvline(0, linestyle='--', color='k', lw=1.5)

a = np.load("jason_actual.spy")
p0 = np.load("jason_p0.npy")
p1 = np.load("jason_p1.npy")
p2 = np.load("jason_p2.npy")

o0 = [1957,19785,16459,14328,4837,20024,24946,14433,9524,2878]
o1 = [7331,6097,14328,12308,12734,17679,20925,20024,10804,20080]
o2 = [22119,24516,12308,6097,2854,12734,16660,20024,3519,24117]

print(len(o0+o1+o2))
print(len(set(o0+o1+o2)))
for i in set(o0+o1+o2):
    print(a[i], p0[i], p1[i], p2[i])
