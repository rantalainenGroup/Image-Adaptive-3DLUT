"""
@author: Hongkai Zhang
@contact: kevin.hkzhang@gmail.com
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D


def vis_lut(lut, lut_dim):
    step = 1.0 / (lut_dim - 1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for b in range(lut_dim):
        for g in range(lut_dim):
            # vectorization for efficiency
            r = np.arange(lut_dim)
            ax.scatter(b * step * np.ones(lut_dim),
                       g * step * np.ones(lut_dim),
                       r * step,
                       c=lut[b, g, r].numpy(),
                       marker='o',
                       alpha=1.0)
    ax.set_xlabel('B')
    ax.set_ylabel('G')
    ax.set_zlabel('R')
    plt.show()

def vis_lut_ax(lut2plot, lut_dim, ax, g_stop=None, b_stop=None, r_stop=None):
    step = 1.0 / (lut_dim - 1)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    if g_stop == None:
        g_stop = lut_dim
    if b_stop == None:
        b_stop = lut_dim
    if r_stop == None:
        r_stop = lut_dim

    if torch.any(lut2plot<0):
        print('Out of range: Negs')
        lut2plot[lut2plot<0]=0.0
    if torch.any(lut2plot>1):
        print('Out of range: Above one')
        lut2plot[lut2plot>1]=1.0

    for b in range(b_stop):
        for g in range(g_stop):
            # vectorization for efficiency
            r = np.arange(r_stop)
            ax.scatter(b * step * np.ones(r_stop),
                       g * step * np.ones(r_stop),
                       r * step,
                       c=lut2plot[b, g, r].numpy(),
                       marker='o',
                       alpha=1.0)
    ax.set_xlabel('B')
    ax.set_xlim(0,1)
    ax.set_ylabel('G')
    ax.set_ylim(0,1)
    ax.set_zlabel('R')
    ax.set_zlim(0,1)
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('lut_path', type=str, help='path to the LUT')
    parser.add_argument('--lut_dim',
                        type=int,
                        default=33,
                        help='dimension of the LUT')
    args = parser.parse_args()

    lut = torch.load(args.lut_path, map_location='cpu')
    lut_dim = args.lut_dim
    lut0, lut1, lut2 = [lut[str(i)]['LUT'] for i in range(3)]

    # convert [3, 17, 17, 17] to [17, 17, 17, 3]
    lut0 = lut0.permute(1, 2, 3, 0)
    lut1 = lut1.permute(1, 2, 3, 0)
    lut2 = lut2.permute(1, 2, 3, 0)

    # TODO: better ways for this process
    # normalization
    lut0 = (lut0 - lut0.min()) / (lut0.max() - lut0.min())
    lut1 = (lut1 - lut1.min()) / (lut1.max() - lut1.min())
    lut2 = (lut2 - lut2.min()) / (lut2.max() - lut2.min())

    # visualize the LUT, take lut0 as an example
    vis_lut(lut0, lut_dim)
