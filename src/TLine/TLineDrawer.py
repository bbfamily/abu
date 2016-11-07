# -*- encoding:utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

__author__ = 'BBFamily'


def plot_xy_with_other(x_org, y_org, other_mark, *other):
    plt.plot(x_org, y_org)
    if other is not None:
        [plt.plot(otList, [y_org[ot] for ot in otList], other_mark) for otList in other]
    plt.show()


def plot_xy_with_other_x_y(x_org, y_org, other_mark, *other):
    plt.plot(x_org, y_org)
    if other is not None:
        [plt.plot(otList[0], otList[1], other_mark) for otList in other if len(otList) == 2]
    plt.show()


def plot_xy_with_mark(x_org, y_org, *other):
    plt.plot(x_org, y_org)
    if other is not None:
        [plt.plot(otList[0], otList[1], otList[2]) for otList in other if len(otList) == 3]
    plt.show()


def plot_elow_k_choice(k_rng, silhouette_score, within_sum_squares, select_k):
    plt.figure(figsize=(7, 8))
    plt.subplot(211)
    plt.title('Using the elbow method to inform k choice')
    plt.plot(k_rng[1:], silhouette_score, 'b*-')
    plt.xlim([1, 15])
    plt.grid(True)
    plt.ylabel('Silhouette Coefficient')

    plt.subplot(212)
    plt.plot(k_rng, within_sum_squares, 'b*-')
    plt.xlim([1, 15])
    plt.grid(True)
    plt.xlabel('k')
    plt.ylabel('Within Sum of Squares')
    plt.plot(select_k, within_sum_squares[select_k], 'ro', markersize=12, markeredgewidth=1.5,
             markerfacecolor='None', markeredgecolor='r')
    plt.show()


def plot_xy_with_scatter_color(x_org, y_org, d, colors_array):
    colors = np.array(['#FF0054', '#FBD039', '#23C2BC',
                       '#CC99CC', '#CC3399', '#33FF99', '#00CCFF', '#66FF66', '#339999',
                       '#6666CC', '#666666', '#663333', '#660033'])
    plt.plot(x_org, y_org, '')
    plt.scatter(d[:, 0], d[:, 1], c=colors[colors_array], s=60)
    plt.show()
