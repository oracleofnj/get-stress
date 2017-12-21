from __future__ import print_function
from __future__ import division
import pickle
import sys
import numpy as np
import matplotlib
from numpy.polynomial.legendre import legval
matplotlib.use('Agg')
np.set_printoptions(precision=3, linewidth=150)
import matplotlib.pyplot as plt


def plot_cluster_centers(cluster_model, output_file=None, n_points=21):
    """Plot the cluster centers."""
    with open(cluster_model, 'rb') as f:
        vc = pickle.load(f)

    x_points = np.linspace(-1, 1, n_points)
    cluster_centers = vc['scaler'].inverse_transform(
        vc['kmeans'].cluster_centers_
    )

    pitch_coefs = cluster_centers[:, :3]
    power_coefs = cluster_centers[:, 3:6]
    dur_coefs = cluster_centers[:, 6]

    mean_pitch_polys = legval(x_points, pitch_coefs.T)
    mean_power_polys = legval(x_points, power_coefs.T)
    pi_min = np.min(mean_pitch_polys)
    pi_max = np.max(mean_pitch_polys)
    po_min = np.min(mean_power_polys)
    po_max = np.max(mean_power_polys)
    fig, axes = plt.subplots(4, 2, figsize=(8, 12))
    fig.subplots_adjust(wspace=0.75)
    for i, (pi, po, d, ax1) in enumerate(zip(
        mean_pitch_polys,
        mean_power_polys,
        dur_coefs,
        axes.ravel()
    )):
        ax1.plot((1000*d/2.)*np.linspace(-1, 1, 21), pi, color='blue')
        ax1.set_ylim(
            pi_min - (pi_max - pi_min)/20.,
            pi_max + (pi_max - pi_min)/20.
        )
        ax2 = ax1.twinx()
        ax2.plot((1000*d/2.)*np.linspace(-1, 1, 21), po, color='red')
        ax2.set_ylim(
            po_min - (po_max - po_min)/20.,
            po_max + (po_max - po_min)/20.
        )
        ax1.set_xticklabels([
            '{0} ms'.format(int(x))
            for x in ax1.get_xticks()
        ])
        ax1.set_ylabel('Pitch', color='blue')
        ax2.set_ylabel('Power', color='red')
    fig.savefig(output_file)


if __name__ == "__main__":
    plot_cluster_centers(
        sys.argv[1],    # vowel_clusters.pkl
        sys.argv[2],    # path_to_output
    )
