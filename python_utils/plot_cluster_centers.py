from __future__ import print_function
from __future__ import division
import pickle
import sys
import numpy as np
import matplotlib
from numpy.polynomial.legendre import legval
matplotlib.use('Agg')
np.set_printoptions(precision=3, linewidth=150)

def plot_cluster_centers(cluster_model, output_file, n_points=21):
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
    mean_pitch_polys = legval(x_points, power_coefs.T)
    print(mean_pitch_polys)
    print(mean_pitch_polys)
    print(dur_coefs)


if __name__ == "__main__":
    plot_cluster_centers(
        sys.argv[1],    # vowel_clusters.pkl
        sys.argv[2],    # path_to_output
    )
