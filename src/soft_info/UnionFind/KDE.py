# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-27

from typing import Union

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

from soft_info import plot_IQ_data
from Scratch import load_calibration_memory


def plot_KDE(data, kde, scaler):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    # Contour plot of original data
    nbins = 50
    hist, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=nbins)
    axs[0].contour(xedges[:-1], yedges[:-1], hist.T, cmap='viridis')
    axs[0].grid()
    axs[0].set_title('Original Data - Contour Plot')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')

    # Contour plot of KDE
    x_grid = np.linspace(min(data[:, 0]) - 1, max(data[:, 0]) + 1, 100)
    y_grid = np.linspace(min(data[:, 1]) - 1, max(data[:, 1]) + 1, 100)
    xv, yv = np.meshgrid(x_grid, y_grid)
    gridpoints = np.array([xv.ravel(), yv.ravel()]).T

    log_dens = kde.score_samples(scaler.transform(gridpoints))
    dens = np.exp(log_dens).reshape(xv.shape)
    axs[1].contour(x_grid, y_grid, dens, cmap='viridis')
    axs[1].grid()
    axs[1].set_title(f'KDE - Contour Plot (bw = {kde.bandwidth})')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')

    plt.tight_layout()
    plt.show()


def fit_KDE(IQ_data, bandwidth=0.1, plot=False, num_samples=1e5, scaler=None):
    data = IQ_data.flatten()
    combined_data = np.column_stack((data.real, data.imag))

    if not scaler:
        scaler = StandardScaler()

    normalized_data = scaler.fit_transform(combined_data)

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(normalized_data)

    if plot:
        plot_IQ_data(IQ_data, figsize=(6, 4),
                     title=f"IQ data: {len(data):.0e} samples")
        kde_samples = kde.sample(int(num_samples))
        kde_samples = scaler.inverse_transform(kde_samples)
        kde_complex_samples = kde_samples[:, 0] + 1j * kde_samples[:, 1]

        plot_IQ_data(kde_complex_samples, figsize=(
            6, 4), title=f"{num_samples:.0e} KDE samples for bandwidth = {bandwidth}")
        plot_KDE(combined_data, kde, scaler)

    return kde, scaler


def get_KDEs(provider, device, qubit, bandwidths: Union[float, list] = 0.2,
             plot: Union[bool, list] = False, num_samples=1e5):
    bw0, bw1 = (bandwidths, bandwidths) if not isinstance(
        bandwidths, list) else bandwidths
    plot0, plot1 = (plot, plot) if not isinstance(plot, list) else plot

    memories = load_calibration_memory(provider, device, qubit)

    kde_0, scaler_0 = fit_KDE(
        memories["mmr_0"], bandwidth=bw0, plot=plot0, num_samples=num_samples)
    kde_1, scaler_1 = fit_KDE(
        memories["mmr_1"], bandwidth=bw1, plot=plot1, num_samples=num_samples)

    return [kde_0, kde_1], [scaler_0, scaler_1]
