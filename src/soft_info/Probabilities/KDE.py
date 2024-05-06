# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-27

from typing import Union, List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

from ..Plotting.IQ_plotter import plot_IQ_data


def plot_KDE(data, kde, scaler):
    '''Plots the KDE and the original data in a contour plot.'''
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    # Contour plot of original data
    nbins = 50
    hist, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=nbins)
    axs[0].contour(xedges[:-1], yedges[:-1], hist.T, cmap='viridis')
    axs[0].grid()
    axs[0].set_title('Original Data - Contour Plot')
    axs[0].set_xlabel("In-Phase [arb.]")
    axs[0].set_ylabel("Quadrature [arb.]")

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
    axs[1].set_xlabel("In-Phase [arb.]")
    axs[1].set_ylabel("Quadrature [arb.]")

    plt.tight_layout()
    plt.show()


def plot_decision_boundary(data, kde_0, kde_1, scaler):
    '''Plots the decision boundary between two KDEs.'''
    # Generate 2D grid of points
    # Contour plot of KDE
    x_grid = np.linspace(min(data[:, 0]) - 1, max(data[:, 0]) + 1, 100)
    y_grid = np.linspace(min(data[:, 1]) - 1, max(data[:, 1]) + 1, 100)
    xv, yv = np.meshgrid(x_grid, y_grid)
    gridpoints = np.array([xv.ravel(), yv.ravel()]).T
    scaled_gridpoints = scaler.transform(gridpoints)

    kde_0_vals = np.exp(kde_0.score_samples(
        scaled_gridpoints)).reshape(xv.shape)
    kde_1_vals = np.exp(kde_1.score_samples(
        scaled_gridpoints)).reshape(xv.shape)

    diff = kde_0_vals - kde_1_vals

    plt.contourf(
        xv, yv, diff, levels=[-np.inf, 0, np.inf], colors=['red', 'blue'], alpha=0.7)

    plt.contour(xv, yv, diff, levels=[0], colors=['black'])

    plt.title("Decision Boundary")
    plt.xlabel("In-Phase [arb.]")
    plt.ylabel("Quadrature [arb.]")
    plt.show()


def fit_KDE(IQ_data, bandwidth=0.2, plot=False, qubit_index='', num_samples=1e5, scaler=None):
    '''Fits a KDE to the IQ data and returns the KDE and the scaler used for normalization.'''
    data = IQ_data.flatten()
    combined_data = np.column_stack((data.real, data.imag))

    if not scaler:
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(combined_data)
    else:
        normalized_data = scaler.transform(combined_data)

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(normalized_data)

    if plot:
        plot_IQ_data(IQ_data, figsize=(6, 4),
                     title=f"IQ data: {len(data):.0e} samples for qubit {qubit_index}")
        kde_samples = kde.sample(int(num_samples))
        kde_samples = scaler.inverse_transform(kde_samples)
        kde_complex_samples = kde_samples[:, 0] + 1j * kde_samples[:, 1]

        plot_IQ_data(kde_complex_samples, figsize=(
            6, 4), title=f"{num_samples:.0e} KDE samples for bandwidth = {bandwidth} for qubit {qubit_index}")
        plot_KDE(combined_data, kde, scaler)

    return kde, scaler


# def get_KDEs(provider, tobecalib_job: Optional[str] = None, tobecalib_backend: Optional[str] = None, qubits: Optional[List[int]] = None,  
#              bandwidths: Union[float, list] = 0.2, plot: Union[bool, list] = False, 
#              plot_db=False, num_samples=1e5, other_date = None) -> Dict[int, List]:
def get_KDEs(all_memories: np.ndarray, qubits: Optional[List[int]] = None,  
             bandwidths: Union[float, list] = 0.1, plot: Union[bool, list] = False, 
             plot_db=False, num_samples=1e5) -> Dict[int, List]:
    """
    Retrieves Kernel Density Estimations (KDEs) for specified qubits using calibration data from a quantum computing service provider. This function is useful for analyzing the state distributions of qubits and understanding their behavior.

    Args:
    - provider: The quantum computing service provider, used to access calibration data.
    - tobecalib_job (str): The job ID for which calibration data is to be retrieved. If provided, the function will use calibration data closest to this job's creation date.
    - qubits (List[int], optional): A list of qubit indices for which KDEs should be retrieved. If not provided, KDEs for all available qubits will be generated.
    - bandwidths (float or list, optional): The bandwidth parameter(s) for KDEs. Can either be a single float or a list of two floats for separate bandwidths for the '0' and '1' states. Defaults to 0.2.
    - plot (bool or list, optional): Indicates whether to plot the KDEs. Can be a single boolean or a list of two booleans for separate plotting options for the '0' and '1' states. Defaults to False.
    - plot_db (bool, optional): If True, plots the decision boundary between the '0' and '1' state distributions. Defaults to False.
    - num_samples (int, optional): The number of samples to use for KDE sample plotting. Defaults to 100,000.

    Returns:
    - Dict[int, List]: A dictionary where keys are qubit indices, and values are lists containing the KDE objects for '0' and '1' states, along with the corresponding scaler for each qubit. For example, all_kdes[qubit] = [kde_0, kde_1], all_scalers[qubit] = scaler.

    Example usage:
    all_kdes, all_scalers = get_KDEs(provider, tobecalib_job='job_id', qubits=[1, 2, 3])
    """
    bw0, bw1 = (bandwidths, bandwidths) if not isinstance(
        bandwidths, list) else bandwidths
    plot0, plot1 = (plot, plot) if not isinstance(plot, list) else plot

    # from Scratch import load_calibration_memory # lazy import to avoid circular import
    # all_memories = load_calibration_memory(provider, tobecalib_job=tobecalib_job, tobecalib_backend=tobecalib_backend, qubits=qubits, other_date=other_date)

    if qubits is None:
        qubits = list(all_memories.keys())

    all_kdes = {}
    all_scalers = {}

    for qubit in qubits:
        memories = all_memories[qubit]
        # Combine both 0 and 1 state data
        combined_data = np.concatenate(
            [memories.get("mmr_0", []), memories.get("mmr_1", [])])
        stacked_data = np.column_stack(
            (combined_data.real, combined_data.imag))

        # Fit the scaler on combined_data
        scaler = StandardScaler()
        scaler.fit(stacked_data)

        # if qubit != 20:
        #     plot0 = False
        #     plot1 = False
        # else:
        #     plot0 = True
        #     plot1 = True
        
        kde_0, _ = fit_KDE(
            memories.get("mmr_0", []), bandwidth=bw0, plot=plot0, qubit_index=qubit, num_samples=num_samples, scaler=scaler)
        kde_1, _ = fit_KDE(
            memories.get("mmr_1", []), bandwidth=bw1, plot=plot1, qubit_index=qubit, num_samples=num_samples, scaler=scaler)

        all_kdes[qubit] = [kde_0, kde_1]
        all_scalers[qubit] = scaler  # single scaler for both states

        if plot_db:
            plot_decision_boundary(
                stacked_data, kde_0, kde_1, scaler=scaler)

    return all_kdes, all_scalers
