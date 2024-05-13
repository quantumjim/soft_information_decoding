# Maurice Hanisch mhanisc@ethz.ch
# Created 2024-05-10

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ..Plotting.general import apply_formatting


def process_gmm_data(IQ_data: np.ndarray, mmr_0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    IQ_real = IQ_data.real
    IQ_imag = IQ_data.imag
    mmr_0_real = mmr_0.real
    mmr_0_imag = mmr_0.imag

    combined_IQ = np.column_stack([IQ_real, IQ_imag])
    combined_mmr_0 = np.column_stack([mmr_0_real, mmr_0_imag])

    combined_data = np.vstack([combined_IQ, combined_mmr_0])

    scaler = StandardScaler()
    # scaled_combined_data = scaler.fit_transform(combined_data)
    # Scaling does not work for some reason so we do not do it!
    scaled_combined_data = combined_data

    scaled_IQ_data = scaled_combined_data[:len(IQ_data)]
    scaled_mmr_0 = scaled_combined_data[len(IQ_data):]

    return scaled_IQ_data, scaled_mmr_0, scaler


def fit_gmm_0_calib(IQ_data_processed):
    gmm_0 = GaussianMixture(n_components=1, covariance_type='diag')
    gmm_0.fit(IQ_data_processed)
    return gmm_0


def fit_RepCode_data(IQ_data_processed, gmm_0, n_components=3, where='above'):
    assert n_components in [2, 3], 'n_components not in [2, 3] is not supported currently'

    # Heuristic parameters (should be good)
    tol = 1e-8
    max_iter = 1_000
    reg_covar = 1e-2

    if n_components == 3:
        mean0 = np.array([gmm_0.means_[0, 0], gmm_0.means_[0, 1]])
        mean1 = np.array([-gmm_0.means_[0, 0], gmm_0.means_[0, 1]])
        mean2 = np.array([0, gmm_0.means_[0, 1] + 0.5*np.abs(gmm_0.means_[0, 1])]) if where == 'above' else np.array(
            [0, gmm_0.means_[0, 1] - 0.5*np.abs(gmm_0.means_[0, 1])])
        means_init = np.array([mean0, mean1, mean2])
        precisions_init = np.array([np.diag(1 / gmm_0.covariances_[0]), 
                                    np.diag(1 / gmm_0.covariances_[0]), 
                                    np.diag(1 / gmm_0.covariances_[0])])

        gmm_RepCode = GaussianMixture(n_components=n_components, covariance_type='full',
                                      means_init=means_init, precisions_init=precisions_init, init_params="k-means++", random_state=42,
                                      tol=tol, max_iter=max_iter, n_init=1, reg_covar=reg_covar)

    elif n_components == 2:
        mean0 = np.array([gmm_0.means_[0, 0], gmm_0.means_[0, 1]])
        mean1 = np.array([-gmm_0.means_[0, 0], gmm_0.means_[0, 1]])
        means_init = np.array([mean0, mean1])
        precisions_init = np.array(
            [np.diag(1 / gmm_0.covariances_[0]), np.diag(1 / gmm_0.covariances_[0])])
        # precisions_init = np.array([1 / gmm_0.covariances_[0], 1 / gmm_0.covariances_[0]])
        precisions_init = np.diag(1 / gmm_0.covariances_[0])

        gmm_RepCode = GaussianMixture(n_components=n_components, covariance_type='tied',
                                      means_init=means_init, precisions_init=precisions_init, init_params="k-means++", random_state=42,
                                      tol=1e-4, max_iter=100, n_init=1, reg_covar=reg_covar)

    gmm_RepCode.fit(IQ_data_processed)
    return gmm_RepCode


def get_gmm_RepCodeData(IQ_data_processed, gmm_0):
    gmm_2 = fit_RepCode_data(IQ_data_processed, gmm_0, n_components=2)
    len_data = len(IQ_data_processed)
    if len_data > 10_000:
        gmm_3_above = fit_RepCode_data(IQ_data_processed, gmm_0, n_components=3, where='above')
        gmm_3_below = fit_RepCode_data(IQ_data_processed, gmm_0, n_components=3, where='below')


        aic2 = gmm_2.aic(IQ_data_processed)
        aic3_above = gmm_3_above.aic(IQ_data_processed)
        aic3_below = gmm_3_below.aic(IQ_data_processed)

        aic_diff_above = abs(aic2 - aic3_above)
        aic_diff_below = abs(aic2 - aic3_below)
        aic_rel_diff_above = aic_diff_above / max(aic2, aic3_above)
        aic_rel_diff_below = aic_diff_below / max(aic2, aic3_below)

        if aic_rel_diff_above < 1e-4 and aic_rel_diff_below < 1e-4: # Heuristic threshold, should work 
            print(f"AIC relative difference above: {aic_rel_diff_above}, below: {aic_rel_diff_below}")
            return gmm_2
    
        return gmm_2 if aic2 < aic3_above else gmm_3_above if aic3_above < aic3_below else gmm_3_below
    
    else:
        print(f"Data too small to compare AICs, returning 2-component GMM.")
        return gmm_2




def plot_RepCode_gmm(IQ_data_processed, gmm_combined, dpi=100, title=''):
    X = IQ_data_processed
    combined_predictions = gmm_combined.predict(X)

    apply_formatting(dpi=dpi)
    fig = plt.figure()
    scatter = plt.scatter(X[:, 0], X[:, 1], s=1,
                          c=combined_predictions, cmap='viridis', alpha=0.1)

    plot_ellipses_comb(gmm_combined, plt.gca())

    plt.xlabel('In-Phase [arb.]')
    plt.ylabel('Quadrature [arb.]')
    plt.title(title)

    return fig


def plot_ellipses_comb(gmm, ax):
    if gmm.covariance_type == 'tied':
        # In 'tied', all components share the same precision matrix, invert it just once
        prec = gmm.precisions_cholesky_ @ gmm.precisions_cholesky_.T
        cov = np.linalg.inv(prec)
        v, w = np.linalg.eigh(cov)
        v = 2. * np.sqrt(2.) * np.sqrt(v)  # Scale the eigenvalues for plotting
        for mean in gmm.means_:
            u = w[:, 0]  # Eigenvector corresponding to the largest eigenvalue
            angle = np.arctan2(u[1], u[0])
            angle = np.degrees(angle)  # Convert to degrees
            ell = Ellipse(xy=mean, width=v[0], height=v[1], angle=angle, color='red', fill=False)
            ax.add_artist(ell)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
    else:
        for mean, prec_chol in zip(gmm.means_, gmm.precisions_cholesky_):
            if gmm.covariance_type == 'full':
                # Construct full precision matrix from its Cholesky decomposition
                prec = prec_chol @ prec_chol.T
            elif gmm.covariance_type == 'diag':
                # For diagonal, the Cholesky decomposition is the square root of the diagonal elements
                prec = np.diag(prec_chol**2)
            else:
                # This block handles the spherical case where the Cholesky factor is just the standard deviation
                prec = np.diag(np.full(gmm.means_.shape[1], prec_chol**2))

            # Invert the precision matrix to get the covariance matrix
            cov = np.linalg.inv(prec)
            # Calculate eigenvalues and eigenvectors for the covariance matrix
            v, w = np.linalg.eigh(cov)
            v = 2. * np.sqrt(2.) * np.sqrt(v)  # Scale the eigenvalues for plotting
            u = w[:, 0]  # Eigenvector corresponding to the largest eigenvalue
            angle = np.arctan2(u[1], u[0])
            angle = np.degrees(angle)  # Convert to degrees
            ell = Ellipse(xy=mean, width=v[0], height=v[1], angle=angle, color='red', fill=False)
            ax.add_artist(ell)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5) 