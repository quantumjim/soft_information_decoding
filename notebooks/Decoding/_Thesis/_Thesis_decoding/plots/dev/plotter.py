from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import colormaps

from scipy.optimize import curve_fit



def plot_error_rates(distances_list: List, errs_list: List, shots_list: List, labels: Optional[List] = None, colors: Optional[List] = None,
                     title: str = None, plot_e_L: bool = False, T = None, mark_outliers = True) -> None:
    # Setup plot parameters for PRX style
    FIGURE_WIDTH_1COL = 3.404
    FIGURE_HEIGHT_1COL_GR = FIGURE_WIDTH_1COL * 2 / (1 + np.sqrt(5)) * 1.1
    font_size = 6
    plt.rcParams.update({
        'font.size': font_size,
        'figure.titlesize': 'medium',
        'figure.dpi': 1000,
        'figure.figsize': (FIGURE_WIDTH_1COL, FIGURE_HEIGHT_1COL_GR),
        'axes.titlesize': 'medium',
        'axes.axisbelow': True,
        'xtick.direction': 'in',
        'xtick.labelsize': 'small',
        'ytick.direction': 'in',
        'ytick.labelsize': 'small',
        'image.interpolation': 'none',
        'legend.fontsize': font_size,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'font.family': 'sans-serif',
    })

    if colors is None:
        colors = ['skyblue', 'darkslategrey', 'steelblue', 'tomato', 'darkred', 'indianred',]

    if labels is None:
        labels = [f"Dataset {i}" for i in range(len(distances_list))]

    # Loop through each dataset
    for distances, errs, shots, label, color in zip(distances_list, errs_list, shots_list, labels, colors):
        label = label_dict[label] if label in label_dict else label
        # Calculate Wilson score intervals for confidence
        lowers = []
        uppers = []
        ps = []
        for err_nb, n in zip(errs, shots):
            p = err_nb / n
            if plot_e_L:
                p = logical_err_rate_per_round(p, T)
                lower, upper = wilson_score_interval(p, n*T) # assumes err rate is err * rounds! (TODO: fix)
            else:
                lower, upper = wilson_score_interval(p, n)
            ps.append(p)
            lowers.append(lower)
            uppers.append(upper)

        # Exclude zero points for fitting
        outlier_nb = 5
        valid_indices = np.where(errs > outlier_nb)
        distances_fit = np.array(distances)[valid_indices]
        ps_fit = np.array(ps)[valid_indices]
        lower_fit = np.array(lowers)[valid_indices]
        upper_fit = np.array(uppers)[valid_indices]

        # Plot error rates and confidence intervals
        plt.plot(distances_fit, ps_fit, label=label, marker='o', color=color, markersize=2, linewidth=1)
        plt.fill_between(distances_fit, lower_fit, upper_fit, color='black', alpha=0.1, edgecolor='none', label='68% CI')

        # Identify outliers
        outlier_indices = np.where(errs <= outlier_nb)
        outlier_distances = np.array(distances)[outlier_indices]
        outlier_ps = np.array(ps)[outlier_indices]
        outlier_lowers = np.array(lowers)[outlier_indices]
        outlier_uppers = np.array(uppers)[outlier_indices]

        # Connect the last non-outlier point to the first outlier if outliers exist
        if outlier_distances.size > 0 and distances_fit.size > 0:
            # Append the last non-outlier point to the beginning of the outlier data
            outlier_distances = np.append(outlier_distances, distances_fit[0])
            outlier_ps = np.append(outlier_ps, ps_fit[0])   
            outlier_lowers = np.append(outlier_lowers, lower_fit[0])
            outlier_uppers = np.append(outlier_uppers, upper_fit[0])

            # Plot outliers with a dashed line starting from the last non-outlier
            plt.plot(outlier_distances, outlier_ps, linestyle=':', color=color, linewidth=0.6)
            plt.plot(outlier_distances[:-1], outlier_ps[:-1], marker='o', color=color, markersize=3, linewidth=0, fillstyle='none', markeredgewidth=0.5)
            plt.fill_between(outlier_distances, outlier_lowers, outlier_uppers, color='black', alpha=0.1, edgecolor='none')


        if plot_e_L and len(distances_fit) > 1:
            log_ps_fit = np.log10(ps_fit)

            popt, pcov = curve_fit(lambda_func, distances_fit, log_ps_fit)
            log10_C, neg_half_log10_Lambda = popt
            C = 10 ** log10_C
            Lambda = 10 ** (-2 * neg_half_log10_Lambda)  # since neg_half_log10_Lambda = -0.5 * log10(Lambda)
            # lambda_error = np.sqrt(pcov[1, 1])
            error_neg_half_log10_Lambda = np.sqrt(pcov[1, 1])
            lambda_error = abs(-2 * np.log(10) * Lambda * error_neg_half_log10_Lambda) # error propagation

            # Plotting the fit
            fit_distances = np.linspace(min(distances), max(distances), 100)
            fit_ps = lambda_func(fit_distances, log10_C, neg_half_log10_Lambda)
            plt.plot(fit_distances, 10**fit_ps, '--', color=color, label=f"Λ-Fit: Λ={Lambda:.2f}±{lambda_error:.2f}", linewidth=0.2)

    plt.yscale("log")
    plt.ylim(1e-6, 1) if not plot_e_L else plt.ylim(2e-8, 2e-2)
    plt.ylabel('Logical error probability') if not plot_e_L else plt.ylabel('Logical error per round')
    plt.xlabel('Distance')
    plt.title(title) if title is not None else None
    plt.grid(True, which="both", linestyle='--', linewidth=0.2)
    plt.xticks(distances_list[0][::2])  # Assuming all lists have similar step, adjust if needed



    plt.legend(fontsize=3, loc='upper right')
    plt.show()


label_dict = {
    's_K': 'soft decoding (not PS)',
    's_KPS': 'soft decoding (PS)',
    'h_K': 'hard decoding (not PS)',
    'h_KPS': 'hard decoding (PS)',
    'h_K_mean': 'Data-informed hard decoding (not PS)',
    'h_K_meanPS': 'Data-informed hard decoding (PS)',
}



def wilson_score_interval(p, n, z=1): # z=1.96 for 95% confidence
    """
    Calculate Wilson score interval for a given success probability (p), number of trials (n), and z-score (z).
    """
    denominator = 1 + z**2 / n
    term = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    lower = (p + z**2 / (2 * n) - term) / denominator
    upper = (p + z**2 / (2 * n) + term) / denominator

    # if p == 0: # For plots with log scale
    #     lower = 0
    #     upper = 0

    return max(lower, 0), min(upper, 1)


def logical_err_rate_per_round(p, T):
    return (1-np.exp(1/T * np.log(1-2*p)))/2


def lambda_func(d, log10_C, neg_half_log10_Lambda):
    return log10_C + neg_half_log10_Lambda * (d + 1)

# def lambda_func(d, C, Lambda):
#     return C * Lambda ** (-(d+1)/2)