from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import colormaps
import matplotlib.lines as mlines
from matplotlib.path import Path

from scipy.optimize import curve_fit

from .data_wrangler import get_err_dicts
from .general import apply_formatting



def plot_error_rates(distances_list: List, 
                     errs_list: List, 
                     shots_list: List, 
                     labels: Optional[List] = None, 
                     title: str = None, 
                     plot_e_L: bool = False, 
                     T = None, 
                     outlier_nb = 5, 
                     state = None,
                     one_collumn = False, 
                     dpi = 1000,
                     ylim = None,
                     plot = True, 
                     colors_TBU = None,
                     T_s = None,
                     small_legend = False) -> None:
    
    if plot == False:
        plt.ioff()  # Turn interactive plotting off
    else:
        plt.ion()
    
    if state:
        marker = get_marker(state)

    if labels is None:
        labels = [f"Dataset {i}" for i in range(len(distances_list))]

    third_height = True if T_s is not None else False
    apply_formatting(one_column=one_collumn, dpi=dpi, third_height=third_height)
    figure = plt.figure()
    lambdas = []
    lambda_errors = []
    colors = []
    iterator = zip(distances_list, errs_list, shots_list, labels, labels) if T_s is None else zip(distances_list, errs_list, shots_list, labels, T_s)
    for idx, (distances, errs, shots, label, Tc) in enumerate(iterator):
        T = Tc if T_s is not None else T
        if colors_TBU is None:
            color = get_color(label, state) if state else 'b'
            colors.append(color)
        else:
            color = colors_TBU[idx]
        label = get_label(label)
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

        # Exclude outlier points for fitting
        valid_indices = np.where(errs > outlier_nb)
        distances_fit = np.array(distances)[valid_indices]
        ps_fit = np.array(ps)[valid_indices]
        lower_fit = np.array(lowers)[valid_indices]
        upper_fit = np.array(uppers)[valid_indices]


        # Plot error rates and confidence intervals
        plt.fill_between(distances_fit, lower_fit, upper_fit, color='black', alpha=0.1, edgecolor='none', label='$68\%$ CI (Wilson)')
        plt.plot(distances_fit, ps_fit, label=label, marker=marker, color=color, markersize=2, linewidth=1)

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
            plt.fill_between(outlier_distances, outlier_lowers, outlier_uppers, color='black', alpha=0.1, edgecolor='none')
            plt.plot(outlier_distances, outlier_ps, linestyle=':', color=color, linewidth=0.6)
            plt.plot(outlier_distances[:-1], outlier_ps[:-1], marker=marker, color=color, markersize=3, linewidth=0, fillstyle='none', markeredgewidth=0.5)


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
            plt.plot(fit_distances, 10**fit_ps, '--', color=color, label="$\Lambda$-Fit: $\Lambda$=", linewidth=0.2)
            # plt.plot(fit_distances, 10**fit_ps, '--', color=color, label="$\Lambda$-Fit: $\Lambda$=" + f"{Lambda:.2f}±{lambda_error:.2f}", linewidth=0.2)
            lambdas.append(Lambda)
            lambda_errors.append(lambda_error)



    # Custom legend
    ci_patch = mpatches.Patch(color='black', alpha=0.1, label='68\% CI (Wilson)')
    
    outlier_handle = mlines.Line2D([], [], color='grey', linestyle=':', marker=marker, 
                               markersize=3, markerfacecolor='none', markeredgewidth=0.5, 
                               label='Outliers $n_{errs} \leq$ ' + str(outlier_nb), linewidth=0.6)
    
    plot_handles = []
    if colors_TBU is not None:
        colors = colors_TBU
    
    iterator = zip(colors, labels, lambdas, lambda_errors) if plot_e_L else zip(colors, labels, colors, labels)

    for color, label, Lambda_, lambda_error_ in iterator:
        if T_s is not None:
            handle = mlines.Line2D([], [], color=color, marker=marker, linestyle='-', 
                                markersize=2, linewidth=1, label=f"{label}, $\Lambda$=" + f"{Lambda_:.2f}$\pm${lambda_error_:.2f}")
            plot_handles.append(handle)

        else:
            if small_legend is True and plot_e_L is True and label in ['s_KPS', 'h_KPS']:
                handle_label = f"{get_label(label)}, $\Lambda$=" + f"{Lambda_:.2f}$\pm${lambda_error_:.2f}"
                handle = mlines.Line2D([], [], color=color, marker=marker, linestyle='-', 
                                markersize=2, linewidth=1, label=handle_label)
            else:
                handle = mlines.Line2D([], [], color=color, marker=marker, linestyle='-', 
                                    markersize=2, linewidth=1, label=get_label(label))
            plot_handles.append(handle)
            
            if plot_e_L:
                if small_legend is True and label in ['s_KPS', 'h_KPS']:
                    continue
        
                handle_lambda = mlines.Line2D([], [], color=color, linestyle='--',
                                    # linewidth=0.2, label=f"Λ-Fit: Λ={Lambda_:.2f}±{lambda_error_:.2f}")
                                    linewidth=0.2, label="$\Lambda$-Fit: $\Lambda$=" + f"{Lambda_:.2f}$\pm${lambda_error_:.2f}")
                plot_handles.append(handle_lambda)    



    # Create the legend by combining all handles
    plt.legend(handles=plot_handles + [ci_patch, outlier_handle], loc='best') if small_legend is False else plt.legend(handles=plot_handles, loc='best')

    plt.yscale("log")
    plt.ylim(5e-6, 0.7) if not plot_e_L else plt.ylim(1e-7, 2e-2)
    if ylim is not None:
        plt.ylim(ylim)
    plt.ylabel('Logical error probability') if not plot_e_L else plt.ylabel('Logical error per round')
    plt.xlabel('Distance')
    plt.title(title) if title is not None else None
    plt.grid(True, which="both", linestyle='--', linewidth=0.2)
    plt.xticks(distances_list[0][::4])  # Assuming all lists have similar step, adjust if needed


    return figure



def get_label(label):
    label_dict = {
        's_K': 'soft decoding (not PS)',
        's_KPS': 'Soft MWPM',
        'h_K': 'hard decoding (not PS)',
        's_KPS_no_ambig': 'Soft MWPM (no leak. handl.)',
        'h_KPS': 'Hard MWPM',
        'h_K_mean': 'Data-informed hard decoding (not PS)',
        'h_K_meanPS': 'Data-informed Hard MWPM',
        'h_G': 'Gaussian Hard MWPM',
        's_G': 'Gaussian Soft MWPM',
    }

    return label_dict[label] if label in label_dict else label

def  get_color(method, state):
    color_dict = {
        's_KPSZ': 'skyblue',
        's_KPSX': 'lightcoral',
        'h_KPSZ': 'midnightblue',
        'h_KPSX': 'darkred',
        'h_K_meanPSZ': 'steelblue',
        'h_K_meanPSX': 'tomato',
        'h_GZ': 'darkviolet',
        'h_GX': 'crimson',
        's_GZ': 'orchid',
        's_GX': 'orangered',
        's_KPS_no_ambigZ': 'deepskyblue',
        's_KPS_no_ambigX': 'red',
    }

    return color_dict[method+state[0]]

def get_marker(state):
    marker_dict = {
        '0': 'o',
        '1': 'd',
    }

    return marker_dict[state[1]]


def wilson_score_interval(p, n, z=1): # z=1.96 for 95% confidence
    """
    Calculate Wilson score interval for a given success probability (p), number of trials (n), and z-score (z).
    """
    denominator = 1 + z**2 / n
    term = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    lower = (p + z**2 / (2 * n) - term) / denominator
    upper = (p + z**2 / (2 * n) + term) / denominator

    return max(lower, 0), min(upper, 1)


def logical_err_rate_per_round(p, T):
    return (1-np.exp(1/T * np.log(1-2*p)))/2


def lambda_func(d, log10_C, neg_half_log10_Lambda):
    return log10_C + neg_half_log10_Lambda * (d + 1)


def plot_info_perfo_ratios(err_per_state_per_method, method='s_KPS', max_ds=[33, 27]):

    num_states = len(err_per_state_per_method)
    if num_states != 2: 
        raise ValueError("Only 2 state plotting is supported for now")
    
    apply_formatting()
    fig, axs = plt.subplots(num_states, 1, sharex=True)  # Added sharey=True


    for ax, (state, err_per_method), max_d in zip(axs, err_per_state_per_method.items(), max_ds):
        cmap = get_cmap(state)
        colors = iter(cmap(np.linspace(0.2, 1, len(err_per_method['s_KPS'].keys()))))
        marker = get_marker(state)
        nBits_list = err_per_method['nBits_list']
        error_dict = err_per_method[method]

        for distance, errs in sorted(error_dict.items()):
            if distance > max_d:
                continue
            color = next(colors)
            err_ratio = errs[:-1] / errs[-1]
            ax.plot(nBits_list[:-1], err_ratio, label=f"d={distance}", color=color, marker=marker, markersize=2, linewidth=1)
        
        ax.grid(True, which="both", linestyle='--', linewidth=0.2)
        columns = 2
        ax.legend(ncol=columns, loc='best')
    
        # Adding a vertical line at 8 bits
        ax.axvline(x=8, color='k', linestyle='--', linewidth=1, alpha=0.6)
        # Adding text with a break in the line for "8 bits"
        middle_y_value = (ax.get_ylim()[1]-ax.get_ylim()[0]) / 2  # Calculate the middle y value based on the current limits
        ax.text(x=8.17, y=middle_y_value+ax.get_ylim()[0], s='1 byte', rotation=90, 
                verticalalignment='center', horizontalalignment='right', 
                color='k', fontsize=8, backgroundcolor='white', alpha=0.6)


    # Set common labels and title
    fig.text(0.53, 0.04, r'Number of bits $b$ used for $P^b_s(\mu)$', ha='center')  # Common x-axis label
    fig.text(0.02, 0.55, r'Logical error probability ratio $\frac{P^{b}_L}{P^{64}_L}$', va='center', rotation='vertical')  # Common y-axis label

    plt.tight_layout(rect=[0.05, 0.05, 1, 1])  # Adjust layout to make room for axis labels and legend
    return fig




def get_cmap(state):
    cmap_dict = {
        'Z': 'Blues',
        'X': 'Reds',
    }
    return colormaps[cmap_dict[state[0]]]






def analyze_lambda_ratios(device, states, distances=range(3, 51, 2), rounds = [10, 20, 30, 40, 50, 75, 100], outlier_nbs=5, plot_individual_states=False):
    lambda_per_state = {}
    lambda_err_per_state = {}
    
    for state in states:
        round_err_dict = {}
        for T in rounds:
            file_name = f'../results/{device}_{state}_{T}.json'
            try:
                _, err_per_method = get_err_dicts(file_name)
                round_err_dict[T] = err_per_method
            except:
                continue

        lambda_per_round = {T: {} for T in round_err_dict.keys()}
        lambda_errors_per_round = {T: {} for T in round_err_dict.keys()}
        for method in ['s_KPS', 'h_KPS', 'h_K_meanPS']:
            for T in round_err_dict.keys():
                ds = np.array(round_err_dict[T][method]['d'])
                errs = np.array(round_err_dict[T][method]['errs'])
                shots = np.array(round_err_dict[T][method]['tot_shots'])
                ps = errs / shots
                e_Ls = logical_err_rate_per_round(ps, T)

                valid_indices = np.where(errs > outlier_nbs)
                d_fit = ds[valid_indices]
                e_Ls_fit = e_Ls[valid_indices]
                log_eLs_fit = np.log10(e_Ls_fit)
                popt, pcov = curve_fit(lambda_func, d_fit, log_eLs_fit)
                log10_C, neg_half_log10_Lambda = popt
                Lambda = 10 ** (-2 * neg_half_log10_Lambda)
                error_neg_half_log10_Lambda = np.sqrt(pcov[1, 1])
                lambda_error = abs(-2 * np.log(10) * Lambda * error_neg_half_log10_Lambda)  # error propagation

                lambda_per_round[T][method] = Lambda
                lambda_errors_per_round[T][method] = lambda_error

        lambda_per_state[state] = lambda_per_round
        lambda_err_per_state[state] = lambda_errors_per_round


        if plot_individual_states:
            plot_state_lambda_ratios(state, lambda_per_state[state], lambda_err_per_state[state])

    fig = plot_mean_lambda_ratios(lambda_per_state, lambda_err_per_state, states)
    return lambda_per_state, lambda_err_per_state, fig




def mean_lambda_plot(lambda_per_state, lambda_errors_per_state, methods=['s_KPS', 'h_KPS', 'h_K_meanPS']):
    # Initialize dictionaries to store aggregated lambdas and errors across rounds and methods
    lambda_means = {method: [] for method in methods}
    lambda_errors = {method: [] for method in methods}
    rounds_list = []

    # Assuming all states have the same rounds, use the first state to get the list of rounds
    first_state = next(iter(lambda_per_state))
    rounds_list = sorted(lambda_per_state[first_state].keys())

    # Aggregate lambda values and errors across all states for each round and method
    for round_val in rounds_list:
        for method in methods:
            lambdas = []
            errors = []
            for state in lambda_per_state:
                if round_val in lambda_per_state[state] and method in lambda_per_state[state][round_val]:
                    lambdas.append(lambda_per_state[state][round_val][method])
                    errors.append(lambda_errors_per_state[state][round_val][method])

            # Calculate mean and propagated error for lambdas
            if lambdas:
                mean_lambda = np.mean(lambdas)
                error_propagation = np.sqrt(sum(np.array(errors) ** 2)) / len(errors)
                lambda_means[method].append(mean_lambda)
                lambda_errors[method].append(error_propagation)

    # Plotting
    apply_formatting()  # Assume this sets up some common plot formatting
    fig, ax = plt.subplots()


    colors = ['forestgreen', 'darkred', 'chocolate']
    # colors = ['forestgreen', 'darkred', 'peru']
    handles = []
    labels = []
    for method, color in zip(methods, colors):
        line = ax.errorbar(rounds_list, lambda_means[method], yerr=lambda_errors[method], label=get_label(method), fmt='-o', 
                    color=color, markersize=5, linewidth=2)
        labels.append(get_label(method))
        handles.append(line[0])
        
    fill_s  = plt.fill_between(rounds_list, lambda_means['s_KPS'], lambda_means['h_K_meanPS'], color='limegreen', alpha=0.2,
                     label='s')
    fill_h  = plt.fill_between(rounds_list, lambda_means['h_K_meanPS'], lambda_means['h_KPS'], color='peachpuff', alpha=0.6, 
                     label='h')
    # handles.append(fill_s)
    # handles.append(fill_h)
    # labels.append('Dynamic reweighting improvement')
    # labels.append('Static leakage-informed improvement')

    # Annotate with arrow
    ax.annotate('Dynamic reweighting', xytext=(65, 2.2), xy=(38, 1.65),
                arrowprops=dict(facecolor='black',  arrowstyle='-'),
                fontsize=9, ha='center', backgroundcolor='white', color='forestgreen')
    
    ax.annotate('Static leakage-informing', xytext=(75, 1.8), xy=(38, 1.5),
                arrowprops=dict(facecolor='black',  arrowstyle='-'),
                fontsize=9, ha='center', backgroundcolor='white', color='chocolate')

    new_order = [0, 2, 1, 3, 4]
    new_order = [0, 2, 1,]
    reordered_handles = [handles[i] for i in new_order]
    reordered_labels = [labels[i] for i in new_order]


    ax.set_xlabel('Rounds')
    ax.set_ylabel('Mean Lambda Value')
    plt.xticks(range(10, 110, 20))
    # ax.set_title('Mean Lambda Values vs Rounds for Different Methods')
    ax.legend(fontsize=10, handles=reordered_handles, labels=reordered_labels, loc='upper right')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.show()
    return fig





def plot_state_lambda_ratios(state, lambda_per_round, lambda_errors_per_round):
    apply_formatting()
    fig = plt.figure()
    rounds = np.array(list(lambda_per_round.keys()))
    lambdas_S = np.array([lambda_per_round[T]['s_KPS'] for T in rounds])
    errors_S = np.array([lambda_errors_per_round[T]['s_KPS'] for T in rounds])
    lambdas_H = np.array([lambda_per_round[T]['h_KPS'] for T in rounds])
    errors_H = np.array([lambda_errors_per_round[T]['h_KPS'] for T in rounds])
    lambdas_H_mean = np.array([lambda_per_round[T]['h_K_meanPS'] for T in rounds])
    errors_H_mean = np.array([lambda_errors_per_round[T]['h_K_meanPS'] for T in rounds])
    
    ratio_S = lambdas_S / lambdas_H - 1
    ratio_S_error = np.sqrt((errors_S / lambdas_H) ** 2 + (lambdas_S * errors_H / lambdas_H**2) ** 2)
    ratio_H_mean = lambdas_H_mean / lambdas_H - 1
    ratio_H_mean_error = np.sqrt((errors_H_mean / lambdas_H) ** 2 + (lambdas_H_mean * errors_H / lambdas_H**2) ** 2)

    plt.errorbar(rounds, ratio_S, yerr=ratio_S_error, label='s_KPS/h_KPS', fmt='-o')
    plt.errorbar(rounds, ratio_H_mean, yerr=ratio_H_mean_error, label='h_K_meanPS/h_KPS', fmt='-o')

    plt.title(f'{state} Logical Error Rate Ratio')
    plt.ylabel('Ratio - 1 (%)')
    plt.xlabel('Rounds')
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    return fig








def plot_mean_lambda_ratios(lambda_per_state, lambda_errors_per_state, states):
    ratio_S_per_round = {}
    ratio_H_mean_per_round = {}
    ratio_S_errors_per_round = {}
    ratio_H_mean_errors_per_round = {}

    for state in states:
        rounds = np.array(list(lambda_per_state[state].keys()))
        for round_idx, round_val in enumerate(rounds):
            lambdas_S = np.array([lambda_per_state[state][T]['s_KPS'] for T in rounds])
            errors_S = np.array([lambda_errors_per_state[state][T]['s_KPS'] for T in rounds])
            lambdas_H = np.array([lambda_per_state[state][T]['h_KPS'] for T in rounds])
            errors_H = np.array([lambda_errors_per_state[state][T]['h_KPS'] for T in rounds])
            lambdas_H_mean = np.array([lambda_per_state[state][T]['h_K_meanPS'] for T in rounds])
            errors_H_mean = np.array([lambda_errors_per_state[state][T]['h_K_meanPS'] for T in rounds])

            ratio_S = lambdas_S / lambdas_H
            ratio_S_error = np.sqrt((errors_S / lambdas_H) ** 2 + (lambdas_S * errors_H / lambdas_H**2) ** 2)
            ratio_H_mean = lambdas_H_mean / lambdas_H
            ratio_H_mean_error = np.sqrt((errors_H_mean / lambdas_H) ** 2 + (lambdas_H_mean * errors_H / lambdas_H**2) ** 2)

            if round_val not in ratio_S_per_round:
                ratio_S_per_round[round_val] = []
                ratio_H_mean_per_round[round_val] = []
                ratio_S_errors_per_round[round_val] = []
                ratio_H_mean_errors_per_round[round_val] = []

            ratio_S_per_round[round_val].append(ratio_S[round_idx])
            ratio_H_mean_per_round[round_val].append(ratio_H_mean[round_idx])
            ratio_S_errors_per_round[round_val].append(ratio_S_error[round_idx])
            ratio_H_mean_errors_per_round[round_val].append(ratio_H_mean_error[round_idx])

    mean_ratio_S = np.array([np.mean(ratio_S_per_round[round_val]) for round_val in sorted(ratio_S_per_round)])
    mean_ratio_H_mean = np.array([np.mean(ratio_H_mean_per_round[round_val]) for round_val in sorted(ratio_H_mean_per_round)])
    mean_ratio_S_errors = np.array([np.sqrt(np.sum(np.array(ratio_S_errors_per_round[round_val])**2)) / len(ratio_S_errors_per_round[round_val]) for round_val in sorted(ratio_S_errors_per_round)])
    mean_ratio_H_mean_errors = np.array([np.sqrt(np.sum(np.array(ratio_H_mean_errors_per_round[round_val])**2)) / len(ratio_H_mean_errors_per_round[round_val]) for round_val in sorted(ratio_H_mean_errors_per_round)])
    sorted_rounds = np.array(sorted(ratio_S_per_round.keys()))



    apply_formatting()
    fig = plt.figure()

    plt.errorbar(sorted_rounds, (mean_ratio_S-1)*100, yerr=mean_ratio_S_errors*100, fmt='o-', color='forestgreen', markersize=5, linewidth=2, label='Soft MWPM')
    plt.errorbar(sorted_rounds, (mean_ratio_H_mean-1)*100, yerr=mean_ratio_H_mean_errors*100, fmt='o-', color='chocolate', markersize=5, linewidth=2, label='Data-Informed Hard MWPM')

    # Define the threshold line at 20%
    threshold = 20
    line_points = interpolate_value(sorted_rounds, (mean_ratio_S-1)*100, query_point=np.linspace(10, 101, 100))
    mask = (line_points >= threshold)
    fill = plt.fill_between(np.linspace(10, 101, 100)[mask], threshold, line_points[mask], color='limegreen', alpha=0.3, edgecolor='none')

    # # Adding a horizontal line at 20% improvement
    alpha = 0.5
    plt.axhline(y=20, color='k', linestyle='--', linewidth=1, alpha=alpha)
    plt.text(x=67, y=19.2, s=r' $\geq$20\% higher threshold', verticalalignment='bottom', color='k', fontweight='bold', alpha=0.7,
             backgroundcolor='white')


    plt.ylabel('Average threshold improvement (\%)')
    plt.xlabel('Rounds')
    plt.ylim(-3, 28)
    plt.xlim(5, 105)
    plt.yticks(range(0, 26, 5))
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=8)

    return fig

    apply_formatting()
    fig = plt.figure()
    plt.fill_between(sorted_rounds, (mean_ratio_S - mean_ratio_S_errors-1)*100, (mean_ratio_S + mean_ratio_S_errors-1)*100, 
                    color='k', alpha=0.2, edgecolor='none', label='Fit Uncertainty')
    plt.plot(sorted_rounds, (mean_ratio_S-1)*100, marker='o', markersize=5, linewidth=2, label='Soft MWPM', color='forestgreen')

    # Adding a horizontal line at 20% improvement
    plt.fill_between(range(0, 120), 20, 50, color='limegreen', alpha=0.1, edgecolor='none')
    color = 'limegreen'
    plt.axhline(y=20, color=color, linestyle='-', linewidth=1, alpha=0.5)
    plt.text(x=65, y=22, s=r' $\geq$20\% higher threshold', verticalalignment='bottom', color=color, fontweight='bold', alpha=0.8)

    plt.fill_between(sorted_rounds, (mean_ratio_H_mean - mean_ratio_H_mean_errors-1)*100, (mean_ratio_H_mean + mean_ratio_H_mean_errors-1)*100, 
                    color='k', alpha=0.2, edgecolor='none', label='Fit Uncertainty')
    plt.plot(sorted_rounds, (mean_ratio_H_mean-1)*100, marker='o', markersize=5, linewidth=2, label='Data-Informed Hard MWPM', color='chocolate')

    plt.ylabel('Average threshold improvement (\%)')
    plt.xlabel('Rounds')
    plt.ylim(-1, 28)
    plt.xlim(5, 105)
    plt.yticks(range(0, 26, 5))
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=10)

    return fig

def interpolate_value(sorted_rounds, data, query_point):
    sorted_rounds = np.array(sorted_rounds)
    data = np.array(data)
    
    # if query_point < sorted_rounds[0] or query_point > sorted_rounds[-1]:
    #     raise ValueError("Query point is outside the range of the data.")
    
    # Perform linear interpolation
    interpolated_value = np.interp(query_point, sorted_rounds, data)
    
    return interpolated_value
