# Maurice Hanisch mhanisc@ethz.ch
# created 24.10.2023

import numpy as np
import matplotlib.pyplot as plt
from .general import apply_formatting


def plot_IQ_data(data, n_bins=250, twod_nbins=100, figsize=(8,4), title=None, dpi = 100):
    apply_formatting()
    real_parts = np.real(data).flatten()
    imag_parts = np.imag(data).flatten()

    fig = plt.figure(dpi=dpi)

    # Main scatter plot
    alpha = min(1, max(2e4 / len(data), 2e-3))
    ax_scatter = plt.subplot2grid((4, 8), (1, 0), rowspan=3, colspan=3)
    ax_scatter.scatter(real_parts, imag_parts,
                        alpha=alpha, marker='.', s=0.1, rasterized=True)
    ax_scatter.set_xlabel("In-Phase [arb.]")
    ax_scatter.set_ylabel("Quadrature [arb.]")
    ax_scatter.grid(True)

    # Histogram for the real parts (on the side of scatter plot)
    ax_hist_x = plt.subplot2grid((4, 8), (0, 0), colspan=3, rowspan=1)
    ax_hist_x.hist(real_parts, bins=n_bins,
                    align='mid', color='blue', alpha=0.3)
    ax_hist_x.set_ylabel("Counts")
    ax_hist_x.xaxis.set_ticklabels([])  # Remove x tick labels
    ax_hist_x.grid(True)
    ax_hist_x.set_title("Scatter Plot")

    # Histogram for the imaginary parts (above the scatter plot)
    ax_hist_y = plt.subplot2grid((4, 8), (1, 3), rowspan=3, colspan=1)
    ax_hist_y.hist(imag_parts, bins=n_bins, orientation='horizontal',
                    align='mid', color='red', alpha=0.3)
    ax_hist_y.set_xlabel("Counts")
    ax_hist_y.yaxis.set_ticklabels([])  # Remove y tick labels
    ax_hist_y.grid(True)

    # 3D Histogram Heatmap
    H, xedges, yedges = np.histogram2d(real_parts, imag_parts, bins=twod_nbins)
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    ax3d = plt.subplot2grid((4, 8), (0, 4), rowspan=4,
                            colspan=4, projection='3d')

    ax3d.plot_surface(X*1e-8, Y*1e-8, H.T, cmap='plasma', rasterized=True)
    ax3d.set_title("3D Histogram Heatmap")

    ax3d.set_xlabel("In-Phase [arb.]", fontsize=8)
    ax3d.set_ylabel("Quadrature [arb.]", fontsize=8)
    ax3d.set_zlabel("Counts", fontsize=8)
    # ax3d.set_xlim(-0.9, 1.2)

    ax3d.tick_params(axis='both', which='major', labelsize=6, pad=-2)
    ax3d.xaxis.offsetText.set_fontsize(6)
    ax3d.yaxis.offsetText.set_fontsize(6)
    ax3d.zaxis.offsetText.set_fontsize(6)

    ax3d.xaxis.labelpad = -5
    ax3d.yaxis.labelpad = -5
    ax3d.zaxis.labelpad = -10

    ax3d.set_box_aspect(None, zoom=1.1)

    if title:
        plt.suptitle(title)
    fig.subplots_adjust(hspace=0.3)  # Adjust the horizontal spacing if needed
    return fig



def plot_IQ_data_pSoft_cmap(data, pSoft, title=None, dpi=100, alpha=0.7):
    real_parts = np.real(data).flatten()
    imag_parts = np.imag(data).flatten()

    apply_formatting(dpi=dpi)
    fig, ax_scatter = plt.subplots()

    # Main scatter plot
    sc = ax_scatter.scatter(real_parts, imag_parts,
                            alpha=alpha, marker='.', s=0.1, c=pSoft, cmap='viridis')
    ax_scatter.set_xlabel("In-Phase [arb.]")
    ax_scatter.set_ylabel("Quadrature [arb.]")
    ax_scatter.grid(True)
    # ax_scatter.set_ylim(-2.7e6, 7.5e6,)
    # ax_scatter.set_xlim(-7.5e6, 7.5e6)

    # Color bar for the scatter plot
    cbar = plt.colorbar(sc)
    cbar.set_label('$p_s(\mu)$')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.yaxis.set_ticks_position('right')
    # cbar.ax.set_alpha(1.0)  # Set colorbar alpha to 100%
    cbar.solids.set_edgecolor("face")  # This fixes the alpha issue on some backends
    

    if title:
        plt.suptitle(title)
    plt.show()
    return fig

# def plot_IQ_data_pSoft_cmap(data, pSoft, n_bins=250, title=None, dpi=100, alpha=0.7):

#     real_parts = np.real(data).flatten()
#     imag_parts = np.imag(data).flatten()

#     apply_formatting(dpi=dpi)
#     fig = plt.figure()

#     # Main scatter plot
#     # alpha = min(1, max(4e4 / len(data), 2e-3))
#     alpha = alpha
#     ax_scatter = plt.subplot2grid((4, 8), (1, 0), rowspan=3, colspan=6)
#     sc = ax_scatter.scatter(real_parts, imag_parts,
#                             alpha=alpha, marker='.', s=0.1, c=pSoft, cmap='viridis')
#     ax_scatter.set_xlabel("In-Phase [arb.]")
#     ax_scatter.set_ylabel("Quadrature [arb.]")
#     ax_scatter.grid(True)

#     # Color bar for the scatter plot
#     # cbar_ax = fig.add_axes([-0.05, 0.05, 0.02, 0.6])  # Adjust these parameters to move and resize the color bar
#     cbar_ax = fig.add_axes([0.85, 0.175, 0.02, 0.6])  # Adjust these parameters to move and resize the color bar
#     cbar = plt.colorbar(sc, cax=cbar_ax)
#     cbar.set_label('$p_s(\mu)$')

#     cbar.ax.yaxis.set_label_position('right')  
#     cbar.ax.yaxis.set_ticks_position('right')  

#     # Histogram for the real parts (on the side of scatter plot)
#     ax_hist_x = plt.subplot2grid((4, 8), (0, 0), colspan=6, rowspan=1)
#     ax_hist_x.hist(real_parts, bins=n_bins, align='mid', color='grey', alpha=0.3)
#     ax_hist_x.set_ylabel("Counts")
#     ax_hist_x.xaxis.set_ticklabels([])  # Remove x tick labels
#     ax_hist_x.grid(True)

#     # Histogram for the imaginary parts (above the scatter plot)
#     ax_hist_y = plt.subplot2grid((4, 8), (1, 6), rowspan=3, colspan=1)
#     ax_hist_y.hist(imag_parts, bins=n_bins, orientation='horizontal',
#                    align='mid', color='grey', alpha=0.3)
#     ax_hist_y.set_xlabel("Counts")
#     ax_hist_y.yaxis.set_ticklabels([])  # Remove y tick labels
#     ax_hist_y.grid(True)

#     fig.subplots_adjust(hspace=0.4, wspace=0.3)
#     if title:
#         plt.suptitle(title)
#     plt.show()
#     return fig



def plot_double_IQ_scatter_outlier(data, pSoft, pSoft_trunc, dpi=1000):
    real_parts = np.real(data).flatten()
    imag_parts = np.imag(data).flatten()

    apply_formatting(third_height=True, dpi=dpi)  # Custom formatting for the plot (assumed to be defined)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)

    # Configure alpha based on data density
    alpha = 0.7

    # First scatter plot
    sc1 = ax1.scatter(real_parts, imag_parts, alpha=alpha, marker='.', s=0.1, c=pSoft, cmap='viridis', rasterized=True, vmin=0, vmax=0.5)
    ax1.set_ylabel("Quadrature [arb.]")
    ax1.set_title("$p_s(\mu)$ without outlier handling")
    ax1.grid(True)

    # Second scatter plot
    sc2 = ax2.scatter(real_parts, imag_parts, alpha=alpha, marker='.', s=0.1, c=pSoft_trunc, cmap='viridis', rasterized=True, vmin=0, vmax=0.5)
    ax2.set_xlabel("In-Phase [arb.]")
    ax2.set_ylabel("Quadrature [arb.]")
    ax2.set_title("$p_s(\mu)$ with outlier handling")
    ax2.grid(True)

    # Color bar for the scatter plots
    cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.77])  # Adjust these parameters to properly place and size the color bar
    cbar = plt.colorbar(sc1, cax=cbar_ax)
    cbar.set_label('Soft flip probability $p_s(\mu)$')
    cbar.ax.set_alpha(1.0)  # Set colorbar alpha to 100%

    # Adjust layout
    fig.subplots_adjust(hspace=0.2)  # Adjust the horizontal spacing if needed
    
    # plt.show()
    return fig


def plot_double_IQ_scatter(data, pSoft, pSoft_trunc, nBits, dpi=2000):
    real_parts = np.real(data).flatten()
    imag_parts = np.imag(data).flatten()

    apply_formatting(third_height=True, dpi=dpi)  # Custom formatting for the plot (assumed to be defined)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)

    # Configure alpha based on data density
    alpha = 0.7

    # First scatter plot
    sc1 = ax1.scatter(real_parts, imag_parts, alpha=alpha, marker='.', s=0.1, c=pSoft, cmap='viridis', rasterized=True, vmin=0, vmax=0.5)
    ax1.set_ylabel("Quadrature [arb.]")
    ax1.set_title("64-bit-accuracy $p^{64}_s(\mu)$")
    ax1.grid(True)

    # Second scatter plot
    sc2 = ax2.scatter(real_parts, imag_parts, alpha=alpha, marker='.', s=0.1, c=pSoft_trunc, cmap='viridis', rasterized=True, vmin=0, vmax=0.5)
    ax2.set_xlabel("In-Phase [arb.]")
    ax2.set_ylabel("Quadrature [arb.]")
    ax2.set_title(f"{nBits}-bit-accuracy $p^{nBits}_s(\mu)$")
    ax2.grid(True)

    # Color bar for the scatter plots
    cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.77])  # Adjust these parameters to properly place and size the color bar
    cbar = plt.colorbar(sc1, cax=cbar_ax)
    cbar.set_label('Soft flip probability $p_s(\mu)$')

    # Adjust layout
    fig.subplots_adjust(hspace=0.2)  # Adjust the horizontal spacing if needed
    
    # plt.show()
    return fig



def plot_IQ_data_with_countMat(data, countMat, n_bins=250, title=None, dpi=100):
    apply_formatting()
    real_parts = np.real(data).flatten()
    imag_parts = np.imag(data).flatten()
    colors = ['steelblue' if value == 0 else 'purple' for value in countMat.flatten()]
    colors = ['steelblue' if value == 0 else 'purple' if value == 1 else 'red' if value == 2 else 'black' for value in countMat.flatten()]

    # fig = plt.figure(figsize=figsize, dpi=dpi)
    apply_formatting(dpi=dpi)
    fig = plt.figure()

    # Main scatter plot
    alpha = min(1, max(2e4 / len(data), 2e-3))
    ax_scatter = plt.subplot2grid((4, 9), (1, 1), rowspan=3, colspan=6)
    ax_scatter.scatter(real_parts, imag_parts, alpha=alpha, marker='.', s=0.1, c=colors, rasterized=True)
    ax_scatter.set_xlabel("In-Phase [arb.]")
    ax_scatter.set_ylabel("Quadrature [arb.]")
    ax_scatter.grid(True)


    # Histogram for the real parts (on the side of scatter plot)
    ax_hist_x = plt.subplot2grid((4, 9), (0, 1), rowspan=1, colspan=6)
    ax_hist_x.hist(real_parts, bins=n_bins, align='mid', color='black', alpha=0.3)
    ax_hist_x.set_ylabel("Counts")
    ax_hist_x.xaxis.set_ticklabels([])  # Remove x tick labels
    ax_hist_x.grid(True)
    # ax_hist_x.set_title("Scatter Plot")

    # Histogram for the imaginary parts (above the scatter plot)
    ax_hist_y = plt.subplot2grid((4, 9), (1, 7), rowspan=3, colspan=2)
    ax_hist_y.hist(imag_parts, bins=n_bins, orientation='horizontal',
                   align='mid', color='black', alpha=0.3)
    ax_hist_y.set_xlabel("Counts")
    ax_hist_y.yaxis.set_ticklabels([])  # Remove y tick labels
    ax_hist_y.grid(True)

    if title:
        plt.suptitle(title)

    plt.subplots_adjust(hspace=0.5)  # Increase the vertical space between subplots
    plt.show()

    return fig



def plot_multiple_IQ_data(datasets, legend_labels=None, figsize = (140/25.4*2, 140/25.4*2/(1 + np.sqrt(5))), alpha = None, n_bins=250, title=None, dpi=100):
    
    apply_formatting(one_column=True, dpi=dpi)
    # if figsize:
    #     fig = plt.figure()
    # else:
    #     fig = plt.figure()
    fig = plt.figure(figsize=figsize)

    colors = ['blue', 'red', 'green', 'purple', 'orange']  # add more colors if needed
    colors = ['steelblue', 'purple', 'green', 'purple', 'orange']  # add more colors if needed

    ax_scatter = plt.subplot2grid((4, 8), (1, 0), rowspan=3, colspan=3)
    ax_hist_x = plt.subplot2grid((4, 8), (0, 0), colspan=3, rowspan=1)
    ax_hist_y = plt.subplot2grid((4, 8), (1, 3), rowspan=3, colspan=1)

    for i, data in enumerate(datasets):
        color = colors[i % len(colors)]
        real_parts = np.real(data).flatten()
        imag_parts = np.imag(data).flatten()

        # Set the label 
        label = legend_labels[i] if legend_labels else f'Dataset {i+1}'

        # Main scatter plot
        if not alpha:
            alpha = min(1, max(2e4 / len(data), 2e-3))
        ax_scatter.scatter(real_parts, imag_parts, alpha=alpha, marker='.', s=0.1, color=color, rasterized=True)

        # Histogram for the real parts
        ax_hist_x.hist(real_parts, bins=n_bins, align='mid', color=color, alpha=0.3, label=label)

        # Histogram for the imaginary parts
        ax_hist_y.hist(imag_parts, bins=n_bins, orientation='horizontal', align='mid', color=color, alpha=0.3)

    # Remaining plot configurations remain largely unchanged
    ax_scatter.set_xlabel("In-Phase [arb.]")
    ax_scatter.set_ylabel("Quadrature [arb.]")
    ax_scatter.grid(True)
    ax_hist_x.set_ylabel("Counts")
    ax_hist_x.xaxis.set_ticklabels([])  # Remove x tick labels
    ax_hist_x.grid(True)
    ax_hist_x.set_title(title if title else "Scatter Plot")
    # ax_hist_x.legend(loc="upper right", bbox_to_anchor=(1.5,1.2), fontsize='small')
    # ax_hist_x.legend(loc="upper right", bbox_to_anchor=(1.35,1), fontsize='small')
    ax_hist_x.legend(loc="upper right", bbox_to_anchor=(1.4,1), fontsize='small')

    ax_hist_y.set_xlabel("Counts")
    ax_hist_y.yaxis.set_ticklabels([])  # Remove y tick labels
    ax_hist_y.grid(True)
    
    plt.subplots_adjust(hspace=0.5)  # Increase the vertical space between subplots
    return fig






