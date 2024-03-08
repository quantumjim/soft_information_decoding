# Maurice Hanisch mhanisc@ethz.ch
# created 24.10.2023

import numpy as np
import matplotlib.pyplot as plt


def plot_IQ_data(data, n_bins=250, twod_nbins=100, figsize=(8,4), title=None, dpi = 100):
    real_parts = np.real(data).flatten()
    imag_parts = np.imag(data).flatten()

    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Main scatter plot
    alpha = min(1, max(2e4 / len(data), 2e-3))
    ax_scatter = plt.subplot2grid((4, 8), (1, 0), rowspan=3, colspan=3)
    ax_scatter.scatter(real_parts, imag_parts,
                        alpha=alpha, marker='.', s=0.1)
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

    ax3d.plot_surface(X, Y, H.T, cmap='plasma')
    ax3d.set_title("3D Histogram Heatmap")

    ax3d.set_xlabel("In-Phase [arb.]", fontsize=8)
    ax3d.set_ylabel("Quadrature [arb.]", fontsize=8)
    ax3d.set_zlabel("Counts", fontsize=8)

    ax3d.tick_params(axis='both', which='major', labelsize=6, pad=-2)
    ax3d.xaxis.offsetText.set_fontsize(6)
    ax3d.yaxis.offsetText.set_fontsize(6)
    ax3d.zaxis.offsetText.set_fontsize(6)

    ax3d.xaxis.labelpad = -7
    ax3d.yaxis.labelpad = -7
    ax3d.zaxis.labelpad = -7

    ax3d.set_box_aspect(None, zoom=1.1)

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()


def plot_multiple_IQ_data(datasets, legend_labels=None, figsize = None, alpha = None, n_bins=250, title=None):
    FIGURE_WIDTH_1COL = 3.404  # For PRX style, change for according to journal
    FIGURE_WIDTH_2COL = 7.057  # For PRX style, change for according to journal
    FIGURE_HEIGHT_1COL_GR = FIGURE_WIDTH_1COL*2/(1 + np.sqrt(5))
    FIGURE_HEIGHT_2COL_GR = FIGURE_WIDTH_2COL*2/(1 + np.sqrt(5))

    font_size = 6 # For PRX style, change for according to journal

    plt.rcParams.update({
        'font.size'           : font_size,  
        'figure.titlesize'    : 'medium',
        'figure.dpi'          : 200,
        'figure.figsize'      : (FIGURE_WIDTH_1COL, FIGURE_HEIGHT_1COL_GR),
        'axes.titlesize'      : 'medium',
        'axes.axisbelow'      : True,
        'xtick.direction'     : 'in',
        'xtick.labelsize'     : 'small',
        'ytick.direction'     : 'in',
        'ytick.labelsize'     : 'small',
        'image.interpolation' : 'none',
        'legend.fontsize'     : font_size,
        'axes.labelsize'      : font_size,
        'axes.titlesize'      : font_size,
        'xtick.labelsize'     : font_size,
        'ytick.labelsize'     : font_size,
    })


    plt.rcParams.update({'font.family':'sans-serif'})
    if figsize:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure(figsize=(8, 4))

    colors = ['red', 'blue', 'green', 'purple', 'orange']  # add more colors if needed

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
        ax_scatter.scatter(real_parts, imag_parts, alpha=alpha, marker='.', s=0.1, color=color)

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
    ax_hist_x.legend(loc="upper right", bbox_to_anchor=(1.5,1.2), fontsize='small')

    ax_hist_y.set_xlabel("Counts")
    ax_hist_y.yaxis.set_ticklabels([])  # Remove y tick labels
    ax_hist_y.grid(True)

    plt.tight_layout()
    plt.show()





