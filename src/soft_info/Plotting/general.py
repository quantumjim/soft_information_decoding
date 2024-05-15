import matplotlib.pyplot as plt
import numpy as np

def apply_formatting(dpi=1000, one_column=True, font_size=10, third_height=False):
    FIGURE_WIDTH_1COL = 3.404  # For PRX style, change for according to journal
    FIGURE_WIDTH_2COL = 7.057  # For PRX style, change for according to journal

    FIGURE_WIDTH_1COL = 140/25.4 # For Thesis style
    FIGURE_WIDTH_2COL = 140/25.4 / 2 # For Thesis style
    FIGURE_HEIGHT_1COL_GR = FIGURE_WIDTH_1COL*2/(1 + np.sqrt(5)) # Golden ratio
    FIGURE_HEIGHT_2COL_GR = FIGURE_WIDTH_2COL*2/(1 + np.sqrt(5)) # Golden ratio
    FIGURE_HEIGHT_2COL_GR = FIGURE_WIDTH_2COL*2/(1 + np.sqrt(5)) * 1.5 # Golden ratio
    FIGURE_HEIGHT_2COL_GR = FIGURE_WIDTH_2COL * 1.25

    if third_height == True:
        FIGURE_HEIGHT_1COL_GR = FIGURE_HEIGHT_1COL_GR * 1.5
        FIGURE_HEIGHT_2COL_GR = FIGURE_HEIGHT_2COL_GR *1.5

    # font_size = font_size if one_column else font_size//1.5
    font_size = font_size if one_column else font_size * 1.2
    legend_font_size = font_size//1.5 if one_column else font_size//1.8

    figsize = (FIGURE_WIDTH_1COL, FIGURE_HEIGHT_1COL_GR) if one_column else (FIGURE_WIDTH_2COL, FIGURE_HEIGHT_2COL_GR)

    plt.rcParams.update({
        'text.usetex'         : True,
        'text.latex.preamble' : r'\usepackage{mathpazo}',
        'font.family'         : 'serif',
        # 'font.serif'          : ['IBM Plex Sans'],
        'font.size'           : font_size,  
        'figure.titlesize'    : 'medium',
        'figure.dpi'          : dpi,
        'figure.figsize'      : figsize,
        'axes.titlesize'      : 'medium',
        'axes.axisbelow'      : True,
        'xtick.direction'     : 'in',
        'xtick.labelsize'     : 'small',
        'ytick.direction'     : 'in',
        'ytick.labelsize'     : 'small',
        'image.interpolation' : 'none',
        # 'legend.fontsize'     : legend_font_size,
        'legend.fontsize'     : font_size,
        'axes.labelsize'      : font_size,
        'axes.titlesize'      : font_size,
        'xtick.labelsize'     : font_size,
        'ytick.labelsize'     : font_size,
    })


def save_plot(plot, filename, format='pdf'):
    '''Filename ex: '4_RepetitionCodes/p_L.pdf'''

    file_name = '/Users/mha/My_Drive/Desktop/Studium/Physik/MSc/Semester_3/IBM/IBM_GIT/MT-Soft-Information-QEC/Thesis/img/'
    file_name += filename

    plot.savefig(file_name, format=format, bbox_inches='tight')
