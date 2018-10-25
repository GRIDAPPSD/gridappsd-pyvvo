import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from zipModel import featureScale

# Set IEEE plotting stuff.
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 6
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['figure.figsize'] = (3.5, 1.64)
mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['lines.markeredgewidth'] = 0.5
mpl.rcParams['figure.dpi'] = 1000

mpl.rcParams['hatch.linewidth'] = 0.25

if __name__ == '__main__':

    # Read metric files.
    metrics_dl = pd.read_csv('metrics_dl.csv')
    metrics_ls = pd.read_csv('metrics_ls.csv')

    # Massage into different format. This isn't efficient, it's just a short
    # way to adapt this script to a new method of data_ls.
    mbe = pd.DataFrame({'ls': metrics_ls['MBE'], 'dl': metrics_dl['MBE']})
    mae = pd.DataFrame({'ls': metrics_ls['MAE'], 'dl': metrics_dl['MAE']})
    rmqe = pd.DataFrame({'ls': metrics_ls['RMQE'], 'dl': metrics_dl['RMQE']})
    rmse = pd.DataFrame({'ls': metrics_ls['RMSE'], 'dl': metrics_dl['RMSE']})
    acc = pd.DataFrame({'ls': metrics_ls['Acc'], 'dl': metrics_dl['Acc']})

    # We'll have double-sized figure.
    figsize = (mpl.rcParams['figure.figsize'][0],
               mpl.rcParams['figure.figsize'][1] * 3)

    # Define colors.
    ls_color = '#1f77b4'
    dl_color = '#ff7f0e'

    # Plot.
    fig, (ax_mbe, ax_mae, ax_rmqe, ax_rmse, ax_acc) = \
        plt.subplots(5, 1, figsize=figsize)

    # Initialize index to use.
    index = np.arange(rmse.shape[0])
    x_labels = [str(i + 1) for i in index]
    # Set bar width.
    bar_width = 0.3

    # Y-ticks
    y_ticks = np.arange(0, 1.2, 0.2)
    y_labels = ['{:.1f}'.format(i) for i in y_ticks]

    d = [
        {'ax': ax_mbe, 'title': 'MBE', 'data_ls': mbe, 'y_label': 'MBE'},
        {'ax': ax_mae, 'title': 'MAE', 'data_ls': mae, 'y_label': 'MAE'},
        {'ax': ax_rmqe, 'title': 'RMQE', 'data_ls': rmqe, 'y_label': 'RMQE'},
        {'ax': ax_rmse, 'title': 'RMSE', 'data_ls': rmse, 'y_label': 'RMSE'},
        {'ax': ax_acc, 'title': 'Accuracy', 'data_ls': acc, 'y_label': 'Acc'}
    ]

    # Plot.
    for k in d:
        # Setup axis.
        k['ax'].set_xticks(index + bar_width/2)
        k['ax'].set_xticklabels(x_labels)
        k['ax'].grid(which='both', axis='y')
        k['ax'].set_axisbelow(True)

        if k['y_label'] == 'Acc':
            # Accuracy should have ranges to 1.
            k['ax'].set_yticks(np.arange(0, 1.01, 0.2))
            k['ax'].set_ylim((0, 1))
            k['ax'].set_yticklabels(y_labels)
        elif k['y_label'] == 'RMSE':
            # HARD-CODE for RMSE. Just keep it simple.
            rmse_range = np.arange(0, 1750, 250)
            k['ax'].set_yticks(rmse_range)
            k['ax'].set_ylim((0, 1500))
            k['ax'].set_yticklabels(['{:.0f}'.format(i) for i in rmse_range])
        elif k['y_label'] == 'RMQE':
            # HARD-CODE
            # Max is ~3500.
            rmqe_range = np.arange(0, 3001, 1000)
            k['ax'].set_yticks(rmqe_range)
            k['ax'].set_ylim((0, 3500))
            k['ax'].set_yticklabels(['{:.0f}'.format(i) for i in rmqe_range])
        elif k['y_label'] == 'MAE':
            # HARD-CODE
            # Max is ~600
            mae_range = np.arange(0, 601, 100)
            k['ax'].set_yticks(mae_range)
            k['ax'].set_ylim((0, 600))
            k['ax'].set_yticklabels(['{:.0f}'.format(i) for i in mae_range])
        elif k['y_label'] == 'MBE':
            # HARD-CODE
            # Let's go -100, 50
            mbe_range = np.arange(-100, 51, 25)
            k['ax'].set_yticks(mbe_range)
            k['ax'].set_ylim((-100, 50))
            k['ax'].set_yticklabels(['{:.0f}'.format(i) for i in mbe_range])

        # Plot.
        rects_ls = k['ax'].bar(index, k['data_ls']['ls'], bar_width,
                               color=ls_color, label='KMLS',
                               hatch='---')
        rects_dl = k['ax'].bar(index + bar_width, k['data_ls']['dl'], bar_width,
                               color=dl_color, label='Deep Learning',
                               hatch='xxx')

        k['ax'].set_title(k['title'], loc='left')
        k['ax'].set_ylabel(k['y_label'])
        k['ax'].set_xlabel('Meter Number')

    # Tighten.
    #plt.tight_layout(rect=(0, 0, 1, 0.8), h_pad=.1, w_pad=.2)

    # Add legend to the figure.
    plt.figlegend((rects_ls, rects_dl), ('KMLS', 'Deep Learning'),
                  loc='upper center', ncol=2)

    # Tighten layout. Use rect to bring subplots down to make room for the
    # legend.
    plt.tight_layout(pad=0.05, h_pad=0, w_pad=0, rect=(0, 0, 1, 0.97))

    plt.savefig('metric_bar.png')
    plt.savefig('metric_bar.eps', type='eps', dpi=1000)
