import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from os.path import join as osp

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


def read_file(f):
    # Read no pq data_ls.
    df = pd.read_csv(f, low_memory=False, na_values='',
                     parse_dates=['T'], infer_datetime_format=True,
                     index_col='T')
    # Drop na values.
    df = df.dropna()

    # Localize time.
    df.index = \
        df.index.tz_localize('UTC').tz_convert('America/Los_Angeles')

    # Convert data_ls to numeric. If only the dtype argument worked on read...
    df = df.apply(pd.to_numeric, errors='ignore')

    return df

'''
def coefficient_of_determination(actual, predicted):
    y_bar = np.mean(actual)
    ss_tot = np.sum(np.square(actual - y_bar))
    ss_res = np.sum(np.square(actual - predicted))
    return 1 - ss_res / ss_tot
'''


if __name__ == '__main__':
    nodes = ['tpm0_R2-12-47-2_tm_1_R2-12-47-2_tn_193',
             'tpm0_R2-12-47-2_tm_6_R2-12-47-2_tn_198',
             'tpm0_R2-12-47-2_tm_11_R2-12-47-2_tn_203',
             'tpm4_R2-12-47-2_tm_80_R2-12-47-2_tn_272',
             'tpm0_R2-12-47-2_tm_187_R2-12-47-2_tn_379',
             'tpm0_R2-12-47-2_tm_7_R2-12-47-2_tn_199',
             'tpm6_R2-12-47-2_tm_32_R2-12-47-2_tn_224',
             'tpm0_R2-12-47-2_tm_4_R2-12-47-2_tn_196',
             'tpm1_R2-12-47-2_tm_22_R2-12-47-2_tn_214',
             'tpm0_R2-12-47-2_tm_145_R2-12-47-2_tn_337',
             'tpm2_R2-12-47-2_tm_29_R2-12-47-2_tn_221',
             'tpm0_R2-12-47-2_tm_152_R2-12-47-2_tn_344',
             'tpm1_R2-12-47-2_tm_136_R2-12-47-2_tn_328',
             'tpm0_R2-12-47-2_tm_135_R2-12-47-2_tn_327',
             'tpm2_R2-12-47-2_tm_137_R2-12-47-2_tn_329',
             'tpm0_R2-12-47-2_tm_168_R2-12-47-2_tn_360'
             ]
    # We only care about meter 13.
    nodes2 = [nodes[12]]

    # Files.
    files = ['cluster_SLSQP_new_obj_fn.csv',
             'cluster_SLSQP_new_obj_fn_pq_avg.csv']

    # Directories
    dirs = ['scatter', 'scatter_pq_avg']
    # Read data_ls for no pq

    for idx in range(len(files)):
        df = read_file(f=files[idx])

        for node in nodes2:
            node_idx = nodes.index(node)

            # Filter.
            node_df = df[df['node'] == node]
            # Reduce number of data_ls points for plotting.
            num_hr = int(np.floor(node_df.shape[0] / 4))
            rem = node_df.shape[0] % 4
            hourly = np.array([True, False, False, False] * num_hr
                              + [False] * rem)

            p_act = node_df['P_actual'].values
            p_est = node_df['P_estimate'].values

            figsize = (mpl.rcParams['figure.figsize'][0],
                       mpl.rcParams['figure.figsize'][1] * 0.75)
            fig, ax = plt.subplots(1, 1, figsize=figsize)

            # Scatter plot.
            ax.plot(p_act[hourly], p_est[hourly], marker='o',
                    markerfacecolor='None', color='#1f77b4', linestyle='None')

            ax.set_ylabel('Predicted Active Power (W)')
            ax.set_xlabel('Actual Active Power (W)')

            # Fit line
            coeff = np.polyfit(p_act, p_est, deg=1)
            p = np.poly1d(coeff)

            # plot.
            x_lim = np.array(ax.get_xlim())
            y_val = p(x_lim)
            ax.plot(x_lim, y_val, marker='None', color='#ff7f0e',
                    linewidth=0.75)

            # Compute coefficient of determination from line.
            y_cod = p(p_act)
            cod = r2_score(p_est, y_cod)

            # ax.set_title('COD: {:.2f}'.format(cod))
            # print('COD: {:.2f}'.format(cod))

            # Tighten and save.
            plt.tight_layout(pad=0.07, h_pad=0, w_pad=0)
            n = 'meter_{}_scatter'.format(node_idx+1)
            fig.savefig(osp(dirs[idx], n + '.png'))
            fig.savefig(osp(dirs[idx], n + '.eps'), type='eps', dpi=1000)
            plt.close(fig)

