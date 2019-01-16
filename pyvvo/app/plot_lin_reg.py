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

def get_p_for_node(df, node):
    # Filter.
    node_df = df[df['node'] == node]
    # Reduce number of data_ls points for plotting.
    num_hr = int(np.floor(node_df.shape[0] / 4))
    rem = node_df.shape[0] % 4
    hourly = np.array([True, False, False, False] * num_hr
                      + [False] * rem)

    p_act = node_df['P_actual'].values
    p_est = node_df['P_estimate'].values

    return p_act, p_est, hourly


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
    node = nodes[12]
    node_idx = nodes.index(node)
    nodes2 = [nodes[12]]

    # Files.
    files = ['cluster_SLSQP_new_obj_fn.csv',
             'cluster_SLSQP_new_obj_fn_pq_avg.csv']

    # Directories
    dirs = ['scatter', 'scatter_pq_avg']

    # Read data_ls for no pq
    df1 = read_file(f=files[0])
    df2 = read_file(f=files[1])

    # Get real power data
    p_act1, p_est1, hourly1 = get_p_for_node(df1, node=node)
    p_act2, p_est2, hourly2 = get_p_for_node(df2, node=node)

    # Define figure size, get axes.
    figsize = (mpl.rcParams['figure.figsize'][0],
               mpl.rcParams['figure.figsize'][1])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Scatter plot.
    ax1.plot(p_act1[hourly1], p_est1[hourly1], marker='o',
             markerfacecolor='None', color='#1f77b4', linestyle='None')
    ax2.plot(p_act2[hourly2], p_est2[hourly2], marker='o',
             markerfacecolor='None', color='#1f77b4', linestyle='None')

    # Labels.
    ax1.set_ylabel('Predicted P (W)')
    ax2.set_ylabel('Predicted P (W)')
    ax2.set_xlabel('Actual P (W)')

    # Shorten ticks for axis 1.
    ax1.tick_params(axis='x', length=2)

    # Titles.
    ax1.set_title('(a)', pad=2.2)
    ax2.set_title('(b)', pad=2.2)

    # Adjust y limits to match for both.
    # HARD-CODING:
    lb = 0
    ub = 6400
    ax1.set_yticks(np.arange(lb, ub, 2000))
    ax1.set_ylim((lb, ub))
    ax2.set_yticks(np.arange(lb, ub, 2000))
    ax2.set_ylim((lb, ub))
    # ax1.set_yticklabels(y_labels)

    # Fit lines.
    coeff1 = np.polyfit(p_act1, p_est1, deg=1)
    p1 = np.poly1d(coeff1)
    coeff2 = np.polyfit(p_act2, p_est2, deg=1)
    p2 = np.poly1d(coeff2)

    # Plot fit lines.
    x_lim1 = np.array(ax1.get_xlim())
    y_val1 = p1(x_lim1)
    ax1.plot(x_lim1, y_val1, marker='None', color='#ff7f0e',
             linewidth=0.75)

    x_lim2 = np.array(ax2.get_xlim())
    y_val2 = p2(x_lim2)
    ax2.plot(x_lim2, y_val2, marker='None', color='#ff7f0e',
             linewidth=0.75)

    # Compute coefficient of determination from line.
    y_cod1 = p1(p_act1)
    cod1 = r2_score(p_est1, y_cod1)
    y_cod2 = p2(p_act2)
    cod2 = r2_score(p_est2, y_cod2)

    # ax.set_title('COD: {:.2f}'.format(cod))
    print('COD1: {:.2f}'.format(cod1))
    print('COD2: {:.2f}'.format(cod2))

    # Tighten and save.
    plt.tight_layout(pad=0.05, h_pad=0.1, w_pad=0, rect=(0, 0, 1, 1))
    n = 'meter_{}_scatter'.format(node_idx+1)
    fig.savefig(n + '.png')
    fig.savefig(n + '.eps', type='eps', dpi=1000)
    plt.close(fig)

