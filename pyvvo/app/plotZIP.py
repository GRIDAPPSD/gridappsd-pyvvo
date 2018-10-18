'''
Created on Jul 6, 2018

@author: thay838
'''
import numpy as np 
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import to_rgb

from zipModel import computeRMSD
from helper import tsToDT

# Load our three files.
FILE = 'cluster_SLSQP_new_obj_fn.csv'

def plot_pq(df, T, node, xlabel_size=8, ylabel_size=8,
            xlabel='Time (First Two Weeks of June)'):
    """Plot actual and expected P and Q on two subplots."""
    P_actual = df['P_actual']
    P_estimate = df['P_estimate']
    Q_actual = df['Q_actual']
    Q_estimate = df['Q_estimate']

    # Define date formatter.
    fmt = mdates.DateFormatter('%m-%d')

    # Plot actual vs. real P and Q
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')

    # Plotting parameters:
    # 'color': to_rgb('0.15')
    # 'color': '#2c7fb8',
    # 'color': '#7fcdbb',
    actual_params = {'marker': 'o', 'markerfacecolor': 'None',
                     'linestyle': 'None',
                     'label': 'Actual', 'zorder': 1, 'markersize': 2.5,
                     'markeredgewidth': 0.25}
    predicted_params = {'marker': 'x',
                        'linestyle': 'None', 'label': 'Predicted',
                        'zorder': 10, 'markersize': 1.5,
                        'markeredgewidth': 0.25}

    # plt.plot_date(T, P_actual, 'o', T, P_estimate, '-')
    ax1.plot_date(T, P_actual, **actual_params)
    ax1.plot_date(T, P_estimate, **predicted_params)
    ax1.set_ylabel('P (W)', fontsize=ylabel_size)
    # ax1.set_title('Real Power, P (W)')
    '''
    ax1.legend(('Actual', 'Predicted'), ncol=2,
               bbox_to_anchor=(0.25, 1.02, 0.5, .102), loc=8, borderaxespad=0.)
    '''

    ax1.xaxis.set_major_formatter(fmt)
    ax1.set_title('Active Power, Actual and Predicted', y=0.90)
    # Shorten xticks.
    ax1.tick_params(axis='x', length=2)
    # plt.subplot(2, 1, 2)
    ax2.plot_date(T, Q_actual, **actual_params)
    ax2.plot_date(T, Q_estimate, **predicted_params)
    # ax2.legend(('Actual', 'Estimate'))
    ax2.set_ylabel('Q (var)', fontsize=ylabel_size)
    ax2.set_xlabel(xlabel, fontsize=xlabel_size, y=2)
    # ax2.set_title('Reactive Power, Q (VA)')
    ax2.xaxis.set_major_formatter(fmt)
    ax2.set_title('Reactive Power, Actual and Predicted', y=0.90)
    ax2.tick_params(axis='x', length=2)
    return fig, ax1, ax2


def read_indra_data(meter_num):
    # Read Indra's data_ls.
    p = pd.read_csv('indra_data/act_{}_node.csv'.format(meter_num),
                    header=None, names=['P_actual', 'P_estimate'])
    q = pd.read_csv('indra_data/react_{}_node.csv'.format(meter_num),
                    header=None, names=['Q_actual', 'Q_estimate'])
    pq = p.join(q)
    # Create time index.
    pq.index = pd.date_range('2016-01-01 00:00:00', freq='15min',
                             periods=pq.shape[0], tz='America/Los_Angeles')
    return pq

def plot_meter_1(data_ls, meter_name):
    """Hard-coded helper for plotting node 1 for the journal paper."""
    # Get meter 1 data.
    meter_1_slsqp = data_ls[data_ls['node'] == meter_name]

    # Just look at June.
    t1 = pd.to_datetime('2016-06-01').tz_localize('America/Los_Angeles')
    t2 = pd.to_datetime('2016-06-15').tz_localize('America/Los_Angeles')

    meter_1_slsqp_filtered = meter_1_slsqp[t1:t2]

    # Plot Dave's file.
    fig_slsqp, ax_p_slsqp, ax_q_slsqp = \
        plot_pq(df=meter_1_slsqp_filtered, T=meter_1_slsqp_filtered.index,
                node='meter 1', xlabel='Time (First Two Weeks of June)')
    leg = ax_p_slsqp.legend(('Actual', 'Predicted'), ncol=2, loc='upper left',
                            bbox_to_anchor=(0, 1.05), markerscale=1.5,
                            borderpad=0.05,
                            handletextpad=0.05)
    frame = leg.get_frame()
    frame.set_linewidth(0.5)

    # Read machine learning data.
    meter_1_dl = read_indra_data(1)

    # Plot.
    meter_1_dl_filtered = meter_1_dl[t1:t2]
    fig_dl, ax_p_dl, ax_q_dl = \
        plot_pq(df=meter_1_dl_filtered, T=meter_1_dl_filtered.index,
                node='meter 1', xlabel='Time (First Two Weeks of June)')
    leg = ax_p_dl.legend(('Actual', 'Predicted'), ncol=2, loc='upper left',
                         bbox_to_anchor=(0, 1.05), markerscale=1.5,
                         borderpad=0.05, handletextpad=0.05)
    frame = leg.get_frame()
    frame.set_linewidth(0.5)

    # Ensure y-axis limits match.
    axes_match((ax_p_slsqp, ax_p_dl))
    axes_match((ax_q_slsqp, ax_q_dl))

    # Tighten and save.
    plt.figure(fig_slsqp.number)
    fig_slsqp.set_tight_layout({'pad': 0.05, 'h_pad': 0.1})
    plt.tight_layout()
    plt.savefig('meter_1_slsqp.png')
    plt.savefig('meter_1_slsqp.eps', format='eps', dpi=1000)

    plt.figure(fig_dl.number)
    fig_dl.set_tight_layout({'pad': 0.05, 'h_pad': 0.1})
    plt.tight_layout()
    plt.savefig('meter_1_dl.png')
    plt.savefig('meter_1_dl.eps', format='eps', dpi=1000)


def plot_meter_9(data_ls, meter_name):
    """Hard-coded helper for plotting meter 9 data."""
    # Get meter 9 data.
    meter_9_slsqp = data_ls[data_ls['node'] == meter_name]

    # Plot mid-Jan to mid-Feb
    t1 = pd.to_datetime('2016-01-15').tz_localize('America/Los_Angeles')
    t2 = pd.to_datetime('2016-01-29').tz_localize('America/Los_Angeles')

    meter_9_slsqp_filtered = meter_9_slsqp[t1:t2]
    fig_slsqp, ax_p_slsqp, ax_q_slsqp = plot_pq(df=meter_9_slsqp_filtered,
                                                T=meter_9_slsqp_filtered.index,
                                                node=nodes[8][-6:],
                                                xlabel='Time (Mid January)')
    ax_q_slsqp.tick_params(axis='x', which='major', pad=1)
    leg = ax_p_slsqp.legend(ncol=2, loc='right', bbox_to_anchor=(0.45, 0.85),
                            markerscale=1.5, borderpad=0.05,
                            handletextpad=0.05)
    frame = leg.get_frame()
    frame.set_linewidth(0.5)

    # Repeat for deep learning data_ls.
    meter_9_dl = read_indra_data(9)
    meter_9_dl_filtered = meter_9_dl[t1:t2]
    fig_dl, ax_p_dl, ax_q_dl = plot_pq(df=meter_9_dl_filtered,
                                       T=meter_9_dl_filtered.index,
                                       node=nodes[8][-6:],
                                       xlabel='Time (Mid January)')
    leg = ax_p_dl.legend(ncol=2, loc='right', bbox_to_anchor=(0.45, 0.85),
                         markerscale=1.5, borderpad=0.05, handletextpad=0.05)
    frame = leg.get_frame()
    frame.set_linewidth(0.5)

    # Ensure y-axis limits match.
    axes_match((ax_p_slsqp, ax_p_dl))
    axes_match((ax_q_slsqp, ax_q_dl))

    # Save
    plt.figure(fig_slsqp.number)
    fig_slsqp.set_tight_layout({'pad': 0.05, 'h_pad': 0.1})
    plt.tight_layout()
    plt.savefig('meter_9_slsqp.png')
    plt.savefig('meter_9_slsqp.eps', format='eps', dpi=1000)

    plt.figure(fig_dl.number)
    fig_dl.set_tight_layout({'pad': 0.05, 'h_pad': 0.1})
    plt.tight_layout()
    plt.savefig('meter_9_dl.png')
    plt.savefig('meter_9_dl.eps', format='eps', dpi=1000)


def axes_match(ax_array):
    """Helper to ensure y-axis limits match.

    Input should be 2x1 arrays of axes.
    """
    # Lower limit
    ax_min = min(ax_array[0].get_ylim()[0], ax_array[1].get_ylim()[0])
    # Upper limit
    ax_max = max(ax_array[0].get_ylim()[1], ax_array[1].get_ylim()[1])

    lim = (ax_min, ax_max)
    # Set.
    ax_array[0].set_ylim(lim)
    ax_array[1].set_ylim(lim)

if __name__ == '__main__':
    # Define styles.
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
    mpl.rcParams['figure.dpi'] = 300

    # Use grayscale.
    # plt.style.use('grayscale')
    # plt.gray()

    # Define list of nodes
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

    # Read the file.
    data_ls = pd.read_csv(FILE, low_memory=False, na_values='', parse_dates=['T'],
                          infer_datetime_format=True, index_col='T')

    # Drop na. There's maybe 100 rows of failures.
    data_ls.dropna(inplace=True)

    # Localize time.
    data_ls.index = \
        data_ls.index.tz_localize('UTC').tz_convert('America/Los_Angeles')

    # Convert data_ls to numeric. If only the dtype argument worked on read...
    data_ls = data_ls.apply(pd.to_numeric, errors='ignore')

    # Plot for meter 1.
    plot_meter_1(data_ls, nodes[1-1])

    # Plot for meter 9.
    plot_meter_9(data_ls, nodes[9-1])


    '''
    # Loop over nodes.
    for idx, node in enumerate(nodes):
        node_num = idx+1
        node_data = data_ls[data_ls['node'] == node]
        fig, ax1, ax2 = plot_pq(node_data, node_data.index, node,
                                xlabel='Time (Year)')
        plt.show()
        pass
    '''