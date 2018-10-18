import pandas as pd
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import datetime

from zipModel import gldZIP

from plotZIP import plot_pq

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
mpl.rcParams['figure.dpi'] = 1000
###############################################################################
# PARAMETERS TO SET

# List of nodes used for journal paper.
'''
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
'''

# Define the node we'll be working with.
# node = 'tpm0_R2-12-47-2_tm_168_R2-12-47-2_tn_360'
# node = 'tpm1_R2-12-47-2_tm_22_R2-12-47-2_tn_214'
# node = 'tpm0_R2-12-47-2_tm_7_R2-12-47-2_tn_199'
# P actual max for tn_379: '2016-07-09 15:00:00-0700'
nodes = ['tpm0_R2-12-47-2_tm_187_R2-12-47-2_tn_379',
         'tpm1_R2-12-47-2_tm_136_R2-12-47-2_tn_328']
node_num = [5, 13]

# Pick a day. We'll go with the peak day for this node.
day = '2016-07-09'

# Times to plot.
# 09:00 is still interesting.
times = ['00:00', '17:30']

###############################################################################
# CONSTANTS
# Nominal voltage is always 240.
Vn = 240
# We'll be sweeping over voltage to generate plots.
V = np.arange(0.9*Vn, 1.1*Vn+1)

# Use ZIP parameters that WSU used. The validity is questionable.
COEFF = {'impedance_fraction': 0.3, 'impedance_pf': 0.97,
         'current_fraction': 0.4, 'current_pf': 0.97, 'power_fraction': 0.3,
         'power_pf': 0.97}

###############################################################################
# FUNCTIONS


def plot_sweep(T, P, Q):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(V, P)
    ax1.set_ylabel('Watts')
    ax2.plot(V, Q)
    ax2.set_ylabel('VARs')
    ax2.set_xlabel('Voltage')
    # Get time components

    plt.savefig('voltage_sweep/{}.png'.format(T.time().strftime('%H%M')))
    plt.close()


def plot_sweep_for_time(axP, axQ, T, P_data, Q_data):
    p_line, = axP.plot(V, P_data[0], linestyle='--', color='k')
    p_line_const, = axP.plot(V, P_data[1],
                             linestyle=':', color='r')
    #axP.legend(('Variable ZIP', 'Constant ZIP'))
    axP.set_ylabel('P (W)')

    axQ.plot(V, Q_data[0], linestyle='--', color='k')
    axQ.plot(V, Q_data[1], linestyle=':', color='r')
    axQ.set_ylabel('Q (VAR)')

    axQ.set_xlabel('Voltage (V)', labelpad=1)

    return p_line, p_line_const

def update_animation(i, sweep_data, sweep_data_constant_ZIP, p_line,
                     p_line_const, q_line, q_line_const, ax2):
    p_line.set_ydata(sweep_data.loc[i]['P'])
    p_line_const.set_ydata(sweep_data_constant_ZIP.loc[i]['P'])

    q_line.set_ydata(sweep_data.loc[i]['Q'])
    q_line_const.set_ydata(sweep_data_constant_ZIP.loc[i]['Q'])

    ax2.set_xlabel('Voltage at time {}'.format(i.time().strftime('%H%M')))

    return p_line, q_line, ax2

if __name__ == '__main__':
    ###########################################################################
    # LOAD AND FILTER DATA

    # Read file. It's huge and has more data_ls than we need, but oh well
    cols = {'P_actual': 'float', 'Q_actual': 'float', 'T': 'str',
            'V': 'float', 'base_power': 'float',
            'current_fraction': 'float', 'current_pf': 'float',
            'impedance_fraction': 'float', 'impedance_pf': 'float',
            'power_fraction': 'float', 'power_pf': 'float', 'node': 'str'}

    # pandas is being terrible with use_cols and dtypes. Just it all
    data = pd.read_csv('cluster_SLSQP_new_obj_fn.csv', low_memory=False,
                       na_values='', parse_dates=['T'],
                       infer_datetime_format=True, index_col='T')

    # Localize time.
    data.index = \
        data.index.tz_localize('UTC').tz_convert('America/Los_Angeles')

    # Convert data_ls to numeric. If only the dtype argument worked on read...
    data = data.apply(pd.to_numeric, errors='ignore')

    # Filter data_ls by node(s).
    node_filter = (data['node'] == nodes[0])
    for node in nodes[1:]:
        node_filter = node_filter | (data['node'] == node)

    node_data = data[node_filter]

    ###########################################################################
    # Plot P/Q for tpm0_R2-12-47-2_tm_187_R2-12-47-2_tn_379 - SLSQP
    this_node = 'tpm0_R2-12-47-2_tm_187_R2-12-47-2_tn_379'
    t1 = pd.to_datetime('2016-07-06').tz_localize('America/Los_Angeles')
    t2 = pd.to_datetime('2016-07-13').tz_localize('America/Los_Angeles')

    tn_379 = node_data[
        node_data['node'] == 'tpm0_R2-12-47-2_tm_187_R2-12-47-2_tn_379']
    node_week = tn_379[t1:t2]
    node_week = node_week.apply(pd.to_numeric, errors='ignore')
    fig, ax1, ax2 = plot_pq(df=node_week, T=node_week.index, node=this_node)
    '''
    plt.tight_layout(pad=0.05)
    plt.savefig('tn_379_slsqp.png')
    plt.savefig('tn_379_slsqp.eps', format='eps', dpi=1000)
    '''
    #  P actual max for tn_379: '2016-07-09 15:00:00-0700'
    # nodes = ['tpm0_R2-12-47-2_tm_187_R2-12-47-2_tn_379',

    # Plot Data from Indra
    p_379 = pd.read_csv('indra_data/act_5_node.csv', header=None,
                        names=['P_actual', 'P_estimate'], dtype=np.float64)

    q_379 = pd.read_csv('indra_data/react_5_node.csv', header=None,
                        names=['Q_actual', 'Q_estimate'], dtype=np.float64)

    # Combine.
    indra_data = p_379.join(q_379)

    # Add time information.
    indra_data.index = pd.date_range('2016-01-01 00:00:00',
                                     periods=indra_data.shape[0], freq='15min',
                                     tz='America/Los_Angeles')

    # Filter.
    indra_week = indra_data[t1:t2]

    ax1.plot_date(indra_week.index, indra_week['P_estimate'], marker='*',
                  markerfacecolor='None', color='blue', linestyle='None',
                  markersize=1)
    ax2.plot_date(indra_week.index, indra_week['Q_estimate'], marker='*',
                  color='blue', linestyle='None', markersize=1)
    ax1.legend(('Actual', 'Least Squares Predicted',
                'Deep Learning Predicted'), ncol=3,
               bbox_to_anchor=(0.25, 1.02, 0.5, .102), loc=8, borderaxespad=0.)

    # Plot
    plt.tight_layout(pad=0.05)
    plt.savefig('tn_379_slsqp_dl.png')
    plt.savefig('tn_379_slsqp_dl.eps', format='eps', dpi=1000)
    '''
    plot_pq(df=indra_week, T=indra_week.index, node=this_node)
    plt.tight_layout(pad=0.05)
    plt.savefig('tn_379_deep_learning.png')
    plt.savefig('tn_379_deep_learning.eps', format='eps', dpi=1000)
    '''

    ###########################################################################
    # GENERATE CURVES FOR NODE

    # Grab data_ls for the given day.
    day_data = node_data[day]

    # Grab index for times.
    dts = [day + ' ' + t for t in times]
    tind = pd.to_datetime(dts).tz_localize('America/Los_Angeles')

    # Initialize DataFrame for sweeping voltages
    sweep_index = pd.MultiIndex.from_product([tind, nodes, V],
                                             names=['T', 'node', 'V'])
    sweep_data = pd.DataFrame(0, index=sweep_index, columns=['P', 'Q'])

    sweep_data_constant_ZIP = pd.DataFrame(0, index=sweep_index,
                                           columns=['P', 'Q'])

    # Loop over nodes and times, get voltage sweep.
    for i in tind:
        for node in nodes:
            # Get filter for node.
            nf = day_data['node'] == node

            # Get the row.
            row = day_data.loc[nf].loc[i]

            # Get ZIP coefficients.
            coeff = {'base_power': row['base_power'],
                     'impedance_fraction': row['impedance_fraction'],
                     'impedance_pf': row['impedance_pf'],
                     'current_fraction': row['current_fraction'],
                     'current_pf': row['current_pf'],
                     'power_fraction': row['power_fraction'],
                     'power_pf': row['power_pf']}

            # Compute P and Q
            P, Q = gldZIP(V=V, coeff=coeff, Vn=Vn)

            # Put it in the DataFrame.
            sweep_data.loc[(i, node), 'P'] = P
            sweep_data.loc[(i, node), 'Q'] = Q

            # Now for constant ZIP.
            COEFF['base_power'] = row['base_power']
            P2, Q2 = gldZIP(V=V, coeff=COEFF, Vn=Vn)
            sweep_data_constant_ZIP.loc[(i, node), 'P'] = P2
            sweep_data_constant_ZIP.loc[(i, node), 'Q'] = Q2

    ###########################################################################
    # PLOT AND SAVE

    '''
    # ANIMATION
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    p_line, = ax1.plot(V, sweep_data.loc[day_data.index[0]]['P'], 'o')
    p_line_const, = ax1.plot(
        V, sweep_data_constant_ZIP.loc[day_data.index[0]]['P'], '-')

    ax1.set_ylabel('Real Power (W)')
    ax1.legend(('Variable ZIP', 'Constant ZIP'))

    q_line, = ax2.plot(V, sweep_data.loc[day_data.index[0]]['Q'], 'o')
    q_line_const, = ax2.plot(
        V, sweep_data_constant_ZIP.loc[day_data.index[0]]['Q'], '-')

    ax2.set_ylabel('Reactive Power (VAR)')
    ax2.set_xlabel('Voltage')

    ax1.set_ybound(sweep_data['P'].min(), sweep_data['P'].max())
    ax2.set_ybound(sweep_data['Q'].min(), sweep_data['Q'].max())

    anim = FuncAnimation(fig=fig, func=update_animation,
                         frames=day_data.index[1:],
                         interval=200, fargs=(sweep_data,
                                              sweep_data_constant_ZIP,
                                              p_line, p_line_const,
                                              q_line, q_line_const, ax2))

    anim.save('anim.mp4')

    plt.show()
    '''

    # Plot the day.
    '''
    plot_pq(df=day_data, T=day_data.index, node=node)

    # Plot our polynomial terms
    poly = {'impedance': (day_data['impedance_fraction']
                          * day_data['impedance_pf']),
            'current': (day_data['current_fraction']
                        * day_data['current_pf']),
            'power': day_data['power_fraction'] * day_data['power_pf']}

    poly_terms = pd.DataFrame(data_ls=poly, index=day_data.index)
    poly_terms.plot(y=['impedance', 'current', 'power'], legend=True,
                    subplots=True, sharex=True, title='Polynomial Terms')
    '''

    # SUBPLOT OF DIFFERENT TIME SNAPSHOTS

    # Make the figure double-tall.
    fig = plt.figure(figsize=(mpl.rcParams['figure.figsize'][0],
                              mpl.rcParams['figure.figsize'][1]*2))
    # Define outer grid - 2x2
    outer = gridspec.GridSpec(2, 2, wspace=0.4, hspace=0.55)

    # Loop over the nodes, as we want a column per node in our 2x2.
    for n, node in enumerate(nodes):
        # Loop over time, as we want a time per row in our 2x2
        for i, t in enumerate(tind):
            # Grab P + Q data_ls.
            P = sweep_data.loc[(t, node), 'P']
            P_const = sweep_data_constant_ZIP.loc[(t, node), 'P']
            Q = sweep_data.loc[(t, node), 'Q']
            Q_const = sweep_data_constant_ZIP.loc[(t, node), 'Q']

            # This ensure column per node, row per time.
            grid_ind = n + (i * 2)

            # Define inner 2x1 grid
            inner = \
                gridspec.GridSpecFromSubplotSpec(2, 1,
                                                 subplot_spec=outer[grid_ind],
                                                 wspace=0.1, hspace=0.1)

            # Plot P and Q for this time.
            axP = plt.subplot(inner[0])
            axQ = plt.subplot(inner[1], sharex=axP)

            # Get datetime string
            T = datetime.datetime.strptime(day + ' ' + times[i],
                                           '%Y-%m-%d %H:%M').strftime("%b %d, %H:%M")

            # Plot.
            p_line, p_line_const = plot_sweep_for_time(axP, axQ, T,
                                                       (P, P_const),
                                                       (Q, Q_const))

            # Set title
            axP.set_title('Meter {}, {}'.format(node_num[n], T))

    '''
    # loop over the outer grid.
    for i in range(4):
        # Define an inner 2x1 grid
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[i],
                                                 wspace=0.1, hspace=0.1)

        # Plot P and Q for this time.
        axP = plt.subplot(inner[0])
        axQ = plt.subplot(inner[1], sharex=axP)

        # Get datetime string
        T = day + ' ' + times[i]

        # Plot.
        p_line, p_line_const = plot_sweep_for_time(axP, axQ, T, sweep_data,
                                                  sweep_data_constant_ZIP)

        # Set title
        axP.set_title('{}'.format(T))
    '''

    # Add a legend to the top.
    plt.figlegend((p_line, p_line_const), ('ZIP from Least Squares', 'Constant ZIP'),
                  loc='upper center', ncol=2)

    # Push subplots down
    plt.subplots_adjust(top=0.86, left=0.11, right=0.99)

    # Tighten figure
    outer.tight_layout(fig, rect=(0, 0, 1, 0.93), pad=0.05, h_pad=.2, w_pad=.2)

    # plt.show(block=False)
    fn = 'voltage_sweep_2_by_2'
    plt.savefig(fn + '.png')
    plt.savefig(fn + '.eps', format='eps', dpi=1000)

    # Save the underlying data_ls.
    sweep_data.to_csv('sweep_data_slsqp.csv')
    sweep_data_constant_ZIP.to_csv('sweep_data_const.csv')
    '''
    # Update.
    for i in day_data.index[1:]:
        # grab data_ls
        d = sweep_data.loc[i]

        # Plot and save.
        plot_sweep(T=i, P=d['P'], Q=d['Q'])
    '''

    print('hooray')
