import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

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
    #mpl.rcParams['lines.markeredgewidth'] = 0.5
    mpl.rcParams['figure.dpi'] = 1000

    # Files.
    d = 'dave_data/'
    f1 = d + 'Meter1-2016-06-01-hr18.csv'
    f2 = d + 'Meter1-2016-06-01-hr18-C0.csv'
    f3 = d + 'Meter1-2016-06-01-hr18-C1.csv'
    f4 = d + 'Meter1-2016-06-01-hr18-C2.csv'

    # Read files.
    all_data = pd.read_csv(f1)
    all_cluster = pd.read_csv(f2)
    cluster_1 = pd.read_csv(f3)
    cluster_2 = pd.read_csv(f4)

    # Sort our cluster data_ls by V, so we don't get line 'backtracking.
    all_cluster.sort_values(by='V', axis=0, inplace=True)
    cluster_1.sort_values(by='V', axis=0, inplace=True)
    cluster_2.sort_values(by='V', axis=0, inplace=True)

    # NOTE: Original plot had both P and Q, but was shortened to just P to
    # reduce the paper length. Uncomment lines below to go back to the
    # double plot.

    # Initialize figure.
    # figsize = (mpl.rcParams['figure.figsize'][0],
    #            mpl.rcParams['figure.figsize'][1]*1.5)
    figsize = (mpl.rcParams['figure.figsize'][0],
               mpl.rcParams['figure.figsize'][1]*0.75)
    # fig, (ax_p, ax_q) = plt.subplots(2, 1, sharex=True, figsize=figsize)
    fig, ax_p = plt.subplots(1, 1, figsize=figsize)

    # Plot actual data_ls on both axes.
    ax_p.plot(all_data['V'], all_data[' P'], color='black', marker='o',
              linestyle='None', markerfacecolor='None', markersize=2,
              markeredgewidth=0.75)
    # ax_q.plot(all_data['V'], all_data[' Q'], color='black', marker='o',
    #           linestyle='None', markerfacecolor='None', markersize=2,
    #           markeredgewidth=0.75)

    # Plot fit of all data_ls in blue.
    ax_p.plot(all_cluster['V'], all_cluster[' P'], color='#1f77b4',
              marker='None', linestyle='-', linewidth=0.75)
    # ax_q.plot(all_cluster['V'], all_cluster[' Q'], color='#1f77b4',
    #           marker='None', linestyle='-', linewidth=0.75)

    # Plot clusters in orange.
    ax_p.plot(cluster_1['V'], cluster_1[' P'], color='#ff7f0e',
              marker='None', linestyle='--', linewidth=0.75)
    # ax_q.plot(cluster_1['V'], cluster_1[' Q'], color='#ff7f0e',
    #           marker='None', linestyle='--', linewidth=0.75)

    ax_p.plot(cluster_2['V'], cluster_2[' P'], color='#ff7f0e',
              marker='None', linestyle='--', linewidth=0.75)
    # ax_q.plot(cluster_2['V'], cluster_2[' Q'], color='#ff7f0e',
    #           marker='None', linestyle='--', linewidth=0.75)

    # Titles and labels
    # ax_p.set_title('Active Power vs. Voltage')
    # ax_p.set_title('Active Power vs. Voltage', pad=12)
    # ax_q.set_title('Reactive Power vs. Voltage')

    # ax_q.set_xlabel('Voltage (V)')
    ax_p.set_xlabel('Voltage (V)')
    ax_p.set_ylabel('P (W)')
    # ax_q.set_ylabel('Q (var)')

    # plt.figlegend(['Actual Data', 'ZIP Fit All Data', 'ZIP Fits Clustered'],
    #               loc='upper center', ncol=3)
    leg = ax_p.legend(['Actual Data', 'ZIP Fit All Data', 'ZIP Fits Clustered'],
                      loc='lower left', ncol=3,
                      bbox_to_anchor=(0.1, 0.97, 1, 0.2), markerscale=1.5,
                      borderpad=0.05, handletextpad=0.05)
    frame = leg.get_frame()
    frame.set_linewidth(0.5)
    # plt.tight_layout(pad=0.05, h_pad=0.4, w_pad=0, rect=(0, 0, 1, 0.90))
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    # Shorten xticks on P axis.
    ax_p.tick_params(axis='x', length=2)
    plt.savefig('cluster.png')
    plt.savefig('cluster.eps', type='eps', dpi=1000)