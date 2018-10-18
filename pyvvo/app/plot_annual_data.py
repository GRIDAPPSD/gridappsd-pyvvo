import pandas as pd
import matplotlib.pyplot as plt

# Constants
# List of nodes used for journal paper.
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

# Define the node we'll be working with.
# node = 'tpm0_R2-12-47-2_tm_168_R2-12-47-2_tn_360'
# node = 'tpm1_R2-12-47-2_tm_22_R2-12-47-2_tn_214'
# node = 'tpm0_R2-12-47-2_tm_7_R2-12-47-2_tn_199'
# node = nodes[1]

out_dir = 'zip_new'

if __name__ == '__main__':
    ###########################################################################
    # PRELIMINARIES

    # Interactive mode on.
    # plt.ion()

    # Load data_ls, filter by node.
    # pandas is being terrible with use_cols and dtypes. Just it all
    data = pd.read_csv('cluster_SLSQP_new_obj_fn.csv', low_memory=False,
                       na_values='', parse_dates=['T'],
                       infer_datetime_format=True, index_col='T')

    # Localize time.
    data.index = \
        data.index.tz_localize('UTC').tz_convert('America/Los_Angeles')

    # Convert data_ls to numeric. dtype arg not working on read...
    data = data.apply(pd.to_numeric, errors='ignore')

    for node in nodes:
        # Filter by node.
        node_data = data[data['node'] == node]

        #######################################################################
        # PLOT

        # Plot fractions.
        node_data.plot(y=['impedance_fraction', 'current_fraction',
                          'power_fraction'], legend=True, subplots=True,
                       sharex=True, title='ZIP Fractions')
        plt.savefig(out_dir + '/' + node[-6:] + '_fractions.png')

        # Plot power factors.
        node_data.plot(y=['impedance_pf', 'current_pf', 'power_pf'],
                       legend=True, subplots=True, sharex=True,
                       title='ZIP PFs')

        plt.savefig(out_dir + '/' + node[-6:] + '_pfs.png')

        # Plot polynomial terms.
        poly = {'impedance': (node_data['impedance_fraction']
                              * node_data['impedance_pf']),
                'current': (node_data['current_fraction']
                            * node_data['current_pf']),
                'power': node_data['power_fraction'] * node_data['power_pf']}

        poly_terms = pd.DataFrame(data=poly, index=node_data.index)
        poly_terms.plot(y=['impedance', 'current', 'power'], legend=True,
                        subplots=True, sharex=True, title='Polynomial Terms')

        plt.savefig(out_dir + '/' + node[-6:] + '_polynomial.png')
