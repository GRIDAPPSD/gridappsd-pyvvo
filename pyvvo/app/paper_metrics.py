from collections import OrderedDict

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

from zipModel import gldZIP

# Use ZIP parameters that WSU used to run a static ZIP model for a year.
# The validity is questionable.
COEFF = {'impedance_fraction': 0.3, 'impedance_pf': 0.97,
         'current_fraction': 0.4, 'current_pf': 0.97, 'power_fraction': 0.3,
         'power_pf': 0.97}

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


def mbe(act, pred):
    """Mean bias error"""
    return (1 / act.shape[0]) * (pred - act).sum()


def mae(act, pred):
    """Mean absolute error"""
    return (1 / act.shape[0]) * (pred - act).abs().sum()


def rmqe(act, pred):
    """Root mean quadratic error"""

    return np.power((1 / act.shape[0]) * (pred - act).pow(4).sum(),
                    0.25)


def rmse(act, pred):
    """Root mean square error"""
    return np.sqrt(mean_squared_error(act, pred))


def accuracy(act, pred):
    return 1 - ((1 / act.shape[0])
                * ((pred - act).abs() / act).sum())


def compute_metrics(act, pred):
    m = OrderedDict()
    m['COD'] = np.round(r2_score(act, pred), 2)
    m['MBE'] = int(np.round(mbe(act, pred)))
    m['MAE'] = int(np.round(mae(act, pred)))
    m['RMQE'] = int(np.round(rmqe(act, pred)))
    m['RMSE'] = int(np.round(rmse(act, pred)))
    m['Acc'] = np.round(accuracy(act, pred), 2)

    return m


if __name__ == '__main__':
    # number nodes
    n = len(nodes)
    # Initialize index for new data_ls.
    metric_index = [i+1 for i in range(len(nodes))]

    df_init = {'COD': [0.0]*n, 'MBE': [0]*n, 'MAE': [0]*n,
               'RMQE': [0]*n, 'RMSE': [0]*n, 'Acc': [0.0]*n}
    # Create DataFrames for each method.
    metrics_ls = pd.DataFrame(data=df_init, index=metric_index)
    metrics_dl = pd.DataFrame(data=df_init, index=metric_index)
    # static ZIP
    metrics_static = pd.DataFrame(data=df_init, index=metric_index)
    metrics_static_sn = pd.DataFrame(data=df_init, index=metric_index)

    ###########################################################################
    # LEAST SQUARES
    print('Starting on least squares...')
    # Read least squares data_ls.
    data_ls = pd.read_csv('cluster_SLSQP_new_obj_fn.csv', low_memory=False,
                          na_values='', parse_dates=['T'],
                          infer_datetime_format=True, index_col='T')
    print('Data read.')

    r_before = data_ls.shape[0]
    # Drop na data_ls.
    data_ls = data_ls.dropna()
    r_after = data_ls.shape[0]
    print('{} rows were dropped due to missing data_ls.'.format(r_before-r_after))

    # Localize time.
    data_ls.index = \
        data_ls.index.tz_localize('UTC').tz_convert('America/Los_Angeles')

    # Convert data_ls to numeric. dtype arg not working on read...
    data_ls = data_ls.apply(pd.to_numeric, errors='ignore')

    # Build up DataFrame to hold all predicted and actual data.
    ls_predictions = pd.DataFrame(0.0, columns=['Actual', 'Predicted'],
                                  index=[-1])

    # Loop through nodes and compute metrics.
    for idx, node in enumerate(nodes):
        # Filter by node.
        node_data = data_ls[data_ls['node'] == node]

        # Stack P and Q.
        actual_ls = pd.concat([node_data['P_actual'], node_data['Q_actual']])
        predicted_ls = pd.concat([node_data['P_estimate'],
                                  node_data['Q_estimate']])

        # Place P and Q in the overall dataframe.
        ls_predictions = \
            pd.concat([ls_predictions,
                       pd.DataFrame({'Actual': actual_ls.values,
                                     'Predicted': predicted_ls.values})])

        # Compute metrics
        i = idx+1
        all_metrics = compute_metrics(actual_ls, predicted_ls)
        metrics_ls.loc[i, all_metrics.keys()] = all_metrics.values()

    # Write to file.
    metrics_ls.to_csv('metrics_ls.csv')
    print('Metrics computed and written to file.')

    # Drop the initial row of all zeros in the overall dataframe.
    ls_predictions.drop(-1, inplace=True)

    ###########################################################################
    # STATIC ZIP
    # TODO: this is silly slow. May be worth parallelizing if we have to run
    # several times.
    print('Starting on static ZIP...', end='')
    # Initialize DataFrame
    data_static = pd.DataFrame(0.0, index=data_ls.index,
                               columns=data_ls.columns)
    data_static_sn = pd.DataFrame(0.0, index=data_ls.index,
                                  columns=data_ls.columns)

    five_pct = np.ceil(data_static.shape[0] / 20).astype(int)

    print('Looping and performing ZIP computation for each row...')
    # Loop over the rows from least squares
    i = 0
    for row in data_ls.itertuples():
        ############# Use LS S_n
        # Assign base power.
        COEFF['base_power'] = row.base_power

        # Compute P and Q given V.
        P, Q = gldZIP(V=row.V, coeff=COEFF, Vn=240)

        # Assign data.
        data_static.loc[row.Index, 'P_estimate'] = P
        data_static.loc[row.Index, 'Q_estimate'] = Q

        # Assign actual data.
        data_static.loc[row.Index, 'P_actual'] = row.P_actual
        data_static.loc[row.Index, 'Q_actual'] = row.Q_actual

        # Assign node.
        data_static.loc[row.Index, 'node'] = row.node

        ############# Use "perfect" s_n
        # Assign base power.
        COEFF['base_power'] = np.sqrt(row.P_actual**2 + row.Q_actual**2)

        # Compute P and Q given V.
        P2, Q2 = gldZIP(V=row.V, coeff=COEFF, Vn=240)

        # Assign data.
        data_static_sn.loc[row.Index, 'P_estimate'] = P2
        data_static_sn.loc[row.Index, 'Q_estimate'] = Q2

        # Assign actual data.
        data_static_sn.loc[row.Index, 'P_actual'] = row.P_actual
        data_static_sn.loc[row.Index, 'Q_actual'] = row.Q_actual

        # Assign node.
        data_static_sn.loc[row.Index, 'node'] = row.node

        if i % five_pct == 0:
            print('{}%...'.format(i / five_pct * 5), end='', flush=True)

        i += 1

    print('100%!', end='', flush=True)

    # Write to file.
    data_static.to_csv('static_ZIP_data.csv',
                       columns=['P_actual', 'P_estimate', 'Q_actual',
                                'Q_estimate'], header=True, index=False)
    data_static_sn.to_csv('static_ZIP_data_sn.csv',
                          columns=['P_actual', 'P_estimate', 'Q_actual',
                                   'Q_estimate'], header=True, index=False)
    print('Static data written to file.')

    print('Computing metrics...')
    # Build up DataFrame to hold all predicted and actual data.
    static_predictions = pd.DataFrame(0.0, columns=['Actual', 'Predicted'],
                                      index=[-1])
    static_predictions_sn = pd.DataFrame(0.0, columns=['Actual', 'Predicted'],
                                         index=[-1])

    # Compute metrics.
    # Loop through nodes and compute metrics.
    for idx, node in enumerate(nodes):
        #
        # Filter by node.
        node_data = data_static[data_static['node'] == node]

        # Stack P and Q.
        actual_static = pd.concat([node_data['P_actual'],
                                   node_data['Q_actual']])
        predicted_static = pd.concat([node_data['P_estimate'],
                                      node_data['Q_estimate']])
        # Compute metrics
        i = idx+1
        all_metrics = compute_metrics(actual_static, predicted_static)
        metrics_static.loc[i, all_metrics.keys()] = all_metrics.values()

        # Place P and Q in the overall dataframe.
        static_predictions = \
            pd.concat([static_predictions,
                       pd.DataFrame({'Actual': actual_static.values,
                                     'Predicted': predicted_static.values})])

        ############# Use "perfect" s_n
        # Filter by node.
        node_data2 = data_static_sn[data_static_sn['node'] == node]

        # Stack P and Q.
        actual_static2 = pd.concat([node_data2['P_actual'],
                                    node_data2['Q_actual']])
        predicted_static2 = pd.concat([node_data2['P_estimate'],
                                       node_data2['Q_estimate']])
        # Compute metrics
        all_metrics2 = compute_metrics(actual_static2, predicted_static2)
        metrics_static_sn.loc[i, all_metrics2.keys()] = all_metrics2.values()

        # Place P and Q in the overall dataframe.
        static_predictions_sn = \
            pd.concat([static_predictions_sn,
                       pd.DataFrame({'Actual': actual_static2.values,
                                     'Predicted': predicted_static2.values})])

    # Write to file.
    metrics_static_sn.to_csv('metrics_static_sn.csv')
    metrics_static.to_csv('metrics_static.csv')
    print('Data written to file.')

    ###########################################################################
    # DEEP LEARNING
    print('Starting on deep learning...')
    # Build up DataFrame to hold all predicted and actual data.
    dl_predictions = pd.DataFrame(0.0, columns=['Actual', 'Predicted'],
                                  index=[-1])

    dl_dir = 'indra_data/'
    # Indra's data_ls is in Actual, predicted format.
    for idx in range(len(nodes)):
        # Get file names.
        dl_p_file = dl_dir + 'act_' + str(idx + 1) + '_node.csv'
        dl_q_file = dl_dir + 'react_' + str(idx + 1) + '_node.csv'

        # Read files.
        dl_p = pd.read_csv(dl_p_file, names=['P_actual', 'P_estimate'],
                           dtype=np.float64, header=None)
        dl_q = pd.read_csv(dl_q_file, names=['Q_actual', 'Q_estimate'],
                           dtype=np.float64, header=None)

        # Stack P and Q.
        actual_dl = pd.concat([dl_p['P_actual'], dl_q['Q_actual']])
        predicted_dl = pd.concat([dl_p['P_estimate'], dl_q['Q_estimate']])

        # Place P and Q in the overall dataframe.
        dl_predictions = \
            pd.concat([dl_predictions,
                       pd.DataFrame({'Actual': actual_dl.values,
                                     'Predicted': predicted_dl.values})])

        # Compute metrics
        i = idx+1
        all_metrics = compute_metrics(actual_dl, predicted_dl)
        metrics_dl.loc[i, all_metrics.keys()] = all_metrics.values()

    # Write to file.
    metrics_dl.to_csv('metrics_dl.csv')
    print('Metrics written to file.')

    # Drop the initial row of all zeros in the overall dataframe.
    dl_predictions.drop(-1, inplace=True)

    ###########################################################################
    # Overall metrics.
    all_ls = compute_metrics(ls_predictions['Actual'],
                             ls_predictions['Predicted'])
    all_dl = compute_metrics(dl_predictions['Actual'],
                             dl_predictions['Predicted'])
    all_static = compute_metrics(static_predictions['Actual'],
                                 static_predictions['Predicted'])
    all_static_sn = compute_metrics(static_predictions_sn['Actual'],
                                    static_predictions_sn['Predicted'])

    # Create a DataFrame
    overall_metrics = pd.DataFrame([all_static_sn, all_static, all_ls, all_dl],
                                   index=['static_sn', 'static', 'ls', 'dl'])
    overall_metrics.to_csv('overall_metrics.csv', header=True, index=True)
