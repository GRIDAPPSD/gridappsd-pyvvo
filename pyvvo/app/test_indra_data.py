import pandas as pd
import numpy as np

in_dir = 'data_for_indra'
out_dir = 'indra_data'

# Loop and test.
for i in range(1, 17):
    in_file = in_dir + '/meter_{}.csv'.format(i)
    out_p = out_dir + '/act_{}_node.csv'.format(i)
    out_q = out_dir + '/react_{}_node.csv'.format(i)

    in_df = pd.read_csv(in_file)
    df_p = pd.read_csv(out_p, header=None, names=['P', 'P_est'])
    df_q = pd.read_csv(out_q, header=None, names=['Q', 'Q_est'])

    # Test p.
    p_match = np.allclose(in_df['P'].values, df_p['P'].values, rtol=0,
                          atol=1)
    # q.
    q_match = np.allclose(in_df['Q'].values, df_q['Q'].values, rtol=0,
                          atol=1)

    print('Meter {}, P matches: {}, Q matches: {}'.format(i, p_match, q_match))

