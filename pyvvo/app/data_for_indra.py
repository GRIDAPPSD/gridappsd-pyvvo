import pandas as pd
from db import db
import helper

# Node we're looking for.
# node = 'tpm0_R2-12-47-2_tm_187_R2-12-47-2_tn_379'
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

# Table info
table = 'r2_12_47_2_ami_triplex_15_min'
climate_table = 'r2_12_47_2_ami_climate_1_min'

# Setup database.
db_inputs = {'password': '', 'pool_size': 1}
db_obj = db(**db_inputs)

st = helper.tsToDT('2016-01-01 00:00:00', timezone='PST+8PDT')
et = helper.tsToDT('2016-12-31 23:45:00', timezone='PST+8PDT')

for idx, node in enumerate(nodes):
    # Get node data_ls.
    data = db_obj.getTPQVForNode(table=table, node=node, starttime=st,
                                 stoptime=et)

    # Get climate data_ls.
    climate_data = db_obj.getTempAndFlux(table=climate_table)

    # Merge.
    all_data = data.join(climate_data)

    all_data.to_csv('data_for_indra/meter_' + str(idx+1) + '.csv')