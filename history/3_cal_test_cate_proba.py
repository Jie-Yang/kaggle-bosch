#import col_stats_utils
import utils
import pandas as pd
import time
import numpy as np


'''
=======================================================
'''
print('loading col_stats_cate...')
col_stats_cate = utils.read_variable('model_stats/col_stats_cate.pkl')
column_names = utils.read_variable('outputs/column_names.pkl')
cols_cate = column_names['categorical']
#%% 
'''
calculate probability of response 0 with categorical col
'''
def cal_0_proba_by_cate(col_name,value):
    stat_0 = col_stats_cate[col_name][0]
    stat_1 = col_stats_cate[col_name][1]
    response0_proba = 0
    if value!=value:
        # nan
        response0_proba = stat_0['nan']/(stat_0['nan']+stat_1['nan'])
    elif value in stat_0 and value not in stat_1:
        response0_proba = 1
    elif value not in stat_0 and value in stat_1:
        response0_proba = 0
    elif value in stat_0 and value in stat_1:
        response0_proba = stat_0[value]/(stat_0[value]+stat_1[value])
    elif value not in stat_0 and value not in stat_1:
        response0_proba = 0
    return response0_proba
    
#%%
'''
testing

'''

import os

chunk_nu = 1184    
chunk_max_length = 1000
#% get sample rows whose reponse is 1
chunks_cate = pd.read_csv('data/test_categorical.csv',index_col='Id',chunksize=chunk_max_length, low_memory=False,iterator=True)

chunk_id = 0
for chunk_id in range(0,chunk_nu,1):
    print('processing chunk:',chunk_id)
    chunk = chunks_cate.get_chunk()
    
    file_path = 'model_stats/test_cate_proba/'+str(chunk_id)+'.pkl'
    if os.path.exists(file_path):
        print('already exist.')
    else:
        ids = chunk.index
        result = np.zeros(chunk.shape)
        time1 = time.time()
        
        r = 0
        c = 0
        for col_name in cols_cate:
            col = chunk[col_name]
            time2 = time.time()
            r = 0
            for index, value in col.iteritems():
                proba = cal_0_proba_by_cate(col_name,value)
                result[r,c]=proba
                r += 1
            c += 1
        print('per chunk:',time.time()-time1)
        df = pd.DataFrame(data=result,index=ids.tolist(),columns=cols_cate)
        utils.save_variable(df,file_path)