import utils
import pandas as pd
import numpy as np
import progressbar

max_chunk_size = 1000

train_rows_nu = 1183747
chunk_nu = 1184

#%
column_names = utils.read_variable('outputs/column_names.pkl')
cols_date = column_names['date']
#%%
"""
process date data
"""

from sklearn.neighbors.kde import KernelDensity
import os.path

responses = utils.read_variable('model_stats/responses.pkl').astype(int)


while True:
    try:
        for col_name in cols_date:
            print('processing date col:',col_name)
            
            file_path = 'model_stats/date/'+col_name+'.pkl'
            if os.path.exists(file_path):
                print('already exist.')
            else:
                cnts = {}
                cnts[0] = {}
                cnts[0]['nan'] = 0
                cnts[0]['nu'] = []
                cnts[1] = {}
                cnts[1]['nan'] = 0
                cnts[1]['nu'] = []
                
                chunks = pd.read_csv('data/train_date.csv',usecols=[col_name],chunksize=max_chunk_size, low_memory=False,iterator=True)
                
                bar = progressbar.ProgressBar()
                
                for chunk_id in bar(range(0,chunk_nu)):
                    
                    col = chunks.get_chunk()[col_name]
                    ys = responses[chunk_id*max_chunk_size:chunk_id*max_chunk_size+col.shape[0]]
                    for i in range(0,col.shape[0],1):
                        value = col.iloc[i]
                        y = ys[i]
                        if value!=value:
                            cnts[y]['nan'] += 1
                        else:
                            cnts[y]['nu'].append(value)
            
                cnts[0]['nu'] = np.asarray(cnts[0]['nu']).reshape(-1, 1)
                cnts[1]['nu'] = np.asarray(cnts[1]['nu']).reshape(-1, 1)
                print('cal kde for 0...')
                if cnts[0]['nu'].size>0:
                    cnts[0]['kde'] = KernelDensity(kernel='gaussian').fit(cnts[0]['nu'])
                print('cal kde for 1...')
                if cnts[1]['nu'].size>0:
                    cnts[1]['kde'] = KernelDensity(kernel='gaussian').fit(cnts[1]['nu'])
                utils.save_variable(cnts,file_path)
        break
    except ValueError:
            print('get ValueError. Restart again.')

#%%

