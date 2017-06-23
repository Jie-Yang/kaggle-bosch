import utils
import pandas as pd
import numpy as np
import progressbar
from collections import defaultdict
max_chunk_size = 1000

train_rows_nu = 1183747
chunk_nu = 10#1184

#%
column_names = utils.read_variable('outputs/column_names.pkl')
cols_numeric = column_names['numeric']

#%%
file_name = 'data/train_numeric.csv'
cols = cols_numeric
nan_counts = np.zeros(cols.size)
col_nan_counts = defaultdict(int)

col_index = 0
bar = progressbar.ProgressBar()
for col_name in bar(cols_numeric):
    #print('processing (',file_name,') col:',col_name)
    
    chunks = pd.read_csv(file_name,usecols=[col_name],chunksize=max_chunk_size, low_memory=False,iterator=True)
    
    nan_count = 0
    for chunk_id in range(0,chunk_nu):
        chunk = chunks.get_chunk()
        nan_count += np.sum(chunk.isnull())
    nan_counts[col_index] = nan_count
    col_nan_counts[col_name] = nan_count
 
    #print('\nnan values:',nan_counts[col_index]/train_rows_nu)
    col_index += 1

del file_name, cols, col_index, chunks, bar, nan_count, chunk_id

#%%

a = nan_counts / (max_chunk_size*chunk_nu)

#%%

nan_threshold_percent = 0.2

threshold = nan_threshold_percent * max_chunk_size * chunk_nu
col_selected = []
bar = progressbar.ProgressBar()
for key, value in bar(col_nan_counts.items()):
    if int(value) < threshold:
        col_selected.append(key)

print(len(col_selected),'cols selected.')
file_path = 'model_stats/col_nu_selected_2percent_Nan.pkl'
utils.save_variable(col_selected, file_path)
print('result is saved to:',file_path)