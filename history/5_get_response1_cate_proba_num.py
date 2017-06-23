import utils
import pandas as pd
import numpy as np
import progressbar

train_row_nu = 1183747
chunk_nu = 1184
max_chunk_size = 1000



'''
proba of categorical cols
'''


print('loading response 1s cate proba and raw num...')

chunks_num = pd.read_csv('data/train_numeric.csv',chunksize=max_chunk_size, low_memory=False,iterator=True)
bar = progressbar.ProgressBar()

# chunk 0
chunk_id = 0
chunk_cate = utils.read_variable('model_stats/train_cate_proba/'+str(chunk_id)+'.pkl')
chunk_num = chunks_num.get_chunk()
response =  chunk_num['Response'].as_matrix()
response_1_index = (response==1)
x_cate = chunk_cate.as_matrix()[response_1_index,:]
x_num = chunk_num.drop(['Response'],axis=1).as_matrix()[response_1_index,:]
x = np.concatenate([x_cate,x_num],axis=1)
x_cate_num_1s = x

# chunk 1...
for chunk_id in bar(range(1,chunk_nu,1)):
    
    chunk_cate = utils.read_variable('model_stats/train_cate_proba/'+str(chunk_id)+'.pkl')
    chunk_num = chunks_num.get_chunk()

    response =  chunk_num['Response'].as_matrix()
    response_1_index = (response==1)
    
    x_cate = chunk_cate.as_matrix()[response_1_index,:]
    x_num = chunk_num.drop(['Response'],axis=1).as_matrix()[response_1_index,:]
    x = np.concatenate([x_cate,x_num],axis=1)
    
    x_cate_num_1s = np.concatenate([x_cate_num_1s, x])

del chunk_id, bar, chunk_num, chunk_cate

print('response 1 samples:',x_cate_num_1s.shape)

utils.save_variable(x_cate_num_1s,'model_stats/x_cate_proba_num_1s.pkl')