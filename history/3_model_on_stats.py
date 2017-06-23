#import col_stats_utils
import utils
import pandas as pd
import time
import numpy as np


'''

proba cal based on kde of num and date cols will take 140 ms for each col in each row, which will cost years to process 1m samples.
Hence, it is impossible to use this kde to process all data
However, since dic used by categorical cols is quick. so it could be used along with chunk_tree_votes which only use num and date cols.

'''
print('loading col_stats_cate...')
col_stats_cate = utils.read_variable('model_stats/col_stats_cate.pkl')
column_names = utils.read_variable('outputs/column_names.pkl')
cols_cate = column_names['categorical']

#%%
#col_name = 'L3_S29_F3348'
#k0 = col_stats_num[col_name][0]['kde']
#k1 = col_stats_num[col_name][1]['kde']
#
#value = [[0.054000000000000006],[4324],[4324],[4324],[4324],[4324],[4324]]
#time1 = time.time()
#print(k0.score_samples(value))
#time2 = time.time()
#print(k1.score_samples(value))
#time3 = time.time()
#print(time2-time1,time3-time2,time3-time1)
#
#
#chunk_max_length = 1000
##% get sample rows whose reponse is 1
#chunks_num = pd.read_csv('data/train_numeric.csv',index_col='Id',usecols=['Id',col_name],chunksize=chunk_max_length, low_memory=False,iterator=True)
#
#chunk_nu = 1184
#
#import progressbar
#bar = progressbar.ProgressBar()
#
#full_col = []
#for chunk_id in range(0,chunk_nu,1):
#
#    chunk = chunks_num.get_chunk()
#    value = chunk[col_name]
#    value = np.nan_to_num(value).reshape((-1,1))
#    time1 = time.time()
#    k0.score_samples(value)
#    time2 = time.time()
#    k1.score_samples(value)
#    time3 = time.time()
#    print(time2-time1,time3-time2,time3-time1)


#%%



#chunk_max_length = 1000
##% get sample rows whose reponse is 1
#chunks_num = pd.read_csv('data/train_numeric.csv',index_col='Id',chunksize=chunk_max_length, low_memory=False,iterator=True)
#chunks_date = pd.read_csv('data/train_date.csv',index_col='Id',chunksize=chunk_max_length, low_memory=False,iterator=True)
#chunks_cate = pd.read_csv('data/train_categorical.csv',index_col='Id',chunksize=chunk_max_length, low_memory=False,iterator=True)
#
##%%
#chunk_id = 0
#while True:
#
#    startTime = time.time()
#    
#    c_n = chunks_num.get_chunk()
#    c_d = chunks_date.get_chunk()
#    c_c = chunks_cate.get_chunk()
#    c = pd.concat([c_n,c_d,c_c],axis=1)
#    
#    x = c.drop('Response',1)
#    y = c['Response']
#
#
#    for index, row in x.iterrows():
#        startTime = time.time()
#        probas = col_stats_utils.cal_0_proba_matrix_by_row(row)
#        print ('took', int(time.time() - startTime),'sec');
#        break
#    break

##%%
#import utils
#train_sample_1 = utils.read_variable('outputs/train_sample_1.pkl')
#train_sample_0 = utils.read_variable('outputs/train_sample_0.pkl')
#
#
#
##%%
#
#column_names = utils.read_variable('outputs/column_names.pkl')
#cols_cate_selected = column_names['categorical']
#cols_date_selected = []
#cols_nu_selected = utils.read_variable('model_stats/col_nu_selected_2percent_Nan.pkl')
#
#
#for index, row in train_sample_1.iterrows():
#        startTime = time.time()
#        probas1 = cal_0_proba_matrix_by_row(row,
#                                            cols_cate = cols_cate_selected,
#                                            cols_num=cols_nu_selected,
#                                            cols_date = [])
#        print ('total:', int(time.time() - startTime),'sec');
#        break
#        #utils.save_variable(probas1,'samples/1_'+str(index)+'.pkl')

