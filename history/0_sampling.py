# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time

#data/train_categorical.csv 2.49 GB as CSV file
#data/train_numeric.csv'    ##### 1.99 GB as CSV file
#data/train_date.csv        ##### 2.69 GB as CSV file

##%%##########################
## get sample data
##############################
#skiprows = 0
#nrows = 1000
#train_categorical_sample = pd.read_csv('data/train_categorical.csv',index_col='Id',skiprows=skiprows,nrows=nrows, low_memory=False)
#train_numeric_sample = pd.read_csv('data/train_numeric.csv',index_col='Id',skiprows=skiprows,nrows=nrows, low_memory=False)
#train_date_sample = pd.read_csv('data/train_date.csv',index_col='Id',skiprows=skiprows,nrows=nrows, low_memory=False)
#
#train_sample = pd.concat([train_categorical_sample,train_numeric_sample,train_date_sample],axis=1)
##%%
#groups = train_sample.groupby('Response')
#train_sample_g0 = groups.get_group(0)
#train_sample_g1 = groups.get_group(1)
#
#print('y=0:',train_sample_g0.shape[0],',y=1:',train_sample_g1.shape[0])
#del groups

#%%#################################
# count how many 0 and 1 in response
####################################
startTime = time.time()
chunks = pd.read_csv('data/train_numeric.csv',chunksize=1000, low_memory=False)
sum=0;
sum_1=0;

for chunk in chunks:
    sum = sum + chunk.shape[0]
    one_sum = chunk['Response'].sum()
    print(sum,'-->',one_sum,'/',chunk.shape[0])
    sum_1 = sum_1+one_sum

print('train_numeric',sum)
print('response 1:',sum_1,sum_1/sum)
print('response 0:',sum-sum_1,1-sum_1/sum)
print ('Training took', int(time.time() - startTime),'sec');

############################
######## Response 0: 0.58%
######## Response 1: 99.42%
############################
#%%#################################
# get row IDs of 1
####################################
startTime = time.time()
chunks = pd.read_csv('data/train_numeric.csv',index_col='Id',chunksize=1000, low_memory=False)
sample_1_nu = 1000
response_1_ids = []
for chunk in chunks:
    rows = chunk.loc[chunk['Response'] ==1]
    response_1_ids.extend(rows.index.get_values())
    print('find response 1: ',rows.shape[0],',total:',len(response_1_ids),',target:',sample_1_nu)
    if len(response_1_ids)>= sample_1_nu:
        break

response_1_ids = response_1_ids[0:sample_1_nu]
print('response 1 founds:',len(response_1_ids),',target:',sample_1_nu)
print ('Training took', int(time.time() - startTime),'sec');

#%%
# get sample rows whose reponse is 0
train_numeric_sample_0 = pd.read_csv('data/train_numeric.csv',index_col='Id',skiprows=response_1_ids,nrows=sample_1_nu, low_memory=False)
train_categorical_sample_0 = pd.read_csv('data/train_categorical.csv',index_col='Id',skiprows=response_1_ids,nrows=sample_1_nu, low_memory=False)
train_date_sample_0 = pd.read_csv('data/train_date.csv',index_col='Id',skiprows=response_1_ids,nrows=sample_1_nu, low_memory=False)
train_sample_0 = pd.concat([train_categorical_sample_0,train_numeric_sample_0,train_date_sample_0],axis=1)

#% get sample rows whose reponse is 1
chunks = pd.read_csv('data/train_numeric.csv',index_col='Id',chunksize=1000, low_memory=False)
rows_total = pd.DataFrame()
for chunk in chunks:
    rows = chunk[chunk.index.isin(response_1_ids)]
    rows_total = rows_total.append(rows)
    print('find response 1:',rows.shape[0],',total:',rows_total.shape[0],',target:',sample_1_nu)
    if rows_total.shape[0]== sample_1_nu:
        break
train_sample_1_numeric = rows_total

chunks = pd.read_csv('data/train_date.csv',index_col='Id',chunksize=1000, low_memory=False)
rows_total = pd.DataFrame()
for chunk in chunks:
    rows = chunk[chunk.index.isin(response_1_ids)]
    rows_total = rows_total.append(rows)
    print('find response 1:',rows.shape[0],',total:',rows_total.shape[0],',target:',sample_1_nu)
    if rows_total.shape[0]== sample_1_nu:
        break
train_sample_1_date = rows_total

chunks = pd.read_csv('data/train_categorical.csv',index_col='Id',chunksize=1000, low_memory=False)
rows_total = pd.DataFrame()
for chunk in chunks:
    rows = chunk[chunk.index.isin(response_1_ids)]
    rows_total = rows_total.append(rows)
    print('find response 1:',rows.shape[0],',total:',rows_total.shape[0],',target:',sample_1_nu)
    if rows_total.shape[0]== sample_1_nu:
        break
train_sample_1_categorical = rows_total

train_sample_1 = pd.concat([train_sample_1_numeric,train_sample_1_date,train_sample_1_categorical],axis=1)

#%%
import utils
utils.save_variable(train_sample_1,'outputs/train_sample_1.pkl')
utils.save_variable(train_sample_0,'outputs/train_sample_0.pkl')


#%% get headers
column_names = {}
chunks = pd.read_csv('data/train_numeric.csv',index_col='Id',chunksize=1, low_memory=False,iterator=True)
column_names['numeric'] = chunks.get_chunk(0).columns.drop('Response')
column_names['response'] = 'Response'
chunks = pd.read_csv('data/train_date.csv',index_col='Id',chunksize=1, low_memory=False,iterator=True)
column_names['date'] = chunks.get_chunk(0).columns
chunks = pd.read_csv('data/train_categorical.csv',index_col='Id',chunksize=1, low_memory=False,iterator=True)
column_names['categorical'] = chunks.get_chunk(0).columns
import utils
utils.save_variable(column_names,'outputs/column_names.pkl')
del chunks

#%%

print(column_names['date'].shape)


#%%#################################
# save all responses to a new file
####################################
import progressbar

chunks = pd.read_csv('data/train_numeric.csv',usecols=['Id','Response'],chunksize=1000, low_memory=False)
responses = np.zeros((1183747, 2),dtype=np.int)
bar = progressbar.ProgressBar()
print('loading cate proba...')
chunk_nu = 1184
max_chunk_size = 1000
for chunk_id in bar(range(0,chunk_nu,1)):
    chunk = chunks.get_chunk()
    responses[chunk_id*max_chunk_size:chunk_id*max_chunk_size+chunk.shape[0],0] = chunk['Id']
    responses[chunk_id*max_chunk_size:chunk_id*max_chunk_size+chunk.shape[0],1] = chunk['Response']
del chunk_id, bar, chunk

df = pd.DataFrame(data = responses[:,1], index =responses[:,0], columns=['Response'])
utils.save_variable(df,'outputs/responses.pkl')



        
