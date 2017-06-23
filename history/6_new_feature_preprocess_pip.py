import utils
import pandas as pd
import numpy as np
import progressbar
from sklearn import tree
from sklearn.metrics import matthews_corrcoef

train_row_nu = 1183747
chunk_idx = range(0,1000,12)
chunk_nu = len(chunk_idx)
max_chunk_size = 1000
col_numeric_nu = 969
col_cate_nu = 2140
col_date_nu = 1157



'''
proba of categorical cols
'''

x_raw_len = chunk_nu*max_chunk_size
if x_raw_len > train_row_nu : x_raw_len = train_row_nu
x_raw = np.zeros((x_raw_len, col_cate_nu+col_numeric_nu+col_date_nu))
y_raw  = np.zeros(x_raw_len)
bar = progressbar.ProgressBar()
print('loading cate proba and raw num, and date...')

x_raw_idx = 0
for chunk_id in bar(chunk_idx):
    
    chunk_cate = utils.read_variable('model_stats/train_cate_proba/'+str(chunk_id)+'.pkl')
    chunk_num = utils.read_variable('data/train_numeric_chunks/chunk_'+str(chunk_id)+'.pkl')
    chunk_date = utils.read_variable('data/train_date_chunks/chunk_'+str(chunk_id)+'.pkl')
    
    row_range = range(x_raw_idx*max_chunk_size,x_raw_idx*max_chunk_size+chunk_cate.shape[0],1)
    x_raw[row_range,:col_cate_nu] = chunk_cate
    x_raw[row_range,col_cate_nu: col_cate_nu+col_numeric_nu] = chunk_num.drop(['Response'],axis=1)
    x_raw[row_range,col_cate_nu+col_numeric_nu:] = chunk_date

    y_raw[row_range] =  chunk_num['Response']

    x_raw_idx +=1

del chunk_id, bar, chunk_num, chunk_cate, row_range


#%%
pip = {}

#%%
'''
Pre-process 0: replace NaN with median value
replace NaN with mean value
The median is a more robust estimator for data with high magnitude variables which could dominate results (otherwise known as a ‘long tail’).
'''
from sklearn.preprocessing import Imputer
import time

X = x_raw

t0 = time.time()
imputer = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer.fit(X)
print('imputer:fit',time.time()-t0,'sec')
t1 = time.time()
x_imputed = imputer.transform(X)
print('imputer:transform',time.time()-t1,'sec')
print('imputer:total',time.time()-t0,'sec')

pip['0_imputer'] = imputer


#%%
'''
Pre-process 1: remove low variance cols
'''

from sklearn.feature_selection import VarianceThreshold
'''
As an example, suppose that we have a dataset with boolean features, 
and we want to remove all features that are either one or zero (on or off) 
in more than 80% of the samples.
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
'''
ratio_1s = 5/1000
sel = VarianceThreshold(threshold=( ratio_1s* (1 - ratio_1s)))
x_high_variance = sel.fit_transform(x_imputed)

print('high variance cols:',x_high_variance.shape[1],'from',x_imputed.shape[1])
pip['1_high_variance'] = sel
#%%

#%%
'''
feature engineering: find the best K, regardless of the config of tree
'''
from sklearn.feature_selection import SelectKBest, f_classif

feature_engine = SelectKBest(f_classif, k='all')
print('SelectKBest...',end='')

t0 = time.time()
feature_engine = feature_engine.fit(x_high_variance,y_raw)
print('cost',time.time()-t0)

feature_engine_pvalues = feature_engine.pvalues_
kbest_cols = []

# pvalue 0.05 or 0.1
pvalue_threshold = 0.1
for idx,pv in enumerate(feature_engine_pvalues):
    if pv > pvalue_threshold:
        kbest_cols.append(idx)

print('selected col:',len(kbest_cols),'from',x_high_variance.shape[1])

pip['2_kbest_cols'] = kbest_cols

#%%

utils.save_variable(pip,'model_stats/pip_1108.pkl')

    
#%%



