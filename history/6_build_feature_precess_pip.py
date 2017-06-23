import utils
import pandas as pd
import numpy as np
import progressbar
from sklearn import tree
from sklearn.metrics import matthews_corrcoef

train_row_nu = 1183747
chunk_nu = 80#1184
max_chunk_size = 1000
col_numeric_nu = 969
col_cate_nu = 2140
col_date_nu = 1157



'''
proba of categorical cols
'''

x_cate_num_len = chunk_nu*max_chunk_size
if x_cate_num_len > train_row_nu : x_cate_num_len = train_row_nu
x_cate_num = np.zeros((x_cate_num_len, col_cate_nu+col_numeric_nu++col_date_nu))
y_cate_num  = np.zeros(x_cate_num_len)
bar = progressbar.ProgressBar()
print('loading cate proba and raw num, and date...')

chunks_num = pd.read_csv('data/train_numeric.csv',chunksize=max_chunk_size, low_memory=False,iterator=True)
chunks_date = pd.read_csv('data/train_date.csv',chunksize=max_chunk_size, low_memory=False,iterator=True)
for chunk_id in bar(range(0,chunk_nu,1)):
    
    chunk_cate = utils.read_variable('model_stats/train_cate_proba/'+str(chunk_id)+'.pkl')
    chunk_num = chunks_num.get_chunk()
    chunk_date = chunks_date.get_chunk()
    row_range = range(chunk_id*max_chunk_size,chunk_id*max_chunk_size+chunk_cate.shape[0],1)
    x_cate_num[row_range,:col_cate_nu] = chunk_cate
    x_cate_num[row_range,col_cate_nu: col_cate_nu+col_numeric_nu] = chunk_num.drop(['Response'],axis=1)
    x_cate_num[row_range,col_cate_nu+col_numeric_nu:] = chunk_date
    y_cate_num[row_range] =  chunk_num['Response']

del chunk_id, bar, chunk_num, chunk_cate, row_range


#%%
'''
remove low density col
'''
pip = {}

nan_ratio_threshold = 0.1
hd_cols = utils.sel_high_density_cols(x_cate_num,nan_ratio_threshold)
x_hd = x_cate_num[:,hd_cols]

pip['0_hd_cols'] = hd_cols
#%% replace NaN with mean value
from sklearn.preprocessing import Imputer
import time
t0 = time.time()
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer.fit(x_hd)
print('imputer:fit',time.time()-t0,'sec')
t1 = time.time()
x_imputed = imputer.transform(x_hd)
print('imputer:transform',time.time()-t1,'sec')
print('imputer:total',time.time()-t0,'sec')

pip['1_imputer'] = imputer

del t0,t1
del x_hd
del x_cate_num
#%%
'''
generate tr and val
'''

tr_X = x_imputed[:70000,:]
tr_Y = y_cate_num[:70000]

val_X = x_imputed[70000:,:]
val_Y = y_cate_num [70000:]


del x_imputed, y_cate_num
#%%
'''
feature engineering: find the best K, regardless of the config of tree
'''
from sklearn.feature_selection import SelectKBest, f_classif

feature_engine = SelectKBest(f_classif, k='all')
print('SelectKBest...',end='')

t0 = time.time()
feature_engine = feature_engine.fit(tr_X,tr_Y)
print('cost',time.time()-t0)

feature_engine_pvalues = feature_engine.pvalues_
kbest_cols = []

# pvalue 0.05 or 0.1
pvalue_threshold = 0.05
for idx,pv in enumerate(feature_engine_pvalues):
    if pv > pvalue_threshold:
        kbest_cols.append(idx)

print('selected col:',len(kbest_cols))

pip['2_kbest_cols'] = kbest_cols

#%%

utils.save_variable(pip,'model_stats/pip.pkl')
#%%
'''
modeling: search for the best max_depth
'''
from sklearn.feature_selection import SelectKBest, f_classif


X = tr_X[:,kbest_cols]
Y = tr_Y
# use random_state to produce repeatable result.
model = tree.DecisionTreeClassifier(random_state=0)
model = model.fit(X,Y)
y_pred = model.predict(X)

print('tree depth:MAX,tr:',matthews_corrcoef(Y , y_pred),end='')
#utils.save_variable(tree_votes_0,'models/tree_votes_0.pkl')
print(',val:',end='')
X = val_X[:,kbest_cols]
Y = val_Y
y_pred = model.predict(X)
print(matthews_corrcoef(Y, y_pred))

max_depth = 1
for max_depth in range(1,100,1):
    X = tr_X[:,kbest_cols]
    Y = tr_Y
    # use random_state to produce repeatable result.
    model = tree.DecisionTreeClassifier(max_depth=max_depth,random_state=0)
    model = model.fit(X,Y)
    y_pred = model.predict(X)
    
    print('tree depth:',max_depth,',tr:',matthews_corrcoef(Y , y_pred),end='')
    #utils.save_variable(tree_votes_0,'models/tree_votes_0.pkl')
    print(',val:',end='')
    X = val_X[:,kbest_cols]
    Y = val_Y
    y_pred = model.predict(X)
    print(matthews_corrcoef(Y, y_pred))


    
#%%



