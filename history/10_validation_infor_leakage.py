from utils import read_variable

all_chunk_idx = range(1184)
tr_chunk_idx = read_variable('final/tr_chunk_idx')
val_chunk_idx = read_variable('final/val_chunk_idx')

#%%
'''
check consistence between chunk index
'''
for chunk_id in all_chunk_idx:
    chunk_y = read_variable('data/train_y_chunks/'+str(chunk_id)+'.pkl')
    chunk_num = read_variable('data/train_numeric_chunks/'+str(chunk_id)+'.pkl')
    chunk_date = read_variable('data/train_date_chunks/'+str(chunk_id)+'.pkl')
    chunk_cate = read_variable('data/train_categorical_chunks/'+str(chunk_id)+'.pkl')
    
    print(chunk_id,end=',')
    diff = chunk_num.index.difference(chunk_date.index)
    if diff.size != 0:
        print('X',end=',')
    else:
        print('v',end=',')
    diff = chunk_num.index.difference(chunk_cate.index)
    if diff.size != 0:
        print('X',end=',')
    else:
        print('v',end=',')
    diff = chunk_num.index.difference(chunk_y.index)
    if diff.size != 0:
        print('X')
    else:
        print('v')

'''
all passed
'''
#%%
'''
check whether there is duplicate Idx
'''
idxs = []
for chunk_id in all_chunk_idx:
    chunk_y = read_variable('data/train_y_chunks/'+str(chunk_id)+'.pkl')
    idxs.extend(chunk_y.index.values)
    
uq = set(idxs)

print(len(idxs)==len(uq))

'''
no duplicate idx
'''
#%%
'''
check distribution of tr and test index
'''
import progressbar

tr_idxs = []
bar = progressbar.ProgressBar()
for chunk_id in bar(all_chunk_idx):
    chunk_y = read_variable('data/train_y_chunks/'+str(chunk_id)+'.pkl')
    tr_idxs.extend(chunk_y.index.values)
    
te_idxs = []
bar = progressbar.ProgressBar()
for chunk_id in all_chunk_idx:
    chunk_date = read_variable('data/test_date_chunks/'+str(chunk_id)+'.pkl')
    te_idxs.extend(chunk_date.index.values)
    
#%
import matplotlib.pyplot as plt
import numpy as np

plt.scatter(tr_idxs, np.zeros(len(tr_idxs)), s=0.2, c='red',label='train',edgecolors='none')
plt.scatter(te_idxs, np.ones(len(te_idxs)), s=0.2, c='green',label='test',edgecolors='none')

#plt.legend()
plt.show()

'''
index of training and testing dataset are well mixed.
'''

#%%
'''
check distribution of tr and val index
'''
tr_tr_idxs = []
for chunk_id in tr_chunk_idx:
    chunk_y = read_variable('data/train_y_chunks/'+str(chunk_id)+'.pkl')
    tr_tr_idxs.extend(chunk_y.index.values)
    
    
tr_val_idxs = []
for chunk_id in val_chunk_idx:
    chunk_y = read_variable('data/train_y_chunks/'+str(chunk_id)+'.pkl')
    tr_val_idxs.extend(chunk_y.index.values)
#%
import matplotlib.pyplot as plt
import numpy as np

plt.scatter(tr_tr_idxs, np.zeros(len(tr_tr_idxs)), s=0.2, c='red',label='train tr',edgecolors='none')
plt.scatter(tr_val_idxs, np.ones(len(tr_val_idxs)), s=0.2, c='green',label='train val',edgecolors='none')

#plt.legend()
plt.show()

'''
index of training and validation dataset are well mixed
'''

#%%
'''
check data pre-process (pip)
suspect: date cols play determinant roles in prediction


chunk_num = read_variable('data/train_numeric_chunks/'+str(chunk_id)+'.pkl')
chunk_date = read_variable('data/train_date_chunks/'+str(chunk_id)+'.pkl')
chunk_cate = read_variable('final/tr_cate_proba/'+str(chunk_id)+'.pkl')

temp_X = np.zeros((chunk_cate.shape[0],col_cate_nu+col_numeric_nu+col_date_nu))
temp_X[:,:col_cate_nu] = chunk_cate
temp_X[:,col_cate_nu: col_cate_nu+col_numeric_nu] = chunk_num
temp_X[:,col_cate_nu+col_numeric_nu:] = chunk_date

pip_X = p3_norm.transform(p1_high_variance.transform(p0_imputer.transform(temp_X))[:,p2_kbest_cols])
'''

col_numeric_nu = 968
col_cate_nu = 2140
col_date_nu = 1156

col_sample = np.zeros(col_cate_nu+col_numeric_nu+col_date_nu)
col_sample[:col_cate_nu] = np.ones(col_cate_nu) # cate
col_sample[col_cate_nu: col_cate_nu+col_numeric_nu] = np.ones(col_numeric_nu)*2 # num
col_sample[col_cate_nu+col_numeric_nu:]  = np.ones(col_date_nu)*3 # num

plt.scatter(range(col_sample.shape[0]), col_sample, s=0.2, c='red',edgecolors='none')
#%%
pip = read_variable('final/feature_sel_pip')
p0_imputer = pip['0_imputer']
p1_high_variance = pip['1_high_variance']
p2_kbest_cols = pip['2_kbest_cols']
p3_norm = pip['3_norm']

p0_filter = (~np.isnan(p0_imputer.statistics_))
p0_o =col_sample[p0_filter]
p1_filter = p1_high_variance.get_support()
p1_o = p0_o[p1_filter]

p2_o = p1_o[p2_kbest_cols]


nu_cate = 0
nu_date = 0
nu_num = 0
for i in p2_o:
    if i==1:
        nu_cate +=1
    elif i==2:
        nu_num +=1
    elif i==3:
        nu_date +=1
        
print('cate:',nu_cate)
print('num:',nu_num)
print('date:',nu_date)
'''
cate: 0
num: 87
date: 592
'''

#%%
'''
check important feature in trees
'''
print('loading trees...')
model_forest = []
bar = progressbar.ProgressBar()
for model_id in bar(range(0,301,1)):
    model = read_variable('final/good_models/'+str(model_id))
    model_forest.append(model)
    
#%
fi = model_forest[1].feature_importances_

'''
based on this plot, date cols play significant roles in prediction
WHICH could be the reason of overfitting
ref: http://decisiontrees.net/decision-trees-tutorial/tutorial-6-entropy-bias/

SUGGEST: model only on number cols
'''

#%%
'''
check important feature in trees
'''
model_id = 0
model = read_variable('final/good_models_onlynum/'+str(model_id))
fi = model.feature_importances_