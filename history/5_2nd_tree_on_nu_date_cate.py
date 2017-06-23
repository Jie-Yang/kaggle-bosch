import utils
import numpy as np
import progressbar
import time

from sklearn.metrics import matthews_corrcoef
from sklearn import tree

chunk_nu = 1184
max_chunk_size = 1000
col_num_date = 1184
col_cate = 2140
tr_chunk_end = 1000 # use the rest as final validation

#%%



# used to find the best SGD model during the training
sgd_val_chunk_nu = 10
sgd_val_chunk_ids = range(0,sgd_val_chunk_nu,1)
sgd_val_nu = sgd_val_chunk_nu*max_chunk_size
sgd_val_y = np.empty(sgd_val_nu,dtype=np.int)
sgd_val_y[:] = np.NAN 
sgd_val_x = np.empty((sgd_val_nu, col_num_date+col_cate))
sgd_val_x[:] = np.NAN

print('loading model_val dataset.','chunk [',sgd_val_chunk_ids,')')
i = 0
for chunk_id in sgd_val_chunk_ids:
    chunk = utils.read_variable('chunk_tree_votes/models/train_y_votes_prob/chunk_'+str(chunk_id)+'.pkl')
    row_range = range(i*max_chunk_size,i*max_chunk_size+chunk.shape[0],1)
    sgd_val_y[row_range] = chunk['Response']
    # num and date
    votes_date_nu = chunk.drop(['Response'],axis = 1)
    sgd_val_x[row_range,0:col_num_date] = votes_date_nu
    # cate
    chunk = utils.read_variable('model_stats/train_cate_proba/'+str(chunk_id)+'.pkl')
    sgd_val_x[row_range,col_num_date:] = chunk

    i += 1

del i, chunk_id, row_range, chunk, votes_date_nu
#%%
'''
build 2nd level decision model
'''
tr_chunk_nu = 300
tree_id = 0
forest = []
tr_chunk_start_index = sgd_val_chunk_nu
tr_chunk_end_index = tr_chunk_start_index+tr_chunk_nu
while True:
    tr_chunk_ids = range(tr_chunk_start_index,tr_chunk_end_index ,1)
    tr_nu = len(tr_chunk_ids) * max_chunk_size
    tr_y = np.empty(tr_nu,dtype=np.int)
    tr_y[:] = np.NAN 
    tr_x = np.empty((tr_nu, col_num_date+col_cate))
    tr_x[:] = np.NAN
    
    print('loading tr subset.','chunk [',tr_chunk_ids,')')
    i = 0
    bar = progressbar.ProgressBar()
    for chunk_id in bar(tr_chunk_ids):
        chunk = utils.read_variable('chunk_tree_votes/models/train_y_votes_prob/chunk_'+str(chunk_id)+'.pkl')

        row_range = range(i*max_chunk_size,i*max_chunk_size+chunk.shape[0],1)
        tr_y [row_range] = chunk['Response']
        # num and date
        votes_date_nu = chunk.drop(['Response'],axis = 1)
        tr_x[row_range,0:col_num_date] = votes_date_nu
        # cate
        chunk = utils.read_variable('model_stats/train_cate_proba/'+str(chunk_id)+'.pkl')
        tr_x[row_range,col_num_date:] = chunk    
        i += 1
    
    print('training model...')
    best_model_mcc = 0
    best_model = 0
    x = tr_x
    y = tr_y
    
    print('tree',str(tree_id),'-->training...',end='')
    time0 = time.time()
    tre = tree.DecisionTreeClassifier().fit(x,y)
    print('(',time.time()-time0,'sec)',end='')
    time0 = time.time()
    y_pred = tre.predict(x)
    mcc = matthews_corrcoef(y, y_pred) 
    print(mcc,end='')
    print('(',time.time()-time0,'sec)',end='')
    print('...val...',end='')
    time0 = time.time()
    y_pred = tre.predict(sgd_val_x)
    mcc = matthews_corrcoef(sgd_val_y, y_pred) 
    print(mcc)
    print('(',time.time()-time0,'sec)')
        
    forest.append(tre)
    tree_id +=1
    
    tr_chunk_start_index = tr_chunk_end_index
    tr_chunk_end_index = tr_chunk_start_index + tr_chunk_nu
    if tr_chunk_end_index > tr_chunk_end:
        tr_chunk_end_index = tr_chunk_end

    if tr_chunk_start_index == tr_chunk_end:
        del tr_chunk_start_index, tr_chunk_end
        break


utils.save_variable(forest,'model_stats/forest_full_trees.pkl')

#%%

#%%
'''
valide over all model mcc with validation dataset
'''
import progressbar
import numpy as np
from sklearn.metrics import matthews_corrcoef
import utils

forest = utils.read_variable('model_stats/forest_full_trees_50minpertree.pkl')

val_chunk_ids = range(tr_chunk_end,chunk_nu ,1)
val_nu = 183747
val_y = np.empty(val_nu,dtype=np.int)
val_y[:] = np.NAN 
val_x = np.empty((val_nu, col_num_date+col_cate))
val_x[:] = np.NAN

print('loading val subset.','chunk [',val_chunk_ids,')')
i = 0
bar = progressbar.ProgressBar()
for chunk_id in bar(val_chunk_ids):
    chunk = utils.read_variable('chunk_tree_votes/models/train_y_votes_prob/chunk_'+str(chunk_id)+'.pkl')
    row_range = range(i*max_chunk_size,i*max_chunk_size+chunk.shape[0],1)
    val_y [row_range] = chunk['Response']
    # num and date
    votes_date_nu = chunk.drop(['Response'],axis = 1)
    val_x[row_range,0:col_num_date] = votes_date_nu
    # cate
    chunk = utils.read_variable('model_stats/train_cate_proba/'+str(chunk_id)+'.pkl')
    val_x[row_range,col_num_date:] = chunk    
    i += 1

print('val model...')

#%
votes_3rd = np.zeros((val_y.shape[0],len(forest)))
votes_3rd_proba = np.zeros((val_y.shape[0],len(forest)))
for model_index, model in enumerate(forest):
     y_pred = model.predict(val_x)
     y_pred_proba = model.predict_proba(val_x)
     mcc = matthews_corrcoef(val_y, y_pred) 
     print('model',model_index,':',mcc)
     votes_3rd[:,model_index] = y_pred
     votes_3rd_proba[:,model_index] = y_pred_proba[:,0]
#%%

from sklearn import tree

tr_3rd_x = votes_3rd_proba[0:100000,:]
tr_3rd_y = val_y[0:100000]
val_3rd_x = votes_3rd_proba[100000:,:]
val_3rd_y = val_y[100000:]


model_3rd = tree.DecisionTreeClassifier().fit(tr_3rd_x, tr_3rd_y)
y_pred = model_3rd.predict(tr_3rd_x)
#y_proba = model_3rd.predict_proba(tr_3rd_x)
mcc = matthews_corrcoef(tr_3rd_y, y_pred) 
print('tr-->real:',sum(tr_3rd_y),',pred:',sum(y_pred),',mcc:',mcc)
y_pred = model_3rd.predict(val_3rd_x)
#y_proba = model_3rd.predict_proba(tr_3rd_x)
mcc = matthews_corrcoef(val_3rd_y, y_pred) 
print('val-->real:',sum(val_3rd_y),',pred:',sum(y_pred),',mcc:',mcc)

#%%
y_proba_0 = y_proba[:,0]
y_proba_1 = y_proba[:,1]
y_pred = (y_proba_1 >0.03).astype(int)
mcc = matthews_corrcoef(val_y, y_pred) 
print(':',mcc)
print('val-->real:',sum(val_y),',pred:',sum(y_pred))

#%%
"""
 produce testing result
"""
import pandas as pd

forest = utils.read_variable('model_stats/forest_full_trees_50minpertree.pkl')

#responses = np.zeros((1183748, 1))
bar = progressbar.ProgressBar()
print('loading testing votes chunks...')
y_sum = np.zeros((1183748, 1))

for chunk_id in bar(range(0,1184,1)):
    # num and date
    chunk = utils.read_variable('chunk_tree_votes/models/test_y_votes_prob/chunk_'+str(chunk_id)+'.pkl')
    row_range = range(chunk_id*max_chunk_size,chunk_id*max_chunk_size+chunk.shape[0],1)
    test_x = np.empty((chunk.shape[0], col_num_date+col_cate))
    test_x[:] = np.NAN 
    test_x[:,0:col_num_date] = chunk
    # cate
    chunk = utils.read_variable('model_stats/test_cate_proba/'+str(chunk_id)+'.pkl')
    test_x[:,col_num_date:] = chunk    
    for tre in forest:            
        y_sum[row_range,0] += tre.predict(test_x)

del chunk_id, bar, chunk   

threshold = 0

y_pred = (y_sum >threshold).astype(int)


# saving to CSV
test_ids = utils.read_variable('outputs/test_ids.pkl')
test_y_ids = pd.DataFrame(test_ids,columns=['Id'])
test_y_y = pd.DataFrame(y_pred,columns=['Response'])
test_y = pd.concat([test_y_ids,test_y_y],axis=1)
test_y = test_y.set_index('Id')
test_y.to_csv('submissions/submission_on_2n_level_fulltree.csv',float_format='%.0f')

print('1s:',np.sum(y_pred))
