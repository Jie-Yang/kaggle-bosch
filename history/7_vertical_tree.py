
import time
from utils import load_training_subset_1108,read_variable,save_variable
import numpy as np
from sklearn import tree
from sklearn.metrics import matthews_corrcoef
from progressbar import ProgressBar


#%%
val_X,val_Y = load_training_subset_1108(range(1000,1010,1))


#tr_X_1s = read_variable('model_stats/tr_pip_data_1s_1108.pkl')


#%%

from sklearn.tree import DecisionTreeClassifier


class_weight = {}
class_weight[0] = 1000
class_weight[1] = 5

for col_set_idx in range(78):
    col_range = range(10*col_set_idx,10*col_set_idx+10,1)

    chunk_size = 1000
    
    tr_X = np.zeros([1000*chunk_size,len(col_range)])
    tr_Y = np.zeros([1000*chunk_size])
    for chunk_id in range(0,1000,1):
    
        t_X,t_Y = load_training_subset_1108([chunk_id])
        
        tr_row_idx = range(chunk_id*chunk_size,chunk_id*chunk_size+chunk_size,1)
        tr_X[tr_row_idx,:] = t_X[:,col_range]
        tr_Y[tr_row_idx] = t_Y

    X,Y = tr_X,tr_Y
    model = DecisionTreeClassifier( class_weight=class_weight)
    t0 = time.time()
    model = model.fit(X,Y)
    
    y_pred = model.predict(X)
    print(col_set_idx,'-->', col_range,end='')
    print(',tr:',matthews_corrcoef(Y , y_pred),end='')
    #print('tr 1s:real',sum(Y),',pred',sum(y_pred))
    #utils.save_variable(tree_votes_0,'models/tree_votes_0.pkl')
    print(',val:',end='')
    X, Y = val_X[:,col_range], val_Y
    y_pred = model.predict(X)
    val_mcc = matthews_corrcoef(Y, y_pred)
    best_val_mcc = val_mcc
    print(val_mcc)
    break
    #save_variable(model,'vert/tree_'+str(col_set_idx)+'.pkl')

#%%

col_range=range(10)
for max_depth in range(100,200,10):
    X,Y = tr_X,tr_Y
    model = DecisionTreeClassifier( min_samples_leaf=2,class_weight=class_weight)
    t0 = time.time()
    model = model.fit(X,Y)
    
    y_pred = model.predict(X)
    print(max_depth,'-->', col_range,end='')
    print(',tr:',matthews_corrcoef(Y , y_pred),end='')
    #print('tr 1s:real',sum(Y),',pred',sum(y_pred))
    #utils.save_variable(tree_votes_0,'models/tree_votes_0.pkl')
    print(',val:',end='')
    X, Y = val_X[:,col_range], val_Y
    y_pred = model.predict(X)
    val_mcc = matthews_corrcoef(Y, y_pred)
    best_val_mcc = val_mcc
    print(val_mcc,end='')
    print(',1s:',sum(y_pred),'/',sum(Y))
#%%
from sklearn.tree import DecisionTreeClassifier
class_weight = {}
class_weight[0] = 1000
class_weight[1] = 10

#%%

col_trees = []
bar = ProgressBar()
for col_set_idx in bar(range(78)):
    model = read_variable('vert/tree_'+str(col_set_idx)+'.pkl')
    col_trees.append(model)
#%%
for col_set_idx,model in enumerate(col_trees):
    col_range = range(10*col_set_idx,10*col_set_idx+10,1)
    
    pred_y = model.predict(val_X[:,col_range])
    print(col_set_idx,matthews_corrcoef(val_Y, pred_y),sum(pred_y))

    
    




