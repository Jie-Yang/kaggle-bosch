
import time
from utils import load_training_subset_1110,read_variable,save_variable
import numpy as np
from sklearn import tree
from sklearn.metrics import matthews_corrcoef


#%%
val_X,val_Y = load_training_subset_1110(range(1000,1020,1))


tr_X_1s = read_variable('model_stats/tr_pip_data_1s_1110.pkl')


#%%

a1 = tr_X_1s[1,:]
a0 = val_X[0,:]
#%%

'''
Model: Tree
'''



'''
Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the dataset prior to fitting with the decision tree.
ref:http://scikit-learn.org/stable/modules/tree.html
'''
from sklearn.ensemble import RandomForestClassifier

len_1s = tr_X_1s.shape[0]
max_depth = 30
min_samples_leaf = 80
n_estimators = 21
for set_id in range(0,166,1):
    n_estimators = 21
    # 6 chunks give about the same 0s (tiny amount of 1s is ignored) as the 1s
    chunk_range = range(set_id,1000,166)
    t_X,t_Y = load_training_subset_1110(chunk_range)
    
    tr_X = np.concatenate([t_X,tr_X_1s])
    tr_Y = np.concatenate([t_Y,np.ones(len_1s)])
    
    X,Y = tr_X,tr_Y


    model = RandomForestClassifier( max_depth=max_depth, min_samples_leaf =min_samples_leaf,n_estimators=n_estimators,n_jobs=3)
    t0 = time.time()
    best_model = model.fit(X,Y)
    
    y_pred = model.predict(X)
    
    print('tree:',set_id,'ini',',tr:',matthews_corrcoef(Y , y_pred),end='')
    #print('tr 1s:real',sum(Y),',pred',sum(y_pred))
    #utils.save_variable(tree_votes_0,'models/tree_votes_0.pkl')
    print(',val:',end='')
    X, Y = val_X, val_Y
    y_pred = model.predict(X)
    val_mcc = matthews_corrcoef(Y, y_pred)
    best_val_mcc = val_mcc
    print(val_mcc)
    #print('val 1s:real',sum(Y),',pred',sum(y_pred))
    print('---------------------------------------')
    round_set = 0
    while best_val_mcc < 0.05 and round_set <3:
        for round_id in range(10):
            X,Y = tr_X,tr_Y
        
            model = RandomForestClassifier( max_depth=max_depth, min_samples_leaf =min_samples_leaf,n_estimators=n_estimators,n_jobs=3)
            t0 = time.time()
            model = model.fit(X,Y)
            
            y_pred = model.predict(X)
            
            print('tree:',set_id,round_id,round_set,',tr:',matthews_corrcoef(Y , y_pred),end='')
            #print('tr 1s:real',sum(Y),',pred',sum(y_pred))
            #utils.save_variable(tree_votes_0,'models/tree_votes_0.pkl')
            print('val:',end='')
            X, Y = val_X, val_Y
            y_pred = model.predict(X)
            val_mcc = matthews_corrcoef(Y, y_pred)
            print(val_mcc,end='')
            #print('val 1s:real',sum(Y),',pred',sum(y_pred))
            
            
            if val_mcc > best_val_mcc:
                best_val_mcc = val_mcc
                best_model = model
                print('-->BEST')
            else:
                print()
            print('---------------------------------------')
            
        round_set += 1
        n_estimators += 10
    
    X,Y = tr_X,tr_Y
    
    y_pred = best_model.predict(X)
    
    print('tree BEST:',set_id,',tr:',matthews_corrcoef(Y , y_pred),end='')

    print(',val:',end='')
    X, Y = val_X, val_Y
    y_pred = best_model.predict(X)
    val_mcc = matthews_corrcoef(Y, y_pred)
    print(val_mcc)
    print('#####################################')
    save_variable(model,'8/forest_'+str(set_id)+'.pkl')

#%%
'''
'''
from sklearn.ensemble import RandomForestClassifier

len_1s = tr_X_1s.shape[0]


set_id=0


chunk_range = range(set_id,1000,166)
t_X,t_Y = load_training_subset_1110(chunk_range)

tr_X = np.concatenate([t_X,tr_X_1s])
tr_Y = np.concatenate([t_Y,np.ones(len_1s)])

X,Y = tr_X,tr_Y

model = RandomForestClassifier( min_samples_leaf =2,n_estimators=10,
                                   n_jobs=3)
t0 = time.time()
best_model = model.fit(X,Y)

y_pred = model.predict(X)

print('tree:',set_id,'ini',',tr:',matthews_corrcoef(Y , y_pred),end='')
#print('tr 1s:real',sum(Y),',pred',sum(y_pred))
#utils.save_variable(tree_votes_0,'models/tree_votes_0.pkl')
print(',val:',end='')
X, Y = val_X, val_Y
y_pred = model.predict(X)
val_mcc = matthews_corrcoef(Y, y_pred)
best_val_mcc = val_mcc
print(val_mcc)
#print('val 1s:real',sum(Y),',pred',sum(y_pred))
print('#####################################')

class_weight = {}
class_weight[0] = 1000
class_weight[1] = 5
for max_depth in range(1,100,2):
    X,Y = tr_X,tr_Y

    model = RandomForestClassifier( min_samples_leaf =2,n_estimators=100,
                                   n_jobs=3)
    t0 = time.time()
    model = model.fit(X,Y)
    
    y_pred = model.predict(X)
    
    print(max_depth,'tree:',set_id,round_id,',tr:',matthews_corrcoef(Y , y_pred),end='')
    #print('tr 1s:real',sum(Y),',pred',sum(y_pred))
    #utils.save_variable(tree_votes_0,'models/tree_votes_0.pkl')
    print('val:',end='')
    X, Y = val_X, val_Y
    y_pred = model.predict(X)
    val_mcc = matthews_corrcoef(Y, y_pred)
    print(val_mcc,end='')
    #print('val 1s:real',sum(Y),',pred',sum(y_pred))
    
    
    if val_mcc > best_val_mcc:
        best_val_mcc = val_mcc
        best_model = model
        print('-->BEST')
    else:
        print()
    print('#####################################')

X,Y = tr_X,tr_Y

y_pred = best_model.predict(X)

print('best tree:',set_id,',tr:',matthews_corrcoef(Y , y_pred),end='')

print(',val:',end='')
X, Y = val_X, val_Y
y_pred = best_model.predict(X)
val_mcc = matthews_corrcoef(Y, y_pred)
print(val_mcc)

#%%
fi = model.feature_importances_