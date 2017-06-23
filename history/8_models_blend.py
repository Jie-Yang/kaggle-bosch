
import progressbar
import numpy as np
from sklearn.metrics import matthews_corrcoef
import time, os
from utils import load_training_subset_1110,read_variable

#%
val_X,val_Y = load_training_subset_1110(range(1000,1184,1))
#%%


print('loading trees...')
model_forest = []
bar = progressbar.ProgressBar()
for set_id in bar(range(0,300,1)):
    model = read_variable('9/forest_'+str(set_id)+'.pkl')
    model_forest.append(model)

#%%
#%%
print('loading logic...')
model_logic = []
bar = progressbar.ProgressBar()
for set_id in bar(range(0,166,1)):
    model = read_variable('7/logic_'+str(set_id)+'.pkl')
    model_logic.append(model)
    #%%
print('loading boost...')
model_boost = []
bar = progressbar.ProgressBar()
for set_id in bar(range(0,166,1)):
    model = read_variable('7/boost_'+str(set_id)+'.pkl')
    model_boost.append(model)

#%%
print('loading sgd...')
model_sgd = []
bar = progressbar.ProgressBar()
for set_id in bar(range(6,1000,6)):
    model = read_variable('7/sgd_'+str(set_id)+'.pkl')
    model_sgd.append(model)
    
    
#%%
'''
following models are not finished completely
'''
import os

print('loading svc...')
# each svc model cost 100 mb
#model_svc = []
#bar = progressbar.ProgressBar()
#for set_id in bar(range(6,1000,6)):
#    model = utils.read_variable('7/svc_'+str(set_id)+'.pkl')
#    model_svc.append(model)


        
#%%
models_single_type = model_forest
X,Y = val_X,val_Y
votes = np.zeros([X.shape[0],len(models_single_type)])
model_mccs = np.zeros(len(models_single_type))
for model_id,model in enumerate(models_single_type):
    t0 = time.time()
    pred_Y = model.predict_proba(X)
    pred_Y_0 = pred_Y[:,0]
    votes[:,model_id] = pred_Y_0
    mcc = matthews_corrcoef(Y , pred_Y_0<=0.5)
    model_mccs[model_id]=mcc
    print(model_id,',mcc:',mcc,',1s:',sum(pred_Y),',cost',int(time.time()-t0))

del models_single_type,model_id,model,X,Y

votes_forest = votes
'''
TREE: 
148 mcc: 0.090866629082
149 mcc: 0.0936175607992
150 mcc: 0.0986567494937
151 mcc: 0.0996349291934
152 mcc: 0.100251554946
153 mcc: 0.0950713035091
154 mcc: 0.0875663229134
155 mcc: 0.0907506385813

Boost:




LOGIC:


SVC:

'''

#%%
stats = np.sum(votes,axis = 1)
# 0.655 give one of the best mcc: 0.0513847845714
for threshold in range(1,260,1):
    pred_Y = (stats>threshold).astype(np.int)
    print(threshold,'mcc:',matthews_corrcoef(val_Y , pred_Y))
    
'''
TREE: 
148 mcc: 0.090866629082
149 mcc: 0.0936175607992
150 mcc: 0.0986567494937
151 mcc: 0.0996349291934
152 mcc: 0.100251554946
153 mcc: 0.0950713035091
154 mcc: 0.0875663229134
155 mcc: 0.0907506385813

BOOST:
1 mcc: 0.0
2 mcc: 0.00519434502942
3 mcc: 0.00252051266942
4 mcc: 0.00570813837665
5 mcc: 0.00201431853502
6 mcc: 0.0020743920144
7 mcc: -0.000530833840134
'''
#%%
stats = np.sum(votes,axis = 1)
opt_threshold = 152
pred_Y = (stats>opt_threshold).astype(np.int)
print(opt_threshold,'mcc:',matthews_corrcoef(val_Y , pred_Y))
#%%
    
    #%%
model_sets = [model_logic,model_forest]

X,Y = val_X,val_Y

votes_len = 0
for model_set in model_sets:
    votes_len += len(model_set)

vote_all = np.zeros([X.shape[0],votes_len])
for model_set_id,model_set in enumerate(model_sets):
    for model_id, model in enumerate(model_set):
        pred_Y = model.predict(X)
        vote_col_index = model_set_id*166+model_id
        vote_all[:,vote_col_index] = pred_Y
        mcc = matthews_corrcoef(Y , pred_Y)
        print(model_set_id, model_id,vote_col_index,',mcc:',mcc,',1s:',sum(pred_Y))


#%%
stats = np.sum(vote_all,axis = 1)
# 0.655 give one of the best mcc: 0.0513847845714
for threshold in range(1,372,1):
    pred_Y = (stats>threshold).astype(np.int)
    print(threshold,'mcc:',matthews_corrcoef(val_Y , pred_Y))
#%%
from sklearn.tree import DecisionTreeClassifier

t_X, t_Y = vote_all[:100000,:], val_Y[:100000]
v_X, v_Y = vote_all[100000:,:], val_Y[100000:]

class_weight = {}
class_weight[0] = 1000
class_weight[1] = 5
for max_depth in range(10,1000,10):
    print(max_depth,end='-->')
    forest_2nd = DecisionTreeClassifier(max_depth=max_depth)
    forest_2nd.fit(t_X, t_Y)
    y_pred= forest_2nd.predict(t_X)
    print('tr:',matthews_corrcoef(t_Y , y_pred),end=',')
    y_pred= forest_2nd.predict(v_X)
    print('val:',matthews_corrcoef(v_Y , y_pred))