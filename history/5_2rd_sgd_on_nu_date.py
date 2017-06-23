import utils
import pandas as pd
import numpy as np
import progressbar
from sklearn import tree
from sklearn.metrics import matthews_corrcoef

#%%

print('reading training num and date col votes matrix...')
# memory can not handle DataFrame
# memory can not handle votes and Response in two separate numpy variables
votes = np.zeros((1183748, 1185))

max_chunk_size = 1000

#responses = np.zeros((1183748, 1))
bar = progressbar.ProgressBar()
for chunk_id in bar(range(0,1184,1)):
    chunk = utils.read_variable('chunk_tree_votes/models/train_y_votes_prob/chunk_'+str(chunk_id)+'.pkl')
    votes[chunk_id*max_chunk_size:chunk_id*max_chunk_size+chunk.shape[0],1:] = chunk.drop(['Response'],axis = 1)
    votes[chunk_id*max_chunk_size:chunk_id*max_chunk_size+chunk.shape[0],0] = chunk['Response']
del chunk_id, bar, chunk

#%

x_tr = votes[:900000,1:]
y_tr = votes[:900000,0]

# used to find best SGD in training
x_val = votes[900000:1000000,1:]
y_val = votes[900000:1000000,0]

# used as testing dataset
x_val_final = votes[1000000:,1:]
y_val_final = votes[1000000:,0]

# save memory
del votes
#%%

from sklearn import linear_model
forest = []

tree_id = 0
id_start = 0
id_length =300000
while True:
    id_end = id_start+id_length    
    if id_end > x_tr.shape[0]:
        id_end = x_tr.shape[0]    
    x = x_tr[id_start:id_end,:]
    y = y_tr[id_start:id_end]
    
    best_tree_mcc = 0
    best_tree = 0
    for candidate_i in range(0,10,1):
        print('tree',str(tree_id)+'.'+str(candidate_i),'training...',end='')
        tre = linear_model.SGDClassifier().fit(x,y)
        y_pred = tre.predict(x)
        mcc = matthews_corrcoef(y, y_pred) 
        print(mcc,end='')
        print(',val...',end='')
        y_pred = tre.predict(x_val)
        mcc = matthews_corrcoef(y_val, y_pred) 
        print(mcc,end='')
        
        if mcc > best_tree_mcc:
            best_tree = tre
            best_tree_mcc = mcc
            print('(best)',end='')
        print()
    print('tree',tree_id,'-->best mcc:',best_tree_mcc)
    forest.append(best_tree)
    if id_end == x_tr.shape[0]:
        break
    tree_id +=1
    id_start = id_end

utils.save_variable(forest,'model_stats/forest.pkl')

del x,y, id_start, id_end, tree_id, best_tree_mcc, best_tree
#%%
#%% validation  based on three second level trees
x = x_val_final
y = y_val_final

y_sum = np.zeros(y.shape[0])
threshold = 0

for tre in forest:
    y_sum += tre.predict(x)

y_pred = (y_sum >threshold).astype(int)

mcc = matthews_corrcoef(y, y_pred) 
print(threshold,'val:',mcc)


#%%


#%%
"""
 produce testing result
"""
max_chunk_size = 1000

forest = utils.read_variable('model_stats/forest.pkl')

#responses = np.zeros((1183748, 1))
bar = progressbar.ProgressBar()
print('loading testing votes chunks...')
y_sum = np.zeros((1183748, 1))

for chunk_id in bar(range(0,1184,1)):
    chunk = utils.read_variable('chunk_tree_votes/models/test_y_votes_prob/chunk_'+str(chunk_id)+'.pkl')
    for tre in forest:            
        y_sum[chunk_id*max_chunk_size:chunk_id*max_chunk_size+chunk.shape[0],0] += tre.predict(chunk)

del chunk_id, bar, chunk   

threshold = 0

y_pred = (y_sum >threshold).astype(int)


# saving to CSV
test_ids = utils.read_variable('outputs/test_ids.pkl')
test_y_ids = pd.DataFrame(test_ids,columns=['Id'])
test_y_y = pd.DataFrame(y_pred,columns=['Response'])
test_y = pd.concat([test_y_ids,test_y_y],axis=1)
test_y = test_y.set_index('Id')
test_y.to_csv('submissions/submission_on_2n_level_votes_sgd.csv',float_format='%.0f')

print('1s:',np.sum(y_pred))

#%%
s1 =  pd.read_csv('submissions/submission_on_2n_level_votes_sgd_2.csv')

print(np.sum(s1['Response']))


#%%


c0 = utils.read_variable('chunk_tree_votes/models/test_y_votes_prob/chunk_0.pkl')
c1 = utils.read_variable('chunk_tree_votes/models/test_y_votes_prob/chunk_399.pkl')