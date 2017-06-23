import progressbar
import numpy as np
from sklearn.metrics import matthews_corrcoef
from utils import read_variable, save_variable
import time

tr_chunk_idx = read_variable('final/tr_chunk_idx')

tr_Y = np.zeros([0])
tr_votes = np.zeros([0,301])
bar = progressbar.ProgressBar()
for chunk_id in bar(tr_chunk_idx):
    chunk_votes = read_variable('final/tr_votes/'+str(chunk_id)+'.pkl')
    tr_votes = np.concatenate([tr_votes,chunk_votes])
    chunk_Y= read_variable('data/train_y_chunks/'+str(chunk_id)+'.pkl')
    tr_Y = np.concatenate([tr_Y,chunk_Y])
    
val_chunk_idx = read_variable('final/val_chunk_idx')
val_Y = np.zeros([0])
val_votes = np.zeros([0,301])
bar = progressbar.ProgressBar()
for chunk_id in bar(val_chunk_idx):
    chunk_votes = read_variable('final/tr_votes/'+str(chunk_id)+'.pkl')
    val_votes = np.concatenate([val_votes,chunk_votes])
    chunk_Y= read_variable('data/train_y_chunks/'+str(chunk_id)+'.pkl')
    val_Y = np.concatenate([val_Y,chunk_Y])
    
#%%
from sklearn.ensemble import RandomForestClassifier

tr_row_nu = tr_Y.shape[0]

tr_range = range(1,tr_row_nu,10)
val_range = range(0,tr_row_nu,10)
t_X, t_Y = tr_votes[tr_range,:], tr_Y[tr_range]
v_X, v_Y = tr_votes[val_range,:], tr_Y[val_range]

class_weight = {}
class_weight[0]=0.995
class_weight[1]=0.005
for max_depth in range(10,500,10):
    t0 = time.time()
    print(max_depth,end='-->')
    forest_2nd = RandomForestClassifier(max_depth=max_depth,n_estimators=11,random_state=12)
    forest_2nd.fit(t_X, t_Y)
    y_pred= forest_2nd.predict(t_X)
    print('tr:',matthews_corrcoef(t_Y , y_pred),end=',')
#    y_pred= forest_2nd.predict(v_X)
#    print('val0:',matthews_corrcoef(v_Y , y_pred),end=',')
    y_pred= forest_2nd.predict(v_X)
    print(' interal_val:',matthews_corrcoef(v_Y , y_pred),end='')
    print(',cost',int(time.time()-t0),'sec')
    
#%%
'''
130-->tr: 0.810655889521, interal_val: 0.528099335162,cost 35 sec
140-->tr: 0.825737217665, interal_val: 0.534956213235,cost 37 sec
150-->tr: 0.842508111335, interal_val: 0.53835228673,cost 40 sec
160-->tr: 0.857037815898, interal_val: 0.55009073362,cost 43 sec
170-->tr: 0.878858001891, interal_val: 0.551474355168,cost 45 sec
180-->tr: 0.89648980268, interal_val: 0.553394933481,cost 48 sec
190-->tr: 0.912885209111, interal_val: 0.55393652612,cost 50 sec
200-->tr: 0.936950628272, interal_val: 0.550903174188,cost 52 sec
210-->tr: 0.954393385139, interal_val: 0.550903174188,cost 54 sec
220-->tr: 0.959565652193, interal_val: 0.550903174188,cost 54 sec
'''
#%%
'''
forest on votes
'''
max_depth_best = 200

class_weight = {}
class_weight[0]=0.995
class_weight[1]=0.005

for forest_idx in range(0,10,1):
    tr_range = range(forest_idx,tr_row_nu,10)
    t_X, t_Y = tr_votes[tr_range,:], tr_Y[tr_range]

    t0 = time.time()
    print('forest',forest_idx,end='-->')
    forest_2nd = RandomForestClassifier(max_depth=max_depth_best,n_estimators=11,random_state=12)
    forest_2nd.fit(t_X, t_Y)
    y_pred= forest_2nd.predict(t_X)
    print('tr:',matthews_corrcoef(t_Y , y_pred),end=',')
    val_y_pred= forest_2nd.predict(val_votes)
    print('val:',matthews_corrcoef(val_Y , val_y_pred),end='')
    print(',cost',int(time.time()-t0))
    save_variable(forest_2nd,'final/2nd_level_models/'+str(forest_idx))

#%%
'''
forest 0-->tr: 0.853781308398,val: 0.547872937046,cost 84
forest 1-->tr: 0.839596341509,val: 0.550393783491,cost 89
forest 2-->tr: 0.868869382821,val: 0.544121676669,cost 110
forest 3-->tr: 0.831067645144,val: 0.556898396227,cost 111
forest 4-->tr: 0.866979504134,val: 0.572140934661,cost 108
forest 5-->tr: 0.865444728731,val: 0.565962549302,cost 105
forest 6-->tr: 0.847071811478,val: 0.550088494652,cost 105
forest 7-->tr: 0.867795690777,val: 0.55928161695,cost 110
forest 8-->tr: 0.849556952293,val: 0.54308323331,cost 111
forest 9-->tr: 0.852024718779,val: 0.558287108709,cost 105
'''
#%%
'''
validating on training dataset
'''

forest_2nd = []
for forest_idx in range(0,10,1):
    model = read_variable('final/2nd_level_models/'+str(forest_idx))
    forest_2nd.append(model)

val_Y_sum = np.zeros(val_Y.shape[0])
for model in forest_2nd:
    val_Y_pred = model.predict(val_votes)
    print('mcc:',matthews_corrcoef(val_Y , val_Y_pred))
    val_Y_sum += val_Y_pred
    
#%%
'''
mcc: 0.572089121234
mcc: 0.584374265357
mcc: 0.574731321479
mcc: 0.561043527534
mcc: 0.56279640286
mcc: 0.568082925197
mcc: 0.593015594965
mcc: 0.565770787296
mcc: 0.56094928499
mcc: 0.577023503516
'''
#%%
for threshold in range(10):
    print(threshold,'total mcc:',matthews_corrcoef(val_Y , val_Y_sum>=threshold))

'''
0 total mcc: 0.0
1 total mcc: 0.686114086329
2 total mcc: 0.672432167471
3 total mcc: 0.690202683729
4 total mcc: 0.685842106258
5 total mcc: 0.657144568806
6 total mcc: 0.61366625163
7 total mcc: 0.551987872277
8 total mcc: 0.491048326685
9 total mcc: 0.400699500447
'''
#%%
threshold_best = 3
print(threshold_best,'total mcc:',matthews_corrcoef(val_Y , val_Y_sum>=threshold_best))
