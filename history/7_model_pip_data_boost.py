
import time
from utils import load_training_subset_1108,read_variable,save_variable
import numpy as np
from sklearn.metrics import matthews_corrcoef


max_chunk_size = 1000
col_cate_nu = 2140
col_numeric_nu = 969
col_date_nu = 1157

#%%
val_X,val_Y = load_training_subset_1108(range(1000,1010,1))


#%%
'''
Model: AdaBoostClassifier: always mcc 0 in val result
'''
from sklearn.ensemble import AdaBoostClassifier
tr_X_1s = read_variable('model_stats/tr_pip_data_1s_1108.pkl')


len_1s = tr_X_1s.shape[0]

for set_id in range(0,166,1):
    # 6 chunks give about the same 0s (tiny amount of 1s is ignored) as the 1s
    chunk_range = range(set_id,1000,166)
    t_X,t_Y = load_training_subset_1108(chunk_range)
    tr_X = np.concatenate([t_X,tr_X_1s])
    tr_Y = np.concatenate([t_Y,np.ones(len_1s)])
    X,Y = tr_X,tr_Y
    model = AdaBoostClassifier(n_estimators=100)
    t0 = time.time()
    model = model.fit(X,Y)
    y_pred = model.predict(X)
    print(set_id,'boost:',',tr:',matthews_corrcoef(Y , y_pred),end='')
    #print('tr 1s:real',sum(Y),',pred',sum(y_pred))
    #utils.save_variable(tree_votes_0,'models/tree_votes_0.pkl')
    print(',val:',end='')
    X, Y = val_X, val_Y
    y_pred = model.predict(X)
    print(matthews_corrcoef(Y, y_pred),end='')
    print(',cost:',int(time.time()-t0),'sec')

    break 
    save_variable(model,'7/boost_'+str(set_id)+'.pkl')

'''


'''
