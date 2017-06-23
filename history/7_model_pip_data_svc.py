
import time
import utils
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import matthews_corrcoef


max_chunk_size = 1000
col_cate_nu = 2140
col_numeric_nu = 969
col_date_nu = 1157

#%%
#val_X,val_Y = utils.load_training_subset(range(1000,1184,1))


#%%

'''
Model: SVC
WARNING: The implementation is based on libsvm.
The fit time complexity is more than quadratic with the number of samples which
makes it hard to scale to dataset with more than a couple of 10000 samples.
HENCE, training dataset is splitted into 10 blocks
'''
from sklearn.svm import SVC
import os

tr_X_1s = utils.read_variable('model_stats/tr_pip_data_1s.pkl')


len_1s = tr_X_1s.shape[0]

for set_id in range(6,1000,6):
    # 6 chunks give about the same 0s (tiny amount of 1s is ignored) as the 1s
    
    file_path = '7/svc_'+str(set_id)+'.pkl'
    if os.path.exists(file_path):
                print('already exist.',file_path)
    else:
        chunk_range = range(set_id-6,set_id,1)
        t_X,t_Y = utils.load_training_subset(chunk_range)
        tr_X = np.concatenate([t_X,tr_X_1s])
        tr_Y = np.concatenate([t_Y,np.ones(len_1s)])
        model = SVC( kernel='rbf', C=1)
        t0 = time.time()
        model = model.fit(tr_X,tr_Y)
        y_pred = model.predict(tr_X)
        print(set_id,'svc:',',tr:',matthews_corrcoef(tr_Y , y_pred))
        
        # do not do val which cost too long.
    #    print(',val:',end='')
    #    X, Y = val_X, val_Y
    #    y_pred = model.predict(X)
    #    print(matthews_corrcoef(Y, y_pred),end='')
    #    print(',cost:',int(time.time()-t0),'sec')
        
    
        utils.save_variable(model,file_path)

