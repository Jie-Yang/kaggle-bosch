
import time
from utils import load_training_subset_1110,read_variable,save_variable
import numpy as np
from sklearn.metrics import matthews_corrcoef


#%%
val_X,val_Y = load_training_subset_1110(range(1000,1010,1))


tr_X_1s = read_variable('model_stats/tr_pip_data_1s_1110.pkl')


#%%
'''
Model: KNN

mcc 0 on training dataset
'''
import os
from sklearn.neighbors import KNeighborsClassifier



len_1s = tr_X_1s.shape[0]
# can not afford smaller tol which will take too long to finish
tol = 1e-3
for set_id in range(0,166,1):

    file_path = '7/knn_'+str(set_id)+'.pkl'
    if os.path.exists(file_path):
                print('already exist.',file_path)
    else:
    
        # 6 chunks give about the same 0s (tiny amount of 1s is ignored) as the 1s
        chunk_range = range(set_id,1000,166)
        t_X,t_Y = load_training_subset_1110(chunk_range)
        tr_X = t_X
        tr_Y = t_Y
        
        '''
        based on experiment, smaller tol give better fit on training dataset
         tol : float, default: 1e-4
        Tolerance for stopping criteria.
        '''

        best_model = KNeighborsClassifier(n_jobs=3)	
        t0 = time.time()
        best_model =best_model.fit(tr_X,tr_Y)
        tr_Y_pred = best_model.predict(tr_X)
        best_tr_mcc = matthews_corrcoef(tr_Y , tr_Y_pred)
        print(set_id,'- i','(',tol,')','logic:',',tr:',best_tr_mcc,end='')
        
#        print(',val:',end='')
#        val_Y_pred = best_model.predict(val_X)
#        print(matthews_corrcoef(val_Y, val_Y_pred),end='')
#        print(',cost:',int(time.time()-t0),'sec')
        #print('val 1s:real',sum(val_Y),',pred',sum(val_Y_pred))
 
        save_variable(best_model,file_path)