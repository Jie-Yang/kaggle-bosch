
import time
from utils import load_pipped_tr_chunks,read_variable,save_variable
import numpy as np
from sklearn.metrics import matthews_corrcoef


#%%
tr_chunk_idx = read_variable('final/tr_chunk_idx')

#%%

'''
Model: Tree
'''

'''
Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the dataset prior to fitting with the decision tree.
ref:http://scikit-learn.org/stable/modules/tree.html
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import os

good_model_val_mcc = 0.2

from sklearn.neighbors import KNeighborsClassifier
for model_idx in range(100):
    
    print(model_idx,end='...')
    file_path = 'final/good_models_onlynum_2/'+str(model_idx) 
    if os.path.isfile(file_path):
        print('exist')
    else:
        print('processing...')
        chunk_range = []
        for i in range(model_idx,len(tr_chunk_idx),100):
            chunk_range.append(tr_chunk_idx[i])
        
        #splite samples into training and validating dataset
        t_X,t_Y = load_pipped_tr_chunks(chunk_range)    
        t_X_0 = t_X[t_Y==0,:]
        t_X_1 = t_X[t_Y==1,:]
    
        t_X_0_len_tr = int(t_X_0.shape[0]*0.9)
        t_X_1_len_tr = int(t_X_1.shape[0]*0.9)
        
        tr_val_range_0 = np.random.permutation(t_X_0.shape[0])
        tr_range_0 = tr_val_range_0[:t_X_0_len_tr]
        val_range_0 = tr_val_range_0[t_X_0_len_tr:]                       
        tr_val_range_1 = np.random.permutation(t_X_1.shape[0])
        tr_range_1 = tr_val_range_1[:t_X_1_len_tr]
        val_range_1 = tr_val_range_1[t_X_1_len_tr:]
                                     
        tr_X = np.concatenate([t_X_0[tr_range_0,:],t_X_1[tr_range_1,:]])
        # only model on numeric cols based on 10_validation_infor_leakage.py
        tr_X = tr_X[:,:87]
        tr_Y = np.concatenate([np.zeros(tr_range_0.shape[0]),np.ones(tr_range_1.shape[0])])
        val_X = np.concatenate([t_X_0[val_range_0,:],t_X_1[val_range_1,:]])
        val_X = val_X[:,:87]
        val_Y = np.concatenate([np.zeros(val_range_0.shape[0]),np.ones(val_range_1.shape[0])])
        
        # training
        model = KNeighborsClassifier(n_jobs=3)	
        t0 = time.time()
        best_model = model.fit(tr_X,tr_Y)
        y_pred = model.predict(tr_X)
        print('tree:',model_idx,'ini',',tr:',matthews_corrcoef(tr_Y , y_pred),end='')
    
        # val
        print(',val:',end='')
        y_pred = model.predict(val_X)

        val_mcc = matthews_corrcoef(val_Y, y_pred)
        best_val_mcc = val_mcc
        print(val_mcc,end='-->BEST')
        
        # test
        print()

        print('---------------------------------------')
        

            
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        
        print('tree:',model_idx,'BEST,val:',best_val_mcc,end='-->')
    
        
        if best_val_mcc > good_model_val_mcc:
            #save_variable(model,file_path)
            print('SAVED'+'('+file_path+')')
        else:
            print('DISCARD')
        print('#####################################')
        #break


    
    
#%%
