
import time
from utils import load_pipped_tr_rows,read_variable,save_variable
import numpy as np
from sklearn.metrics import matthews_corrcoef

#%%

'''
Model: Tree
'''

'''
Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the dataset prior to fitting with the decision tree.
ref:http://scikit-learn.org/stable/modules/tree.html
'''
from sklearn.ensemble import RandomForestClassifier
import os

good_model_val_mcc = 0.2

for gp_idx in range(119):
    row_group = read_variable('final/row_groups/'+str(gp_idx))
    
    print(gp_idx,end='...')
    file_path = 'final/model_1L/'+str(gp_idx) 
    if os.path.isfile(file_path):
        print('exist')
    else:
        print('processing...')
        
        final_val_X, final_val_Y = load_pipped_tr_rows(row_group['val'])

        t_X,t_Y = load_pipped_tr_rows(row_group['train'])    
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
        model = RandomForestClassifier()
        t0 = time.time()
        best_model = model.fit(tr_X,tr_Y)
        y_pred = model.predict(tr_X)
        print('tree:',gp_idx,'ini',',tr:',matthews_corrcoef(tr_Y , y_pred),end='')
    
        # val
        print(',val:',end='')
        y_pred = model.predict(val_X)
        val_mcc = matthews_corrcoef(val_Y, y_pred)
        best_val_mcc = val_mcc
        print(val_mcc,end='-->BEST')
        
        # val
        print(',test:',end='')
        final_y_pred = model.predict(final_val_X)
        final_val_mcc = matthews_corrcoef(final_val_Y, final_y_pred)
        print(final_val_mcc,end='')
        
        # test
        print()

        print('---------------------------------------')
    
        round_id =0
        round_max = 20
        shuffle_tr_val_count = 0
        while round_id <round_max:
            
            model = RandomForestClassifier()
            t0 = time.time()
            model = model.fit(tr_X,tr_Y)
            
            y_pred = model.predict(tr_X)
            
            print('tree:',gp_idx,round_id,',tr:',matthews_corrcoef(tr_Y , y_pred),end='')
            print(',val:',end='')
            y_pred = model.predict(val_X)
            val_mcc = matthews_corrcoef(val_Y, y_pred)
            print(val_mcc,end='')
    
            if val_mcc > best_val_mcc:
                best_val_mcc = val_mcc
                best_model = model
                print('-->BEST',end='')
            else:
                print(end='')
                    # val
            print(',final:',end='')
            final_y_pred = model.predict(final_val_X)
            final_val_mcc = matthews_corrcoef(final_val_Y, final_y_pred)
            print(final_val_mcc)
            print('---------------------------------------')
            round_id += 1
            
            if round_id ==3 and best_val_mcc <=0:
                shuffle_tr_val_count +=1
                round_id = 0
                # generate new tr and val dataset
                tr_val_range_0 = np.random.permutation(t_X_0.shape[0])
                tr_range_0 = tr_val_range_0[:t_X_0_len_tr]
                val_range_0 = tr_val_range_0[t_X_0_len_tr:]                       
                tr_val_range_1 = np.random.permutation(t_X_1.shape[0])
                tr_range_1 = tr_val_range_1[:t_X_1_len_tr]
                val_range_1 = tr_val_range_1[t_X_1_len_tr:]
                                             
                tr_X = np.concatenate([t_X_0[tr_range_0,:],t_X_1[tr_range_1,:]])
                # only model on numeric cols based on 10_validation_infor_leakage.py
                tr_Y = np.concatenate([np.zeros(tr_range_0.shape[0]),np.ones(tr_range_1.shape[0])])
                val_X = np.concatenate([t_X_0[val_range_0,:],t_X_1[val_range_1,:]])
                val_Y = np.concatenate([np.zeros(val_range_0.shape[0]),np.ones(val_range_1.shape[0])])
                
            if shuffle_tr_val_count ==10:
                break
                
    
        if best_val_mcc <=0:
            for max_depth in range(1,200,10):
            
                model = RandomForestClassifier()
                t0 = time.time()
                model = model.fit(tr_X,tr_Y)
                
                y_pred = model.predict(tr_X)
                
                print('tree:',gp_idx, 'max_depth;',max_depth,',tr:',matthews_corrcoef(tr_Y , y_pred),end='')
                #print('tr 1s:real',sum(Y),',pred',sum(y_pred))
                #utils.save_variable(tree_votes_0,'models/tree_votes_0.pkl')
                print(',val:',end='')
                y_pred = model.predict(val_X)
                val_mcc = matthews_corrcoef(val_Y, y_pred)
                print(val_mcc,end='')
        
        #        print(',test:',end='')
        #        f_y_pred = model.predict(f_X)
        #        f_mcc = matthews_corrcoef(f_Y, f_y_pred)
        #        print(f_mcc,end='')
                
                if val_mcc > best_val_mcc:
                    best_val_mcc = val_mcc
                    best_model = model
                    print('-->BEST')
                else:
                    print()
                print(',test:',end='')
                final_y_pred = model.predict(final_val_X)
                final_val_mcc = matthews_corrcoef(final_val_Y, final_y_pred)
                print(final_val_mcc)
                print('---------------------------------------')
            
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        
        print('tree:',gp_idx,'BEST,val:',best_val_mcc,end='-->')
    
        
        if best_val_mcc > good_model_val_mcc:
            save_variable(model,file_path)
            print('SAVED'+'('+file_path+')')
        else:
            print('DISCARD')
        print('#####################################')
 


    
    
#%%
'''
TEST
'''

from sklearn.ensemble import RandomForestClassifier
import os

good_model_val_mcc = 0.2

gp_idx = 1
row_group = read_variable('final/row_groups/'+str(gp_idx))

print(gp_idx,end='...')
file_path = 'final/model_1L/'+str(gp_idx) 


final_val_X, final_val_Y = load_pipped_tr_rows(row_group['val'])
t_X,t_Y = load_pipped_tr_rows(row_group['train'])   

#%% 
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


tr_Y = np.concatenate([np.zeros(tr_range_0.shape[0]),np.ones(tr_range_1.shape[0])])
val_X = np.concatenate([t_X_0[val_range_0,:],t_X_1[val_range_1,:]])

val_Y = np.concatenate([np.zeros(val_range_0.shape[0]),np.ones(val_range_1.shape[0])])

# training
model = RandomForestClassifier()
t0 = time.time()
best_model = model.fit(tr_X,tr_Y)
y_pred = model.predict(tr_X)
print('tree:',gp_idx,'ini',',tr:',matthews_corrcoef(tr_Y , y_pred),end='')

# val
print(',val:',end='')
y_pred = model.predict(val_X)
val_mcc = matthews_corrcoef(val_Y, y_pred)
best_val_mcc = val_mcc
print(val_mcc,end='')

# val
print(',test:',end='')
final_y_pred = model.predict(final_val_X)
final_val_mcc = matthews_corrcoef(final_val_Y, final_y_pred)
print(final_val_mcc,',',str(int(sum(final_y_pred)))+'/'+str(sum(final_val_Y)),end='')

# test
print()

print('---------------------------------------')
