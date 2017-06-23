
import progressbar
import numpy as np
from sklearn.metrics import matthews_corrcoef
import time, os
from utils import load_training_subset_1110,read_variable, save_variable

#%
val0_X,val0_Y = load_training_subset_1110(range(0,100,1))
val1_X,val1_Y = load_training_subset_1110(range(1000,1100,1))
#%


print('loading trees...')
model_forest = []
bar = progressbar.ProgressBar()
for set_id in bar(range(0,300,1)):
    model = read_variable('9/forest_'+str(set_id)+'.pkl')
    model_forest.append(model)
    
forest_2nd = read_variable('forest_2nd.pkl') 
#%%
models_single_type = model_forest
X,Y = val0_X,val0_Y
votes0 = np.zeros([X.shape[0],len(models_single_type)])
model_mccs = np.zeros(len(models_single_type))
for model_id,model in enumerate(models_single_type):
    t0 = time.time()
    pred_Y = model.predict_proba(X)
    pred_Y_0 = pred_Y[:,0]
    votes0[:,model_id] = pred_Y_0
    mcc = matthews_corrcoef(Y , pred_Y_0<=0.2)
    model_mccs[model_id]=mcc
    print(model_id,',mcc:',mcc,',1s:',sum(pred_Y_0<=0.2),',cost',int(time.time()-t0))

X,Y = val1_X,val1_Y
votes1 = np.zeros([X.shape[0],len(models_single_type)])
model_mccs = np.zeros(len(models_single_type))
for model_id,model in enumerate(models_single_type):
    t0 = time.time()
    pred_Y = model.predict_proba(X)
    pred_Y_0 = pred_Y[:,0]
    votes1[:,model_id] = pred_Y_0
    mcc = matthews_corrcoef(Y , pred_Y_0<=0.2)
    model_mccs[model_id]=mcc
    print(model_id,',mcc:',mcc,',1s:',sum(pred_Y_0<=0.2),',cost',int(time.time()-t0))

del models_single_type,model_id,model,X,Y



#%%


    

 
#%%
from sklearn.ensemble import RandomForestClassifier
t_X, t_Y = votes0, val0_Y

class_weight = {}
class_weight[0] = 1000
class_weight[1] = 1000
for max_depth in range(50,100,10):
    print(max_depth,end='-->')
    forest_2nd = RandomForestClassifier(max_depth=max_depth,n_estimators=11,random_state=12)
    forest_2nd.fit(t_X, t_Y)
    y_pred= forest_2nd.predict(t_X)
    print('tr:',matthews_corrcoef(t_Y , y_pred),end=',')
#    y_pred= forest_2nd.predict(v_X)
#    print('val0:',matthews_corrcoef(v_Y , y_pred),end=',')
    y_pred= forest_2nd.predict(votes1)
    print('val1:',matthews_corrcoef(val1_Y , y_pred))

'''

'''
#%%
from sklearn.ensemble import RandomForestClassifier

t_X, t_Y = votes[:90000,:], val_Y[:90000]
v_X, v_Y = votes[90000:,:], val_Y[90000:]

max_depth=59
print(max_depth,end='-->')
forest_2nd = RandomForestClassifier(max_depth=max_depth,n_estimators=11,random_state=12)
forest_2nd.fit(t_X, t_Y)
y_pred= forest_2nd.predict(t_X)
print('tr:',matthews_corrcoef(t_Y , y_pred),end=',')
y_pred= forest_2nd.predict(v_X)
print('val:',matthews_corrcoef(v_Y , y_pred))

save_variable(forest_2nd,'forest_2nd_14.pkl')


#%%    
"""
 produce testing result
"""

max_chunk_size = 1000
col_cate_nu = 2140
col_numeric_nu = 969
col_date_nu = 1157


pip = read_variable('model_stats/pip_1110.pkl')

p0_imputer = pip['0_imputer']
p1_high_variance = pip['1_high_variance']
p2_kbest_cols = pip['2_kbest_cols']
p3_norm = pip['3_norm']


y_sum = np.zeros(1183748)
   
for chunk_id in range(1184):
    
    path = 'final/test/'+str(chunk_id)+'.pkl'
    if os.path.isfile(path):
        print(chunk_id,'exist')
    else:
        chunk_num = read_variable('data/test_numeric_chunks/chunk_'+str(chunk_id)+'.pkl')
        chunk_date = read_variable('data/test_date_chunks/chunk_'+str(chunk_id)+'.pkl')
        chunk_cate = read_variable('model_stats/test_cate_proba/'+str(chunk_id)+'.pkl')
        
        temp_X = np.zeros((chunk_cate.shape[0],col_cate_nu+col_numeric_nu+col_date_nu))
        temp_X[:,:col_cate_nu] = chunk_cate
        temp_X[:,col_cate_nu: col_cate_nu+col_numeric_nu] = chunk_num
        temp_X[:,col_cate_nu+col_numeric_nu:] = chunk_date
    
        pip_X = p3_norm.transform(p1_high_variance.transform(p0_imputer.transform(temp_X))[:,p2_kbest_cols])
    
        test_X = pip_X
    
        # predit
        votes = np.zeros([test_X.shape[0],len(model_forest)])
        bar = progressbar.ProgressBar()
        model_id = 0
        for model in bar(model_forest):
            t0 = time.time()
            pred_Y = model.predict_proba(test_X)
            pred_Y_0 = pred_Y[:,0]
            votes[:,model_id] = pred_Y_0
            model_id +=1
        y_pred= forest_2nd.predict(votes)
        print(chunk_id,'1s:',sum(y_pred))
        end_idx = (chunk_id+1)*1000
        if chunk_id == 1183: end_idx = chunk_id*1000+748
        save_variable(y_pred,path)

#%%
for chunk_id in range(1184):
    chunk_pred = read_variable('final/'+str(chunk_id)+'.pkl')
    print(chunk_id,int(sum(chunk_pred)))
#%%
y_pred = np.zeros(1183748)
bar = progressbar.ProgressBar()
for chunk_id in bar(range(1184)):
    chunk_pred = read_variable('final/'+str(chunk_id)+'.pkl')
    end_idx = (chunk_id+1)*1000
    if chunk_id == 1183: end_idx = chunk_id*1000+748
    y_pred[chunk_id*1000:end_idx] = chunk_pred

# saving to CSV
test_ids = read_variable('outputs/test_ids.pkl')
test_y_ids = pd.DataFrame(test_ids,columns=['Id'])
test_y_y = pd.DataFrame(y_pred,columns=['Response'])
test_y = pd.concat([test_y_ids,test_y_y],axis=1)
test_y = test_y.set_index('Id')
test_y.to_csv('submissions/submission_final.csv',float_format='%.0f')

print('1s:',np.sum(y_pred))
