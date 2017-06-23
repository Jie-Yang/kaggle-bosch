import progressbar
import numpy as np
from utils import read_variable

#%%

def get_votes(rows_range):
    votes = np.zeros([len(rows_range),104])
    Y = np.zeros(len(rows_range))
    for i, row_id in enumerate(rows_range):
        votes[i,:] = read_variable('final/tr_votes_1L/'+str(row_id))
        Y[i] = read_variable('data/train_y_rows/'+str(row_id)+'.pkl')
    return votes, Y

def get_votes_large(rows_range):
    votes = np.zeros([len(rows_range),104])
    Y = np.zeros(len(rows_range))
    bar = progressbar.ProgressBar()
    i = 0
    for row_id in bar(rows_range):
        votes[i,:] = read_variable('final/tr_votes_1L/'+str(row_id))
        Y[i] = read_variable('data/train_y_rows/'+str(row_id)+'.pkl')
        i += 1
    return votes, Y
#%%
#all_test_ids = []
#for gp_idx in range(119):
#    row_group = read_variable('final/row_groups/'+str(gp_idx))
#    test_row_ids = row_group['test']
#    all_test_ids.extend(test_row_ids)
#
##%
#tr_rows_range = []
#val_rows_range = []
#test_rows_range = []
#
#ids = range(400000)
#ids_shuffled = np.random.permutation(ids)
#tr_val_ids = []
#bar = progressbar.ProgressBar()
#for i in bar(ids_shuffled):
#    if i in all_test_ids:
#        test_rows_range.append(i)
#    else:
#        tr_val_ids.append(i)
        
#%% Read all trainingd dataset
all_test_ids = []
tr_val_ids = []
bar = progressbar.ProgressBar()
for gp_idx in bar(range(119)):
    row_group = read_variable('final/row_groups/'+str(gp_idx))
    test_row_ids = row_group['test']
    all_test_ids.extend(test_row_ids)
    tr_row_ids = row_group['train']
    tr_val_ids.extend(tr_row_ids)

#%
tr_rows_range = []
val_rows_range = []
test_rows_range = all_test_ids

ids = range(len(tr_val_ids))
ids_shuffled = np.random.permutation(ids)

#%
tr_rows_range = tr_val_ids[:int(len(tr_val_ids)*0.9)]
val_rows_range = tr_val_ids[int(len(tr_val_ids)*0.9):]
#%%
tr_X, tr_Y = get_votes_large(tr_rows_range)
val_X, val_Y = get_votes_large(val_rows_range)
test_X, test_Y = get_votes_large(test_rows_range)

print('tr',sum(tr_Y))
print('val',sum(val_Y))
print('test',sum(test_Y))

'''
tr 5637.0
val 584.0
test 658.0
'''
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import matthews_corrcoef
import time
#%%
#%%
'''
RandomForestClassifier
'''
model = RandomForestClassifier(n_estimators = 100)
t0 = time.time()
model.fit(tr_X,tr_Y)
print(model)


tr_Y_pred = model.predict(tr_X)
print('tr mcc',matthews_corrcoef(tr_Y ,  tr_Y_pred),',',str(int(sum(tr_Y_pred)))+'/'+str(int(sum(tr_Y))))
val_Y_pred = model.predict(val_X)
print('val mcc',matthews_corrcoef(val_Y ,  val_Y_pred),',',str(int(sum(val_Y_pred)))+'/'+str(int(sum(val_Y))))
test_Y_pred = model.predict(test_X)
print('test mcc',matthews_corrcoef(test_Y ,  test_Y_pred),',',str(int(sum(test_Y_pred)))+'/'+str(int(sum(test_Y))))
print('cost',int(time.time()-t0),'sec')
'''
tr mcc 0.662095028116 , 4044/5637
val mcc 0.0684906061566 , 78/584
test mcc 0.0740118961185 , 216/658
cost 1084 sec
'''
rfc_tr_Y_pred = tr_Y_pred
rfc_val_Y_pred = val_Y_pred
rfc_test_Y_pred = test_Y_pred
#%%
'''
AdaBoostClassifier
'''
model = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
t0 = time.time()
model.fit(tr_X,tr_Y)
print(model)

tr_Y_pred = model.predict(tr_X)
print('tr mcc',matthews_corrcoef(tr_Y ,  tr_Y_pred),',',str(int(sum(tr_Y_pred)))+'/'+str(int(sum(tr_Y))))
val_Y_pred = model.predict(val_X)
print('val mcc',matthews_corrcoef(val_Y ,  val_Y_pred),',',str(int(sum(val_Y_pred)))+'/'+str(int(sum(val_Y))))
test_Y_pred = model.predict(test_X)
print('test mcc',matthews_corrcoef(test_Y ,  test_Y_pred),',',str(int(sum(test_Y_pred)))+'/'+str(int(sum(test_Y))))
print('cost',int(time.time()-t0),'sec')
'''
tr mcc 0.380128699224 , 1875/5637
val mcc -0.00128687302386 , 32/584
test mcc 0.0435735019425 , 105/658
cost 92 sec
'''
abc_tr_Y_pred = tr_Y_pred
abc_val_Y_pred = val_Y_pred
abc_test_Y_pred = test_Y_pred
#%%
'''
GradientBoostClassifer
'''
model = GradientBoostingClassifier(n_estimators=200)
t0 = time.time()
model.fit(tr_X,tr_Y)
print(model)

tr_Y_pred = model.predict(tr_X)
print('tr mcc',matthews_corrcoef(tr_Y ,  tr_Y_pred),',',str(int(sum(tr_Y_pred)))+'/'+str(int(sum(tr_Y))))
val_Y_pred = model.predict(val_X)
print('val mcc',matthews_corrcoef(val_Y ,  val_Y_pred),',',str(int(sum(val_Y_pred)))+'/'+str(int(sum(val_Y))))
test_Y_pred = model.predict(test_X)
print('test mcc',matthews_corrcoef(test_Y ,  test_Y_pred),',',str(int(sum(test_Y_pred)))+'/'+str(int(sum(test_Y))))
print('cost',int(time.time()-t0),'sec')
'''
tr mcc 0.659520889853 , 4418/5637
val mcc 0.0644919719458 , 99/584
test mcc 0.0709679701093 , 249/658
cost 878 sec
'''
gbc_tr_Y_pred = tr_Y_pred
gbc_val_Y_pred = val_Y_pred
gbc_test_Y_pred = test_Y_pred
#%%
'''
MLPClassifier
'''

model = MLPClassifier(solver='lbfgs',activation='logistic')
t0 = time.time()
model.fit(tr_X,tr_Y)
print(model)

tr_Y_pred = model.predict(tr_X)
print('tr mcc',matthews_corrcoef(tr_Y ,  tr_Y_pred),',',str(int(sum(tr_Y_pred)))+'/'+str(int(sum(tr_Y))))
val_Y_pred = model.predict(val_X)
print('val mcc',matthews_corrcoef(val_Y ,  val_Y_pred),',',str(int(sum(val_Y_pred)))+'/'+str(int(sum(val_Y))))
test_Y_pred = model.predict(test_X)
print('test mcc',matthews_corrcoef(test_Y ,  test_Y_pred),',',str(int(sum(test_Y_pred)))+'/'+str(int(sum(test_Y))))
print('cost',int(time.time()-t0),'sec')
'''
tr mcc 0.0 , 0/5637
val mcc 0.0 , 0/584
test mcc 0.0 , 0/658
cost 23 sec
'''
mlp_tr_Y_pred = tr_Y_pred
mlp_val_Y_pred = val_Y_pred
mlp_test_Y_pred = test_Y_pred

#%%
import warnings
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")

steps = 30000
for tr_id_start in range(0,len(tr_Y),steps):
    
    tr_id_end = tr_id_start+steps if tr_id_start+steps < len(tr_Y) else len(tr_Y)
    print('[',tr_id_start,tr_id_end,')')
    temp_tr_X, temp_tr_Y = tr_X[tr_id_start:tr_id_end,:],tr_Y[tr_id_start:tr_id_end]
    temp_test_X, temp_test_Y = test_X, test_Y
    
    
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    model_cv = []
    k_fold_i = 0
    kf_mccs = []
    for tr_idx, val_idx in skf.split(temp_tr_X, temp_tr_Y):
        kf_tr_X, kf_tr_Y = temp_tr_X[tr_idx,:], temp_tr_Y[tr_idx]
        kf_val_X, kf_val_Y = temp_tr_X[val_idx,:], temp_tr_Y[val_idx]
        print('   kfold['+str(k_fold_i)+']')

        for random_state in [1,2,3]:
            print('         seed['+str(random_state)+']',end='')
            model = GradientBoostingClassifier(n_estimators=200,random_state=random_state)
            t0 = time.time()
            model.fit(kf_tr_X,kf_tr_Y)
            tr_Y_pred = model.predict(kf_tr_X)
            tr_mcc = matthews_corrcoef(kf_tr_Y , tr_Y_pred)
            print('tr:',tr_mcc,end=',')
            val_Y_pred = model.predict(kf_val_X)
            val_mcc = matthews_corrcoef(kf_val_Y , val_Y_pred)
            print('val:',val_mcc,end=',')
            test_Y_pred = model.predict(temp_test_X)
            test_mcc = matthews_corrcoef(temp_test_Y , test_Y_pred)
            print('test:',test_mcc)
            
            kf_mccs.append(val_mcc)
        
        k_fold_i += 1
        
    print('KFOLD MCC STD:',np.std(kf_mccs))
    
    

#%%

steps = 30000
for tr_id_start in [270000]:
    
    tr_id_end = tr_id_start+steps if tr_id_start+steps < len(tr_Y) else len(tr_Y)
    print('[',tr_id_start,tr_id_end,')')
    temp_tr_X, temp_tr_Y = tr_X[tr_id_start:tr_id_end,:],tr_Y[tr_id_start:tr_id_end]
    temp_test_X, temp_test_Y = test_X, test_Y
    
    
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    model_cv = []
    k_fold_i = 0
    kf_mccs = []
    for tr_idx, val_idx in skf.split(temp_tr_X, temp_tr_Y):
        kf_tr_X, kf_tr_Y = temp_tr_X[tr_idx,:], temp_tr_Y[tr_idx]
        kf_val_X, kf_val_Y = temp_tr_X[val_idx,:], temp_tr_Y[val_idx]
        print('   kfold['+str(k_fold_i)+']')

        for random_state in [1,2,3]:
            print('         seed['+str(random_state)+']',end='')
            model = GradientBoostingClassifier(n_estimators=200,random_state=random_state)
            t0 = time.time()
            model.fit(kf_tr_X,kf_tr_Y)
            tr_Y_pred = model.predict(kf_tr_X)
            tr_mcc = matthews_corrcoef(kf_tr_Y , tr_Y_pred)
            print('tr:',tr_mcc,end=',')
            val_Y_pred = model.predict(kf_val_X)
            val_mcc = matthews_corrcoef(kf_val_Y , val_Y_pred)
            print('val:',val_mcc,end=',')
            test_Y_pred = model.predict(temp_test_X)
            test_mcc = matthews_corrcoef(temp_test_Y , test_Y_pred)
            print('test:',test_mcc)
            
            kf_mccs.append(val_mcc)
            break
        break
        k_fold_i += 1
        
    print('KFOLD MCC STD:',np.std(kf_mccs))
#%%
for random_state in [1,2,3]:
    model = GradientBoostingClassifier(n_estimators=200,random_state=random_state)
    t0 = time.time()
    model.fit(tr_X,tr_Y)
    #print(model)
    
    tr_Y_pred = model.predict(tr_X)
    print('tr mcc',matthews_corrcoef(tr_Y ,  tr_Y_pred),',',str(int(sum(tr_Y_pred)))+'/'+str(int(sum(tr_Y))))
    val_Y_pred = model.predict(val_X)
    print('val mcc',matthews_corrcoef(val_Y ,  val_Y_pred),',',str(int(sum(val_Y_pred)))+'/'+str(int(sum(val_Y))))
    test_Y_pred = model.predict(test_X)
    print('test mcc',matthews_corrcoef(test_Y ,  test_Y_pred),',',str(int(sum(test_Y_pred)))+'/'+str(int(sum(test_Y))))
    print('cost',int(time.time()-t0),'sec')
    
'''
tr mcc 0.659520889853 , 4418/5637
val mcc 0.0644919719458 , 99/584
test mcc 0.0709679701093 , 249/658
cost 905 sec

tr mcc 0.659520889853 , 4418/5637
val mcc 0.0644919719458 , 99/584
test mcc 0.0709679701093 , 249/658
cost 879 sec

tr mcc 0.659520889853 , 4418/5637
val mcc 0.0644919719458 , 99/584
test mcc 0.0709679701093 , 249/658
cost 882 sec
'''
#%%

a = model.feature_importances_
