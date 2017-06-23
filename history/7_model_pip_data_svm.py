
import time
from utils import load_training_subset_1108,read_variable,save_variable
import numpy as np
from sklearn.metrics import matthews_corrcoef

#%%
val_X,val_Y = load_training_subset_1108(range(1000,1002,1))


tr_X_1s = read_variable('model_stats/tr_pip_data_1s_1108.pkl')

#%%
'''
Model: SVM

The implementation is based on libsvm. 
The fit time complexity is more than quadratic with the number of samples which
 makes it hard to scale to dataset with more than a couple of 10000 samples.
'''
from sklearn import svm


len_1s = tr_X_1s.shape[0]

for set_id in range(0,166,1):

    chunk_range = range(set_id,1000,166)
    t_X,t_Y = load_training_subset_1108(chunk_range)
    tr_X = np.concatenate([t_X,tr_X_1s])
    tr_Y = np.concatenate([t_Y,np.ones(len_1s)])
    
    model = svm.SVC()
    t0 = time.time()
    model = model.fit(tr_X,tr_Y)
    tr_Y_pred = model.predict(tr_X)
    best_tr_mcc = matthews_corrcoef(tr_Y , tr_Y_pred)
    print(set_id,'sgd:',',tr:',best_tr_mcc,end='')
    best_model = model
    for round_id in range(20):
        model = svm.SVC()
        t0 = time.time()
        model = model.fit(tr_X,tr_Y)
        tr_Y_pred = model.predict(tr_X)
        tr_mcc = matthews_corrcoef(tr_Y , tr_Y_pred)
        print(set_id,'/',round_id,'svm:',',tr:',tr_mcc,end='')
        print(',val:',end='')
        val_Y_pred = model.predict(val_X)
        print(matthews_corrcoef(val_Y, val_Y_pred),'(',int(sum(val_Y_pred)),')',end='')

        if tr_mcc > best_tr_mcc:
            best_model = model
            best_tr_mcc = tr_mcc
            print('<---best')
        else:
            print()
        if best_tr_mcc > 0.9:
            break;
    tr_Y_pred = best_model.predict(tr_X)
    best_tr_mcc = matthews_corrcoef(tr_Y , tr_Y_pred)
    print(set_id,'best svm:',',tr:',best_tr_mcc,end='')
    print(',val:',end='')
    val_Y_pred = best_model.predict(val_X)
    print(matthews_corrcoef(val_Y, val_Y_pred),end='')
    print(',cost:',int(time.time()-t0),'sec')


    break
    save_variable(best_model,'7/svm_'+str(set_id)+'.pkl')

#%%
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
len_1s = tr_X_1s.shape[0]

set_id=0

chunk_range = range(set_id,1000,166)
t_X,t_Y = load_training_subset_1108(chunk_range)

tr_X = np.concatenate([t_X,tr_X_1s])
tr_Y = np.concatenate([t_Y,np.ones(len_1s)])

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(tr_X, tr_Y)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

