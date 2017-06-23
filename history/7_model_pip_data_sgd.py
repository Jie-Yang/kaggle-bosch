
import time
from utils import load_training_subset_1110,read_variable,save_variable
import numpy as np
from sklearn.metrics import matthews_corrcoef


#%%
val_X,val_Y = load_training_subset_1110(range(1000,1010,1))


tr_X_1s = read_variable('model_stats/tr_pip_data_1s_1110.pkl')

#%%
'''
Model: SGD
'''
from sklearn.linear_model import SGDClassifier


len_1s = tr_X_1s.shape[0]

for set_id in range(0,166,1):

    chunk_range = range(set_id,1000,166)
    t_X,t_Y = load_training_subset_1110(chunk_range)
    tr_X = np.concatenate([t_X,tr_X_1s])
    tr_Y = np.concatenate([t_Y,np.ones(len_1s)])


    alpha = 1e-4 # default
    #‘none’, ‘l2’, ‘l1’, or ‘elasticnet’
    penalty='l1'
    model = SGDClassifier(alpha=alpha,shuffle=True,n_jobs=3,penalty=penalty)
    t0 = time.time()
    model = model.fit(tr_X,tr_Y)
    tr_Y_pred = model.predict(tr_X)
    best_tr_mcc = matthews_corrcoef(tr_Y , tr_Y_pred)
    print(set_id,'sgd:',',tr:',best_tr_mcc)
    best_model = model
    for round_id in range(20):
        model = SGDClassifier(alpha=alpha,shuffle=True,n_jobs=3,penalty=penalty)
        t0 = time.time()
        model = model.fit(tr_X,tr_Y)
        tr_Y_pred = model.predict(tr_X)
        tr_mcc = matthews_corrcoef(tr_Y , tr_Y_pred)
        print(set_id,'/',round_id,'sgd:',',tr:',tr_mcc,end='')
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
    print(set_id,'best sgd:',',tr:',best_tr_mcc,end='')
    print(',val:',end='')
    val_Y_pred = best_model.predict(val_X)
    print(matthews_corrcoef(val_Y, val_Y_pred))


    break
    save_variable(best_model,'7/sgd_'+str(set_id)+'.pkl')

#%%
'''

No matter what config, SGD give really bad pred even on training.

0 / 1 sgd: ,tr: 0.0130409009077,val:-0.00894843344066 ( 3325 )
0 / 2 sgd: ,tr: -0.0126166788456,val:-0.00494400796572 ( 3414 )
0 / 3 sgd: ,tr: 0.0210647425353,val:-0.00608883260096 ( 2121 )
0 / 4 sgd: ,tr: 0.0409474491906,val:0.00673896161817 ( 1048 )
0 / 5 sgd: ,tr: 0.0178420591587,val:-0.00513487374047 ( 2921 )
0 / 6 sgd: ,tr: 0.0812392407412,val:0.0155743265752 ( 4745 )<---best
0 / 7 sgd: ,tr: 0.0667857466617,val:0.0115874549738 ( 1881 )
0 / 8 sgd: ,tr: 0.0161787956102,val:-0.00146447562307 ( 2706 )
0 / 9 sgd: ,tr: 0.0362361124868,val:-0.0127457101921 ( 2143 )
0 / 10 sgd: ,tr: -0.000802056866675,val:-0.000745157056783 ( 8233 )
0 / 11 sgd: ,tr: 0.0401933579907,val:0.00931908809466 ( 5974 )
0 / 12 sgd: ,tr: 0.039276842256,val:-0.0110381829447 ( 2584 )
0 / 13 sgd: ,tr: -0.0223510367086,val:-0.0131685046733 ( 7898 )
0 / 14 sgd: ,tr: 0.0915034786479,val:-0.00615669975639 ( 3152 )<---best
0 / 15 sgd: ,tr: 0.0437200637915,val:-0.00421107256128 ( 3034 )
0 / 16 sgd: ,tr: 0.032846353564,val:0.017828289759 ( 4443 )
0 / 17 sgd: ,tr: 0.0877407301394,val:0.0090269913826 ( 7361 )
0 / 18 sgd: ,tr: 0.0577555689646,val:-0.00489172034875 ( 3578 )
0 / 19 sgd: ,tr: 0.0300959160778,val:0.000297348176367 ( 1462 )
'''