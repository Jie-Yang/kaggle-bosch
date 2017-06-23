from utils import read_variable, save_variable
import numpy as np
import progressbar


col_numeric_nu = 968


def load_tr_chunk(group_ids):
    gp_X = np.zeros([0,col_numeric_nu])
    gp_Y = np.zeros(0)
    for group_id in group_ids:
        tr_chunk = read_variable('final/tr_groups/'+str(group_id))
        chunk_X = tr_chunk['x']
        chunk_Y = tr_chunk['y']

        gp_X = np.concatenate([gp_X,chunk_X])
        gp_Y = np.concatenate([gp_Y,chunk_Y])
    return gp_X, gp_Y

#%%
'''
16GB memory can not handle all training dataset, so only use subset of it
'''
pip_X, pip_Y = load_tr_chunk([1,11,37,67,83,101,131,157,170])

print('1s:',sum(pip_Y),',1s%:',sum(pip_Y)/len(pip_Y))
#%%
pip = {}

#%%
'''
Pre-process 0: replace NaN with median value
replace NaN with mean value
The median is a more robust estimator for data with high magnitude variables which could dominate results (otherwise known as a ‘long tail’).
'''
from sklearn.preprocessing import Imputer
import time

X = pip_X

t0 = time.time()
imputer = Imputer(missing_values='NaN', strategy='median', axis=0)
imputer.fit(X)
print('imputer:fit',time.time()-t0,'sec')
t1 = time.time()
x_imputed = imputer.transform(X)
print('imputer:transform',time.time()-t1,'sec')
print('imputer:total',time.time()-t0,'sec')

pip['0_imputer'] = imputer


#%%
'''
Pre-process 1: remove low variance cols
'''

from sklearn.feature_selection import VarianceThreshold
'''
threshold : float, optional
Features with a training-set variance lower than this threshold will be removed.
The default is to keep all features with non-zero variance, i.e. remove the features that have the same value in all samples.
'''
sel = VarianceThreshold()
x_high_variance = sel.fit_transform(x_imputed)

print('high variance cols:',x_high_variance.shape[1],'from',x_imputed.shape[1])
pip['1_high_variance'] = sel

#%%
'''
feature engineering: find the best K, regardless of the config of tree
'''
from sklearn.feature_selection import SelectKBest, f_classif

feature_engine = SelectKBest(f_classif, k='all')
print('SelectKBest...',end='')

t0 = time.time()
feature_engine = feature_engine.fit(x_high_variance,pip_Y)
print('cost',time.time()-t0)

feature_engine_pvalues = feature_engine.pvalues_
kbest_cols = []

# pvalue 0.05 or 0.1
pvalue_threshold = 0.05
for idx,pv in enumerate(feature_engine_pvalues):
    if pv > pvalue_threshold:
        kbest_cols.append(idx)

print('selected col[pvalue>'+str(pvalue_threshold)+']:',len(kbest_cols),'from',x_high_variance.shape[1])

pip['2_kbest_cols'] = kbest_cols

x_kbest = x_high_variance[:,kbest_cols]

#%%
'''
 many elements used in the objective function of a learning algorithm 
 (such as the RBF kernel of Support Vector Machines or the l1 and l2 regularizers of linear models) assume that all features are centered around zero and have variance in the same order. If a feature has a variance that is orders of magnitude larger than others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.

'''
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(x_kbest)
pip['3_norm'] = scaler
#%%

save_variable(pip,'final/feature_sel_pip')




