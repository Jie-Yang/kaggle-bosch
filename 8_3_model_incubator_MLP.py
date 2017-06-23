

from utils import load_pipped_tr_chunk,load_pipped_test_chunk,read_variable,save_variable
import numpy as np
from sklearn.metrics import matthews_corrcoef
from model_class_mlp import MLPClassifierWithSeeds, MLPChunkClassifierWithKFolds


test_X, test_Y = load_pipped_test_chunk()


#%%

#%%

'''
Model: Tree
'''

'''
Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the dataset prior to fitting with the decision tree.
ref:http://scikit-learn.org/stable/modules/tree.html
'''
from sklearn.model_selection import StratifiedKFold
'''
StratifiedKFold is a variation of k-fold which returns stratified folds:
each set contains approximately the same percentage of samples of each target
class as the complete set.
'''
import os

for root, dirs, files in os.walk('final/tr_groups'):
    for chunk_idx in files:
        print('chunk',chunk_idx,end='...')
        chunk_path = os.path.join(root, chunk_idx)
        tr_chunk = read_variable(chunk_path)
        model_path = 'final/1L_mlp/'+str(chunk_idx) 
        if os.path.isfile(model_path):
            print('model exist')
        else:
            save_variable({},model_path)
            print('processing...')

            chunk_X, chunk_Y = load_pipped_tr_chunk([chunk_idx])
            
            # based on experiment, k=3 give the smallest STD
            chunk_model = MLPChunkClassifierWithKFolds(k=3,seeds=[13,11,193])
            chunk_model.fit(chunk_X, chunk_Y)
            chunk_Y_pred = chunk_model.predict(chunk_X)
            chunk_mcc = matthews_corrcoef(chunk_Y , chunk_Y_pred)
            print('OVERALL tr:',chunk_mcc,end='')
            test_Y_pred = chunk_model.predict(test_X)
            test_mcc = matthews_corrcoef(test_Y , test_Y_pred)
            print(',test:',test_mcc,end='-->')
            

            save_variable(chunk_model,model_path)
            print('[save]')

    
                

#%%
'''
check STD and MEAN
'''
for root, dirs, files in os.walk('final/1L_mlp'):
    for model_idx in files:
        model = read_variable('final/1L_mlp/'+str(model_idx) )
        print(model_idx,'std:',model.val_std,',mean:',model.val_mean)
       
