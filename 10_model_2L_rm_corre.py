import os
import progressbar
import matplotlib.pyplot as plt
from utils import read_variable
from model_class_tree import RandomForestClassifierWithSeeds, ForestChunkClassifierWithKFolds
from model_class_ada import AdaBoostChunkClassifierWithKFolds, AdaBoostClassifierWithSeeds
from model_class_gb import GradientBoostingChunkClassifierWithKFolds, GradientBoostingClassifierWithSeeds
from model_class_mlp import MLPClassifierWithSeeds, MLPChunkClassifierWithKFolds

from utils import load_pipped_test_chunk,save_variable
import numpy as np
from sklearn.metrics import matthews_corrcoef

    
#%%
model_folders = ['final/1L_tree','final/1L_ada','final/1L_gb','final/1L_mlp']
# remove correlation between models by using models based on different datasets
model_idx_ranges = [range(43),range(43,86),range(86,129),range(129,172)]
#model_idx_ranges = [range(172),range(172),range(172),range(172)]

packs = []
model_idx_ranges_idx = 0
for model_folder in model_folders:
    print('Processing model cluster:',model_folder)
    for root, dirs, files in os.walk(model_folder):
        bar = progressbar.ProgressBar()
        for model_idx in bar(model_idx_ranges[model_idx_ranges_idx]):
            model_path = os.path.join(root, str(model_idx))
            model = read_variable(model_path)
            
            pack = {}
            pack['root'] = root
            pack['chunk_id'] = model_idx
            pack['model'] = model
            
            packs.append(pack)
    model_idx_ranges_idx += 1

print('model loaded:',len(packs))


test_X, test_Y = load_pipped_test_chunk()


#%%
'''
Read test dataset
'''
test_Y_votes = np.zeros([test_X.shape[0],len(packs)])

bar = progressbar.ProgressBar()
for pack_id, pack in enumerate(packs):
    file_path = pack['root']+'_test_y_pred/'+str(pack['chunk_id'])
    test_Y_pred = read_variable(file_path)
    print(file_path,sum(test_Y_pred))
    test_Y_votes[:,pack_id]= test_Y_pred


#%%
tr_Y_votes_chunkids = list(range(41))
tr_Y_votes_chunkids.append(139)
tr_Y_votes_chunkids.append(171)

tr_Y_votes = np.zeros([0,len(packs)])
tr_Y = np.zeros(0)


bar = progressbar.ProgressBar()

for chunk_idx in tr_Y_votes_chunkids:
    chunk_Y =read_variable('final/tr_groups/'+str(chunk_idx))['y']
    chunk_Y_votes = np.zeros([len(chunk_Y),len(packs)])
    
    for pack_id, pack in enumerate(packs):
        file_path = pack['root']+'_tr_y_pred/'+str(pack['chunk_id'])+'_'+str(chunk_idx)
        tr_Y_pred = read_variable(file_path)
        print(file_path,sum(tr_Y_pred))
        chunk_Y_votes[:,pack_id]= tr_Y_pred
         
    tr_Y_votes = np.concatenate([tr_Y_votes, chunk_Y_votes])
    tr_Y = np.concatenate([tr_Y, chunk_Y])
    
#%%
'''
Forward Selection of Level 1 models
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

best_mcc = 0
selected_model_idxs = []
remain_model_idxs = list(range(len(packs)))
overall_best_mcc = 0
seeds=[11]
while len(remain_model_idxs) > 0:
    best_mcc = 0
    best_idx_candidate = -1
    for idx in remain_model_idxs:
        temp_selected_model_idxs = selected_model_idxs + [idx]
        print(temp_selected_model_idxs,end=',')
        fold_k = 2
        skf = StratifiedKFold(n_splits=fold_k)
        k_fold_i = 0
        val_mcc_sum = 0
        test_mcc_sum = 0
        
        threshold = len(temp_selected_model_idxs)/2
        temp_X = tr_Y_votes[:,temp_selected_model_idxs]
        temp_Y = tr_Y
        for tr_idx, val_idx in skf.split(temp_X, temp_Y):
            tr_X_tr, tr_Y_tr = temp_X[tr_idx,:], temp_Y[tr_idx]
            tr_X_val, tr_Y_val = temp_X[val_idx,:], temp_Y[val_idx]
            for seed in seeds:
                #model = RandomForestClassifier(random_state=seed)
                #model.fit(tr_X_tr, tr_Y_tr)
                
                #print(k_fold_i,seed,'2L tr:',matthews_corrcoef(tr_Y_tr, tr_Y_pred),end=',')
                val_Y_pred = (np.sum(tr_X_val,axis=1)>threshold).astype(np.int)
                val_mcc = matthews_corrcoef(tr_Y_val, val_Y_pred)
                #print('val:',val_mcc)
                val_mcc_sum += val_mcc
                
                test_Y_pred = (np.sum(test_X[:,temp_selected_model_idxs],axis=1)>threshold).astype(np.int)
                test_mcc = matthews_corrcoef(test_Y, test_Y_pred)
                test_mcc_sum += test_mcc
            k_fold_i +=1
        val_mcc_mean = val_mcc_sum/(fold_k*len(seeds))
        print('average val CV mcc:',val_mcc_mean,end='')
        test_mcc_mean = test_mcc_sum/(fold_k*len(seeds))
        print(',test CV mcc:',test_mcc_mean,end='')
        
        if val_mcc_mean > best_mcc:
            best_mcc = val_mcc_mean
            best_idx_candidate = idx
            print('-->BEST',end='')
        print()
    
    print('selected idx:',best_idx_candidate,',mcc:',best_mcc)
    if overall_best_mcc >= best_mcc:
        print('no improvement by adding new model',best_idx_candidate,',pre_mcc:',overall_best_mcc , ',new_mcc:',best_mcc)
        break
    overall_best_mcc = best_mcc
    remain_model_idxs.remove(best_idx_candidate)
    selected_model_idxs.append(best_idx_candidate)
print('final selected model idxs:',selected_model_idxs)

#%%
model2 = []
temp_X = tr_Y_votes[:,selected_model_idxs]
temp_Y = tr_Y
for tr_idx, val_idx in skf.split(temp_X, temp_Y):
    tr_X_tr, tr_Y_tr = temp_X[tr_idx,:], temp_Y[tr_idx]
    tr_X_val, tr_Y_val = temp_X[val_idx,:], temp_Y[val_idx]
    for seed in seeds:
        model = RandomForestClassifier(random_state=seed)
        model.fit(tr_X_tr, tr_Y_tr)
        model2.append(model)
#%%
'''
try on test dataset
'''
temp_X = test_X[:,selected_model_idxs]
temp_Y = test_Y
for model in model2:
    temp_Y_pred = model.predict(temp_X)
    temp_mcc = matthews_corrcoef(temp_Y, temp_Y_pred)
    print('test mcc:',temp_mcc,sum(temp_Y_pred),'/',sum(temp_Y))
    
#%%
'''
OUTPUT

test mcc: -0.000743770969988 1.0
test mcc: -0.000743770969988 1.0
'''
