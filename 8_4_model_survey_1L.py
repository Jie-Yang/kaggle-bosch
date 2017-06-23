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

for model_folder in model_folders:
    print('Processing model cluster:',model_folder)

    model_ids = []
    model_stds = []
    model_means = []
    
    for root, dirs, files in os.walk(model_folder):
        bar = progressbar.ProgressBar()
        for model_idx in bar(files):
            model_path = os.path.join(root, model_idx)
            model = read_variable(model_path)
            #print(model_idx,'std:',model.val_std,',mean:',model.val_mean)
            model_stds.append(model.val_std)
            model_means.append(model.val_mean)
            model_ids.append(model_idx)
    
    #%
    plt.figure()
    plt.errorbar(model_ids, model_means, yerr=model_stds, fmt='o')
    plt.title(root)

#%%
model_folders = ['final/1L_tree','final/1L_ada','final/1L_gb','final/1L_mlp']

packs = []
for model_folder in model_folders:
    print('Processing model cluster:',model_folder)
    for root, dirs, files in os.walk(model_folder):
        bar = progressbar.ProgressBar()
        for model_idx in bar(files):
            model_path = os.path.join(root, model_idx)
            model = read_variable(model_path)
            
            pack = {}
            pack['root'] = root
            pack['chunk_id'] = model_idx
            pack['model'] = model
            
            packs.append(pack)

print('model loaded:',len(packs))



#%%

test_X, test_Y = load_pipped_test_chunk()

real_test_Y_1s_count = str(sum(test_Y))
for pack in packs:
    model = pack['model']
    test_Y_pred = model.predict(test_X)
    file_path = pack['root']+'_test_y_pred/'+str(pack['chunk_id'])
    save_variable(test_Y_pred,file_path)
    print(file_path,'-->',matthews_corrcoef(test_Y, test_Y_pred),',1s:',str(sum(test_Y_pred))+'/'+real_test_Y_1s_count)
    
#%%
model_folders = ['final/1L_tree','final/1L_ada','final/1L_gb','final/1L_mlp']

packs = []
for model_folder in model_folders:
    print('Processing model cluster:',model_folder)
    for root, dirs, files in os.walk(model_folder):
        bar = progressbar.ProgressBar()
        for model_idx in bar(files):
            model_path = os.path.join(root, model_idx)
            model = read_variable(model_path)
            
            pack = {}
            pack['root'] = root
            pack['chunk_id'] = model_idx
            pack['model'] = model
            
            packs.append(pack)

print('model loaded:',len(packs))


from sklearn.metrics import matthews_corrcoef
test_X, test_Y = load_pipped_test_chunk()
#%%

test_Y_sum = np.zeros(test_X.shape[0])

bar = progressbar.ProgressBar()
for pack in packs:
    file_path = pack['root']+'_test_y_pred/'+str(pack['chunk_id'])
    test_Y_pred = read_variable(file_path)
    print(file_path,sum(test_Y_pred))
    test_Y_sum += test_Y_pred
#%%
for votes_threshold in range(1,len(packs)):
    #votes_threshold = int(len(packs)/2)
    test_Y_pred = (test_Y_sum > votes_threshold).astype(np.int)

    print(votes_threshold,'2L test mcc:',matthews_corrcoef(test_Y, test_Y_pred),',1s:',sum(test_Y_pred))
    
#%%

#%%
test_Y_votes = np.zeros([test_X.shape[0],len(packs)])

bar = progressbar.ProgressBar()
for pack_id, pack in enumerate(packs):
    file_path = pack['root']+'_test_y_pred/'+str(pack['chunk_id'])
    test_Y_pred = read_variable(file_path)
    print(file_path,sum(test_Y_pred))
    test_Y_votes[:,pack_id]= test_Y_pred


#%%
tr_Y_votes_chunkids = [1,139,171]

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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
seeds=[13,11,193]

models_2L = []

skf = StratifiedKFold(n_splits=4)
k_fold_i = 0
for tr_idx, val_idx in skf.split(tr_Y_votes, tr_Y):
    tr_X_tr, tr_Y_tr = tr_Y_votes[tr_idx,:], tr_Y[tr_idx]
    tr_X_val, tr_Y_val = tr_Y_votes[val_idx,:], tr_Y[val_idx]
    for seed in seeds:
        model = RandomForestClassifier(random_state=seed)
        model.fit(tr_X_tr, tr_Y_tr)
        tr_Y_pred = model.predict(tr_X_tr)
        print(k_fold_i,seed,'2L tr:',matthews_corrcoef(tr_Y_tr, tr_Y_pred),end=',')
        val_Y_pred = model.predict(tr_X_val)
        print('val:',matthews_corrcoef(tr_Y_val, val_Y_pred))
        
        models_2L.append(model)
    k_fold_i +=1
    

#%% try on testing dataset
'''
check model on testing dataset
'''
from sklearn.metrics import confusion_matrix
test_Y_sum = np.zeros(len(test_Y))
for model2 in models_2L:
    test_Y_pred = model2.predict(test_Y_votes)
    print('2L test:',matthews_corrcoef(test_Y, test_Y_pred),',1s:',sum(test_Y_pred),'/',sum(test_Y))
    test_Y_sum += test_Y_pred

for t in range(len(models_2L)):
    test_Y_pred = (test_Y_sum>t).astype(np.int)
    print(t,matthews_corrcoef(test_Y, test_Y_pred),',1s:',sum(test_Y_pred),'/',sum(test_Y))
    print(confusion_matrix(test_Y, test_Y_pred))
    
    
#%%
