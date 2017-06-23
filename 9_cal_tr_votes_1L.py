import os
import progressbar
from utils import read_variable
from model_class_tree import RandomForestClassifierWithSeeds, ForestChunkClassifierWithKFolds
from model_class_ada import AdaBoostChunkClassifierWithKFolds, AdaBoostClassifierWithSeeds
from model_class_gb import GradientBoostingChunkClassifierWithKFolds, GradientBoostingClassifierWithSeeds
from model_class_mlp import MLPClassifierWithSeeds, MLPChunkClassifierWithKFolds


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
from utils import load_pipped_tr_chunk,save_variable
from sklearn.metrics import matthews_corrcoef

'''
WARNING: 41 chunks cost about 20 hrs to finish
'''
for chunk_idx in range(41):
    print('chunk',chunk_idx,end='...')
    chunk_X, chunk_Y = load_pipped_tr_chunk([chunk_idx])

    for pack in packs:
        model = pack['model']
        chunk_Y_pred = model.predict(chunk_X)
        file_path = pack['root']+'_tr_y_pred/'+str(pack['chunk_id'])+'_'+str(chunk_idx)
        save_variable(chunk_Y_pred,file_path)
        print(file_path,'-->',matthews_corrcoef(chunk_Y, chunk_Y_pred),',1s:',str(sum(chunk_Y_pred)))
