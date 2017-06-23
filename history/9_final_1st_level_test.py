
import progressbar
import numpy as np
import time, os
from utils import read_variable, save_variable,load_pipped_test_chunks


print('loading trees...')
model_forest = []
bar = progressbar.ProgressBar()
for model_id in bar(range(0,301,1)):
    model = read_variable('final/good_models/'+str(model_id))
    model_forest.append(model)

#%%

for chunk_id in range(1184):
    
    path = 'final/test_votes/'+str(chunk_id)+'.pkl'
    print('checking chunk:',chunk_id,end='...')
    if os.path.isfile(path):
        print('exist')
    else:
        save_variable({},path)
        print('processing',end='...')
        chunk_X = load_pipped_test_chunks([chunk_id])
    
        # predit
        votes = np.zeros([chunk_X.shape[0],len(model_forest)])
        bar = progressbar.ProgressBar()
        model_id = 0
        for model in bar(model_forest):
            t0 = time.time()
            pred_Y = model.predict_proba(chunk_X)
            pred_Y_0 = pred_Y[:,0]
            votes[:,model_id] = pred_Y_0
            model_id +=1
        save_variable(votes,path)
        print('saved to',path)
