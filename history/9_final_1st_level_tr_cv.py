
import progressbar
import numpy as np
import time, os
from utils import read_variable, save_variable,load_pipped_tr_chunks
from sklearn.metrics import matthews_corrcoef
#%%
def gp_model_predit(model_cv,gp_X):
    gp_Y_pred_votes_count = np.zeros(gp_X.shape[0])
    for model in model_cv:
        gp_Y_pred = model.predict(gp_X)
        gp_Y_pred_votes_count += gp_Y_pred
        
    count_threshold=1
    gp_Y_pred_2nd = (gp_Y_pred_votes_count>count_threshold).astype(np.int)
    return gp_Y_pred_2nd

#%%

print('loading trees...')
model_forest = []
bar = progressbar.ProgressBar()
for model_id in bar(range(0,301,1)):
    model = read_variable('final/model_1L_cv/'+str(model_id))
    model_forest.append(model)

#%%

for chunk_id in range(1184):
    
    path = 'final/tr_votes/'+str(chunk_id)+'.pkl'
    print('checking chunk:',chunk_id,end='...')
    if os.path.isfile(path):
        print('exist')
    else:
        save_variable({},path)
        print('processing',end='...')
        chunk_X, chunk_Y = load_pipped_tr_chunks([chunk_id])
    
        # predit
        votes = np.zeros([chunk_X.shape[0],len(model_forest)])
        bar = progressbar.ProgressBar()
        model_id = 0
        for model in bar(model_forest):
            t0 = time.time()
            pred_Y = gp_model_predit(chunk_X)
            print(matthews_corrcoef(chunk_Y , pred_Y),end=',')
            votes[:,model_id] = pred_Y
            model_id +=1
        save_variable(votes,path)
        print('saved to',path)

