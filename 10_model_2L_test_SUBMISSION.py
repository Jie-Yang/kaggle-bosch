
import numpy as np

from utils import read_variable


test_Y = np.zeros(1183748)
row_start = 0
row_end = 0
for chunk_id in range(1184):
    
    path = 'final/test_votes_1L/'+str(chunk_id)
    votes = read_variable(path)

    chunk_Y_pred = (np.sum(votes,axis=1)>=1).astype(np.int)
    print(chunk_id,'1s:',sum(chunk_Y_pred))
    
    row_end = row_start+len(chunk_Y_pred)

    test_Y[row_start:row_end] = chunk_Y_pred
    row_start = row_end
    
print('FINAL 1s:',sum(test_Y))
#%%
import pandas as pd
# saving to CSV
test_ids = read_variable('outputs/test_ids.pkl')
test_y_ids = pd.DataFrame(test_ids,columns=['Id'])
test_y_y = pd.DataFrame(test_Y,columns=['Response'])
test_y = pd.concat([test_y_ids,test_y_y],axis=1)
test_y = test_y.set_index('Id')
test_y.to_csv('submissions/submission_1130.csv',float_format='%.0f')


