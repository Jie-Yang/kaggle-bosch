from utils import save_variable, read_variable
import numpy as np
import progressbar
import os

row_total = 1183747
#%%
'''
get 1% samples as testing dataset for the final testing

NOTE: use permutation to cover information through the all training dataset, 
and try to avoid information unbias between samples at different sequential positions
'''
idx = np.random.permutation(row_total)
tr_test_split = int(1183747*0.01)
test_row_idx = idx[:tr_test_split]
tr_row_idx = idx[tr_test_split:]

save_variable(tr_row_idx,'final/tr_row_idx')
save_variable(test_row_idx,'final/test_row_idx')

# check test dataset have the same distributions of outcomes as training (about 0.5% 1s)
count_1 = 0
bar = progressbar.ProgressBar()
for row_id in bar(test_row_idx):
    row_y = read_variable('data/train_y_rows/'+str(row_id)+'.pkl')
    count_1 += row_y.values

print()        
print('test 1s:',count_1,'(',count_1/len(test_row_idx),')')
#%%
'''
rebalance 1s and 0s in the rest of dataset
'''
'''
    step 1, find idx of all 1s
'''

tr_row_idx = read_variable('final/tr_row_idx')
tr_1s_idx = []
tr_0s_idx = []
bar = progressbar.ProgressBar()
for row_id in bar(tr_row_idx):
    row_y = read_variable('data/train_y_rows/'+str(row_id)+'.pkl')
    if row_y.values ==1: 
        tr_1s_idx.append(row_id)
    else:
        tr_0s_idx.append(row_id)
print('find 1s:',len(tr_1s_idx))
save_variable(tr_1s_idx,'final/tr_1s_idx')
#%%
'''
    step 2, create training groups with the same amount of 1s and 0s
'''
len_0s = len(tr_0s_idx)
len_1s = len(tr_1s_idx)
group_nu = 1+int(len_0s/len_1s)
tr_0s_groups = []
bar = progressbar.ProgressBar()
for tr_0s_group_id in bar(range(group_nu)):
    
    tr_0s_group = []
    for i in range(tr_0s_group_id,len_0s,group_nu):
        tr_0s_group.append(tr_0s_idx[i])
    tr_0s_groups.append(tr_0s_group)
save_variable(tr_0s_groups,'final/tr_0s_groups')

    
# check the result
group_0s_count = 0
tr_0s_groups = read_variable('final/tr_0s_groups')
for tr_0s_group in tr_0s_groups:
    group_0s_count += len(tr_0s_group)
    
print('is tr_0s_groups generated correct?',len(tr_0s_idx)==group_0s_count)


#%%



#%%
'''
generating training chunks which will reduce IO times for data batch processing
comparing with IOs of processing each row individually.

'''
col_numeric_nu = 968
tr_1s_idx= read_variable('final/tr_1s_idx')
tr_0s_groups = read_variable('final/tr_0s_groups')
def load_tr_XY(group_id):
    gp_0s_idx = tr_0s_groups[group_id]

    gp_idx = np.concatenate([tr_1s_idx,gp_0s_idx])
    gp_idx = np.random.permutation(gp_idx)
    
    gp_X = np.zeros([len(gp_idx),col_numeric_nu])
    gp_Y = np.zeros(len(gp_idx))
    bar = progressbar.ProgressBar()
    i = 0
    for row_id in bar(gp_idx):
        row_num = read_variable('data/train_numeric_rows/'+str(row_id)+'.pkl')
        gp_Y[i] = read_variable('data/train_y_rows/'+str(row_id)+'.pkl')
        gp_X[i,:] = row_num
        i +=1
        
    return gp_X, gp_Y
    
bar = progressbar.ProgressBar()
for group_id in range(len(tr_0s_groups)):
    file_path = 'final/tr_groups/'+str(group_id)
    if os.path.isfile(file_path):
        print('skip')
    else:
        gp_X, gp_Y = load_tr_XY(group_id)
        gp = {}
        gp['x'] = gp_X
        gp['y'] = gp_Y
        save_variable(gp,file_path)

#%%
'''
check all traing chunk has balanced 1s and 0s
'''

for root, dirs, files in os.walk('final/tr_groups'):
    for file in files:
        file_path = os.path.join(root, file)
        tr_chunk = read_variable(file_path)
        tr_chunk_Y = tr_chunk['y']
        print(file,'->1s:',sum(tr_chunk_Y),',1s%:',sum(tr_chunk_Y)/len(tr_chunk_Y))    

#%% 
'''
generate test chunk
'''        
test_row_idx = read_variable('final/test_row_idx')
gp_X = np.zeros([len(test_row_idx),col_numeric_nu])
gp_Y = np.zeros(len(test_row_idx))
bar = progressbar.ProgressBar()
i = 0
for row_id in bar(test_row_idx):
    row_num = read_variable('data/train_numeric_rows/'+str(row_id)+'.pkl')
    gp_Y[i] = read_variable('data/train_y_rows/'+str(row_id)+'.pkl')
    gp_X[i,:] = row_num
    i +=1
    
gp = {}
gp['x'] = gp_X
gp['y'] = gp_Y
save_variable(gp,'final/test_chunk')
print('test chunk->1s:',sum(gp_Y),',1s%:',sum(gp_Y)/len(gp_Y))  
#%%
'''
in the following data processing, 1L models can be built on a every combination 
of tr_1s_idx and tr_0s_group[?]
'''