import utils
import pandas as pd
import numpy as np
import progressbar
from sklearn import tree
from sklearn.metrics import matthews_corrcoef

train_row_nu = 1183747
chunk_nu = 50
max_chunk_size = 1000


'''
proba of categorical cols
'''
column_names = utils.read_variable('outputs/column_names.pkl')
cols_cate = column_names['categorical']
votes_cate = np.zeros((train_row_nu, cols_cate.size))
bar = progressbar.ProgressBar()
print('loading cate proba...')
for chunk_id in bar(range(0,chunk_nu,1)):
    chunk = utils.read_variable('model_stats/train_cate_proba/'+str(chunk_id)+'.pkl')
    votes_cate[chunk_id*max_chunk_size:chunk_id*max_chunk_size+chunk.shape[0],:] = chunk
del chunk_id, bar, chunk

#%
responses = utils.read_variable('outputs/responses.pkl').astype(int)
#%%

x_tr = votes_cate[:45000,:]
y_tr = responses[:45000]

x_val = votes_cate[45000:50000,:]
y_val = responses[45000:50000]

#%%

for i in range(1000,2000,1):

    if y_tr.iloc[i]['Response']==1:
        a2 = votes_cate[i,:]
        break
a0= votes_cate[1,:]

#%%
import matplotlib.pyplot as plt
t = np.arange(0, 2140, 1)
plt.plot(a1,a2)

plt.ylim([0.9940,0.9950])
plt.xlim([0.9940,0.9950])

plt.show()

#%%
for i in range(0,len(a0),1):
    if a1[i] != a2[i]:
        print(i, a1[i],a2[i])

#%%
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
model = SelectKBest( k=1).fit(x_tr, y_tr)
a = model.scores_

feature_selected = []
for i in range(0,len(a),1):
    if a[i]>1:
        feature_selected.append(i)        
print(model.scores_)

#%%

min_samples_leaf =1
max_depth=2


# training 0
x = x_tr[:,feature_selected]
y = y_tr[:]
# overfitting problem solution: smaller max_depth or larger min_sample_leaf
print(max_depth,min_samples_leaf, end='->')
print('model 0:','training...',end='')
tree_votes_0 = tree.DecisionTreeClassifier().fit(x,y)
y_pred = tree_votes_0.predict(x)
mcc = matthews_corrcoef(y, y_pred) 
print('mcc:',mcc,end='')
#utils.save_variable(tree_votes_0,'models/tree_votes_0.pkl')
print(',val...',end='')
y_pred = tree_votes_0.predict(x_val[:,feature_selected])
mcc = matthews_corrcoef(y_val, y_pred) 
print('mcc:',mcc)

