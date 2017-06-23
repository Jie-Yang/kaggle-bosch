import progressbar
import numpy as np
from utils import read_variable

#%%

def get_votes(rows_range):
    votes = np.zeros([len(rows_range),104])
    Y = np.zeros(len(rows_range))
    for i, row_id in enumerate(rows_range):
        votes[i,:] = read_variable('final/tr_votes_1L/'+str(row_id))
        Y[i] = read_variable('data/train_y_rows/'+str(row_id)+'.pkl')
    return votes, Y

#%%
all_test_ids = []
for gp_idx in range(119):
    row_group = read_variable('final/row_groups/'+str(gp_idx))
    test_row_ids = row_group['test']
    all_test_ids.extend(test_row_ids)

#%%
tr_rows_range = []
val_rows_range = []
test_rows_range = []

ids = range(28000)
ids_shuffled = np.random.permutation(ids)
tr_val_ids = []
bar = progressbar.ProgressBar()
for i in bar(ids_shuffled):
    if i in all_test_ids:
        test_rows_range.append(i)
    else:
        tr_val_ids.append(i)
    
tr_rows_range = tr_val_ids[:int(len(tr_val_ids)*0.9)]
val_rows_range = tr_val_ids[int(len(tr_val_ids)*0.9):]

tr_X, tr_Y = get_votes(tr_rows_range)
val_X, val_Y = get_votes(val_rows_range)
test_X, test_Y = get_votes(test_rows_range)

print('tr',sum(tr_Y))
print('val',sum(val_Y))
print('test',sum(test_Y))

#%%

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import matthews_corrcoef

#%%
'''
SIMPLE VOTES MODEL
'''
class SimpleVoteModel:
    vote_threshold = 1
    
    def fit(self, X, Y):
        return self
    def predict(self,votes):
        return (np.sum(votes,axis=1)>=self.vote_threshold).astype(np.int)
    def __str__(self):
        return "SimpleVoteModel(vote_threshold="+str(self.vote_threshold)+')'

        
model = SimpleVoteModel()
model.fit(tr_X,tr_Y)
print(model)

tr_Y_pred = model.predict(tr_X)
print('tr mcc',matthews_corrcoef(tr_Y ,  tr_Y_pred),',',str(int(sum(tr_Y_pred)))+'/'+str(int(sum(tr_Y))))
val_Y_pred = model.predict(val_X)
print('val mcc',matthews_corrcoef(val_Y ,  val_Y_pred),',',str(int(sum(val_Y_pred)))+'/'+str(int(sum(val_Y))))
test_Y_pred = model.predict(test_X)
print('test mcc',matthews_corrcoef(test_Y ,  test_Y_pred),',',str(int(sum(test_Y_pred)))+'/'+str(int(sum(test_Y))))
'''
SimpleVoteModel(vote_threshold=1)
tr mcc 0.404491673505 , 397/119
val mcc 0.295791062959 , 55/16
test mcc 0.131001784943 , 39/21

SimpleVoteModel(vote_threshold=2)
tr mcc 0.392090845463 , 52/119
val mcc 0.280553343263 , 7/16
test mcc 0.215921852961 , 4/21
'''

svm_tr_Y_pred = tr_Y_pred
svm_val_Y_pred = val_Y_pred
svm_test_Y_pred = test_Y_pred
#%%
'''
RandomForestClassifier
'''
model = RandomForestClassifier(n_estimators = 100)
model.fit(tr_X,tr_Y)
print(model)

tr_Y_pred = model.predict(tr_X)
print('tr mcc',matthews_corrcoef(tr_Y ,  tr_Y_pred),',',str(int(sum(tr_Y_pred)))+'/'+str(int(sum(tr_Y))))
val_Y_pred = model.predict(val_X)
print('val mcc',matthews_corrcoef(val_Y ,  val_Y_pred),',',str(int(sum(val_Y_pred)))+'/'+str(int(sum(val_Y))))
test_Y_pred = model.predict(test_X)
print('test mcc',matthews_corrcoef(test_Y ,  test_Y_pred),',',str(int(sum(test_Y_pred)))+'/'+str(int(sum(test_Y))))
'''
tr mcc 0.775122145005 , 108/119
val mcc 0.647617806267 , 12/16
test mcc 0.272378608562 , 10/21

Notes: no difference in output when use different n_estimators in [10,100,1000]
'''
rfc_tr_Y_pred = tr_Y_pred
rfc_val_Y_pred = val_Y_pred
rfc_test_Y_pred = test_Y_pred
#%%
'''
AdaBoostClassifier
'''
model = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
model.fit(tr_X,tr_Y)
print(model)

tr_Y_pred = model.predict(tr_X)
print('tr mcc',matthews_corrcoef(tr_Y ,  tr_Y_pred),',',str(int(sum(tr_Y_pred)))+'/'+str(int(sum(tr_Y))))
val_Y_pred = model.predict(val_X)
print('val mcc',matthews_corrcoef(val_Y ,  val_Y_pred),',',str(int(sum(val_Y_pred)))+'/'+str(int(sum(val_Y))))
test_Y_pred = model.predict(test_X)
print('test mcc',matthews_corrcoef(test_Y ,  test_Y_pred),',',str(int(sum(test_Y_pred)))+'/'+str(int(sum(test_Y))))
'''
tr mcc 0.773511547384 , 106/119
val mcc 0.647617806267 , 12/16
test mcc 0.272378608562 , 10/21
'''
abc_tr_Y_pred = tr_Y_pred
abc_val_Y_pred = val_Y_pred
abc_test_Y_pred = test_Y_pred
#%%
'''
GradientBoostClassifer
'''
model = GradientBoostingClassifier(n_estimators=200)
model.fit(tr_X,tr_Y)
print(model)

tr_Y_pred = model.predict(tr_X)
print('tr mcc',matthews_corrcoef(tr_Y ,  tr_Y_pred),',',str(int(sum(tr_Y_pred)))+'/'+str(int(sum(tr_Y))))
val_Y_pred = model.predict(val_X)
print('val mcc',matthews_corrcoef(val_Y ,  val_Y_pred),',',str(int(sum(val_Y_pred)))+'/'+str(int(sum(val_Y))))
test_Y_pred = model.predict(test_X)
print('test mcc',matthews_corrcoef(test_Y ,  test_Y_pred),',',str(int(sum(test_Y_pred)))+'/'+str(int(sum(test_Y))))
'''
tr mcc 0.696303521215 , 92/119
val mcc 0.630689348394 , 10/16
test mcc 0.228025757503 , 8/21
'''
gbc_tr_Y_pred = tr_Y_pred
gbc_val_Y_pred = val_Y_pred
gbc_test_Y_pred = test_Y_pred
#%%
'''
MLPClassifier
'''

model = MLPClassifier(solver='lbfgs',activation='logistic')
model.fit(tr_X,tr_Y)
print(model)

tr_Y_pred = model.predict(tr_X)
print('tr mcc',matthews_corrcoef(tr_Y ,  tr_Y_pred),',',str(int(sum(tr_Y_pred)))+'/'+str(int(sum(tr_Y))))
val_Y_pred = model.predict(val_X)
print('val mcc',matthews_corrcoef(val_Y ,  val_Y_pred),',',str(int(sum(val_Y_pred)))+'/'+str(int(sum(val_Y))))
test_Y_pred = model.predict(test_X)
print('test mcc',matthews_corrcoef(test_Y ,  test_Y_pred),',',str(int(sum(test_Y_pred)))+'/'+str(int(sum(test_Y))))
'''
tr mcc 0.775122145005 , 108/119
val mcc 0.647617806267 , 12/16
test mcc 0.272378608562 , 10/21
'''
mlp_tr_Y_pred = tr_Y_pred
mlp_val_Y_pred = val_Y_pred
mlp_test_Y_pred = test_Y_pred


#%%
'''
3L model

Observation: 3L modeling give no improvement on prediction
'''
tr_votes_2L = np.zeros([len(mlp_tr_Y_pred),5])
tr_votes_2L[:,0] = svm_tr_Y_pred
tr_votes_2L[:,1] = rfc_tr_Y_pred
tr_votes_2L[:,2] = abc_tr_Y_pred
tr_votes_2L[:,3] = gbc_tr_Y_pred
tr_votes_2L[:,4] = mlp_tr_Y_pred

val_votes_2L = np.zeros([len(mlp_val_Y_pred),5])
val_votes_2L[:,0] = svm_val_Y_pred
val_votes_2L[:,1] = rfc_val_Y_pred
val_votes_2L[:,2] = abc_val_Y_pred
val_votes_2L[:,3] = gbc_val_Y_pred
val_votes_2L[:,4] = mlp_val_Y_pred

test_votes_2L = np.zeros([len(mlp_test_Y_pred),5])
test_votes_2L[:,0] = svm_test_Y_pred
test_votes_2L[:,1] = rfc_test_Y_pred
test_votes_2L[:,2] = abc_test_Y_pred
test_votes_2L[:,3] = gbc_test_Y_pred
test_votes_2L[:,4] = mlp_test_Y_pred

#%%
model =RandomForestClassifier(n_estimators = 100)
model.fit(tr_votes_2L,tr_Y)
tr_Y_pred = model.predict(tr_votes_2L)
print(model)
print('tr mcc',matthews_corrcoef(tr_Y ,  tr_Y_pred))
val_Y_pred = model.predict(val_votes_2L)
print('val mcc',matthews_corrcoef(val_Y ,  val_Y_pred))
test_Y_pred = model.predict(test_votes_2L)
print('test mcc',matthews_corrcoef(test_Y ,  test_Y_pred))
'''
tr mcc 0.773511547384
val mcc 0.647617806267
test mcc 0.272378608562
'''
