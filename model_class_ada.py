import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
#%%
class AdaBoostClassifierWithSeeds:
    def __init__(self, seeds):
        self.seeds = seeds
    def fit(self, X, Y):
        self.models = []
        for seed in self.seeds:
            model = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=seed)
            model.fit(X,Y)
            self.models.append(model)
        self.votes_threshold = int(len(self.models)/2)
        return self
    def predict(self,X):
        Y_sum = np.zeros(X.shape[0])
        for model in self.models:
            Y_sum += model.predict(X)
                 
        return (Y_sum>self.votes_threshold).astype(np.int)
    def __str__(self):
        return "AdaBoostClassifierWithSeeds(seeds="+str(self.seeds)+',votes_threshold='+str(self.votes_threshold)+')'


#%%
class AdaBoostChunkClassifierWithKFolds:
    def __init__(self, k, seeds):
        self.k = k
        self.seeds = seeds
        self.votes_threshold = int(self.k/2)
    def fit(self, X, Y, test_X, test_Y):
        skf = StratifiedKFold(n_splits=self.k)
        self.chunk_models = []
        chunk_models_val_mcc = []
        k_fold_i = 0
        for tr_idx, val_idx in skf.split(X, Y):
            tr_X, tr_Y = X[tr_idx,:], Y[tr_idx]
            val_X, val_Y = X[val_idx,:], Y[val_idx]
            print('   kfold['+str(k_fold_i)+']',end='')

            model =  AdaBoostClassifierWithSeeds(seeds=self.seeds)	
            model.fit(tr_X,tr_Y)
            
            tr_Y_pred = model.predict(tr_X)
            tr_mcc = matthews_corrcoef(tr_Y , tr_Y_pred)
            print('tr:',tr_mcc,end=',')
            val_Y_pred = model.predict(val_X)
            val_mcc = matthews_corrcoef(val_Y , val_Y_pred)
            print('val:',val_mcc,end=',')
            test_Y_pred = model.predict(test_X)
            test_mcc = matthews_corrcoef(test_Y , test_Y_pred)
            print('test:',test_mcc,end='')
            
            self.chunk_models.append(model)
            chunk_models_val_mcc.append(val_mcc)
            k_fold_i += 1
            print()
        
        self.val_std = np.std(chunk_models_val_mcc)
        self.val_mean = np.mean(chunk_models_val_mcc)
        print('     val std:',self.val_std,',mean:',self.val_mean)

        return self
    def predict(self,X):
        Y_sum = np.zeros(X.shape[0])
        for model in self.chunk_models:
            Y_sum += model.predict(X)
                 
        return (Y_sum>self.votes_threshold).astype(np.int)
    def __str__(self):
        return "AdaBoostChunkClassifierWithKFolds(kfold="+str(self.k)+",seeds="+str(self.seeds)+',votes_threshold='+str(self.votes_threshold)+')'
    