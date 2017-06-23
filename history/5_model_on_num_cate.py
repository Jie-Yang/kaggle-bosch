import utils
import pandas as pd
import numpy as np
import progressbar
from sklearn import tree
from sklearn.metrics import matthews_corrcoef

train_row_nu = 1183747
chunk_nu = 10#1184
max_chunk_size = 1000
col_numeric_nu = 969
col_cate_nu = 2140



'''
proba of categorical cols
'''

x_cate_num_len = chunk_nu*max_chunk_size
if x_cate_num_len > train_row_nu : x_cate_num_len = train_row_nu
x_cate_num = np.zeros((x_cate_num_len, col_cate_nu+col_numeric_nu))
y_cate_num  = np.zeros(x_cate_num_len)
bar = progressbar.ProgressBar()
print('loading cate proba and raw num...')

chunks_num = pd.read_csv('data/train_numeric.csv',chunksize=max_chunk_size, low_memory=False,iterator=True)
for chunk_id in bar(range(0,chunk_nu,1)):
    
    chunk_cate = utils.read_variable('model_stats/train_cate_proba/'+str(chunk_id)+'.pkl')
    chunk_num = chunks_num.get_chunk()
    row_range = range(chunk_id*max_chunk_size,chunk_id*max_chunk_size+chunk_cate.shape[0],1)
    x_cate_num[row_range,:col_cate_nu] = chunk_cate
    x_cate_num[row_range,col_cate_nu:] = chunk_num.drop(['Response'],axis=1)
    y_cate_num [row_range] =  chunk_num['Response']

del chunk_id, bar, chunk_num, chunk_cate, row_range

#%% read all response 1 data

x_cate_num_1s = utils.read_variable('model_stats/x_cate_proba_num_1s.pkl')

#%% remove low density col

x = x_cate_num_1s
x_len = x.shape[0]

is_nan = np.isnan(x)
nan_count = np.sum(is_nan.astype(np.int),axis=0)
selected_cols = []

nan_ratio_threshold = 0.1
for idx,c in enumerate(nan_count):
    if c/x_len < nan_ratio_threshold:
        selected_cols.append(idx)

print('selected col:',len(selected_cols))
x_hd_mix = x_cate_num[:,selected_cols]
x_hd_1 = x_cate_num_1s[:,selected_cols]

#%% replace NaN with mean value
from sklearn.preprocessing import Imputer
import time
x= np.concatenate([x_hd_mix,x_hd_1])
t0 = time.time()
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer.fit(x)
print('imputer:fit',time.time()-t0,'sec')
t1 = time.time()
x_imputed_mix = imputer.transform(x_hd_mix)
x_imputed_1 = imputer.transform(x_hd_1)
print('imputer:transform',time.time()-t1,'sec')
print('imputer:total',time.time()-t0,'sec')

del t0,t1

#%%
'''
generate tr and val
'''
tr_ratio = 0.9

tr_val_cutoff = int(x_imputed_mix.shape[0]*tr_ratio)
x_tr = np.concatenate([x_imputed_mix[:tr_val_cutoff,:],x_imputed_1[:6780]])
y_tr = np.concatenate([y_cate_num[:tr_val_cutoff],np.ones(6780)])

x_euqal01 = np.concatenate([x_imputed_mix[:x_imputed_1.shape[0],:],x_imputed_1])
y_equal01 = np.concatenate([y_cate_num[:x_imputed_1.shape[0]],np.ones(x_imputed_1.shape[0])])
#x_tr = x_imputed_mix[:tr_val_cutoff,:]
#y_tr = y_cate_num[:tr_val_cutoff]

x_val = x_imputed_mix[tr_val_cutoff:,:]
y_val = y_cate_num [tr_val_cutoff:]


#%%
'''
modeling
'''

#%%
'''
SVC
hard to scale to dataset with more than a couple of 10000 samples
'''
#from sklearn.svm import SVC
#model = SVC(kernel= 'poly')
#t0 = time.time()
#
#print('SVC tr...',end='')
#model.fit(x_tr,y_tr)
#
#y_tr_pred = model.predict(x_tr)
#print(matthews_corrcoef(y_tr, y_tr_pred) ,end='')
#print(',val...',end='')
#y_val_pred = model.predict(x_val[:,:])
#print('mcc:',matthews_corrcoef(y_val, y_val_pred),end='' )
#
#print(',cost',time.time()-t0,'sec')


#%%
'''
Tree
'''
#min_samples_leaf = 1
#max_depth=32
#
## training 0
#x = x_tr
#y = y_tr
## overfitting problem solution: smaller max_depth or larger min_sample_leaf
#print('max_depth','min_samples_leaf')
#
#print('model:','training...',end='')
#model = tree.DecisionTreeClassifier()
#model = model.fit(x,y)
#y_pred = model.predict(x)
#mcc = matthews_corrcoef(y, y_pred) 
#print('mcc:',mcc,end='')
##utils.save_variable(tree_votes_0,'models/tree_votes_0.pkl')
#print(',val...',end='')
#y_pred = model.predict(x_val[:,:])
#mcc = matthews_corrcoef(y_val, y_pred) 
#print('mcc:',mcc)
#
#for max_depth in range(1,100,1):
#    print(max_depth,min_samples_leaf, end='->')
#    print('model:','training...',end='')
#    model = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf=min_samples_leaf)
#    model = model.fit(x,y)
#    y_pred = model.predict(x)
#    mcc = matthews_corrcoef(y, y_pred) 
#    print('mcc:',mcc,end='')
#    #utils.save_variable(tree_votes_0,'models/tree_votes_0.pkl')
#    print(',val...',end='')
#    y_pred = model.predict(x_val[:,:])
#    mcc = matthews_corrcoef(y_val, y_pred) 
#    print('mcc:',mcc)
    
'''
with mix 0s and 1s, AND all 1s, training...mcc: 0.979500912256,val...mcc: 0.0278246564691
with mix 0s and 1s, training...mcc: 1.0,val...mcc: -0.00354884485712
'''
#%%
'''
KNN
'''
#from sklearn.neighbors import KNeighborsClassifier
#x = x_tr
#y = y_tr
#
#for n_neighbors in range(1,100,2):
#    t0 = time.time()
#    model = KNeighborsClassifier(n_neighbors=n_neighbors)
#    print('knn',n_neighbors,',tr...',end='')
#    model = model.fit(x,y)
#    y_pred = model.predict(x)
#    mcc = matthews_corrcoef(y, y_pred) 
#    print(mcc,end='')
#    print(',val...',end='')
#    y_pred = model.predict(x_val[:,:])
#    mcc = matthews_corrcoef(y_val, y_pred) 
#    print(mcc,end='')
#    print(',cost',time.time()-t0,'sec')
    
'''
knn 1 ,tr...1.0,val...0.0278246564691,cost 21.692999839782715 sec
knn 3 ,tr...0.990035485737,val...0.0299539330585,cost 21.877000093460083 sec
knn 5 ,tr...0.981084972371,val...0.0305343085104,cost 22.081000089645386 sec
knn 7 ,tr...0.98067361513,val...0.0318541807108,cost 22.29099988937378 sec
knn 9 ,tr...0.980000675259,val...0.0396172617323,cost 22.417999982833862 sec
knn 11 ,tr...0.979492090566,val...-0.00689520910228,cost 22.752000093460083 sec
knn 13 ,tr...0.979099346894,val...-7.2365000332e-05,cost 23.052000045776367 sec
knn 15 ,tr...0.979099346894,val...0.00528786391558,cost 23.051999807357788 sec
knn 17 ,tr...0.979099346894,val...0.00761804467117,cost 23.230000019073486 sec
knn 19 ,tr...0.979099346894,val...0.00754755987924,cost 23.40000009536743 sec
knn 21 ,tr...0.978998957547,val...0.000354445033095,cost 23.58999991416931 sec
knn 23 ,tr...0.978998957547,val...0.00226958217973,cost 23.81500005722046 sec
knn 25 ,tr...0.978998957547,val...0.00734051801358,cost 23.90400004386902 sec
knn 27 ,tr...0.978998957547,val...0.00914893704465,cost 24.134000062942505 sec
knn 29 ,tr...0.978998957547,val...0.0162164051765,cost 24.282999992370605 sec
knn 31 ,tr...0.978998957547,val...0.0182251713932,cost 24.562000036239624 sec
knn 33 ,tr...0.978998957547,val...0.0191661201628,cost 24.729999780654907 sec
knn 35 ,tr...0.978998957547,val...0.0200370194851,cost 24.925000190734863 sec
knn 37 ,tr...0.978998957547,val...0.0222932825474,cost 25.151000022888184 sec
knn 39 ,tr...0.978998957547,val...0.0230315461674,cost 27.537999868392944 sec
knn 41 ,tr...0.978998957547,val...0.0331873887064,cost 26.09599995613098 sec
knn 43 ,tr...0.978998957547,val...0.0332828905881,cost 25.664000034332275 sec
knn 45 ,tr...0.978898568995,val...0.0391185958041,cost 25.80299997329712 sec
knn 47 ,tr...0.978798181235,val...0.0273335483739,cost 25.99500012397766 sec
knn 49 ,tr...0.978798181235,val...0.0383778121066,cost 27.258999824523926 sec
knn 51 ,tr...0.978497022696,val...-0.0173470940646,cost 27.98099994659424 sec
knn 53 ,tr...0.978396638091,val...0.0,cost 28.1489999294281 sec
knn 55 ,tr...0.978396638091,val...0.0,cost 28.289999961853027 sec
knn 57 ,tr...0.97829625427,val...0.0,cost 28.449000120162964 sec
knn 59 ,tr...0.97829625427,val...0.0,cost 28.414000034332275 sec
knn 61 ,tr...0.97829625427,val...0.0,cost 28.674000024795532 sec
knn 63 ,tr...0.978195871233,val...0.0,cost 28.86299991607666 sec
knn 65 ,tr...0.978195871233,val...0.0,cost 29.079999923706055 sec
knn 67 ,tr...0.978195871233,val...0.0,cost 29.243000030517578 sec
knn 69 ,tr...0.978095488978,val...0.0,cost 29.29700016975403 sec
knn 71 ,tr...0.977995107502,val...0.0,cost 29.478999853134155 sec
knn 73 ,tr...0.977593589367,val...0.0,cost 29.58400011062622 sec
knn 75 ,tr...0.977493211767,val...0.0,cost 29.64199995994568 sec
knn 77 ,tr...0.977493211767,val...0.0,cost 29.799999952316284 sec
knn 79 ,tr...0.977493211767,val...0.0,cost 29.86400008201599 sec
knn 81 ,tr...0.977292458876,val...0.0,cost 29.96499991416931 sec
knn 83 ,tr...0.977192083583,val...0.0,cost 30.064000129699707 sec
knn 85 ,tr...0.977192083583,val...0.0,cost 30.127999782562256 sec
knn 87 ,tr...0.977192083583,val...0.0,cost 30.16000008583069 sec
knn 89 ,tr...0.977091709055,val...0.0,cost 30.48099994659424 sec
knn 91 ,tr...0.977091709055,val...0.0,cost 30.603000164031982 sec
knn 93 ,tr...0.976991335291,val...0.0,cost 30.744999885559082 sec
knn 95 ,tr...0.976890962289,val...0.0,cost 30.740999937057495 sec
knn 97 ,tr...0.976890962289,val...0.0,cost 31.396000146865845 sec
knn 99 ,tr...0.976890962289,val...0.0,cost 33.27799987792969 sec
'''
#%%
#%% PCA only on samples with euqal response 1 and 0
from sklearn.decomposition import PCA


x_pca_i = x_euqal01

pca = PCA(n_components=x_pca_i.shape[1])
t0 = time.time()
pca.fit(x_pca_i)
print(time.time()-t0,'sec')
print(pca.explained_variance_ratio_) 
cs = np.cumsum(pca.explained_variance_ratio_)
print(cs)

'''
based on plot of cs, pc 50 give a good variances
'''
#%%
pca = PCA(n_components=50)
t0 = time.time()
pca.fit(x_euqal01)
print(time.time()-t0,'sec')
x_pca_o = pca.transform(x_euqal01)

print('total PCA variance:',sum(pca.explained_variance_ratio_))


#%%

x_tr_pcs = pca.transform(x_tr)
x_val_pcs = pca.transform(x_val)

min_samples_leaf = 1
max_depth=32

# overfitting problem solution: smaller max_depth or larger min_sample_leaf
print('max_depth','min_samples_leaf')
for min_samples_leaf in range(1,100,1):
    print('(',min_samples_leaf,max_depth,')','training...',end='')
    model = tree.DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
    model = model.fit(x_tr_pcs,y_tr)
    y_tr_pred = model.predict(x_tr_pcs)
    
    print('mcc:',matthews_corrcoef(y_tr, y_tr_pred),end='')
    #utils.save_variable(tree_votes_0,'models/tree_votes_0.pkl')
    print(',val...',end='')
    y_pred = model.predict(x_val_pcs)
    mcc = matthews_corrcoef(y_val, y_pred) 
    print('mcc:',mcc)

#%%

#%%
'''


Model with hd filter, imputer, PCA, tree


'''
chunk_nu_large = 1184#1184
x_large_len = train_row_nu
x_large= np.zeros((x_large_len, 50))
y_large  = np.zeros(x_large_len)
chunks_num = pd.read_csv('data/train_numeric.csv',chunksize=max_chunk_size, low_memory=False,iterator=True)
bar = progressbar.ProgressBar()
for chunk_id in bar(range(chunk_nu_large)):
    
    chunk_cate = utils.read_variable('model_stats/train_cate_proba/'+str(chunk_id)+'.pkl')
    chunk_num = chunks_num.get_chunk()

    chunk_x= np.zeros((chunk_num.shape[0], col_cate_nu+col_numeric_nu))
    chunk_x[:,:col_cate_nu] = chunk_cate
    chunk_x[:,col_cate_nu:] = chunk_num.drop(['Response'],axis=1)
    chunk_x_hd = chunk_x[:,selected_cols]
    chunk_x_imputed = imputer.transform(chunk_x_hd)
    chunk_x_pcs = pca.transform(chunk_x_imputed)
    
    row_range = range(chunk_id*max_chunk_size,chunk_id*max_chunk_size+chunk_cate.shape[0],1)
    x_large[row_range,:] = chunk_x_pcs
    y_large[row_range] =  chunk_num['Response']
    

del chunk_id, chunks_num, chunk_num, chunk_x, chunk_x_hd,chunk_x_imputed, chunk_x_pcs, row_range    


#%% tr and val
large_tr_ratio = 0.9

large_tr_val_cutoff = int(x_large.shape[0]*large_tr_ratio)

large_x_tr = x_large[:large_tr_val_cutoff,:]
large_y_tr = y_large[:large_tr_val_cutoff]

large_x_val = x_large[large_tr_val_cutoff:,:]
large_y_val = y_large[large_tr_val_cutoff:]



#%%

#print('(null,null)','training...',end='')
#model = tree.DecisionTreeClassifier()
#model = model.fit(large_x_tr,large_y_tr)
#y_tr_pred = model.predict(large_x_tr)
#
#print('mcc:',matthews_corrcoef(large_y_tr , y_tr_pred),end='')
##utils.save_variable(tree_votes_0,'models/tree_votes_0.pkl')
#print(',val...',end='')
#y_pred = model.predict(large_x_val)
#mcc = matthews_corrcoef(large_y_val, y_pred) 
#print('mcc:',mcc)
#
#min_samples_leaf = 1
#max_depth=32
#
## overfitting problem solution: smaller max_depth or larger min_sample_leaf
#print('max_depth','min_samples_leaf')
#for max_depthin in range(1,100,1):
#    print('(',min_samples_leaf,max_depth,')','training...',end='')
#    model = tree.DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf)
#    model = model.fit(large_x_tr,large_y_tr)
#    y_tr_pred = model.predict(large_x_tr)
#    
#    print('mcc:',matthews_corrcoef(large_y_tr , y_tr_pred),end='')
#    #utils.save_variable(tree_votes_0,'models/tree_votes_0.pkl')
#    print(',val...',end='')
#    y_pred = model.predict(large_x_val)
#    mcc = matthews_corrcoef(large_y_val, y_pred) 
#    print('mcc:',mcc)
    
'''
(null,null) training...mcc: 1.0,val...mcc: 0.0092392140463
observation: overfitting problem can not be addressed by adjusting max_depyth or min_sample_leaf
'''

#%%
'''
forest
'''
#from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier(n_estimators=10,n_jobs = 3)
#t0 = time.time()
#model = model.fit(large_x_tr,large_y_tr)
#t1 = time.time()
#y_tr_pred = model.predict(large_x_tr)
#
#print('cost:','tr:',t1-t0, ',total:',time.time()-t0)
#
#print('mcc:',matthews_corrcoef(large_y_tr , y_tr_pred),end='')
##utils.save_variable(tree_votes_0,'models/tree_votes_0.pkl')
#print(',val...',end='')
#y_pred = model.predict(large_x_val)
#mcc = matthews_corrcoef(large_y_val, y_pred)
#print('mcc:',mcc)
#
#'''
#cost: tr: 187.72373700141907 ,total: 189.83285808563232
#mcc: 0.860321663357,val...mcc: -0.000217919779316
#
#'''
##%%
#
##%%
#y_pred_proba = model.predict_proba(large_x_val)
#
#for threshold in np.arange(0.1,0.4,0.001):
#    y_pred = (y_pred_proba[:,1]>threshold).astype(np.int)
#    print(threshold,end=',')
#    print('1s:',sum(y_pred),end=',')
#    print('mcc:',matthews_corrcoef(large_y_val, y_pred))
#    
#%%
row_per_tree = 300000
tree_id =0
for max_depth in range(10,30,2):
    row_range = range(tree_id*row_per_tree,(tree_id+1)*row_per_tree,1)
    x = large_x_tr[row_range,:]
    y = large_y_tr[row_range]

    time0 = time.time()
    print('tree',tree_id,max_depth,'training...',end='')

    model = tree.DecisionTreeClassifier(max_depth=max_depth)
    model = model.fit(x,y)
    y_tr_pred = model.predict(x)
    
    print('mcc:',matthews_corrcoef(y , y_tr_pred),end='')
    #utils.save_variable(tree_votes_0,'models/tree_votes_0.pkl')
    print(',val...',end='')
    y_pred = model.predict(large_x_val)
    mcc = matthews_corrcoef(large_y_val, y_pred) 
    print('mcc:',mcc,end=',')
    print('cost',int(time.time()-time0),'sec')
    
'''
tree 0 1 training...mcc: 0.0,val...mcc: 0.0,cost 2 sec
tree 0 2 training...mcc: 0.0560717697863,val...mcc: 0.0,cost 4 sec
tree 0 3 training...mcc: 0.0686506138537,val...mcc: 0.0,cost 7 sec
tree 0 4 training...mcc: 0.085686935304,val...mcc: -0.000218131423364,cost 9 sec
tree 0 5 training...mcc: 0.117972983136,val...mcc: -0.000218131423364,cost 12 sec
tree 0 6 training...mcc: 0.136173838007,val...mcc: -0.00104621878164,cost 14 sec
tree 0 7 training...mcc: 0.167711838852,val...mcc: 0.00636244755147,cost 16 sec
tree 0 8 training...mcc: 0.212944830207,val...mcc: -0.00113356862093,cost 18 sec
tree 0 9 training...mcc: 0.254223024544,val...mcc: 0.00578116593891,cost 20 sec
'''
#%%
from sklearn.ensemble import GradientBoostingClassifier
x0 = large_x_tr[:10000,:]
y0 = large_y_tr[:10000]

x1 = large_x_tr[-10000:,:]
y1 = large_y_tr[-10000:]

for super_value in range(1,10,1):
    print('Supervalue:',super_value,end=',')
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                       max_depth=super_value, random_state=0)
    model = model.fit(x0,y0)
    y_tr_pred = model.predict(x0)
    
    print('mcc:',matthews_corrcoef(y0, y_tr_pred),end='')
    #utils.save_variable(tree_votes_0,'models/tree_votes_0.pkl')
    print(',val...',end='')
    y_pred = model.predict(x1)
    mcc = matthews_corrcoef(y1, y_pred) 
    print('mcc:',mcc)
#%%
# overfitting problem solution: smaller max_depth or larger min_sample_leaf

row_per_tree = 300000
forest = []
for tree_id in range(3):
    row_range = range(tree_id*row_per_tree,(tree_id+1)*row_per_tree,1)
    x = large_x_tr[row_range,:]
    y = large_y_tr[row_range]

    time0 = time.time()
    print('tree',tree_id,'training...',end='')

    model = tree.DecisionTreeClassifier()
    model = model.fit(x,y)
    y_tr_pred = model.predict(x)
    
    print('mcc:',matthews_corrcoef(y , y_tr_pred),end='')
    #utils.save_variable(tree_votes_0,'models/tree_votes_0.pkl')
    print(',val...',end='')
    y_pred = model.predict(large_x_val)
    mcc = matthews_corrcoef(large_y_val, y_pred) 
    print('mcc:',mcc,end=',')
    print('cost',int(time.time()-time0),'sec')
    
    
    forest.append(model)
    
'''
tree 0 training...mcc: 1.0,val...mcc: -0.00371071812233,cost 104 sec
tree 1 training...mcc: 1.0,val...mcc: 0.00828514607201,cost 94 sec
tree 2 training...mcc: 1.0,val...mcc: 0.0135077035197,cost 102 sec
'''   
#%%
y_sum = np.zeros(large_y_val.shape)
for model in forest:
    y_pred = model.predict(large_x_val)
    y_sum += y_pred

y_pred = (y_sum>0).astype(np.int)
mcc = matthews_corrcoef(large_y_val, y_pred) 
print('mcc:',mcc,end=',')