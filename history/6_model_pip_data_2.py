
import time
import utils
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import matthews_corrcoef


max_chunk_size = 1000
col_cate_nu = 2140
col_numeric_nu = 969
col_date_nu = 1157

#%%
val_X,val_Y = utils.load_training_subset(range(1000,1184,1))

forest_important_features = utils.read_variable('model_stats/forest_important_features.pkl')

#%%
forest = []
chunk_steps = 333
for set_id in range(chunk_steps,1001,chunk_steps):
    # 6 chunks give about the same 0s (tiny amount of 1s is ignored) as the 1s
    chunk_range = range(set_id-chunk_steps,set_id,1)
    t_X,t_Y = utils.load_training_subset(chunk_range)
    
    X,Y = t_X[:,forest_important_features],t_Y
    model = tree.DecisionTreeClassifier(random_state=0)
    t0 = time.time()
    model = model.fit(X,Y)
    
    y_pred = model.predict(X)
    
    print('TREE tr:',matthews_corrcoef(Y , y_pred),end='')
    #print('   tr 1s:real',sum(Y),',pred',sum(y_pred))
    #utils.save_variable(tree_votes_0,'models/tree_votes_0.pkl')
    print(',val:',end='')
    X, Y = val_X[:,forest_important_features], val_Y
    y_pred = model.predict(X)
    print(matthews_corrcoef(Y, y_pred),end='')
    print('(',int(time.time()-t0),'sec)')
        #print('   val 1s:real',sum(Y),',pred',sum(y_pred))
    
    forest.append(model)
    
#%%

y_votes = np.zeros([val_X.shape[0],len(forest)])

for idx,model in enumerate(forest):
    print(idx,'/',len(forest))
    y_votes[:,idx] = model.predict(val_X[:,forest_important_features])
    print('1s:',sum(y_votes[:,idx]),'/',int(sum(val_Y)))
#%%
for threshold in range(0,4,1):
    y_pred = (np.sum(y_votes,axis=1)>threshold).astype(np.int)
    print('1s:',sum(y_pred),'/',int(sum(val_Y)),',mcc:',matthews_corrcoef(val_Y, y_pred))
    
    
#%%
chunk_range = range(0,1000,1)
t_X,t_Y = utils.load_training_subset(chunk_range)


#%%
'''
Model: Random Forest Classifier
'''
from sklearn.ensemble import RandomForestClassifier

n_estimators = 5
for max_depth in range(1,100,2):
    X,Y = t_X[:,forest_important_features],t_Y
    model = RandomForestClassifier(n_estimators=n_estimators, n_jobs = 3, max_depth=max_depth)
    t0 = time.time()
    model = model.fit(X,Y)
    
    y_pred = model.predict(X)
    
    print(max_depth,'model tr:',matthews_corrcoef(Y , y_pred),end='')
    print(',val:',end='')
    X, Y = val_X[:,forest_important_features], val_Y
    y_pred = model.predict(X)
    print(matthews_corrcoef(Y, y_pred),end='')
    print('(',int(time.time()-t0),'sec)')
    utils.save_variable(model,'models/rfc_'+str(n_estimators)+'_'+str(max_depth)+'.pkl')

'''
n_estimators = 10

10 model tr: 0.104941652899,val:0.0407162855692( 148 sec)
12 model tr: 0.167247274569,val:0.0337964523864( 124 sec)
14 model tr: 0.250074784343,val:0.0545491302518( 119 sec)
16 model tr: 0.32368663829,val:0.0535392979611( 129 sec)
18 model tr: 0.371770954583,val:0.0303294881435( 136 sec)
20 model tr: 0.456984005851,val:0.0327101060725( 143 sec)
22 model tr: 0.500885401677,val:0.0233665138915( 136 sec)
24 model tr: 0.568135683718,val:0.0168382023099( 144 sec)
26 model tr: 0.630927848449,val:0.0458359043299( 144 sec)
28 model tr: 0.690003573808,val:0.036139147729( 136 sec)
30 model tr: 0.725650208595,val:0.0293765141473( 150 sec)
32 model tr: 0.772076711911,val:0.0286930446929( 199 sec)
34 model tr: 0.814062391401,val:0.0381273258981( 167 sec)
36 model tr: 0.839536551355,val:0.0329434363665( 169 sec)
38 model tr: 0.845561161964,val:0.0329434363665( 162 sec)
40 model tr: 0.856632602346,val:0.0243542361641( 152 sec)
42 model tr: 0.854639830688,val:0.022916642459( 155 sec)
44 model tr: 0.858456203079,val:0.0414139977494( 151 sec)
46 model tr: 0.854138656855,val:0.0656871382918( 162 sec)
48 model tr: 0.859025209149,val:0.026343934481( 156 sec)
50 model tr: 0.860849669939,val:0.0460176784769( 161 sec)
52 model tr: 0.864010023053,val:0.0597697133966( 164 sec)
54 model tr: 0.86553831412,val:0.0407571273802( 167 sec)
56 model tr: 0.867109095561,val:0.0288495263513( 149 sec)
58 model tr: 0.863950572046,val:0.0233665138915( 151 sec)
60 model tr: 0.86502113908,val:0.0433374692086( 161 sec)
62 model tr: 0.867408379738,val:0.0138326852892( 151 sec)
64 model tr: 0.866331185275,val:0.0379621191348( 151 sec)
66 model tr: 0.862432769162,val:0.0232774875104( 166 sec)
68 model tr: 0.867009857187,val:0.0282963084686( 158 sec)
70 model tr: 0.866215548235,val:0.0211243933352( 162 sec)

############################################################
n_estimators = 20

50 model tr: 0.919417398887,val:-0.000353884411721( 270 sec)

############################################################
n_estimators = 5

39 model tr: 0.818204519508,val:0.0359580036731( 101 sec)
41 model tr: 0.830062215805,val:0.0592229160536( 99 sec)
43 model tr: 0.840819049658,val:0.0372936195088( 83 sec)
45 model tr: 0.831089458522,val:0.0354700964484( 92 sec)
47 model tr: 0.84761623254,val:0.0124939063032( 103 sec)
49 model tr: 0.85484280941,val:0.0363171370956( 109 sec)
51 model tr: 0.842516058744,val:0.0460663215554( 101 sec)
53 model tr: 0.846442767342,val:0.0446618178854( 103 sec)
55 model tr: 0.849810288191,val:0.044481220252( 91 sec)
57 model tr: 0.856060273776,val:0.0176843155636( 113 sec)
59 model tr: 0.853671409736,val:0.0560941159602( 107 sec)
61 model tr: 0.853996629087,val:0.0559819287688( 118 sec)
63 model tr: 0.855636811275,val:0.0475390340034( 103 sec)
65 model tr: 0.85930010062,val:0.0722365957925( 114 sec)
67 model tr: 0.857300161886,val:0.0354441343705( 96 sec)
69 model tr: 0.853849978723,val:0.0554220623073( 88 sec)
71 model tr: 0.857572499915,val:0.0186381954315( 100 sec)
73 model tr: 0.853420047561,val:0.0293567157338( 91 sec)
75 model tr: 0.85529789117,val:0.0376089918172( 98 sec)
77 model tr: 0.853763740229,val:0.0366512829462( 96 sec)
79 model tr: 0.850295177555,val:0.0534141242089( 98 sec)
81 model tr: 0.859679622916,val:0.0618513004398( 93 sec)
83 model tr: 0.863435036673,val:0.0566055590135( 103 sec)
85 model tr: 0.856826465471,val:0.0505160867104( 90 sec)
87 model tr: 0.855165521613,val:0.0410472389987( 98 sec)
89 model tr: 0.858621741174,val:0.0502097370136( 97 sec)
91 model tr: 0.861716398631,val:0.0350909678187( 98 sec)
93 model tr: 0.852017803052,val:0.0354441343705( 109 sec)
95 model tr: 0.855077929441,val:0.0670130849357( 99 sec)
97 model tr: 0.855002619069,val:0.0372337083269( 99 sec)
99 model tr: 0.851434957701,val:0.0298068590249( 92 sec)

'''
#%%
'''
Model: Support Vector Machine

WARNING: The implementation is based on libsvm.
The fit time complexity is more than quadratic with the number of samples which
makes it hard to scale to dataset with more than a couple of 10000 samples.
HENCE, training dataset is splitted into 10 blocks

'''
print('SVC')
from sklearn.svm import SVC

for sub_id in range(10):
    row_from = sub_id*1000
    row_to = (sub_id+1)*1000
    print('SVC:',row_from,row_to)
    X,Y = t_X[row_from:row_to,forest_important_features],t_Y[row_from:row_to]
    model = SVC( kernel='rbf', C=1)
    t0 = time.time()
    model = model.fit(X,Y)
    
    y_pred = model.predict(X)
    
    print(sub_id,'model tr:',matthews_corrcoef(Y , y_pred),end='')
    print(',val:',end='')
    X, Y = val_X[:,forest_important_features], val_Y
    y_pred = model.predict(X)
    print(matthews_corrcoef(Y, y_pred),end='')
    print('(',int(time.time()-t0),'sec)')
    utils.save_variable(model,'models/svc_'+str(sub_id)+'.pkl')


#%%
'''
Model: SGD
'''

from sklearn.linear_model import SGDClassifier
print('SGDClassifier')
X,Y = t_X[:,forest_important_features],t_Y
model = SGDClassifier()
t0 = time.time()
model = model.fit(X,Y)

y_pred = model.predict(X)

print('model tr:',matthews_corrcoef(Y , y_pred),end='')
print(',val:',end='')
X, Y = val_X[:,forest_important_features], val_Y
y_pred = model.predict(X)
print(matthews_corrcoef(Y, y_pred),end='')
print('(',int(time.time()-t0),'sec)')
utils.save_variable(model,'models/sgd.pkl')


#%%
'''
Model: LogisticRegression
'''
from sklearn.linear_model import LogisticRegression
print('LogisticRegression')

X,Y = t_X[:,forest_important_features],t_Y
model = LogisticRegression()
t0 = time.time()
model = model.fit(X,Y)

y_pred = model.predict(X)

print('model tr:',matthews_corrcoef(Y , y_pred),end='')
print(',val:',end='')
X, Y = val_X[:,forest_important_features], val_Y
y_pred = model.predict(X)
print(matthews_corrcoef(Y, y_pred),end='')
print('(',int(time.time()-t0),'sec)')
utils.save_variable(model,'models/lg.pkl')

#%%
'''
Model: AdaBoostClassifier
''' 

from sklearn.ensemble import AdaBoostClassifier

print('AdaBoostClassifier')
for learning_rate in range(1,12,1):
    X,Y = t_X[:,forest_important_features],t_Y
    model = AdaBoostClassifier(learning_rate=learning_rate)
    t0 = time.time()
    model = model.fit(X,Y)
    
    y_pred = model.predict(X)
    
    print(c,'model tr:',matthews_corrcoef(Y , y_pred),end='')
    print(',val:',end='')
    X, Y = val_X[:,forest_important_features], val_Y
    y_pred = model.predict(X)
    print(matthews_corrcoef(Y, y_pred),end='')
    print('(',int(time.time()-t0),'sec)')
    utils.save_variable(model,'models/abc_'+str(learning_rate)+'.pkl')