import utils
import pandas as pd


train_sample_1 = utils.read_variable('outputs/train_sample_1.pkl')
train_sample_0 = utils.read_variable('outputs/train_sample_0.pkl')
column_names = utils.read_variable('outputs/column_names.pkl')

#%
cols_numeric = column_names['numeric']
cols_date = column_names['date']
cols_categorical = column_names['categorical']
#%%
cols = cols_categorical
print('categorical')
data = train_sample_0[cols]
print('Not Null Cell Ratio in 0:',data.count().sum()/(data.shape[0]*data.shape[1]))
data = train_sample_1[cols]
print('Not Null Cell Ratio in 1:',data.count().sum()/(data.shape[0]*data.shape[1]))

cols = cols_numeric
print('numeric')
data = train_sample_0[cols]
print('Not Null Cell Ratio in 0:',data.count().sum()/(data.shape[0]*data.shape[1]))
data = train_sample_1[cols]
print('Not Null Cell Ratio in 1:',data.count().sum()/(data.shape[0]*data.shape[1]))

cols = cols_date 
print('date')
data = train_sample_0[cols]
print('Not Null Cell Ratio in 0:',data.count().sum()/(data.shape[0]*data.shape[1]))
data = train_sample_1[cols]
print('Not Null Cell Ratio in 1:',data.count().sum()/(data.shape[0]*data.shape[1]))


#####RESULT#########
######## Data is really sparse
# categorical features: about 2.5% not null cells
# numeric features: about 18.9% not null cells
# date features: about 17.6% not null cells

#categorical
#Not Null Cell Ratio in 0: 0.026426168224299065
#Not Null Cell Ratio in 1: 0.02935327102803738
#numeric
#Not Null Cell Ratio in 0: 0.18965428276573787
#Not Null Cell Ratio in 1: 0.18909803921568627
#date
#Not Null Cell Ratio in 0: 0.17659688581314878
#Not Null Cell Ratio in 1: 0.1782811418685121
############################

#%% view the values in categorical features

cols = cols_categorical
print('examine categorical cols:')

options=[]
def check_element(x):
    if pd.notnull(x):
        print(x)
        global options
        options.append(x)
        
data = train_sample_0[cols]
data.applymap(check_element)
data = train_sample_1[cols]
data.applymap(check_element)
# remove duplicate
options = list(set(options))
print(options)

#####RESULT#########
# in sampling dataset there are only categorical 22 values

#['T512', 'T24', 'T4', 'T65536', 'T1372', 'T128', 'T143', 'T5', 'T16777557', 'T2', 'T16', 'T-2147483648', 'T48', 'T3', 'T8', 'T256', 'T1', 'T98', 'T16777232', 'T145', 'T6', 'T786432']


#%% view the values in categorical features
cols = cols_date
print('examine date cols:')

options=[]
def check_element(x):
    if pd.notnull(x):
        print(x)
        global options
        options.append(x)
        
data = train_sample_0[cols]
data.applymap(check_element)
data = train_sample_1[cols]
data.applymap(check_element)
# remove duplicate
options = list(set(options))
print(options)

######RESULT#########
# all date values are in float format as 1496.11, not in text nor string.


#%%#####################################
# view the sparsity of data
#############################################
import matplotlib.pyplot as plt
import time

cols = cols_numeric.append(cols_date)
X = pd.concat([train_sample_0[cols],train_sample_1[cols]],axis=0)
# remove rows with Nan Values
matrix = X.notnull().astype(int).as_matrix()
plt.matshow(matrix)
plt.pause(100)# solution for unresponse intactive window
plt.close()
#%%#####################################
# PCA can NOT be used if X contain Nan values
#############################################
from sklearn.decomposition import PCA
startTime = time.time()
n_components=100
pca = PCA(n_components=n_components)
pca.fit(X)
print ('Training took', int(time.time() - startTime),'sec');
pca_variance = pca.explained_variance_ratio_

#%%
plt.figure(1) 
plt.title('Variances')
plt.plot(pca_variance)
plt.figure(2) 
plt.title('Accumulated Variance')
plt.plot(pca_variance.cumsum())