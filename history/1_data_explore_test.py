import numpy as np
import pandas as pd
import time
import utils

startTime = time.time()
chunks = pd.read_csv('data/test_numeric.csv',index_col='Id',chunksize=1000, low_memory=False)
sum=0
for chunk in chunks:
    sum = sum + chunk.shape[0]
    print(chunk.shape,'/',sum)

print('Test Row Nu:',sum)
print ('Training took', int(time.time() - startTime),'sec')

##########RESULT########################
# rows: 1183748
########################################

#%% get test id for further use
startTime = time.time()
chunks = pd.read_csv('data/test_numeric.csv',index_col='Id',chunksize=1000, low_memory=False)
sum=0
test_ids = []
for chunk in chunks:
    sum = sum + chunk.shape[0]
    print(chunk.shape,'/',sum)
    test_ids.extend(chunk.index.values.tolist())

print('Test Row Nu:',sum)
print ('Training took', int(time.time() - startTime),'sec')

utils.save_variable(test_ids,'outputs/test_ids.pkl')
#%% check sample_submission

startTime = time.time()
chunks = pd.read_csv('data/sample_submission.csv',index_col='Id',chunksize=1000, low_memory=False)

sum = 0
sum_1 = 0
for chunk in chunks:
    sum = sum+chunk.shape[0]
    rows_nu_1 = chunk[chunk.Response==1].shape[0]
    sum_1 = sum_1+rows_nu_1
    print(rows_nu_1, '/',sum_1,'/',sum)

print('sample submission with response 1:',sum_1,'/',sum)