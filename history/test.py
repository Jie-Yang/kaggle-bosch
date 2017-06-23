#import col_stats_utils
import utils
import pandas as pd
import time
import numpy as np

'''

proba cal based on kde of num and date cols will take 140 ms for each col in each row, which will cost years to process 1m samples.
Hence, it is impossible to use this kde to process all data
However, since dic used by categorical cols is quick. so it could be used along with chunk_tree_votes which only use num and date cols.

'''
print('loading col_stats_cate...')
stats = utils.read_variable('model_stats/num/L3_S51_F4262.pkl')



#%%
a = stats[0]['nu']
b = stats[1]['nu']



#%%
from sklearn.neighbors import KernelDensity
kd0 = KernelDensity(kernel='tophat',metric='manhattan').fit(a.reshape(-1, 1))
kd1 = KernelDensity(kernel='tophat',metric='manhattan').fit(b.reshape(-1, 1))

value=0
time1 = time.time()
s0 = kd0.score_samples(value)
s0 = np.exp(s0)
time2 = time.time()
s1 = kd1.score_samples(value)
s1 = np.exp(s1)
time3 = time.time()
print(s0,s1,'cost','s0:',time2-time1,'s1:',time3-time2,'total:',time3-time1)
