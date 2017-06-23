import utils

import numpy as np
import time
import pandas as pd

#%%

print('loading col_stats_cate...')
col_stats_cate = utils.read_variable('model_stats/col_stats_cate.pkl')
print('loading col_stats_date...')
col_stats_date = utils.read_variable('model_stats/col_stats_date.pkl')
print('loading col_stats_num...')
col_stats_num = utils.read_variable('model_stats/col_stats_num.pkl')
#%% 
'''
calculate probability of response 0 with categorical col
'''
def cal_0_proba_by_cate(col_name,value):
    stat_0 = col_stats_cate[col_name][0]
    stat_1 = col_stats_cate[col_name][1]
    response0_proba = 0
    if value!=value:
        # nan
        response0_proba = stat_0['nan']/(stat_0['nan']+stat_1['nan'])
    elif value in stat_0 and value not in stat_1:
        response0_proba = 1
        
    elif value not in stat_0 and value in stat_1:
        response0_proba = 0
        
    elif value in stat_0 and value in stat_1:
        response0_proba = stat_0[value]/(stat_0[value]+stat_1[value])
    
    elif value not in stat_0 and value not in stat_1:
        response0_proba = 0
        
    return response0_proba
    
#%%   
'''
calculate probability of response 0 with date col
'''
def cal_0_proba_by_date(col_name, value):
    stat_0 = col_stats_date[col_name][0]
    stat_1 = col_stats_date[col_name][1]
    response0_proba = 0
    
    if value!=value:
        # nan
        response0_proba = stat_0['nan']/(stat_0['nan']+stat_1['nan'])
    else:
        proba0 = 0
        proba1 = 0
        if 'kde' in stat_0:
            kde0 = stat_0['kde']
            proba0 = np.exp(kde0.score_samples(value))
        if 'kde' in stat_1: 
            kde1 = stat_1['kde']
            proba1 = np.exp(kde1.score_samples(value))
        if proba0 == 0 and proba1 == 0:
            response0_proba = 0
        else:
            response0_proba = proba0/(proba0+proba1)
        
    return response0_proba

#%%    
'''
calculate probability of response 0 with num col
'''
def cal_0_proba_by_num(col_name, value):
    stat_0 = col_stats_num[col_name][0]
    stat_1 = col_stats_num[col_name][1]
    response0_proba = 0
    
    if value!=value:
        # nan
        response0_proba = stat_0['nan']/(stat_0['nan']+stat_1['nan'])
    else:
        proba0 = 0
        proba1 = 0
        if 'kde' in stat_0:
            kde0 = stat_0['kde']
            proba0 = np.exp(kde0.score_samples(value))
        if 'kde' in stat_1: 
            kde1 = stat_1['kde']
            proba1 = np.exp(kde1.score_samples(value))
        if proba0 == 0 and proba1 == 0:
            response0_proba = 0
        else:
            response0_proba = proba0/(proba0+proba1)
        
    return response0_proba
    
#%%
'''
calculate proba of each col for response 0 for a complete row
'''
def cal_0_proba_matrix_by_row(row, cols_cate = [], cols_date = [], cols_num = []):
    #col_selected = cols_cate_selected.union(cols_date_selected).union(cols_numeric_selected)
    #result = pd.DataFrame(columns=col_selected)
    result=0
    print('cal cate cols...',end='')
    startTime = time.time()
    for col_name in cols_cate:
        value = row[col_name]
        proba = cal_0_proba_by_cate(col_name,value)
        #result.loc['0',col_name] = proba
    print ('(', int(time.time() - startTime),'sec)',end='...');

    print('cal date cols...',end='')
    startTime = time.time()
    for col_name in cols_date:
        value = row[col_name]
        proba = cal_0_proba_by_date(col_name,value)
        #result.loc['0',col_name] = proba
    print ('(', int(time.time() - startTime),'sec)',end='...');

    print('cal num cols...',end='')
    startTime = time.time()
    for col_name in cols_num:
        value = row[col_name]
        proba = cal_0_proba_by_num(col_name,value)
        #result.loc['0',col_name] = proba
    print ('(', time.time() - startTime,'sec)');
    return result
    