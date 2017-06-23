import utils
import progressbar



column_names = utils.read_variable('outputs/column_names.pkl')
cols_categorical = column_names['categorical']
cols_date = column_names['date']
cols_numeric = column_names['numeric']

col_len = cols_categorical.size+cols_date.size+cols_numeric.size

del column_names
#%% 
'''
calculate probability of response 0 with categorical col
'''
col_stats_cate = {}
print('import col statistic:','categorical columns')
bar = progressbar.ProgressBar()
for col_name in bar(cols_categorical):
    col_stats_cate[col_name] = utils.read_variable('model_stats/cate/'+col_name+'.pkl')

utils.save_variable(col_stats_cate,'model_stats/col_stats_cate.pkl')

      
'''
calculate probability of response 0 with date col
'''
col_stats_date = {}
print('import col statistic:','date columns')
bar = progressbar.ProgressBar()
for col_name in bar(cols_date):
    stat = utils.read_variable('model_stats/date/'+col_name+'.pkl')
    #remove nu list to save memory
    del stat[0]['nu'],stat[1]['nu']
    col_stats_date[col_name] = stat

utils.save_variable(col_stats_date,'model_stats/col_stats_date.pkl')
  
'''
calculate probability of response 0 with num col
'''
col_stats_num = {}
print('import col statistic:','date columns')
bar = progressbar.ProgressBar()
for col_name in bar(cols_numeric):
    stat = utils.read_variable('model_stats/num/'+col_name+'.pkl')
        #remove nu list to save memory
    del stat[0]['nu'],stat[1]['nu']
    col_stats_num[col_name] = stat

utils.save_variable(col_stats_num,'model_stats/col_stats_num.pkl')