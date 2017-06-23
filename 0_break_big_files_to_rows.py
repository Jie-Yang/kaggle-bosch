#%%
'''
Good practise: break a big file into small ones which can accelerate data access, since file access have to be done line by line from the very begining
break num and date csv file into chunks which can improve the efficiency to access certain parts of data

However, IO will be really heavy which could slow down things
'''
import pandas as pd
import utils
import progressbar
max_chunk_size = 1


chunks_num = pd.read_csv('data/train_numeric.csv',index_col='Id',chunksize=max_chunk_size, low_memory=False,iterator=True)

bar = progressbar.ProgressBar()

for chunk_id in bar(range(1183747)):
    # chunk has to be read one by one in sequence
    chunk_num_response = chunks_num.get_chunk()
    chunk_num = chunk_num_response.drop(['Response'],axis=1)
    chunk_y = chunk_num_response['Response']
    
    utils.save_variable(chunk_y, 'data/train_y_rows/'+str(chunk_id)+'.pkl')
    utils.save_variable(chunk_num, 'data/train_numeric_rows/'+str(chunk_id)+'.pkl')

    
#%%

chunks_num = pd.read_csv('data/test_numeric.csv',index_col='Id',chunksize=max_chunk_size, low_memory=False,iterator=True)


bar = progressbar.ProgressBar()

for chunk_id in bar(range(1183748)):
    # chunk has to be read one by one in sequence
    chunk_num = chunks_num.get_chunk()
    
    utils.save_variable(chunk_num, 'data/test_numeric_rows/'+str(chunk_id)+'.pkl')