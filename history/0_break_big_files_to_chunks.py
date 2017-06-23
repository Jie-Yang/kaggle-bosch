#%%
'''
Good practise: break a big file into small ones which can accelerate data access, since file access have to be done line by line from the very begining
break num and date csv file into chunks which can improve the efficiency to access certain parts of data
'''
import pandas as pd
import utils
import progressbar
max_chunk_size = 1000
max_chunk_nu = 1184

chunks_num = pd.read_csv('data/train_numeric.csv',index_col='Id',chunksize=max_chunk_size, low_memory=False,iterator=True)
chunks_date = pd.read_csv('data/train_date.csv',index_col='Id',chunksize=max_chunk_size, low_memory=False,iterator=True)
chunks_cate = pd.read_csv('data/train_categorical.csv',index_col='Id',chunksize=max_chunk_size, low_memory=False,iterator=True)


bar = progressbar.ProgressBar()

for chunk_id in bar(range(max_chunk_nu)):
    # chunk has to be read one by one in sequence
    chunk_num_response = chunks_num.get_chunk()
    chunk_num = chunk_num_response.drop(['Response'],axis=1)
    chunk_y = chunk_num_response['Response']
    chunk_date = chunks_date.get_chunk()
    chunk_cate = chunks_cate.get_chunk()
    
    utils.save_variable(chunk_y, 'data/train_y_chunks/'+str(chunk_id)+'.pkl')
    utils.save_variable(chunk_num, 'data/train_numeric_chunks/'+str(chunk_id)+'.pkl')
    utils.save_variable(chunk_date, 'data/train_date_chunks/'+str(chunk_id)+'.pkl')
    utils.save_variable(chunk_cate, 'data/train_categorical_chunks/'+str(chunk_id)+'.pkl')
    
#%%

chunks_num = pd.read_csv('data/test_numeric.csv',index_col='Id',chunksize=max_chunk_size, low_memory=False,iterator=True)
chunks_date = pd.read_csv('data/test_date.csv',index_col='Id',chunksize=max_chunk_size, low_memory=False,iterator=True)
chunks_cate = pd.read_csv('data/test_categorical.csv',index_col='Id',chunksize=max_chunk_size, low_memory=False,iterator=True)


bar = progressbar.ProgressBar()

for chunk_id in bar(range(max_chunk_nu)):
    # chunk has to be read one by one in sequence
    chunk_num = chunks_num.get_chunk()
    chunk_date = chunks_date.get_chunk()
    chunk_cate = chunks_cate.get_chunk()
    
    utils.save_variable(chunk_num, 'data/test_numeric_chunks/'+str(chunk_id)+'.pkl')
    utils.save_variable(chunk_date, 'data/test_date_chunks/'+str(chunk_id)+'.pkl')
    utils.save_variable(chunk_cate, 'data/test_categorical_chunks/'+str(chunk_id)+'.pkl')