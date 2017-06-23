import pickle
import sklearn.metrics as mx
import numpy as np
import progressbar

    
def save_variable(var,file_name):
    pkl_file = open(file_name, 'wb')
    pickle.dump(var, pkl_file, -1)
    pkl_file.close()
    
def read_variable(file_name):
    pkl_file = open(file_name, 'rb')
    var = pickle.load(pkl_file)
    pkl_file.close()
    return var

def validate_prediction(y,f):
    #best value at 1 and worst score at 0
    f1 = mx.f1_score(y,f)
    # statistic used by Kaggle
    auc = mx.roc_auc_score(y,f)
    confusion = mx.confusion_matrix(y,f)
    print('f1:',f1)
    print('AUC:',auc)
    print('conf:\n',confusion)
    return (f1,auc, confusion)

def sel_high_density_cols(x, nan_ratio):
    x_len = x.shape[0]
    
    is_nan = np.isnan(x)
    nan_count = np.sum(is_nan.astype(np.int),axis=0)
    selected_cols = []
    
    for idx,c in enumerate(nan_count):
        if c/x_len < nan_ratio:
            selected_cols.append(idx)
    
    print('selected col:',len(selected_cols))
    
    return selected_cols

def load_pipped_tr_rows(row_id_range):
    pip = read_variable('final/feature_sel_pip')
    p0_imputer = pip['0_imputer']
    p1_high_variance = pip['1_high_variance']
    p2_kbest_cols = pip['2_kbest_cols']
    p3_norm = pip['3_norm']

    col_len = len(p2_kbest_cols)
    
    all_X = np.zeros([len(row_id_range) , col_len])
    all_Y  = np.zeros(len(row_id_range))
    
    #print('loading & preprocessing (imputer,high variance, kbest, norm) data...',chunk_id_range)

    bar = progressbar.ProgressBar()
    for i, row_id in enumerate(row_id_range):
        chunk_num = read_variable('data/train_numeric_rows/'+str(row_id)+'.pkl')
        all_Y[i] = read_variable('data/train_y_rows/'+str(row_id)+'.pkl')
        all_X[i,:] = p3_norm.transform(p1_high_variance.transform(p0_imputer.transform(chunk_num))[:,p2_kbest_cols])
        
    return all_X, all_Y.astype(np.int)

    
def load_pipped_tr_chunk(chunk_id_range):
    pip = read_variable('final/feature_sel_pip')
    p0_imputer = pip['0_imputer']
    p1_high_variance = pip['1_high_variance']
    p2_kbest_cols = pip['2_kbest_cols']
    p3_norm = pip['3_norm']

    col_len = len(p2_kbest_cols)
    
    all_X = np.zeros([0, col_len])
    all_Y  = np.zeros(0)
    
    #print('loading & preprocessing (imputer,high variance, kbest, norm) data...',chunk_id_range)

    bar = progressbar.ProgressBar()
    for chunk_id in chunk_id_range:
        chunk = read_variable('final/tr_groups/'+str(chunk_id))
        
        all_Y = np.concatenate([all_Y,chunk['y']])
        
        pipped_X = p3_norm.transform(p1_high_variance.transform(p0_imputer.transform(chunk['x']))[:,p2_kbest_cols])
        all_X = np.concatenate([all_X,pipped_X])
        
    return all_X, all_Y.astype(np.int)

def load_pipped_test_chunk():
    pip = read_variable('final/feature_sel_pip')
    p0_imputer = pip['0_imputer']
    p1_high_variance = pip['1_high_variance']
    p2_kbest_cols = pip['2_kbest_cols']
    p3_norm = pip['3_norm']

    chunk = read_variable('final/test_chunk')    
    pipped_X = p3_norm.transform(p1_high_variance.transform(p0_imputer.transform(chunk['x']))[:,p2_kbest_cols])
        
    return pipped_X, chunk['y'].astype(np.int)    
    
max_chunk_size = 1000
col_numeric_nu = 968
col_cate_nu = 2140
col_date_nu = 1156


def load_pipped_tr_chunks(chunk_id_range):
    pip = read_variable('final/feature_sel_pip')
    p0_imputer = pip['0_imputer']
    p1_high_variance = pip['1_high_variance']
    p2_kbest_cols = pip['2_kbest_cols']
    p3_norm = pip['3_norm']

    col_len = len(p2_kbest_cols)
    
    all_X = np.zeros([0 , col_len])
    all_Y  = np.zeros(0)
    
    for chunk_id in chunk_id_range:
        # chunk has to be read one by one in sequence
        chunk_num = read_variable('data/train_numeric_chunks/'+str(chunk_id)+'.pkl')
        chunk_Y = read_variable('data/train_y_chunks/'+str(chunk_id)+'.pkl')
        temp_X= chunk_num

    
        pip_X = p3_norm.transform(p1_high_variance.transform(p0_imputer.transform(temp_X))[:,p2_kbest_cols])
    
        all_X = np.concatenate([all_X,pip_X])
        all_Y = np.concatenate([all_Y,chunk_Y])
        
    return all_X, all_Y.astype(np.int)

def load_pipped_test_chunks(chunk_id_range):
    pip = read_variable('final/feature_sel_pip')
    p0_imputer = pip['0_imputer']
    p1_high_variance = pip['1_high_variance']
    p2_kbest_cols = pip['2_kbest_cols']
    p3_norm = pip['3_norm']

    col_len = len(p2_kbest_cols)
    
    all_X = np.zeros([0 , col_len])
    
    #print('loading & preprocessing (hd, imputer, kbest) data...',chunk_id_range)
   
    for chunk_id in chunk_id_range:
        # chunk has to be read one by one in sequence
        chunk_num = read_variable('data/test_numeric_chunks/'+str(chunk_id)+'.pkl')

        temp_X = chunk_num

    
        pip_X = p3_norm.transform(p1_high_variance.transform(p0_imputer.transform(temp_X))[:,p2_kbest_cols])
    
        all_X = np.concatenate([all_X,pip_X])

    return all_X




    