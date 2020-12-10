import os
import json
from termcolor import colored
from datetime import datetime

def get_support_accuracy_metrics():
    return ['rmse', 'mape', 'mae', 'r2']

def get_support_devices():
    dict_ = {
        '1080ti': 'gpu',
        'p1000': 'gpu',
        'p2000': 'gpu',
        'p4000': 'gpu',
        'p5000': 'gpu',
    }
    return dict_

def get_support_layers():
    return ['convolution', 'pooling', 'dense']

def get_verify_models():
    return ['lenet', 'alexnet', 'vgg16']

def get_model_csv_total_columns():
    total_cols = ['layers', 'name', 'operation']
    total_cols = total_cols + [i for i in get_cov_colnames()   if i not in total_cols]
    total_cols = total_cols + [i for i in get_dense_colnames() if i not in total_cols]
    total_cols = total_cols + [i for i in get_pool_colnames()  if i not in total_cols]
    total_cols = total_cols + [i for i in get_time_colnames() if i not in total_cols]
    total_cols = total_cols + [i for i in get_profile_colnames()  if i not in total_cols]
    return total_cols

def get_model_predict_total_columns():
    total_cols = ['layers', 'name', 'operation']
    total_cols = total_cols + [i for i in get_cov_colnames()   if i not in total_cols]
    total_cols = total_cols + [i for i in get_dense_colnames() if i not in total_cols]
    total_cols = total_cols + [i for i in get_pool_colnames()  if i not in total_cols]
    total_cols = total_cols + [i for i in get_time_colnames() if i not in total_cols]
    total_cols = total_cols + [i for i in get_profile_colnames()  if i not in total_cols]
    total_cols = total_cols + [i for i in get_predict_time_colnames()  if i not in total_cols]
    return total_cols

def get_colnames(typename):
    if typename == 'convolution':
        return get_cov_colnames()
    elif typename == 'dense':
        return get_dense_colnames()
    elif typename == 'pooling':
        return get_pool_colnames()
    else:
        print("This type of layer is not support!")
        return

def get_hash_colnames():
    return ['hashkey']

def get_support_activation():
    return ['None', 'tf.nn.relu'] 

def get_cov_colnames():
    cols_ = ['batchsize', 'matsize', 'kernelsize', 'channels_in', 'channels_out', 'strides', 
        'padding', 'activation_fct', 'use_bias', 'elements_matrix', 'elements_kernel']
    return cols_ 

def get_dense_colnames():
    cols_ = ['batchsize', 'dim_input', 'dim_output', 'use_bias', 'activation_fct']
    return cols_

def get_pool_colnames():
    cols_ = ['batchsize', 'matsize', 'channels_in', 'poolsize', 'strides', 'padding', 'elements_matrix']
    return cols_

def get_time_colnames():
    return ['time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean']

def get_profile_colnames():
    return ['preprocess_time', 'execution_time', 'postprocess_time', 'sess_time']

def get_predict_time_colnames():
    return ['pre_time_abse', 'pre_time_re', 'pre_time_rmse']

def get_colnames_from_dict():
    time_colnames    = get_time_colnames()
    profile_colnames = get_profile_colnames()
    conv_colnames  = get_cov_colnames()
    dense_colnames = get_dense_colnames()
    pool_colnames  = get_pool_colnames()
    cols_dict = {
        'convolution': conv_colnames,
        'dense': dense_colnames,
        'pooling': pool_colnames,
        'profile': profile_colnames,
        'time': time_colnames,
        'hash': get_hash_colnames()
    }
    return cols_dict

def backup_file(file_path):
    ### Backup the Output CSV file
    warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
    base_name = os.path.basename(file_path)
    path = os.path.dirname(file_path)
    split_basname = os.path.splitext(base_name)
    bk_filename = split_basname[0] + '_' + datetime.now().strftime('%m%d-%H%M%S') + split_basname[1]
    print(warn_tag + 'Ouput CSV: ' + file_path + ' is existed, backup as ' + bk_filename)
    os.rename(file_path, os.path.join(path, bk_filename))

def write_file(data, path, file):
    file_path = os.path.join(path, file)
    warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
    if not os.path.isdir(path):
        os.makedirs(path)
    else:
        if os.path.isfile(file_path):
            backup_file(file_path)
    print(warn_tag + 'Auto create file: ' + file_path)
    data.to_csv(file_path, index=False)

def append_file(data, path, file):
    file_path = os.path.join(path, file)
    data.to_csv(file_path, index=False, mode='a', header=False)

def get_feature_target(filename): ### OK 
    print("==> get the feature and target...")
    if os.path.isfile(filename):
        with open(filename) as json_file:
            data = json.load(json_file)
            feature = data["feature"]
            target = data["target"]
    else:
        print("[FT] feature/target json file is not found, use the default opt")
        feature = get_cov_colnames()
        target  = get_time_colnames()[3]
    return feature, target

def _get_time_str():
    return datetime.now().strftime('%m%d-%H%M%S')