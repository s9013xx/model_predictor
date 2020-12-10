import os
import re
import sys
import time
import shutil
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from termcolor import colored
from utils.utils import *
from utils.model import Model
from utils.network import get_nn_list
from utils.meter import RegErrorMeter
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

activation_list = ['None', 'tf.nn.relu']

def pred_data_preparation(flags, df_test, feature=list()):
    print("==> Do data preparation ...")

    ### Get Data from csv
    df_train = pd.read_csv(flags.train_csv)

    ### Feature transformations
    ft_str = "[Features] Transformation: {} to Features: {}".format(flags.feature_transform, flags.ft_list)
    if flags.feature_transform == "":
        ft_str = "[Features] No transformation to the Features"
    elif flags.feature_transform == "boxcox":
        for ft in flags.ft_list:
            df_train[ft], maxlog = stats.boxcox(df_train[ft])
            df_test[ft]  =  stats.boxcox(df_test[ft], maxlog)
    print(ft_str)
    train_f = df_train[feature]
    test_f  = df_test[feature]

    ### Feature StandardScale
    print("[Features] Standardscale to all Features")
    scaler = StandardScaler()
    scaler.fit(train_f.astype(float))
    train_scale = scaler.transform(train_f[feature].astype(float))
    test_scale  = scaler.transform(test_f[feature].astype(float))
    return train_scale, test_scale  #df_train[target], test_scale, df_test[target]

def pred_validation(flags, model, ckpt_path_name, testdata, testlabel): 
    print("==> Do inference...")
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    te_num_datapoints = testdata.shape[0]
    te_list_datapoints = np.arange(0,te_num_datapoints)
    te_num_batches = np.int(np.ceil(te_num_datapoints/model.batch_size))
    with tf.Session() as sess:
        sess.run(init)
        print("==> Resuming model from checkpoint..")
        print(ckpt_path_name)
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_path_name))
        regm_test = RegErrorMeter(scaler = flags.scaler)

        for i in range(0, te_num_batches):
            cur_batch_size = min((i+1)*model.batch_size,te_num_datapoints) - (i*model.batch_size)
            batch = te_list_datapoints[i*model.batch_size:min((i+1)*model.batch_size,te_num_datapoints)]
            testloss, test_pred_time = sess.run(
                [model.loss, model.prediction],
                    feed_dict={
                        model.tf_inputs: testdata[batch,:],
                        model.tf_targets: testlabel[batch],
                        model.tf_istraining: False})
            # update to meter
            regm_test.error.update(testlabel[batch], test_pred_time)
            regm_test.loss.update(testloss, cur_batch_size)
        regm_test.error.summary()
    return regm_test.error.prediction, regm_test.error.answer

def read_verify_model_flags():
    parser = argparse.ArgumentParser('Model Verify Parameters Parser')

    # General parameters
    parser.add_argument('--predition_device', '-pd', default='1080ti', 
                    type=str, help='predition device for training or testing')
    parser.add_argument('--network_name', '-n', default='perfnetA',
                    type=str, choices=get_nn_list(), help='network name for training or testing')
    parser.add_argument('--loss_name', '-lf', default='maple',
                    type=str, choices=['msle', 'mse', 'maple', 'poisson', 'mae'], help='loss function name for training or testing')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size you want to run')
    parser.add_argument('--iter_warmup', type=int, default=5, help='Number of iterations for warm-up')
    parser.add_argument('--iter_benchmark', type=int, default=10, help='Number of iterations for benchmark')

    #Data path Parameters
    parser.add_argument('--data_dirname', '-dd', default='data_full_model', 
        type=str, help='data dirname')
    parser.add_argument('--data_path', '-dp', default='', 
        type=str, help='data path')
   
    #Transformation Parameters
    parser.add_argument('--feature_transform', '-ft', default='',
                    type=str, choices=['', 'boxcox'], help='transofrmation for features')
    parser.add_argument('--ft_list', '-ftl', default=['elements_matrix', 'elements_kernel'],
                    type=str, nargs='+', help='list of features needed to do feature transformations')
    parser.add_argument('--scaler', '-scaler', default=10, 
                        type=int, help='scaler for smoothing poisson reg prediction')
 
    #Model and model path Parameters  
    parser.add_argument('--model_dirname', '-md', default='model', 
        type=str, help='model dirname')
    parser.add_argument('--model_path', '-mp', default='', 
        type=str, help='model path')
    
    parser.add_argument('--accuracy_metric', '-am', default='mape', type=str, metavar='PATH',
                       choices=get_support_accuracy_metrics(), help='best accuracy metric for loading the model')



    ### model parameters ### TBD
    parser.add_argument('--start_epoch', default=0, 
                        type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--epochs', '-e', default=200, 
                        type=int,  help='number of total epochs to run')
    parser.add_argument('--learning_rate', '-lr', default=0.1, 
                        type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, 
                        type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, 
                        type=float, metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--scheduler_step', '-st', default=150, 
                        type=int, help="step size of scheduler")
    parser.add_argument('--scheduler_gamma', '-sg', default=0.1, 
                        type=float, help="decay rate of scheduler")
                    

    #Feature and Target filename Parameters
    parser.add_argument('--ft_filename', '-ftfname', 
        default='', type=str, help='Feature and Target file name')
    parser.add_argument('--ft_dirname', '-ftdname', 
        default='Feature_Target', 
        type=str, help='Feature and Target dir name')
    parser.add_argument('--ft_filepath', '-ftf', 
        default='', 
        type=str, help='Feature and Target full path name')

    ### Predict model parameters
    parser.add_argument('--model_name', '-mn',default='lenet', type=str, 
        choices=['lenet', 'alexnet', 'vgg16'], help='Neural networks models') ## Can arbitrarily add any model if you have model structures
    parser.add_argument('--train_filename', '-tf', type=str, default='train.csv', help='The input train filename(need all csv to be same)')
    parser.add_argument('--train_path', '-tp', type=str, default='tensorflow_data', 
                    choices=['tensorflow_data', 'tensorRT_data'], help='The input main train path')
    parser.add_argument('--convoulution_sub_path', '-csp', type=str, default='', help='The conv sub train path')
    parser.add_argument('--dense_sub_path', '-dsp', type=str, default='', help='The desne sub train path')
    parser.add_argument('--pooling_sub_path', '-psp', type=str, default='', help='The pooling sub train path')
    parser.add_argument('--output_model_predict_filename', '-ompdf', type=str, default='', help='The output model csv file name')
    parser.add_argument('--output_model_predict_dirname', '-ompdd', type=str, default='model_predict', help='The dirname of the output model csv filename in generation model data step')
    parser.add_argument('--output_model_predict_path', '-ompdp', type=str, default='', help='The path of the output model csv filename in generation model data step')

    parser.add_argument('--input_model_csv_filename', '-imcf', type=str, default='', help='The input params csv file')
    parser.add_argument('--input_model_csv_dirname', '-imcd', type=str, default='model_csv', help='The dirname of the input csv')
    parser.add_argument('--input_model_csv_path', '-imcp', type=str, default='', help='The path of the input csv')

    #Feature and Target filename 
    parser.add_argument('--feature_convolution_path', default= os.path.join(os.getcwd(), 'Feature_Target', 'convolution.json'))
    parser.add_argument('--feature_dense_path', default= os.path.join(os.getcwd(), 'Feature_Target', 'dense.json'))
    parser.add_argument('--feature_pooling_path', default= os.path.join(os.getcwd(), 'Feature_Target', 'pooling.json'))
    args = parser.parse_args()
    return args

def fullproof_flags(flags):
    flags.model_batch_name = flags.model_name + '_' + str(flags.batch_size)
    flags.model_batch_device_name = flags.model_batch_name + '_' + flags.predition_device
    
    if not flags.data_path:
        flags.data_path = os.path.join(os.getcwd(), flags.data_dirname)
   
    flags.network_loss_name = flags.network_name + '_' + flags.loss_name
    if not flags.output_model_predict_filename:
        flags.output_model_predict_filename = flags.model_batch_device_name + '.csv'
    if not flags.output_model_predict_path:
        flags.output_model_predict_path = os.path.join(flags.data_path, flags.output_model_predict_dirname, flags.output_model_predict_filename)
    
    if not flags.convoulution_sub_path:
        flags.convoulution_sub_path = 'convolution_' + flags.predition_device
    if not flags.dense_sub_path:
        flags.dense_sub_path = 'dense_' + flags.predition_device
    if not flags.pooling_sub_path:
        flags.pooling_sub_path = 'pooling_' + flags.predition_device

    if not flags.input_model_csv_filename:
            flags.input_model_csv_filename = flags.model_batch_name + '.csv'
    if not flags.input_model_csv_path:
        flags.input_model_csv_path = os.path.join(flags.data_path, flags.input_model_csv_dirname, flags.input_model_csv_filename)
       
    if flags.accuracy_metric == 'mae':
        flags.amdirname = flags.network_loss_name + '_mae'
    elif flags.accuracy_metric == 'mape':
        flags.amdirname = flags.network_loss_name + '_mape'
    elif flags.accuracy_metric == 'rmse':
        flags.amdirname = flags.network_loss_name + '_rmse'
    else:
        flags.amdirname = flags.network_loss_name + '_r2'    
    return flags

def auto_create_dir(flags):
    def create_dir_elemenet(path):
        warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
        if not os.path.isdir(path):
            os.makedirs(path)
            print(warn_tag + 'Auto create dir: ' + path)
    create_dir_elemenet(flags.data_path)
    create_dir_elemenet(os.path.dirname(flags.output_model_predict_path))
    return

def main():
    warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
    success_tag = colored('[Success] ', 'green')
    flags = read_verify_model_flags()
    flags = fullproof_flags(flags)
    auto_create_dir(flags)
    model_csv_columns = get_model_csv_total_columns()

    if not os.path.isfile(flags.input_model_csv_path):
        print(flags.input_model_csv_path)
        print(warn_tag, "Please create model csv file or open the '-gmc' tag for the supported model at first!")
        return
    print("[Predict Model]")
    if os.path.isfile(flags.output_model_predict_path):
        print(warn_tag, "Alreadly have the data in %s, pass this step!" % flags.output_model_predict_path)
        df_  = pd.read_csv(flags.input_model_csv_path)
        return
    feature_conv, _    = get_feature_target(flags.feature_convolution_path) # get the feature and target 
    feature_pooling, _ = get_feature_target(flags.feature_pooling_path) # get the feature and target 
    feature_dense, _   = get_feature_target(flags.feature_dense_path) # get the feature and target 
    df_  = pd.read_csv(flags.input_model_csv_path)
    target_list = ['preprocess_time', 'execution_time', 'postprocess_time']
    dict_target_list ={
        'preprocess_time' : [],
        'execution_time' : [],
        'postprocess_time' : []
    }
    dict_target ={
        'preprocess_time' : 'pre',
        'execution_time' : 'exe',
        'postprocess_time' : 'post'
    }
    for index in range(df_.shape[0]):
        print(' ======== %d ========' % index)
        layer = df_.loc[index, :]
        if re.search('conv2d', layer['operation'], re.M|re.I):
            df_test = layer.fillna(0)
            for target in target_list:
                if index > 0 and target == 'preprocess_time':
                    dict_target_list[target].append(0)
                    continue
                if index != (df_.shape[0]-1) and (target == 'postprocess_time'):
                    dict_target_list[target].append(0)
                    continue
                df_test = layer.fillna(0)
                df_test = pd.DataFrame(df_test.values.reshape(-1, len(layer)), columns=df_.columns)     
                df_test[target] = 0 ## Init prediction
                df_test[feature_conv] = df_test[feature_conv].astype(int)
                model = Model(flags, len(feature_conv))
                flags.train_csv = os.path.join(os.getcwd(), flags.train_path, flags.convoulution_sub_path, flags.train_filename)
                flags.ft_list   = ['elements_matrix', 'elements_kernel']
                train_feature, test_feature = pred_data_preparation(flags, df_test, feature_conv)
                print("----------->", dict_target[target])
                if not dict_target[target]:
                    if not train_feature:
                        continue
                    flags.data_ex_basename = 'convolution_' + flags.predition_device 
                else:
                    flags.data_ex_basename = 'convolution_' + dict_target[target] + '_' + flags.predition_device 
                ckpt_path_name = os.path.join(os.getcwd(), flags.model_dirname, flags.data_ex_basename, flags.amdirname)
                print(ckpt_path_name)
                pred_ele, anw_t  = pred_validation(flags, model, ckpt_path_name, test_feature, df_test[target])                    
                dict_target_list[target].append(pred_ele[0])
                print(dict_target_list)
        
        elif re.search('pooling2d', layer['operation'], re.M|re.I):
            df_test = layer.fillna(0)
            for target in target_list:
                if index > 0 and target == 'preprocess_time':
                    dict_target_list[target].append(0)
                    continue
                if index != (df_.shape[0]-1) and (target == 'postprocess_time'):
                    dict_target_list[target].append(0)
                    continue
                df_test = layer.fillna(0)
                df_test = pd.DataFrame(df_test.values.reshape(-1, len(layer)), columns=df_.columns)     
                df_test[target] = 0 ## Init prediction
                df_test[feature_pooling] = df_test[feature_pooling].astype(int)
                model = Model(flags, len(feature_pooling))
                flags.train_csv = os.path.join(os.getcwd(), flags.train_path, flags.pooling_sub_path, flags.train_filename)
                flags.ft_list   = ['elements_matrix']
                train_feature, test_feature = pred_data_preparation(flags, df_test, feature_pooling)
                print("----------->", dict_target[target])
                if not dict_target[target]:
                    if not train_feature:
                        continue
                    flags.data_ex_basename = 'pooling_' + flags.predition_device 
                else:
                    flags.data_ex_basename = 'pooling_' + dict_target[target] + '_' + flags.predition_device 
                ckpt_path_name = os.path.join(os.getcwd(), flags.model_dirname, flags.data_ex_basename, flags.amdirname)
                pred_ele, anw_t  = pred_validation(flags, model, ckpt_path_name, test_feature, df_test[target])         
                dict_target_list[target].append(pred_ele[0])
                print(dict_target_list)
         
        elif re.search('dense', layer['operation'], re.M|re.I):
            df_test = layer.fillna(0)
            for target in target_list:
                if index > 0 and target == 'preprocess_time':
                    dict_target_list[target].append(0)
                    continue
                if index != (df_.shape[0]-1) and (target == 'postprocess_time'):
                    dict_target_list[target].append(0)
                    continue
                df_test = layer.fillna(0)
                df_test = pd.DataFrame(df_test.values.reshape(-1, len(layer)), columns=df_.columns)
                df_test[target] = 0 ## Init prediction
                df_test[feature_dense] = df_test[feature_dense].astype(int)
                model = Model(flags, len(feature_dense))
                flags.train_csv = os.path.join(os.getcwd(), flags.train_path, flags.dense_sub_path, flags.train_filename)
                flags.ft_list = ""
                train_feature, test_feature = pred_data_preparation(flags, df_test, feature_dense)
                print("----------->", dict_target[target])
                if not dict_target[target]:
                    if not train_feature:
                        continue
                    flags.data_ex_basename = 'dense_' + flags.predition_device 
                else:
                    flags.data_ex_basename = 'dense_' + dict_target[target] + '_' + flags.predition_device 
                ckpt_path_name = os.path.join(os.getcwd(), flags.model_dirname, flags.data_ex_basename, flags.amdirname)
                pred_ele, anw_t  = pred_validation(flags, model, ckpt_path_name, test_feature, df_test[target])
                dict_target_list[target].append(pred_ele[0])
                print(dict_target_list)
 
    if dict_target_list['preprocess_time']:
        df_['pred_pre_time'] = dict_target_list['preprocess_time']
    if dict_target_list['execution_time']:
        df_['pred_exe_time'] = dict_target_list['execution_time']
    if dict_target_list['postprocess_time']:
        df_['pred_post_time'] = dict_target_list['postprocess_time']
    if dict_target_list['preprocess_time'] and  dict_target_list['execution_time'] and dict_target_list['postprocess_time']:
        sum_time = np.sum(df_['pred_exe_time']) + df_.iloc[0]['pred_pre_time'] + df_.iloc[-1]['pred_post_time']
        #df_['sum_time'] = sum_time
    df_.to_csv(flags.output_model_predict_path, index=False)
    print('Total_Execution_Time = ', sum_time)
    print(success_tag, "Create file to %s!" % flags.output_model_predict_path)

if __name__ == '__main__':
    main()
