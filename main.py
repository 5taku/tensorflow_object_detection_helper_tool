import os
import subprocess
from utils.utils import download_model, remove_model_tar_file, model_input, model_dict, remake_config, check_time, set_log
import argparse
import shutil
import time

def user_input():
    config = argparse.ArgumentParser()
    config.add_argument('-l', '--label_file', help='Label file Location', default='./label_map.pbtxt', type=str,required=False)
    config.add_argument('-log_level', '--log_level', help='Logger Level [DEBUG, INFO(Default), WARNING, ERROR, CRITICAL]', default='INFO', type=str, required=False)
    config.add_argument('-r', '--reset', help='Training Resset configration [ Default = False ]', default=False, type=str,required=False)
    config.add_argument('-e', '--evaluate', help='Perform an evaluate every evaluate_number times.  [ Default = True ]', default=True, type=str, required=False)
    config.add_argument('-n', '--evaluate_number', help='Perform an evaluate every evaluate_number times.  [ Default = 2000 ]', default='2000', type=str, required=False)
    args = config.parse_args()
    arguments = vars(args)

    return arguments


# re-training func
def transfer_learning(logger, model,args):
    start_time = time.time()
    logger.info('Transfer learning start')

    if args['reset']:
        shutil.rmtree('./train_dir/' + model_dict[model][0])
        os.mkdir('./train_dir/' + model_dict[model][0])
        args['reset'] = False

    train_dir = './train_dir/' + model_dict[model][0]
    config_file = './model_conf/' + model_dict[model][1]
    try:
        subprocess.check_output(['python', 'object_detection/train.py', ' --logtostderr', '--train_dir', train_dir,
                     '--pipeline_config_path', config_file])
    except:
        logger.error('Transfer leaarning Error')
        exit()
    end_time = time.time()
    h,m,s = check_time(int(end_time-start_time))
    logger.info('Transfer learning Success [ Total learning time : '+h+" Hour "+m+" Minute "+s+" Second ]")

# export func
def export_model(logger, model, exam_num):
    logger.info('Export model start')
    if os.path.isdir('./export_dir/' + model_dict[model][0]):
        shutil.rmtree('./export_dir/' + model_dict[model][0])
    export_dir = './export_dir/' + model_dict[model][0]
    config_file = './model_conf/' + model_dict[model][1]
    trained_checkpoint = './train_dir/' + model_dict[model][0] + '/model.ckpt-' + str(exam_num)
    try:
        subprocess.check_output(['python', 'object_detection/export_inference_graph.py',
                     '--input_type', 'image_tensor',
                     '--pipeline_config_path', config_file,
                     '--trained_checkpoint_prefix', trained_checkpoint,
                     '--output_directory', export_dir])
    except:
        logger.error('Export Model Error')
        exit()
    logger.info('Export model Success')

# evaluate func
def evaluate_model(logger, model, num_steps):
    logger.info('Evaluate model start [ Step number : ' + str(num_steps) + " ]")
    config_file = './model_conf/' + model_dict[model][1]
    try:
        subprocess.check_output(['python', 'object_detection/eval.py',
                     '--logtostderr',
                     '--pipeline_config_path', config_file,
                     '--checkpoint_dir', './train_dir/' + model_dict[model][0],
                     '--eval_dir',  './eval_dir/' + model_dict[model][0] ,
                     '--run_once','True'])
    except:
        logger.error('Evaluate Model Error')
        exit()
    logger.info('Evaluate model Success')

def main():

    args = user_input()

    # logger setting
    logger = set_log(args['log_level'])

    model = model_input()

    print("")
    num_steps = int(input('Input number steps : '))
    print("")

    total_start_time = time.time()
    logger.info('Program start [ model : ' + model_dict[model][0] + ', num steps : ' + str(num_steps) + ' ]')

    # Download model zoo file into the device
    download_model(logger,model)
    remove_model_tar_file(model)

    if args['reset']:
        if os.path.isdir('./eval_dir/' + model_dict[model][0]):
            shutil.rmtree('./eval_dir/' + model_dict[model][0])

    if args['evaluate']:
        tmp_num = int(args['evaluate_number'])
        while tmp_num < num_steps:
            remake_config(model, tmp_num, args)
            transfer_learning(logger, model, args)
            evaluate_model(logger, model, tmp_num)
            tmp_num += int(args['evaluate_number'])
        remake_config(model, num_steps, args)
        transfer_learning(logger, model, args)
        evaluate_model(logger, model, num_steps)
    else:
        remake_config(model, num_steps, args)
        transfer_learning(logger, model, args)

    export_model(logger, model, num_steps)
    total_end_time = time.time()
    h, m, s = check_time(int(total_end_time - total_start_time))
    logger.info('Program end [ Total time : '+h+" Hour "+m+" Minute "+s+" Second ]")
    logger.info('')

main()

#TO-DO visualization func add
#TO-DO Active learning check
#TO-D0 model download and test

