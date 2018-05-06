import os
import subprocess
from utils.utils import download_model, remove_model_tar_file, model_input, model_dict, remake_config
import logging
import argparse
import shutil

def set_log(log_level):
    global logger
    logger = logging.getLogger('5takulogger')
    fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')

    fileHandler = logging.FileHandler('./process.log')
    streamHandler = logging.StreamHandler()

    fileHandler.setFormatter(fomatter)
    streamHandler.setFormatter(fomatter)

    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    level = logging.getLevelName(log_level)
    logger.setLevel(level)

def user_input():
    config = argparse.ArgumentParser()
    config.add_argument('-l', '--label_file', help='Label file Location', default='./label_map.pbtxt', type=str,required=False)
    config.add_argument('-log_level', '--log_level', help='Logger Level [DEBUG, INFO(Default), WARNING, ERROR, CRITICAL]', default='INFO', type=str, required=False)
    config.add_argument('-r', '--reset', help='Training Resset configration [ Default = False ]', default=False, type=str,required=False)
    args = config.parse_args()
    arguments = vars(args)
    records = []
    records.append(arguments)
    return records


# re-training 수행
def transfer_learning(model,reset):
    logger.info('Transfer learning start')

    if reset:
        shutil.rmtree('./train_dir/' + model_dict[model][0])
        os.mkdir('./train_dir/' + model_dict[model][0])

    train_dir = './train_dir/' + model_dict[model][0]
    config_file = './model_conf/' + model_dict[model][1]
    subprocess.call(['python', 'object_detection/train.py', ' --logtostderr', '', '--train_dir', train_dir,
                     '--pipeline_config_path', config_file])
    logger.info('Transfer learning end')


# export 수행
def export_model(model, exam_num):
    logger.info('Export model start')
    if os.path.isdir('./export_dir/' + model_dict[model][0]):
        shutil.rmtree('./export_dir/' + model_dict[model][0])
    export_dir = './export_dir/' + model_dict[model][0]
    config_file = './model_conf/' + model_dict[model][1]
    trained_checkpoint = './train_dir/' + model_dict[model][0] + '/model.ckpt-' + str(exam_num)
    subprocess.call(['python', 'object_detection/export_inference_graph.py',
                     '--input_type', 'image_tensor',
                     '--pipeline_config_path', config_file,
                     '--trained_checkpoint_prefix', trained_checkpoint,
                     '--output_directory', export_dir])
    logger.info('Export model end')


# 완료


def main():

    record = user_input()

    for arguments in record:
         if arguments['log_level']:
             log_level = arguments['log_level']
         if arguments['reset']:
             reset = arguments['reset']

    #logger setting
    set_log(log_level)

    model = model_input()

    print("")
    num_steps = int(input('Input number steps : '))
    print("")

    logger.info('Program start [ model : ' + model_dict[model][0] + ', num steps : ' + num_steps)

    # Download model zoo file into the device
    if download_model(model):
        remove_model_tar_file(model)

    remake_config(model, num_steps, record)
    transfer_learning(model,reset)
    export_model(model, num_steps)

    logger.info('Program end [ model : ' + model_dict[model][0] + ', num steps : ' + num_steps)

main()
