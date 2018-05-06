import subprocess
from utils.utils import download_model, remove_model_tar_file, model_input, model_dict, remake_config
import logging
import argparse

def user_input():
    config = argparse.ArgumentParser()
    config.add_argument('-tr', '--train_record_output', help='Train record output file Location',default='./dataset/train.record',type=str, required=False)
    config.add_argument('-vr', '--validate_record_output', help='Validate record output file Location',default='./dataset/validate.record',type=str, required=False)
    config.add_argument('-m', '--max_num_classes', help='Maximum class number', default='90', type=str,required=False)
    config.add_argument('-i','--input_folder',help='Input Images Forlder',default='./images/',type=str, required=False)
    config.add_argument('-l', '--label_file', help='Label file Location', default='./label_map.pbtxt', type=str,required=False)
    config.add_argument('-tc', '--train_csv_output', help='Train csv output file Location', default='./dataset/train.csv', type=str,required=False)
    config.add_argument('-vc', '--validate_csv_output', help='Validate csv output file Location', default='./dataset/validate.csv', type=str,required=False)
    config.add_argument('-sr', '--split_rate', help='Dataset split rate ( 8 = train 80 | validate 20 )', default='8', type=str, required=False)
    args = config.parse_args()
    arguments = vars(args)
    records = []
    records.append(arguments)
    return records

# re-training 수행
def transfer_learning(model):
    print("+++++++++++++++++++++++++++++++++++")
    print("++++++  Transfer learning    ++++++")
    print("+++++++++++++++++++++++++++++++++++")
    print("transfer_learning process")
    train_dir = './train_dir' + model_dict[model][0]
    #subprocess.call("object_detection/train.py --logtostderr --train_dir=" + train_dir + " --pipeline_config="+ model_dict[model][1],shell=True)
    subprocess.call(['python', 'object_detection/train.py',' --logtostderr','','--train_dir',train_dir,'--pipeline_config',model_dict[model][1]])
    #subprocess.call(['python', 'tfgenerator.py','-sr','8'])

# export 수행
def export_model():
    print("+++++++++++++++++++++++++++++++++++")
    print("++++++    Export learning    ++++++")
    print("+++++++++++++++++++++++++++++++++++")
    print("transfer_learning process")
# 완료


def main():

    record = user_input()

    model = model_input()

    #Download model zoo file into the device
    #download_model(model)
    #remove_model_tar_file(model)

    print("")
    exam_num = int(input('Input example count : '))
    print("")

    for arguments in record:
        if arguments['label_file']:
            label_file = arguments['label_file']

    remake_config(model,exam_num,record)
    transfer_learning(model)
    #export_model()

main()