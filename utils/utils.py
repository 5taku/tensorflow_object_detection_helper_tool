import os
import tarfile
import fileinput
import sys
from tqdm import tqdm
import requests
import math
import logging

def set_log(log_level):
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

    return logger

global model_dict
model_dict= {1:['ssd_mobilenet_v1_coco_2017_11_17', 'ssd_mobilenet_v1_coco.config'],
              2:['ssd_mobilenet_v2_coco_2018_03_29' ,'ssd_mobilenet_v2_coco.config'] ,
              3:['ssd_inception_v2_coco_2017_11_17' ,'ssd_inception_v2_coco.config' ],
              4:['faster_rcnn_inception_v2_coco_2018_01_28','faster_rcnn_inception_v2_coco.config' ],
              5:['faster_rcnn_resnet50_coco_2018_01_28','faster_rcnn_resnet50_coco.config' ],
              6:['faster_rcnn_resnet50_lowproposals_coco_2018_01_28','faster_rcnn_resnet50_coco.config' ],
              7:['rfcn_resnet101_coco_2018_01_28','rfcn_resnet101_coco.config' ],
              8:['faster_rcnn_resnet101_coco_2018_01_28' ,'faster_rcnn_resnet101_coco.config'],
              9:['faster_rcnn_resnet101_lowproposals_coco_2018_01_28','faster_rcnn_resnet101_coco.config' ],
              10:['faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28','faster_rcnn_inception_resnet_v2_atrous_coco.config' ],
              11:['faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28','faster_rcnn_inception_resnet_v2_atrous_coco.config' ],
              12:['faster_rcnn_nas_coco_2018_01_28','faster_rcnn_nas_coco.config' ],
              13:['faster_rcnn_nas_lowproposals_coco_2018_01_28','faster_rcnn_nas_coco.config']}

def model_input():
    print("+++++++++++++++++++++++++++++++++++")
    print("++++++ Auto re training tool ++++++")
    print("++++++        5TAKU          ++++++")
    print("+++++++++++++++++++++++++++++++++++")
    print("")
    print("Select Model ")
    print("")
    print("1. ssd_mobilenet_v1_coco ")
    print("2. ssd_mobilenet_v2_coco ")
    print("3. ssd_inception_v2_coco ")
    print("4. faster_rcnn_inception_v2_coco ")
    print("5. faster_rcnn_resnet50_coco ")
    print("6. faster_rcnn_resnet50_lowproposals_coco ")
    print("7. rfcn_resnet101_coco ")
    print("8. faster_rcnn_resnet101_coco ")
    print("9. faster_rcnn_resnet101_lowproposals_coco ")
    print("10. faster_rcnn_inception_resnet_v2_atrous_coco ")
    print("11. faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco ")
    print("12. faster_rcnn_nas ")
    print("13. faster_rcnn_nas_lowproposals_coco ")
    print("")

    model = int(input('Select Model Number : '))

    return model

def download_model(logger,modelnum):
    MODEL_NAME = model_dict[modelnum][0]
    DIC_NAME = './model_zoo/'
    MODEL_DIC_NAME = DIC_NAME + MODEL_NAME
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    FULL_FILE_PATH = MODEL_DIC_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    if not os.path.isdir(MODEL_DIC_NAME):
        logger.info(MODEL_NAME + " Model not Exist. Download start")

        # make taring_dir folder
        if not os.path.isdir('./train_dir/'+MODEL_NAME):
            os.mkdir('./train_dir/'+MODEL_NAME)

        # make taring_dir folder
        if not os.path.isdir('./export_dir/' + MODEL_NAME):
            os.mkdir('./export_dir/' + MODEL_NAME)

        # make taring_dir folder
        if not os.path.isdir('./eval_dir/' + MODEL_NAME):
                os.mkdir('./eval_dir/' + MODEL_NAME)

        #  Streaming, so we can iterate over the response.
        r = requests.get(DOWNLOAD_BASE + MODEL_FILE, stream=True)

        # Total size in bytes.
        total_size = int(r.headers.get('content-length', 0));
        block_size = 1024
        wrote = 0
        with open(FULL_FILE_PATH, 'wb') as f:
            for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size // block_size), unit='KB',
                             unit_scale=True):
                wrote = wrote + len(data)
                f.write(data)
        if total_size != 0 and wrote != total_size:
            logger.error("ERROR, something went wrong")
        tar_file = tarfile.open(FULL_FILE_PATH)
        tar_file.extractall(DIC_NAME)
        tar_file.close()
        logger.info(MODEL_NAME + " Download success")

def remove_model_tar_file(modelnum):
    MODEL_NAME = model_dict[modelnum][0]
    DIC_NAME = './model_zoo/'
    MODEL_DIC_NAME = DIC_NAME + MODEL_NAME
    FULL_FILE_PATH = MODEL_DIC_NAME + '.tar.gz'

    if os.path.exists(FULL_FILE_PATH):
        os.remove(FULL_FILE_PATH)

def remake_config(model,exam_num,args):
    config_file = './model_conf/'+model_dict[model][1]
    model_checkpoint_path = './model_zoo/'+model_dict[model][0]+'/model.ckpt'
    num_steps = 'num_steps: '
    num_classes = 'num_classes: '
    label_map_path = 'label_map_path: '
    fine_tune_checkpoint = 'fine_tune_checkpoint: '

    class_num = 0
    for line in fileinput.input(args['label_file'], inplace=1):
        if 'id: ' in line:
            class_num += 1
        sys.stdout.write(line)

    for line in fileinput.input(config_file, inplace=1):
        if num_steps in line:
            line = line.replace(line,'  '+num_steps + str(exam_num) + '\n')
        if num_classes in line:
            line = line.replace(line,'  '+num_classes + str(class_num) + '\n')
        if label_map_path in line:
            line = line.replace(line, label_map_path + '"' + args['label_file'] + '"' + '\n')
        if fine_tune_checkpoint in line:
            line = line.replace(line, fine_tune_checkpoint + '"' + model_checkpoint_path + '"' + '\n')
        sys.stdout.write(line)

def check_time(time):
    second = str(int(time % 60))
    time /= 60
    min = str(int(time % 60))
    time /= 60
    hour = str(int(time))

    return hour,min,second
