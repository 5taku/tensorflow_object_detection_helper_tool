import os
import tarfile
from tqdm import tqdm
import requests
import math

model_dict = {1:'ssd_mobilenet_v1_coco_2017_11_17' ,
              2:'ssd_mobilenet_v2_coco_2018_03_29' ,
              3:'ssd_inception_v2_coco_2017_11_17' ,
              4:'faster_rcnn_inception_v2_coco_2018_01_28' ,
              5:'faster_rcnn_resnet50_coco_2018_01_28' ,
              6:'faster_rcnn_resnet50_lowproposals_coco_2018_01_28' ,
              7:'rfcn_resnet101_coco_2018_01_28' ,
              8:'faster_rcnn_resnet101_coco_2018_01_28' ,
              9:'faster_rcnn_resnet101_lowproposals_coco_2018_01_28' ,
              10:'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28' ,
              11:'faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28' ,
              12:'faster_rcnn_nas_2018_01_28' ,
              13:'faster_rcnn_nas_lowproposals_coco_2018_01_28'}

def download_model(modelnum):

    MODEL_NAME = model_dict[modelnum]
    DIC_NAME = './model_zoo/'
    MODEL_DIC_NAME = DIC_NAME + MODEL_NAME
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    FULL_FILE_PATH = MODEL_DIC_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    if not os.path.isdir(MODEL_DIC_NAME):
        print(MODEL_NAME + " Model not Exist. Download start.")

        # Streaming, so we can iterate over the response.
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
            print("ERROR, something went wrong")
        tar_file = tarfile.open(FULL_FILE_PATH)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, DIC_NAME )


def remove_model_tar_file(modelnum):

    MODEL_NAME = model_dict[modelnum]
    DIC_NAME = './model_zoo/'
    MODEL_DIC_NAME = DIC_NAME + MODEL_NAME
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    FULL_FILE_PATH = MODEL_DIC_NAME + '.tar.gz'

    if os.path.exists(FULL_FILE_PATH):
        os.remove(FULL_FILE_PATH)

def change_exam_count(modelnum, examnum):
    MODEL_NAME = model_dict[modelnum]
    MODEL_CONF_NAME = MODEL_NAME[0:MODEL_NAME.find('_201')] + '.config'
    f = open('./model_conf/'+MODEL_CONF_NAME)

    while True:
        line = f.readline()
        if "num_examples" in line:
            print("Okay")
        if not line: break
    f.close()
