import os
import io
import time
from PIL import Image
import tensorflow as tf
import argparse
import numpy as np
import glob
import pandas as pd
from random import shuffle
from object_detection.utils import label_map_util
from object_detection.utils import dataset_util
import xml.etree.ElementTree as ET
from collections import namedtuple
from utils.utils import set_log, check_time

global max_num_classes

def make_summary(rows):
    logger.info('{0:^50}'.format('TF Record Summary'))
    logger.info('{0:^10}'.format('ID') + '{0:^20}'.format('NAME') + '{0:^10}'.format('Train') + '{0:^10}'.format('Validate'))
    for i in rows:
        logger.info('{0:^10}'.format(i[0]) + '{0:^20}'.format(i[1]) + '{0:^10}'.format(i[2]) + '{0:^10}'.format(i[3]))


def get_label_category(label_file):
    label_map = label_map_util.load_labelmap(label_file)
    return label_map_util.convert_label_map_to_categories(label_map, max_num_classes=max_num_classes,use_display_name=True)

def make_category_dict(categories):
    category_dict = {}
    for i in range(len(categories)):
        category_dict[categories[i]['name']] = categories[i]['id']
    return category_dict

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def user_input():
    config = argparse.ArgumentParser()
    config.add_argument('-m', '--max_num_classes', help='Maximum class number', default='90', type=str,required=False)
    config.add_argument('-i','--input_folder',help='Input Images Forlder',default='./images/',type=str, required=False)
    config.add_argument('-l', '--label_file', help='Label file Location', default='./label_map.pbtxt', type=str,required=False)
    config.add_argument('-tc', '--train_csv_output', help='Train csv output file Location', default='./dataset/train.csv', type=str,required=False)
    config.add_argument('-vc', '--validate_csv_output', help='Validate csv output file Location', default='./dataset/validate.csv', type=str,required=False)
    config.add_argument('-sr', '--split_rate', help='Dataset split rate ( 8 = train 80 | validate 20 )', default='8', type=str, required=False)
    config.add_argument('-lv', '--log_level',help='Logger Level [DEBUG, INFO(Default), WARNING, ERROR, CRITICAL]', default='INFO', type=str,required=False)
    args = config.parse_args()
    arguments = vars(args)
    records = []
    records.append(arguments)
    return records

def xml_to_csv(record):

    for arguments in record:
        if arguments['split_rate']:
            split_rate = int(arguments['split_rate'])
        if arguments['input_folder']:
            input_folder = arguments['input_folder']
        if arguments['label_file']:
            label_file = arguments['label_file']
        if arguments['train_csv_output']:
            train_csv_output = arguments['train_csv_output']
        if arguments['validate_csv_output']:
            validate_csv_output = arguments['validate_csv_output']

    labels = []
    for i in range(len(categories)):
        labels.append(categories[i]['name'])

    xml_list = [[]]
    for i in range(len(labels)):
        xml_list.append([])

    for xml_file in glob.glob(input_folder + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list[labels.index(member[0].text)].append(value)


    for i in range(len(labels)):
        shuffle(xml_list[i])

    train = []
    validate = []
    summaries = []

    for i in range(len(labels)):
        tmptrain, tmpvalidate = np.split(xml_list[i],[int((split_rate/10.0)*len(xml_list[i]))])

        summary = (category_dict[xml_list[i][0][3]], xml_list[i][0][3],len(tmptrain),len(tmpvalidate))
        summaries.append(summary)

        train.extend(tmptrain)
        validate.extend(tmpvalidate)

    make_summary(summaries)
    shuffle(train)
    shuffle(validate)

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    train_df = pd.DataFrame(train, columns=column_name)
    validate_df = pd.DataFrame(validate, columns=column_name)

    train_df.to_csv(train_csv_output, index=None)
    validate_df.to_csv(validate_csv_output, index=None)

    return train_df, validate_df

def create_tf_example(group, path):

    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(int(row['xmin']) / width)
        xmaxs.append(int(row['xmax']) / width)
        ymins.append(int(row['ymin']) / height)
        ymaxs.append(int(row['ymax']) / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(category_dict[row['class']])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main():

    record = user_input()
    start_time = time.time()
    for arguments in record:
        if arguments['max_num_classes']:
            max_num_classes = int(arguments['max_num_classes'])
        if arguments['input_folder']:
            input_folder = arguments['input_folder']
        if arguments['label_file']:
            label_file = arguments['label_file']
        if arguments['log_level']:
            log_level = arguments['log_level']

    # logger setting
    global logger
    logger = set_log(log_level)
    logger.info('TF Record Generator Start')
    global categories
    categories = get_label_category(label_file)

    global category_dict
    category_dict = make_category_dict(categories)

    #make xml file to dataframe
    train_df, validate_df = xml_to_csv(record)

    #make train record
    grouped = split(train_df, 'filename')
    writer = tf.python_io.TFRecordWriter('./dataset/train.record')
    for group in grouped:
        tf_example = create_tf_example(group, input_folder)
        writer.write(tf_example.SerializeToString())

    writer.close()

    #make validation record
    grouped = split(validate_df, 'filename')
    writer = tf.python_io.TFRecordWriter('./dataset/validate.record')
    for group in grouped:
        tf_example = create_tf_example(group, input_folder)
        writer.write(tf_example.SerializeToString())

    writer.close()
    end_time = time.time()
    h, m, s = check_time(int(end_time - start_time))
    logger.info('TF Record Generator End [ Total Generator time : '+h+' Hour '+m+' Minute '+s+' Second ]')

main()