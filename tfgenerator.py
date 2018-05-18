from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import time
import glob
import argparse
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET
from PIL import Image
from random import shuffle
from object_detection.utils import label_map_util
from object_detection.utils import dataset_util
from collections import namedtuple
from utils.utils import set_log, check_time

def make_summary(logger,rows):
    logger.info('{0:^50}'.format('TF Record Summary'))
    logger.info('{0:^10}'.format('ID') + '{0:^20}'.format('NAME') + '{0:^10}'.format('Train') + '{0:^10}'.format('Validate'))
    for i in rows:
        logger.info('{0:^10}'.format(i[0]) + '{0:^20}'.format(i[1]) + '{0:^10}'.format(i[2]) + '{0:^10}'.format(i[3]))


def get_label_category(args):
    label_map = label_map_util.load_labelmap(args['label_file'])
    return label_map_util.convert_label_map_to_categories(label_map, max_num_classes=int(args['max_num_classes']),use_display_name=True)

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
    config.add_argument('-m', '--max_num_classes', help='Maximum class number', default='90', type=int,required=False)
    config.add_argument('-i','--input_folder',help='Input Images Forlder',default='./images/',type=str, required=False)
    config.add_argument('-l', '--label_file', help='Label file Location', default='./label_map.pbtxt', type=str,required=False)
    config.add_argument('-c', '--custom_csv', help='Custom csv', default=False, type=str,required=False)
    config.add_argument('-tc', '--train_csv_output', help='Train csv output file Location', default='./dataset/train.csv', type=str,required=False)
    config.add_argument('-vc', '--validate_csv_output', help='Validate csv output file Location', default='./dataset/validate.csv', type=str,required=False)
    config.add_argument('-sr', '--split_rate', help='Dataset split rate ( 8 = train 80 | validate 20 )', default='8', type=int, required=False)
    config.add_argument('-lv', '--log_level',help='Logger Level [DEBUG, INFO(Default), WARNING, ERROR, CRITICAL]', default='INFO', type=str,required=False)
    args = config.parse_args()
    arguments = vars(args)

    return arguments

def xml_to_csv(logger,args):

    label_cnt = len(categories)

    labels = []
    for i in range(label_cnt):
        labels.append(categories[i]['name'])

    xml_list = []
    for i in range(label_cnt):
        xml_list.append([])

    for xml_file in glob.glob(args['input_folder'] + '/*.xml'):
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


    for i in range(label_cnt):
        shuffle(xml_list[i])

    train = []
    validate = []
    summaries = []

    for i in range(label_cnt):
        rate = int( len(xml_list[i]) * (float(args['split_rate'])/10.0))
        tmptrain = xml_list[i][:rate]
        tmpvalidate = xml_list[i][rate:]

        summary = (category_dict[xml_list[i][0][3]], xml_list[i][0][3],len(tmptrain),len(tmpvalidate))
        summaries.append(summary)

        train.extend(tmptrain)
        validate.extend(tmpvalidate)

    make_summary(logger,summaries)
    shuffle(train)
    shuffle(validate)

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    train_df = pd.DataFrame(train, columns=column_name)
    validate_df = pd.DataFrame(validate, columns=column_name)

    train_df.to_csv(args['train_csv_output'], index=None)
    validate_df.to_csv(args['validate_csv_output'], index=None)

    train_csv = pd.read_csv(args['train_csv_output'])
    validate_csv = pd.read_csv(args['validate_csv_output'])

    return train_csv, validate_csv

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

    args = user_input()
    start_time = time.time()

    # logger setting
    logger = set_log(args['log_level'])
    logger.info('TF Record Generator Start')
    global categories
    categories = get_label_category(args)

    global category_dict
    category_dict = make_category_dict(categories)

    #make xml file to dataframe
    if not args['custom_csv']:
        train, validate = xml_to_csv(logger, args)
    else:
        train = pd.read_csv(args['train_csv_output'])
        validate = pd.read_csv(args['validate_csv_output'])

    #make train record
    grouped = split(train, 'filename')
    writer = tf.python_io.TFRecordWriter('./dataset/train.record')
    for group in grouped:
        tf_example = create_tf_example(group, args['input_folder'])
        writer.write(tf_example.SerializeToString())

    writer.close()

    #make validation record
    grouped = split(validate, 'filename')
    writer = tf.python_io.TFRecordWriter('./dataset/validate.record')
    for group in grouped:
        tf_example = create_tf_example(group, args['input_folder'])
        writer.write(tf_example.SerializeToString())

    writer.close()
    end_time = time.time()
    h, m, s = check_time(int(end_time - start_time))
    logger.info('TF Record Generator End [ Total Generator time : '+h+' Hour '+m+' Minute '+s+' Second ]')

main()