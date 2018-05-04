import argparse
import numpy as np
from random import shuffle
from object_detection.utils import label_map_util
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def user_input():
    config = argparse.ArgumentParser()
    config.add_argument('-m', '--max_num_classes', help='Maximum class number', default='90', type=str,required=False)
    config.add_argument('-i','--input_folder',help='Input Images Forlder',default='./images/',type=str, required=False)
    config.add_argument('-l', '--label_file', help='Label file Location', default='./label_map.pbtxt', type=str,required=False)
    config.add_argument('-t', '--train_output', help='Train output file Location', default='./train.csv', type=str,required=False)
    config.add_argument('-v', '--validate_output', help='Validate output file Location', default='./validate.csv', type=str,required=False)
    config.add_argument('-sr', '--split_rate', help='Dataset split rate ( 8 = train 80 | validate 20 )', default='8', type=str, required=False)
    args = config.parse_args()
    arguments = vars(args)
    records = []
    records.append(arguments)
    return records

def xml_to_csv(record):

    for arguments in record:
        if arguments['max_num_classes']:
            max_num_classes = int(arguments['max_num_classes'])
        if arguments['split_rate']:
            split_rate = int(arguments['split_rate'])
        if arguments['input_folder']:
            input_folder = arguments['input_folder']
        if arguments['label_file']:
            label_file = arguments['label_file']
        if arguments['train_output']:
            train_output = arguments['train_output']
        if arguments['validate_output']:
            validate_output = arguments['validate_output']

    label_map = label_map_util.load_labelmap(label_file)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=max_num_classes,use_display_name=True)

    labels = []
    for i in range(len(categories)):
        labels.append(categories[i]['name'])
    print(labels)
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
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

    for i in range(len(labels)):
        shuffle(xml_list[i])

    train = []
    validate = []

    for i in range(len(labels)):
        tmptrain, tmpvalidate = np.split(xml_list[i],[int((split_rate/10)*len(xml_list[i]))])
        train.extend(tmptrain)
        validate.extend(tmpvalidate)

    shuffle(train)
    shuffle(validate)

    train_df = pd.DataFrame(train, columns=column_name)
    validate_df = pd.DataFrame(validate, columns=column_name)

    train_df.to_csv(train_output, index=None)
    validate_df.to_csv(validate_output, index=None)

def main():

    record = user_input()
    xml_df = xml_to_csv(record)

main()