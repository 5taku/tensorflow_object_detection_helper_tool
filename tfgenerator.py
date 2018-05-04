import os
import argparse
from random import shuffle
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def user_input():
    config = argparse.ArgumentParser()
    config.add_argument('-i','--input_folder',help='Input Images Forlder',default='./images/',type=str, required=False)
    args = config.parse_args()
    arguments = vars(args)
    records = []
    records.append(arguments)
    return records


def xml_to_csv(path):

    print(path)

    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
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
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    print(len(xml_df))
    return xml_df


def main():

    record = user_input()

    for arguments in record:
        if arguments['input_folder']:
            input_folder = arguments['input_folder']

    xml_df = xml_to_csv(input_folder)

    #xml_df.to_csv('raccoon_labels.csv', index=None)
    #print('Successfully converted xml to csv.')


main()