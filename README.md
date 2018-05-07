# **Object Detection with Tensorflow Helper Tool**

### Summary

This is Helper Tool for Google Tensorflow Object Detection API.

Key features include:
> 1. Create tfrecord file
>2. Re-training Automation
>3. Active Learning Assistant ( Not yet )

#### 1. tfrecord Generator

##### Advance Preparation

Dataset collection using [labelimg](https://github.com/tzutalin/labelImg)

It is recommended that both the original image and the xml file be placed in the image folder of the corresponding git.
(You can also put it in your custom folder.)

##### Usage - Using Command Line Interface

    python tfgenerator.py [Arguments...]

##### Arguments

| Argument               | Short Hand | Default                   | Description                                       |
|------------------------|------------|---------------------------|---------------------------------------------------|
| max_num_classes        | m          | 90                        | Maximum class number                              |
| input_folder           | i          | ./images/                 | Input Images Folder                               |
| label_file             | l          | ./label_map.pbtxt         | Label file Location                               |
| train_csv_output       | tc         | ./dataset/train.csv       | Train csv output file Location                    |
| validate_csv_output    | vc         | ./dataset/validate.csv    | Validate csv output file Location                 |
| split_rate             | sr         | 8                         | Dataset split rate ( 8 = train 80 | validate 20 ) |
