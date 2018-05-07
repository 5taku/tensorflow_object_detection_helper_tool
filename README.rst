Tensorflow Object Detection Api Helper Tool
###########################################

Contents

.. contents:: :local:

Summary
=======

Google Tensorflow 의 Object Detection API Helper Tool 입니다.

주요기능은 다음과 같습니다.

- Auto Generate TF Record File
- Auto Transfer learning & export model
- Active learning ( Not yet )

Compatibility
=============

파이썬 2.x , 3.x와 호환됩니다.

Installation
============

Manually using CLI

.. code-block:: bash

    $ git clone https://github.com/5taku/auto_re_training_tool.git

Manually using UI

Go to the `repo on github <https://github.com/5taku/auto_re_training_tool.git>`__ ==> Click on 'Clone or Download' ==> Click on 'Download ZIP' and save it on your local disk.

Directory Structure
===================



Usage - Using Command Line Interface
====================================

- TF Record Generator

.. code-block:: bash

    $ python3 tfgenerator.py [Arguments...]
    OR
    $ python tfgenerator.py [Arguments...]

- Auto Transfer learning

.. code-block:: bash

    $ python3 main.py [Arguments...]
    OR
    $ python main.py [Arguments...]


Arguments
=========

- TF Record Generator

+--------------------+-------------+-------------------------------------------------------------------------------------------------------------------------------+
| Argument           | Short hand  | Description                                                                                                                   |
+====================+=============+===============================================================================================================================+
| log level          | lv          | 로그 레벨을 지정합니다.                                                                                                       |
|                    |             |                                                                                                                               |
|                    |             | 로그는 [DEBUG , INFO , WARNING , ERROR , CRITICAL] 로 구성되어 있습니다.                                                      |
|                    |             | 기본값은 INFO 입니다.                                                                                                         |
|                    |             |                                                                                                                               |
|                    |             | * 현재는 INFO 레벨의 로그밖에 존재하지 않습니다.                                                                              |
+--------------------+-------------+-------------------------------------------------------------------------------------------------------------------------------+
| split rate         | sr          | Train, Validate 비율을 조정합니다. 기본은 Train 80% , Validate 20% 입니다.                                                    |
|                    |             |                                                                                                                               |
|                    |             | 사용 :                                                                                                                        |
|                    |             |                                                                                                                               |
|                    |             | * 만약 split rate 값이 7이라면 Train 70% , validate 30% 로 분할됩니다.                                                        |
|                    |             | * 만약 split rate 값이 5이라면 Train 50% , validate 50% 로 분할됩니다.                                                        |
|                    |             | * 만약 split rate 값이 2이라면 Train 20% , validate 80% 로 분할됩니다.                                                        |
+--------------------+-------------+-------------------------------------------------------------------------------------------------------------------------------+
| label_file         | l           | 레이블 파일 위치를 설정합니다.                                                                                                |
|                    |             | 기본 레이블 파일의 위치는 './label_map.pbtxt' 입니다.                                                                         |
|                    |             |                                                                                                                               |
|                    |             | 레이블 파일의 구성은 아래와 같습니다. ( 예 : [1:dog , 2:cat , 3:human] 의 경우 )                                              |                                                                                |
|                    |             |                                                                                                                               |
|                    |             | item {                                                                                                                        |
|                    |             |   id: 1                                                                                                                       |
|                    |             |   name: 'dog'                                                                                                                 |
|                    |             | }                                                                                                                             |
|                    |             |                                                                                                                               |
|                    |             | item {                                                                                                                        |
|                    |             |   id: 2                                                                                                                       |
|                    |             |   name: 'cat'                                                                                                                 |
|                    |             | }                                                                                                                             |
|                    |             |                                                                                                                               |
|                    |             | item {                                                                                                                        |
|                    |             |   id: 3                                                                                                                       |
|                    |             |   name: 'human'                                                                                                               |
|                    |             | }                                                                                                                             |
|                    |             |                                                                                                                               |
+--------------------+-------------+-------------------------------------------------------------------------------------------------------------------------------+
| input_folder       | i           | Image 와 xml 파일이 존재하는 폴더 위치입니다.                                                                                 |
|                    |             |                                                                                                                               |
|                    |             | 기본값은 './images' 입니다.                                                                                                   |
+--------------------+-------------+-------------------------------------------------------------------------------------------------------------------------------+
| train_csv_output   | tc          | train 부분의 xml 을 csv 형태로 변환하고, 저장할 위치입니다.                                                                   |
|                    |             |                                                                                                                               |
|                    |             | 기본값은 './dataset/train.csv' 입니다.                                                                                        |
+--------------------+-------------+-------------------------------------------------------------------------------------------------------------------------------+
| validate_csv_output| vc          | validate 부분의 xml 을 csv 형태로 변환하고, 저장할 위치입니다.                                                                |
|                    |             |                                                                                                                               |
|                    |             | 기본값은 './dataset/validate.csv' 입니다.                                                                                     |
+--------------------+-------------+-------------------------------------------------------------------------------------------------------------------------------+
| max_num_classe     | m           | 클래스의 최대 갯수 입니다.                                                                                                    |
|                    |             |                                                                                                                               |
|                    |             | 기본값은 90입니다. ( 클래스 갯수가 90이 넘을 경우 수정하세요. )                                                               |
+--------------------+-------------+-------------------------------------------------------------------------------------------------------------------------------+
| help               | h           | 위의 인자에 대한 설명을 볼 수 있습니다.                                                                                       |
+--------------------+-------------+-------------------------------------------------------------------------------------------------------------------------------+

Examples
========

- TF Record Generator

    1, 이미지폴더내에 원본 이미지와 , object 영역의 위치정보가 담긴 xml이 존재하여야 합니다.

    2. label_map 은 원하는 데이터셋에 맞게 수정하여야 합니다.

    3. 결과 record 파일은 './dataset/train.record' , './dataset/validate.record' 에서 확인할 수 있습니다.

    4. 결과에 대한 요약은 process.log 에서 확인할 수 있습니다. ( Auto Transfer learning & export model 과 로그를 공유합니다.)

- 이미지 폴더의 데이터를 Train 60% ,Validate 40% 로 분할하여 TF Record 생성

.. code-block:: bash

    $ ptyhon tfrecord.py -sr=6

--------------

Troubleshooting
===============

Contribute
==========

비효율적인 코드가 많습니다.
누구나 이 git를 수정할 수 있습니다.
만약 수정을 원하시면, open 하여 pull request 를 수행하실 수 있습니다.
For issues and discussion visit the
`Issue Tracker <https://github.com/5taku/auto_re_training_tool/issues>`__.
