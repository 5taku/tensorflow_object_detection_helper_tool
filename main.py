import os
import utils

print("+++++++++++++++++++++++++++++++++++")
print("++++++ Auto re training tool ++++++")
print("+++++++++++++++++++++++++++++++++++")
print("")
print("Step 1 Choose model ")
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

#download models
utils.download_model(model)
utils.remove_model_tar_file(model)

print("")
exam_num = int(input('Input example count : '))
print("")

print(exam_num)