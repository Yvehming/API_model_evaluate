# import os
# import argparse
# import cv2
# import numpy as np
# import glob
#
# MODEL_NAME = ""
# GRAPH_NAME = "pepper_detect3.tflite"
# LABELMAP_NAME = "model_data/pepper_classs.txt"
# min_conf_threshold = 0.5
# use_TPU = False
# IM_NAME = 'image/phone33_y1.jpg'
#
# # Import TensorFlow libraries
# # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# # If using Coral Edge TPU, import the load_delegate library
#
# from tflite_runtime.interpreter import Interpreter
# # Get path to current working directory
# CWD_PATH = os.getcwd()
#
# # Path to .tflite file, which contains the model that is used for object detection
# PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
#
# # Path to label map file
# PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)
#
# # Load the label map
# with open(PATH_TO_LABELS, 'r') as f:
#     labels = [line.strip() for line in f.readlines()]
#
# # Load the Tensorflow Lite model.
# interpreter = Interpreter(model_path=PATH_TO_CKPT)
# interpreter.allocate_tensors()
#
# # Get model details
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# height = input_details[0]['shape'][1]
# width = input_details[0]['shape'][2]
#
# floating_model = (input_details[0]['dtype'] == np.float32)
#
# input_mean = 127.5
# input_std = 127.5
#
# # Loop over every image and perform detection
# image_path = "image/phone33_y1.jpg"
# # Load image and resize to expected shape [1xHxWx3]
# image = cv2.imread(image_path)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# imH, imW, _ = image.shape
# image_resized = cv2.resize(image_rgb, (width, height))
# input_data = np.expand_dims(image_resized, axis=0)
#
# # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
# if floating_model:
#     input_data = (np.float32(input_data) - input_mean) / input_std
#
# # Perform the actual detection by running the model with the image as input
# interpreter.set_tensor(input_details[0]['index'], input_data)
# interpreter.invoke()
#
# # Retrieve detection results
# boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
# classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
# scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
# # num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
# print(scores)
# # Loop over all detections and draw detection box if confidence is above minimum threshold
# for i in range(len(scores)):
#     if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
#         # Get bounding box coordinates and draw box
#         # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
#         ymin = int(max(1, (boxes[i][0] * imH)))
#         xmin = int(max(1, (boxes[i][1] * imW)))
#         ymax = int(min(imH, (boxes[i][2] * imH)))
#         xmax = int(min(imW, (boxes[i][3] * imW)))
#
#         cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
#
#         # Draw label
#         object_name = labels[int(classes[i])]  # Look up object name from "labels" array using class index
#         label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
#         labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
#         label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
#         cv2.rectangle(image, (xmin, label_ymin - labelSize[1] - 10),
#                       (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
#                       cv2.FILLED)  # Draw white box to put label text in
#         cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
#                     2)  # Draw label text
#
# # All the results have been drawn on the image, now display the image
# cv2.imshow('Object detector', image)
# # cv2.imwrite('result.jpg', image)
# # Press any key to continue to next image, or press 'q' to quit
# cv2.waitKey(0)
#
# # Clean up
# cv2.destroyAllWindows()

import os
import xml.etree.ElementTree as ET

import cv2

# import show_testing_result
import tensorflow as tf
from tqdm import tqdm

from get_map_text import get_map_lite_txt
from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":
    '''
    Recall和Precision不像AP是一个面积的概念，在门限值不同时，网络的Recall和Precision值是不同的。
    map计算结果中的Recall和Precision代表的是当预测时，门限置信度为0.5时，所对应的Recall和Precision值。

    此处获得的./map_out/detection-results/里面的txt的框的数量会比直接predict多一些，这是因为这里的门限低，
    目的是为了计算不同门限条件下的Recall和Precision值，从而实现map的计算。
    '''
    # ------------------------------------------------------------------------------------------------------------------#
    #   map_mode用于指定该文件运行时计算的内容
    #   map_mode为0代表整个map计算流程，包括获得预测结果、获得真实框、计算VOC_map。
    #   map_mode为1代表仅仅获得预测结果。
    #   map_mode为2代表仅仅获得真实框。
    #   map_mode为3代表仅仅计算VOC_map。
    #   map_mode为4代表利用COCO工具箱计算当前数据集的0.50:0.95map。需要获得预测结果、获得真实框后并安装pycocotools才行
    # -------------------------------------------------------------------------------------------------------------------#
    map_mode = 0
    # -------------------------------------------------------#
    #   此处的classes_path用于指定需要测量VOC_map的类别
    #   一般情况下与训练和预测所用的classes_path一致即可
    # -------------------------------------------------------#
    classes_path = 'model_data/pepper_classs.txt'
    # -------------------------------------------------------#
    #   MINOVERLAP用于指定想要获得的mAP0.x
    #   比如计算mAP0.75，可以设定MINOVERLAP = 0.75。
    # -------------------------------------------------------#
    MINOVERLAP = 0.5
    # -------------------------------------------------------#
    #   map_vis用于指定是否开启VOC_map计算的可视化
    # -------------------------------------------------------#
    map_vis = False
    # -------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    # -------------------------------------------------------#
    VOCdevkit_path = 'VOCdevkit'
    # -------------------------------------------------------#
    #   结果输出的文件夹，默认为map_out
    # -------------------------------------------------------#
    map_out_path = 'map_out'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")

        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
            image = cv2.imread(image_path)
            # if map_vis:
            #     image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            get_map_lite_txt(class_names, image_id, image, map_out_path)
        print("Get predict result done.")

    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/" + image_id + ".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult') != None:
                        difficult = obj.find('difficult').text
                        if int(difficult) == 1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, path=map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names=class_names, path=map_out_path)
        print("Get map done.")
    if map_vis:
        pass
        # show_testing_result.show_testing_result()
