import os
import numpy as np
import tensorflow as tf
import cv2


def get_map_txt(model, class_names, pruned_classes, pruned_scores, pruned_boxes, image_id, image, map_out_path):
    # 载入模型
    load_model = model


    f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
    # ---------------------------------------------------------#
    #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    # ---------------------------------------------------------#
    imH, imW, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (300, 300))
    image_expanded = np.expand_dims(image_resized, axis=0)
    classes = pruned_classes(tf.constant(image_expanded, dtype=tf.uint8)).numpy()
    scores = pruned_scores(tf.constant(image_expanded, dtype=tf.uint8)).numpy()
    boxes = pruned_boxes(tf.constant(image_expanded, dtype=tf.uint8)).numpy()
    for i in range(boxes.shape[1]):
        boxes[0][i][0] = int(max(1, (boxes[0][i][0] * imH)))
        boxes[0][i][1] = int(max(1, (boxes[0][i][1] * imW)))
        boxes[0][i][2] = int(min(imH, (boxes[0][i][2] * imH)))
        boxes[0][i][3] = int(min(imW, (boxes[0][i][3] * imW)))

    # --------------------------------------#
    #   如果没有检测到物体，则返回原图
    # --------------------------------------#
    if scores.shape[1] <= 0:
        return
    for i in range(boxes.shape[1]):
        if scores[0][i] > 0.01:
            predicted_class = class_names[int(classes[0][i]-1)]
            score = round(scores[0][i], 4)
            left = boxes[0][i][1]
            top = boxes[0][i][0]
            right = boxes[0][i][3]
            bottom = boxes[0][i][2]
            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, str(score), str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))
    f.close()
    return
