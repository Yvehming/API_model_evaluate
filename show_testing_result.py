import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from ssd import SSD

# 显示检测结果与真实值
def show_testing_result():
    sum_ground_truth = 0
    sum_above05 = 0
    sum_detected = 0
    # 检测结果所在的文件夹
    detection_path = "map_out/detection-results"
    # ground truth所在的文件夹
    ground_truth_path = "map_out/ground-truth"
    # 将检测框与真实框比较，保存在文件夹中，若没有，则创建
    save_path = "map_out/visualization"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    detection_txt = os.listdir(detection_path)
    ground_truth_txt = os.listdir(ground_truth_path)
    print(detection_txt)
    print(ground_truth_txt)
    for i in range(len(detection_txt)):
        txt = []
        box = []
        truth = []
        rects = []
        # 读取第i个检测结果
        if detection_txt[i][-3:] == 'txt':
            f = open(detection_path + '/' + detection_txt[i])
            # 读取第i个ground truth
            g = open(ground_truth_path + '/' + detection_txt[i])
            # 对应的图片名
            img = str(detection_txt[i]).replace(".txt", ".jpg")
            image = cv2.imread("VOCdevkit/VOC2007/JPEGImages" + '/' + img)
        # 检测结果的txt文件中，将文件内容格式化
        for line in f:
            txt.append(line.strip())
        # txt[0] :'person 0.9987 1103 235 1199 467'
        for j in range(len(txt)):
            rects.append(txt[j].split(" "))
        # 选出置信度大于0.5的
        for rect in rects:
            sum_detected += 1
            if eval(rect[1]) > 0.5:
                box.append(rect)
                sum_above05 += 1
        print("检测到的box数量:", len(box))
        print(box)
        f.close()
        # 清空列表
        txt.clear(); rects.clear()
        # ground truth的txt文件中，将文件内容格式化
        for line in g:
            txt.append(line.strip())
        # txt[0] :'0.9987 1103 235 1199 467'
        for j in range(len(txt)):
            rects.append(txt[j].split(" "))
        # 由于ground truth没有置信度，因此不用筛选
        for rect in rects:
            sum_ground_truth += 1
            truth.append(rect)
        g.close()
        print("图片中ground_truth的数量{}".format(len(truth)))
        print("ground_truth:", truth)
        # 绘制矩形框，检测结果为绿色
        for k in range(len(box)):
            cv2.rectangle(image, (eval(box[k][2]), eval(box[k][3])), (eval(box[k][4]), eval(box[k][5])),
                          (0, 255, 0), 2)
        # 绘制矩形框，ground truth为红色
        for k in range(len(truth)):
            cv2.rectangle(image, (eval(truth[k][1]), eval(truth[k][2])), (eval(truth[k][3]), eval(truth[k][4])),
                          (0, 0, 255), 2)
        cv2.imshow('Result', image)
        # 将图片保存至map_out/visualization
        cv2.imwrite(save_path + '/' + str(i) + '.jpg', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # print("ground truth:", sum_ground_truth)
    # print("all detected:", sum_detected)
    # print("detected confidence>0.5", sum_above05)
if __name__ == "__main__":
    show_testing_result()
    # ground truth: 9446
    # all detected: 150000
    # detected confidence > 0.5 3542
