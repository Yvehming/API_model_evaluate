import tensorflow as tf
import cv2
import numpy as np
# https://yinguobing.com/load-savedmodel-of-estimator-by-keras/
# https://tensorflow.google.cn/versions/r2.2/api_docs/python/tf/saved_model/load
if __name__ == "__main__":
    load_model = tf.saved_model.load("pepper_inference_graph_0.5mnetv1.pb/saved_model")
    pruned_classes = load_model.prune("image_tensor:0", "FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D_Fold:0")
    image = cv2.imread("image/03.jpg")
    imH, imW, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (300, 300))
    image_expanded = np.expand_dims(image_resized, axis=0)
    classes = pruned_classes(tf.constant(image_expanded, dtype=tf.uint8)).numpy()
    print(classes)
#     scores = pruned_scores(tf.constant(image_expanded, dtype=tf.uint8)).numpy()
#     boxes = pruned_boxes(tf.constant(image_expanded, dtype=tf.uint8)).numpy()
#     for i in range(boxes.shape[1]):
#         boxes[0][i][0] = int(max(1, (boxes[0][i][0] * imH)))
#         boxes[0][i][1] = int(max(1, (boxes[0][i][1] * imW)))
#         boxes[0][i][2] = int(min(imH, (boxes[0][i][2] * imH)))
#         boxes[0][i][3] = int(min(imW, (boxes[0][i][3] * imW)))
#     print(classes)
#     print(scores)
#     print(boxes)
#     # img = cv2.imread("image/IMG_20211227_141738.jpg")
#     for i in range(scores.shape[1]):
#         if 0.5 < scores[0][i] < 1:
#             cv2.rectangle(image, (boxes[0][i][1], boxes[0][i][0]), (boxes[0][i][3], boxes[0][i][2]), (10, 255, 0), 4)
#             print(boxes[0][i][1])
#             print(boxes[0][i][0])
#             print(boxes[0][i][3])
#             print(boxes[0][i][2])
#     cv2.imshow("result", image)
#     cv2.imwrite("result.jpg", image)
#     # left = img[int(boxes[0][1][0]):int(boxes[0][1][2]), int(boxes[0][1][1]):int(boxes[0][1][3])]
#     # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # cv2.imshow('left', left)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# #
# load_model.predict(image_expanded)
