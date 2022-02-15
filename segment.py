import cv2
import numpy as np
import matplotlib.pyplot as plt
# https://blog.csdn.net/lights_joy/article/details/46291229
box = [23,57,437,706]
src = cv2.imread("image/03.jpg")
# src = cv2.GaussianBlur(src, (3, 3), 0)  # 高斯滤波
src = src[box[1]:box[3], box[0]:box[2]]
h, w, _ = src.shape
# cut = src[2*h//3:h, 0:w]
fsrc = np.array(src, dtype=np.float32) / 255.0
(b, g, r) = cv2.split(fsrc)
gray = 2 * g - b - r
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
(thresh, bin_img) = cv2.threshold(np.array(gray_u8, dtype=np.uint8), -1.0, 255, cv2.THRESH_OTSU)
hist = cv2.calcHist([gray], [0], None, [256], [minVal, maxVal])
plt.plot(hist)
# plt.savefig("curve.svg")
plt.show()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # 定义结构元素
dilate = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel)
closing = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel)
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
drawing = opening.copy()
# lines = cv2.HoughLinesP(opening, 0.8, np.pi / 180, 90, minLineLength=50, maxLineGap=10)
# print(lines)
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(drawing, (x1, y1), (x2, y2), (0, 255, 0), 2)
# cv2.imwrite('houghlines.jpg', drawing)
contour_img = opening.copy()
_, contours, hierarchy = cv2.findContours(contour_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
area = []
for j in range(len(contours)):
    area.append(cv2.contourArea(contours[j]))
max_idx = np.argmax(area)
max_area = cv2.contourArea(contours[max_idx])
for k in range(len(contours)):
    if k != max_idx:
        cv2.fillPoly(contour_img, [contours[k]], 0)
# for i in range(2):
#     closing = cv2.morphologyEx(closing, cv2.MORPH_DILATE, kernel)
cv2.imshow("origin", bin_img)
cv2.imshow("morph", opening)
# cv2.imwrite("detect.jpg", src)
cv2.imshow("result", contour_img)
# cv2.imshow("result", max_area)
# cv2.imwrite("segment.jpg", bin_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

