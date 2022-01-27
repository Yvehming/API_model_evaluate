import cv2
import numpy as np
import matplotlib.pyplot as plt
# https://blog.csdn.net/lights_joy/article/details/46291229
box = [23,57,437,706]
src = cv2.imread("image/03.jpg")
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
contour_img = bin_img.copy()
_, contours, hierarchy = cv2.findContours(contour_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
area = []
for j in range(len(contours)):
    area.append(cv2.contourArea(contours[j]))
max_idx = np.argmax(area)
max_area = cv2.contourArea(contours[max_idx])
for k in range(len(contours)):
    if k != max_idx:
        cv2.fillPoly(contour_img, [contours[k]], 0)
# cv2.rectangle(src, (0, 2*h//3), (w, h), (10, 255, 0), 4)
cv2.imshow("origin", bin_img)
# cv2.imwrite("detect.jpg", src)
cv2.imshow("result0", contour_img)
# cv2.imshow("result", max_area)
# cv2.imwrite("segment.jpg", bin_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

