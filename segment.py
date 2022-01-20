import cv2
import numpy as np

src = cv2.imread("image/03.jpg")
src = src[2*(23+706)//3:706, 57:437]
fsrc = np.array(src, dtype=np.float32)/255.0
(b, g, r) = cv2.split(fsrc)
gray = 2*g-b-r
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
gray_u8 = np.array((gray-minVal)/(maxVal-minVal)*255, dtype=np.uint8)
(thresh, bin_img) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)
cv2.imshow("origin", src)
cv2.imshow("result", bin_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
