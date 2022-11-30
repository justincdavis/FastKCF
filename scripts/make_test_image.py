# make an opencv image that is all white and of size 4000x4000
# save it as test_image.png

import cv2
import numpy as np

img = np.zeros((4000,4000,3), np.uint8)
img[:] = (255,255,255)
cv2.imwrite('test_image.png',img)
