import cv2
import os


for root, dirs, _ in os.walk("test"):
    for directory in dirs:
        for _, _, files in os.walk("test/" + directory):
            out = None
            for file in files:
                if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                    img = cv2.imread("test/" + directory + "/" + file)
                    height, width, layers = img.shape
                    size = (width,height)
                    if out is None:
                        out = cv2.VideoWriter(directory + '.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
                    out.write(img)
            out.release()
