import cv2
import os

high_dir = "."

# delete the jpegs/jpgs
for root, dirs, _ in os.walk(high_dir):
    for directory in dirs:
        for _, _, files in os.walk(high_dir + "/" + directory):
            for file in files:
                if "ground" in file:
                    continue
                if file.endswith(".avi"):
                    continue
                os.remove(high_dir + "/" + directory + "/" + file)

# move the video files into the 
for root, dirs, files in os.walk(high_dir):
    for file in files:
        if file.endswith(".avi"):
            os.replace(file, file[0:len(file)-4] + "/" + file)
