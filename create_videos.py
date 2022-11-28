import cv2
import os

high_dir = "test"
video_type = "avi"

for root, dirs, _ in os.walk(high_dir):
    dir_list = sorted([_dir for _dir in dirs])
    for directory in dir_list:
        print("Creating video for " + directory)
        for _, _, files in os.walk(high_dir + "/" + directory):
            out = None
            file_list = sorted([file for file in files])
            for file in file_list:
                # print(file)
                if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                    img = cv2.imread(high_dir + "/" + directory + "/" + file)
                    height, width, layers = img.shape
                    size = (width,height)
                    if out is None:
                        # create a video writer for avi
                        out = cv2.VideoWriter(high_dir + "/" + directory + "/" + directory + f".{video_type}",cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
                    out.write(img)
            out.release()

# delete the jpegs/jpgs
for root, dirs, _ in os.walk(high_dir):
    for directory in dirs:
        for _, _, files in os.walk(high_dir + "/" + directory):
            for file in files:
                if "ground" in file:
                    continue
                if file.endswith(f".{video_type}"):
                    continue
                os.remove(high_dir + "/" + directory + "/" + file)

# move the video files into the 
for root, dirs, files in os.walk(high_dir):
    for file in files:
        if file.endswith(f".{video_type}"):
            os.replace(file, file[0:len(file)-4] + "/" + file)
