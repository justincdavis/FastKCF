import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_data_files():
    # using os.walk
    for root, _, files in os.walk('data'):
        for file in files:
            if file.endswith('.txt'):
                yield os.path.join(root, file)

def load_data_from_file(filename):
    bbox_size, std_init_time, fast_init_time = None, None, None
    std_update_times, fast_update_times = [], []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            # there are 5 'templates' for each line
            # BBOX_SIZE:SIZE
            # STD_INIT_TIME:TIME
            # FAST_INIT_TIME:TIME
            # STD_UPDATE_TIME:TIME
            # FAST_UPDATE_TIME:TIME
            linedata = line.split(":")
            if linedata[0] == "BBOX_SIZE":
                bbox_size = int(linedata[1])
            elif linedata[0] == "STD_INIT_TIME":
                std_init_time = float(linedata[1])
            elif linedata[0] == "FAST_INIT_TIME":
                fast_init_time = float(linedata[1])
            elif linedata[0] == "STD_UPDATE_TIME":
                std_update_times.append(float(linedata[1]))
            elif linedata[0] == "FAST_UPDATE_TIME":
                fast_update_times.append(float(linedata[1]))
    return bbox_size, std_init_time, fast_init_time, std_update_times, fast_update_times

def aggregate_file_data():
    # aggregate data from all files
    # return a dict with bbox_size as key and a list of tuples as value
    # each tuple contains std_init_time, fast_init_time, std_update_times, fast_update_times
    data = defaultdict(list)
    for filename in load_data_files():
        bbox_size, std_init_time, fast_init_time, std_update_times, fast_update_times = load_data_from_file(filename)
        data[bbox_size].append((std_init_time, fast_init_time, std_update_times, fast_update_times))
    return data

def calculate_average(data):
    # calculate average for each bbox_size
    # return a dict with bbox_size as key and a list of tuples as value
    # each tuple contains std_init_time, fast_init_time, std_update_times, fast_update_times
    average_data = defaultdict(list)
    for bbox_size, values in data.items():
        std_init_time, fast_init_time, std_update_times, fast_update_times = 0, 0, [], []
        try:
            for value in values:
                std_init_time += value[0]
                fast_init_time += value[1]
                std_update_times += value[2]
                fast_update_times += value[3]
        except TypeError:
            continue
        std_init_time /= len(values)
        fast_init_time /= len(values)
        std_update_time = np.average(std_update_times)
        fast_update_time = np.average(fast_update_times)
        average_data[bbox_size] = (std_init_time, fast_init_time, std_update_time, fast_update_time)
    return average_data

def plot_data(data):
    # plot data using matplotlib
    # data is a dict with bbox_size as key and a list of tuples as value
    # each tuple contains std_init_time, fast_init_time, std_update_times, fast_update_times
    std_points = []
    std_bboxes = []
    fast_points = []
    fast_bboxes = []
    for bbox_size, values in data.items():
        std_points.append(values[2])
        std_bboxes.append(bbox_size)
        fast_points.append(values[3])
        fast_bboxes.append(bbox_size)
    # plt.plot(bbox_sizes, std_init_times, label="std_init_time")
    # plt.plot(bbox_sizes, fast_init_times, label="fast_init_time")
    plt.scatter(std_bboxes, std_points, label="std_update_time")
    plt.scatter(fast_bboxes, fast_points, label="fast_update_time")
    plt.legend()
    plt.savefig("result.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    data = aggregate_file_data()
    average_data = calculate_average(data)
    plot_data(average_data)
