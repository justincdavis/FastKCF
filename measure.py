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
    bbox_size = None
    base_init_time = None
    mp_init_time = None
    cuda_init_time = None
    mp_cuda_init_time = None
    base_update_times = []
    mp_update_times = []
    cuda_update_times = []
    mp_cuda_update_times = []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            # there are 5 'templates' for each line
            # BBOX_SIZE:SIZE
            # BASE_INIT_TIME:TIME
            # MP_INIT_TIME:TIME
            # CUDA_INIT_TIME:TIME
            # MP_CUDA_INIT_TIME:TIME
            # BASE_UPDATE_TIME:TIME
            # MP_UPDATE_TIME:TIME
            # CUDA_UPDATE_TIME:TIME
            # MP_CUDA_UPDATE_TIME:TIME
            linedata = line.split(":")
            if linedata[0] == "BBOX_SIZE":
                bbox_size = int(linedata[1])
            elif linedata[0] == "BASE_INIT_TIME":
                base_init_time = float(linedata[1])
            elif linedata[0] == "MP_INIT_TIME":
                mp_init_time = float(linedata[1])
            elif linedata[0] == "CUDA_INIT_TIME":
                cuda_init_time = float(linedata[1])
            elif linedata[0] == "MP_CUDA_INIT_TIME":
                mp_cuda_init_time = float(linedata[1])
            elif linedata[0] == "BASE_UPDATE_TIME":
                base_update_times.append(float(linedata[1]))
            elif linedata[0] == "MP_UPDATE_TIME":
                mp_update_times.append(float(linedata[1]))
            elif linedata[0] == "CUDA_UPDATE_TIME":
                cuda_update_times.append(float(linedata[1]))
            elif linedata[0] == "MP_CUDA_UPDATE_TIME":
                mp_cuda_update_times.append(float(linedata[1]))

    return bbox_size, base_init_time, mp_init_time, cuda_init_time, mp_cuda_init_time, base_update_times, mp_update_times, cuda_update_times, mp_cuda_update_times

def aggregate_file_data():
    # aggregate data from all files
    # return a dict with bbox_size as key and a list of tuples as value
    # each tuple contains base_init_time, mp_init_time, cuda_init_time, mp_cuda_init_time, base_update_times, mp_update_times, cuda_update_times, mp_cuda_update_times
    data = defaultdict(list)
    for filename in load_data_files():
        retval = load_data_from_file(filename)
        data[retval[0]].append(retval[1:-1])
    return data

def calculate_average(data):
    # calculate average for each bbox_size
    # return a dict with bbox_size as key and a list of tuples as value
    # each tuple contains base_init_time, mp_init_time, cuda_init_time, mp_cuda_init_time, base_update_times, mp_update_times, cuda_update_times, mp_cuda_update_times
    average_data = defaultdict(list)
    for bbox_size, values in data.items():
        base_init_times = []
        mp_init_times = []
        cuda_init_times = []
        mp_cuda_init_times = []
        base_update_times = []
        mp_update_times = []
        cuda_update_times = []
        mp_cuda_update_times = []
        for value in values:
            base_init_times.append(value[0])
            mp_init_times.append(value[1])
            cuda_init_times.append(value[2])
            mp_cuda_init_times.append(value[3])
            base_update_times.append(value[4])
            mp_update_times.append(value[5])
            cuda_update_times.append(value[6])
            mp_cuda_update_times.append(value[7])
        base_init_time = np.mean(base_init_times)
        mp_init_time = np.mean(mp_init_times)
        cuda_init_time = np.mean(cuda_init_times)
        mp_cuda_init_time = np.mean(mp_cuda_init_times)
        base_update_time = np.mean(base_update_times)
        mp_update_time = np.mean(mp_update_times)
        cuda_update_time = np.mean(cuda_update_times)
        mp_cuda_update_time = np.mean(mp_cuda_update_times)
        average_data[bbox_size].append((base_init_time, mp_init_time, cuda_init_time, mp_cuda_init_time, base_update_time, mp_update_time, cuda_update_time, mp_cuda_update_time))
    return average_data

def plot_data(data):
    # plot data using matplotlib
    # data is a dict with bbox_size as key and a list of tuples as value
    # each tuple contains base_init_time, mp_init_time, cuda_init_time, mp_cuda_init_time, base_update_times, mp_update_times, cuda_update_times, mp_cuda_update_times
    bbox_sizes = []
    base_init_times = []
    mp_init_times = []
    cuda_init_times = []
    mp_cuda_init_times = []
    base_update_times = []
    mp_update_times = []
    cuda_update_times = []
    mp_cuda_update_times = []
    for bbox_size, values in data.items():
        bbox_sizes.append(bbox_size)
        for data in values:
            base_init_times.append(data[0])
            mp_init_times.append(data[1])
            cuda_init_times.append(data[2])
            mp_cuda_init_times.append(data[3])
            base_update_times.append(data[4])
            mp_update_times.append(data[5])
            cuda_update_times.append(data[6])
            mp_cuda_update_times.append(data[7])
        
    # plt.plot(bbox_sizes, std_init_times, label="std_init_time")
    # plt.plot(bbox_sizes, fast_init_times, label="fast_init_time")
    plt.scatter(bbox_sizes, base_update_times, label="baseline")
    plt.scatter(bbox_sizes, mp_update_times, label="openmp + optimizations")
    plt.legend()
    plt.savefig("result.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    data = aggregate_file_data()
    average_data = calculate_average(data)
    plot_data(average_data)
