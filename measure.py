import os
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt


DIRECTORY1='data/got10k'
DIRECTORY2='data/benchmarking'

def load_data_files():
    # using os.walk
    files = []
    for root, _, files in os.walk(DIRECTORY1):
        for file in files:
            if file.endswith('.txt'):
                print("file: ", file)
                files.append(os.path.join(root, file))
    for root, _, files in os.walk(DIRECTORY2):
        for file in files:
            if file.endswith('.txt'):
                print("file: ", file)
                files.append(os.path.join(root, file))
    for file in files:
        yield file

def load_data_from_file(filename):
    bbox_size = 0
    base_init_points: Dict[int, List[float]] = defaultdict(list)
    mp_init_points: Dict[int, List[float]] = defaultdict(list)
    cuda_init_points: Dict[int, List[float]] = defaultdict(list)
    mp_cuda_init_points: Dict[int, List[float]] = defaultdict(list)
    base_update_points: Dict[int, List[float]] = defaultdict(list)
    mp_update_points: Dict[int, List[float]] = defaultdict(list)
    cuda_update_points: Dict[int, List[float]] = defaultdict(list)
    mp_cuda_update_points: Dict[int, List[float]] = defaultdict(list)

    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            # BBOX_SIZE:SIZE
            # BASE_INIT:TIME
            # MP_INIT:TIME
            # CUDA_INIT:TIME
            # MP_CUDA_INIT:TIME
            # BASE_UPDATE:TIME
            # MP_UPDATE:TIME
            # CUDA_UPDATE:TIME
            # MP_CUDA_UPDATE:TIME
            linedata = line.split(":")
            if linedata[0] == "BBOX_SIZE":
                bbox_size = int(linedata[1])
            elif linedata[0] == "BASE_INIT":
                base_init_points[bbox_size].append(float(linedata[1]))
            elif linedata[0] == "MP_INIT":
                mp_init_points[bbox_size].append(float(linedata[1]))
            elif linedata[0] == "CUDA_INIT":
                cuda_init_points[bbox_size].append(float(linedata[1]))
            elif linedata[0] == "MP_CUDA_INIT":
                mp_cuda_init_points[bbox_size].append(float(linedata[1]))
            elif linedata[0] == "BASE_UPDATE":
                base_update_points[bbox_size].append(float(linedata[1]))
            elif linedata[0] == "MP_UPDATE":
                mp_update_points[bbox_size].append(float(linedata[1]))
            elif linedata[0] == "CUDA_UPDATE":
                cuda_update_points[bbox_size].append(float(linedata[1]))
            elif linedata[0] == "MP_CUDA_UPDATE":
                mp_cuda_update_points[bbox_size].append(float(linedata[1]))

    # return bbox_size, base_init_time, mp_init_time, cuda_init_time, mp_cuda_init_time, base_update_times, mp_update_times, cuda_update_times, mp_cuda_update_times
    return bbox_size, base_init_points, mp_init_points, cuda_init_points, mp_cuda_init_points, base_update_points, mp_update_points, cuda_update_points, mp_cuda_update_points

def aggregate_file_data():
    # aggregate data from all files
    # return a dict with bbox_size as key and a list of tuples as value
    # each tuple contains base_init_time, mp_init_time, cuda_init_time, mp_cuda_init_time, base_update_times, mp_update_times, cuda_update_times, mp_cuda_update_times
    base_init_points, mp_init_points, cuda_init_points, mp_cuda_init_points, base_update_points, mp_update_points, cuda_update_points, mp_cuda_update_points = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for filename in load_data_files():
        retval = list(load_data_from_file(filename))
        bbox_size = retval.pop(0)
        base_init_points[bbox_size] += retval.pop(0)
        mp_init_points[bbox_size] += retval.pop(0)
        cuda_init_points[bbox_size] += retval.pop(0)
        mp_cuda_init_points[bbox_size] += retval.pop(0)
        base_update_points[bbox_size] += retval.pop(0)
        mp_update_points[bbox_size] += retval.pop(0)
        cuda_update_points[bbox_size] += retval.pop(0)
        mp_cuda_update_points[bbox_size] += retval.pop(0)
    return base_init_points, mp_init_points, cuda_init_points, mp_cuda_init_points, base_update_points, mp_update_points, cuda_update_points, mp_cuda_update_points

def plot_data(data, title, label, xlabel, ylabel):
    # data is a list of points (x,y)
    # plot the data
    x, y = zip(*data)
    plt.plot(x, y, label)
    # fit a linear regression to the plot
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.show()

    # save the plt 
    plt.savefig(f'graphs/{title.replace(" ", "")}.png')

def plot_multi_data(data, title, labels, xlabel, ylabel, colors):
    # data is a list of points (x,y)
    # plot the data
    for i, d in enumerate(data):
        x, y = zip(*d)
        plt.scatter(x, y, label=labels[i])
        # fit a linear regression to the plot
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        plt.plot(x, p(x), colors[i])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    # plt.show()

    # save the plt 
    plt.savefig(f'graphs/{title.replace(" ", "")}.png')

def remove_outliers(data):
    # input is a dict of bbox_size as key and a list of times as value
    # output is a dict of bbox_size as key and a list of times as value
    
    # for each list in the dict
    # remove values more than 2 standard deviations away from the mean
    for key in data:
        mean = np.mean(data[key])
        std = 2 * np.std(data[key])
        data[key] = [x for x in data[key] if x < mean + std]
        # data[key] = [x for x in data[key] if (x < np.mean(data[key]) + 2 * np.std(data[key]))]
    return data

def take_average(data):
    # input is a dict of bbox_size as key and a list of times as value
    # output is a list of (x,y) points
    # x is the bbox_size
    # y is the average time
    for key in data:
        data[key] = np.mean(data[key])
    return [(key, data[key]) for key in data]

if __name__ == "__main__":
    base = aggregate_file_data()
    base_init_points, mp_init_points, cuda_init_points, mp_cuda_init_points, base_update_points, mp_update_points, cuda_update_points, mp_cuda_update_points = base

    # remove outliers for each set of points
    base_init_points = take_average(base_init_points)
    mp_init_points = take_average(mp_init_points)
    cuda_init_points = take_average(cuda_init_points)
    mp_cuda_init_points = take_average(mp_cuda_init_points)
    base_update_points = take_average(base_update_points)
    mp_update_points = take_average(mp_update_points)
    cuda_update_points = take_average(cuda_update_points)
    mp_cuda_update_points = take_average(mp_cuda_update_points)

    # PLOT INIT DATA
    plot_data(base_init_points, "Base Init", "o", "BBox Size", "Time (ms)")
    plot_data(mp_init_points, "MP Init", "o", "BBox Size", "Time (ms)")
    plot_data(cuda_init_points, "CUDA Init", "o", "BBox Size", "Time (ms)")
    plot_data(mp_cuda_init_points, "MP CUDA Init", "o", "BBox Size", "Time (ms)")
    plot_multi_data([base_init_points, mp_init_points, cuda_init_points, mp_cuda_init_points], "Init", ["Base", "MP", "CUDA", "MP+CUDA"], "BBox Size", "Time (ms)", ["b", "y", "g", "r"])

    # PLOT UPDATE DATA
    plot_data(base_update_points, "Base Update", "o", "BBox Size", "Time (ms)")
    plot_data(mp_update_points, "MP Update", "o", "BBox Size", "Time (ms)")
    plot_data(cuda_update_points, "CUDA Update", "o", "BBox Size", "Time (ms)")
    plot_data(mp_cuda_update_points, "MP CUDA Update", "o", "BBox Size", "Time (ms)")
    plot_multi_data([base_update_points, mp_update_points, cuda_update_points, mp_cuda_update_points], "Update", ["Base", "MP", "CUDA", "MP+CUDA"], "BBox Size", "Time (ms)", ["b", "y", "g", "r"])
