import models
import os
import subprocess
import time
import random
from math import sqrt, e, pi
import numpy as np
import matplotlib.pyplot as plt

random_seed = 177
num_samples = models.len_chrom
num_chrom = 100
start_time = time.time()
sample_width = models.sample_width

random.seed(random_seed)

sigma = 2.5

assert sample_width >= 1 and sample_width % 2 == 1

def GetMemory():
    if os.name == 'posix':
        mem_info = subprocess.check_output(['free','-b']).decode().split()
        total_memory = int(mem_info[7]) - int(mem_info[8]) 
        total_memory *= 10**-9
        
    elif os.name == 'nt':
        mem_info = subprocess.check_output(['wmic','OS','get','FreePhysicalMemory']).decode().split()
        total_memory = int(mem_info[1]) * 1024 * 10**-9
        
    print(f"Available memory: {total_memory:0.2f} GB")
    
def GetTime():
    seconds = int(time.time() - start_time)
    hours = seconds // 3600
    seconds = seconds % 3600
    minutes = seconds // 60
    seconds = seconds % 60
    print(f"Total time elapsed: {hours}h {minutes}m {seconds}s")

Normal = lambda x, mean: e**(-0.5*((x-mean)/sigma)**2) / (sigma * sqrt(2*pi))

def split_to_sample(split_file, sampled_sites):
    with open(split_file, "r") as f:
        lines = f.readlines()

    lines = [[float(l) for l in line] for line in lines]

    min_site = sampled_sites[0]

    X = [[0 for _ in range(len(lines)//2)] for _ in range(len(sampled_sites))]
    for i in range(0,len(lines),2):
        split_points = lines[i]
        categories = lines[i+1]

        # find correct starting index
        ind = min_site * len(split_points)
        if split_points[ind] < min_site:  
            while split_points[ind] < min_site:
                ind += 1
            ind -= 1
        else:
            while split_points[ind] >= min_site:
                ind -= 1

        # sample sites
        for j,site in enumerate(sampled_sites):
            while site > split_points[ind+1]:
                ind += 1

            X[j][i // 2] = categories[ind]




def convert_files(sampling_file, command_file):

    if command_file.endswith("000"):
        print(command_file[-5:])

    with open(sampling_file, "r") as f:
        lines = f.readlines()

    if len(lines) < 200:
        return None

    X1 = [[int(l) for l in line[:-1]] for line in lines[::2]]  #first chromosome
    X2 = [[int(l) for l in line[:-1]] for line in lines[1::2]] #second chromosome

    with open(command_file, "r") as f:
        s = f.readlines()[0].split()

    command_idx = [6,7,10,11]
    points = [float(s[idx]) for idx in command_idx]
    ind11, ind12, ind22, ind21 = [round(1000*point - 0.5) for point in points]
    # indAB: A is pair number and B is chromosome number

    site11 = [sample[ind11] for sample in X1]
    site12 = [sample[ind12] for sample in X2]
    site21 = [sample[ind21] for sample in X1]
    site22 = [sample[ind22] for sample in X2]

    for offset in range(1, sample_width // 2 + 1):
        site_prev = [sample[ind11 - offset] for sample in X1] if ind11 - offset >= 0 else [1 for _ in range(len(X1))]
        site_next = [sample[ind11 + offset] for sample in X1] if ind11 + offset < num_samples else [1 for _ in range(len(X1))]
        site11 = site_prev + site11 + site_next

        site_prev = [sample[ind12 - offset] for sample in X2] if ind12 - offset >= 0 else [1 for _ in range(len(X2))]
        site_next = [sample[ind12 + offset] for sample in X2] if ind12 + offset < num_samples else [1 for _ in range(len(X2))]
        site12 = site_prev + site12 + site_next

        site_prev = [sample[ind21 - offset] for sample in X1] if ind21 - offset >= 0 else [1 for _ in range(len(X1))]
        site_next = [sample[ind21 + offset] for sample in X1] if ind21 + offset < num_samples else [1 for _ in range(len(X1))]
        site21 = site_prev + site21 + site_next

        site_prev = [sample[ind22 - offset] for sample in X2] if ind22 - offset >= 0 else [1 for _ in range(len(X2))]
        site_next = [sample[ind22 + offset] for sample in X2] if ind22 + offset < num_samples else [1 for _ in range(len(X2))]
        site22 = site_prev + site22 + site_next
        
    # print()
    # print(sum(site11) - sample_width*100)
    # print(sum(site12) - sample_width*100)
    # print(sum(site21) - sample_width*100)
    # print(sum(site22) - sample_width*100)
    # True, False, True, False
    return [site11 + site12, site11 + site21, site21 + site22, site22 + site12]

def convert_command_file1(file):
    
    with open(file,"r") as f:
        lines, = f.readlines()
    

    flist = [float(x) for x in lines.split()[1:]]

    return [flist,flist]