import models
import sys
import os
import subprocess
import time
import random
import numpy as np


sys.stdout.flush()

random.seed(1337)
num_samples = models.len_chrom
num_chrom = models.num_chrom
piece_size = models.piece_size


start_time = time.time()

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




#genotypes, indvs, pieces

def multi_one_hot(one_hot):

    multi_hot = [None]*9

    for i in range(9):
        elem = [None]*models.num_indv
        for j in range(models.num_indv):
            el = [0]*piece_size
            elem[j] = el
        multi_hot[i] = elem

    ##good till here

    for i in range(len(one_hot[0])):
        for x in range(3):
            for y in range(3):
                for j in range(5):
                    multi_hot[x*3 + y][i][j] = one_hot[x][i][j] * one_hot[3 + y][i][j]
    
    return one_hot + multi_hot
        



def create_input1(sample_file, command_file):


    num = sample_file.split('_')[-1]
    if int(num) % 1000 == 0:
        print(num)
        sys.stdout.flush()
    
    if not os.path.isfile(sample_file):
        return None
    if not os.path.isfile(command_file):
        return None
    
    with open(sample_file, "r") as f:
        lines = f.readlines()
    
    if len(lines) < 198:
        return None

    
    out = [[float(l) for l in line[:-1]] for line in lines]
    

    with open(command_file,"r") as f:
        command_line, = f.readlines()
    
    ep_locations = []

    ep_locations.append(int(round(float(command_line.split()[6])*models.samples_per_morgan - 0.5)))  #chrom 0 #epistatic pair
    ep_locations.append(int(round(float(command_line.split()[7])*models.samples_per_morgan - 0.5)))  #chrom 1 #

    ep_locations.append(int(round(float(command_line.split()[10])*models.samples_per_morgan - 0.5)))  #chrom 1 #epistatic pair
    ep_locations.append(int(round(float(command_line.split()[11])*models.samples_per_morgan - 0.5)))  #chrom 0 #



    local_one_hot = []   #sites, genotypes, indvs, pieces
    for j in range(4):
        elem = [[],[],[]]
        local_one_hot.append(elem)
    

    for i in range(num_chrom//2):
        pieces = []
        for s in range(4):
            pieces.append([1]*piece_size)

        for j in range(piece_size):
            

            index_0 = (j - piece_size//2) + ep_locations[0]
            if index_0 >= 0 and index_0 < models.len_chrom:
                pieces[0][j] = int(out[i*2][index_0])

            
            index_1 = (j - piece_size//2) + ep_locations[1]
            if index_1 >= 0 and index_1 < models.len_chrom:
                pieces[1][j] = int(out[i*2 + 1][index_1])

            
            index_2 = (j - piece_size//2) + ep_locations[2]
            if index_2 >= 0 and index_2 < models.len_chrom:
                pieces[2][j] = int(out[i*2 + 1][index_2])

            
            index_3 = (j - piece_size//2) + ep_locations[3]
            if index_3 >= 0 and index_3 < models.len_chrom:
                pieces[3][j] = int(out[i*2][index_3])

        

        for j in range(4):
            for l in range(3):
                local_one_hot[j][l].append([0]*piece_size)
        
        for j in range(4):
            for l in range(piece_size):
                local_one_hot[j][pieces[j][l]][-1][l] = 1
    
    negated_local_one_hot = [None]*4
    for i in range(4):
        negated_local_one_hot[i] = [local_one_hot[i][2], local_one_hot[i][1], local_one_hot[i][0]]

    
    # 1 1 
    """
    print("OOOH")
    print(len(local_one_hot[0] + local_one_hot[1]))
    print(len(multi_one_hot(local_one_hot[0] + local_one_hot[1])))
    ans = multi_one_hot(local_one_hot[0] + local_one_hot[1])
    for j in range(len(ans[0])):
        for i in range(len(ans[0][0])):
            for k in range(len(ans)):
                print(ans[k][j][i], end='')
                if k %3 ==2:
                    print(end=' ')
            print(end='   ')
        print()
    print("oh shit")
    """
    #true false order
    #1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0
    return [
        multi_one_hot(local_one_hot[0] + local_one_hot[1]),
        multi_one_hot(local_one_hot[1] + local_one_hot[0]),
        multi_one_hot(local_one_hot[2] + local_one_hot[3]),
        multi_one_hot(local_one_hot[3] + local_one_hot[2]),

        multi_one_hot(local_one_hot[0] + local_one_hot[3]),
        multi_one_hot(local_one_hot[3] + local_one_hot[0]),
        multi_one_hot(local_one_hot[2] + local_one_hot[1]),
        multi_one_hot(local_one_hot[1] + local_one_hot[2]),

        multi_one_hot(negated_local_one_hot[0] + negated_local_one_hot[1]),
        multi_one_hot(negated_local_one_hot[1] + negated_local_one_hot[0]),
        multi_one_hot(negated_local_one_hot[2] + negated_local_one_hot[3]),
        multi_one_hot(negated_local_one_hot[3] + negated_local_one_hot[2]),

        multi_one_hot(negated_local_one_hot[0] + negated_local_one_hot[3]),
        multi_one_hot(negated_local_one_hot[3] + negated_local_one_hot[0]),
        multi_one_hot(negated_local_one_hot[2] + negated_local_one_hot[1]),
        multi_one_hot(negated_local_one_hot[1] + negated_local_one_hot[2]),
    ]





def convert_command_file1(sample_file, command_file):


    num = sample_file.split('_')[-1]
    if int(num) % 1000 == 0:
        print("C",num)
        sys.stdout.flush()
    
    if not os.path.isfile(sample_file):
        return None
    if not os.path.isfile(command_file):
        return None
    
    with open(sample_file, "r") as f:
        lines = f.readlines()
    
    if len(lines) < 198:
        return None


    with open(command_file,"r") as f:
        lines, = f.readlines()
    

    flist = [float(x) for x in lines.split()]
    ans = []
    for i in range(16):
        ans.append(flist + [i])
    return ans


