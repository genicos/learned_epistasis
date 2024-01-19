from processing import *
from models import *
import multiprocessing
import torch
import torch.nn as nn

sample_file_name = "../../data/sampled_genotypes/sample_0"
command_file_name = "../../data/commands/command_0"



num_files = 10

X1 = [convert_sampling_file("../../data/sampled_genotypes/sample_" + str(i)) for i in range(num_files)]
X1 = torch.tensor(X1) - 1

print(X1)


test = create_input(sample_file_name, command_file_name)
X = torch.tensor(test) - 1



