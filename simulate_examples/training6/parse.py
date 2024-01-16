#save list of indices where epistatic strength is smaller
import numpy as np
import matplotlib.pyplot as plt
import random

num_samples = 1000

# l = []
# for i in range(50000):
#     with open("../../test_training/commands/command_stronger_" + str(i), "r") as f:
#         s = f.readlines()[0].split()
#         if float(s[-1]) > float(s[-4]):
#             l.append(i)

# with open("smaller_epistatic.txt", "w") as f:
#     f.write(str(l))

# exit()
for i in range(87500):
    with open("../../test_training/commands/command_stronger_" + str(i), "r") as f:
        s = f.readlines()[0].split()

    if int(s[3]) > 550:
        print(i)
        exit()


l = []

y = []
x = []
idx = list(range(87500))
random.shuffle(idx)
for p,i in enumerate(idx):

    sampling_file = "../../test_training/sampled_genotypes/sample_stronger_" + str(i)
    try:
        with open(sampling_file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        continue

    X = [[float(l) for l in line[:-1]] for line in lines]

    X = np.array(X) - 1
    with open("../../test_training/commands/command_stronger_" + str(i), "r") as f:
        s = f.readlines()[0].split()

    X = X.sum(axis=0)

    points = [float(s[6]), float(s[7])]
    inds = [round(num_samples*point - 0.5) for point in points]
    regular_site = round(num_samples*float(s[10]) - 0.5)

    # x.append(float(s[8]) + float(s[11]))
    # y.append(min(X))
    # y.append(min(X[inds]) - X[regular_site])

    if min(X) > -85:
        l.append(i)

    if p % 1000 == 0:
        print(p)
    #     print(s)
    # else: 
    #     print(s)

with open("smaller_strength.txt", "w") as f:
    f.write(str(l))

# plt.scatter(x,y)
# plt.show()


print(len(l))



