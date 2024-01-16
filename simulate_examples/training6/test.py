from processing import *
import models
import torch

train_prop = 0.9
num_files = 100_000
data_dir = "../Data3"
max_batch_size = 128
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Define model
model1 = models.TransformerModel1()
model1.load_state_dict(torch.load("site1_model.pth", map_location=torch.device(device)))
model1 = model1.to(device)
model1.eval()
# Define model
model2 = models.TransformerModel1()
model2.load_state_dict(torch.load("site2_model.pth", map_location=torch.device(device)))
model2 = model2.to(device)
model2.eval()

# Recreate test indices
torch.manual_seed(random_seed)
ind = int(train_prop * num_files)
idx = torch.randperm(num_files)
test_indices = idx[ind:].tolist()

# Load data from test indices
X = [convert_files(data_dir + "/sampled_genotypes/sample_" + str(i), data_dir + "/commands/command_" + str(i)) for i in test_indices] 

C = [convert_command_file1(data_dir + "/commands/command_" + str(i)) for i in test_indices]
C = [c for x,c in zip(X,C) if x is not None]

X = [x for x in X if x is not None]

X = torch.tensor(X) - 1
X = X.reshape(-1, sample_width*num_chrom * 2)

C = torch.tensor(C)
C = C.reshape(-1, 11)

y = torch.tensor([i % 2 for i in range(1, X.shape[0] + 1)])

print(X.shape)
print(y.shape)
print(C.shape)

# Test model on test set
with torch.no_grad():
    y_pred = torch.zeros(X.shape[0])
    for i in range(0, X.shape[0], max_batch_size):
        try:
            X_batch = X[i:i+max_batch_size].to(device)
            y_pred[i:i+max_batch_size] = (model1(X_batch) + model2(X_batch))/2
        except IndexError:
            X_batch = X[i:].to(device)
            y_pred[i:] = (model1(X_batch) + model2(X_batch))/2              

    weak_epistasis = (C[:,7] < C[:,10])
    weak_count = weak_epistasis.sum().item()

    # loss = criterion(y_pred, y.float())
    predictions = (y_pred >= 0.5).int()

    print()
    print("full set:")

    num_correct = (predictions == y).sum().item()
    print(f"Accuracy: {num_correct/y.shape[0]:0.4f}")

    print("\t0\t1")
    for i in range(2):
        called_0 = torch.sum((predictions == 0) & (y == i)).item()
        called_1 = torch.sum((predictions == 1) & (y == i)).item()
        sums = called_0 + called_1
        called_0 /= sums
        called_1 /= sums
        print(i, f"{called_0:.3f}" , f"{called_1:.3f}", sep='\t')
        
    called_epistasic = (predictions == 1)

    print()
    print("M and T evaluation")

    for i in range(8):
        minim = 0.1 + i/10
        maxum = 0.2 + i/10

        subset_0 = ((C[:,1] >= minim) & (C[:,1] < maxum))

        for j in range(10):

            minim = 100 + j*50
            maxum = 150 + j*50

            subset_1 = ((C[:,2] >= minim) & (C[:,2] < maxum))

            subset = (subset_0 & subset_1 & called_epistasic)

            #Counting proportion of those called epistasis which are correct

            subset_count = subset.sum().item()
            num_correct = ((predictions == y) & subset).sum().item()

            if subset_count == 0:
                print("0",end='')
                continue
            
            ratio = num_correct / subset_count

            if ratio < 0.5:
                print(".",end='')
            elif ratio < 0.66:
                print("-",end='')
            elif ratio < 0.80:
                print("+",end='')
            elif ratio < 0.95:
                print("$",end='')
            else:
                print("#",end='')
        
        print()


    print()

    print("epistrength and normstrength evaluation")

    for i in range(9):
        minim = 0.01 + i * 0.01
        maxum = 0.02 + i * 0.01

        subset_0 = ((C[:,7] >= minim) & (C[:,7] < maxum))

        for j in range(20):

            minim = -0.05  + j * 0.005
            maxum = -0.045 + j * 0.005

            subset_1 = ((C[:,10] >= minim) & (C[:,10] < maxum))

            subset = (subset_0 & subset_1 & called_epistasic)

            subset_count = subset.sum().item()
            num_correct = ((predictions == y) & subset).sum().item()

            if subset_count == 0:
                print("0",end='')
                continue
            
            ratio = num_correct / subset_count

            if ratio < 0.5:
                print(".",end='')
            elif ratio < 0.66:
                print("-",end='')
            elif ratio < 0.80:
                print("+",end='')
            elif ratio < 0.95:
                print("$",end='')
            else:
                print("#",end='')
        
        print()


    print()


exit()

from processing import *
import models
import torch

train_prop = 0.9
num_files = 100_000
data_dir = "../Data3"
max_batch_size = 128
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Define model
model = models.TransformerModel1()
model.load_state_dict(torch.load("site2_model.pth", map_location=torch.device(device)))
model = model.to(device)
model.eval()

# Recreate test indices
torch.manual_seed(random_seed)
ind = int(train_prop * num_files)
idx = torch.randperm(num_files)
test_indices = idx[ind:].tolist()

# Load data from test indices
X = [convert_files(data_dir + "/sampled_genotypes/sample_" + str(i), data_dir + "/commands/command_" + str(i)) for i in test_indices] 

C = [convert_command_file1(data_dir + "/commands/command_" + str(i)) for i in test_indices]
C = [c for x,c in zip(X,C) if x is not None]

X = [x for x in X if x is not None]

X = torch.tensor(X) - 1
X = X.reshape(-1, sample_width*num_chrom * 2)

C = torch.tensor(C)
C = C.reshape(-1, 11)

y = torch.tensor([i % 2 for i in range(1, X.shape[0] + 1)])

print(X.shape)
print(y.shape)
print(C.shape)

# Test model on test set
with torch.no_grad():
    y_pred = torch.zeros(X.shape[0])
    for i in range(0, X.shape[0], max_batch_size):
        try:
            X_batch = X[i:i+max_batch_size].to(device)
            y_pred[i:i+max_batch_size] = model(X_batch)
        except IndexError:
            X_batch = X[i:].to(device)
            y_pred[i:] = model(X_batch)                   

    weak_epistasis = (C[:,7] < C[:,10])
    weak_count = weak_epistasis.sum().item()

    # loss = criterion(y_pred, y.float())
    predictions = (y_pred >= 0.5).int()

    print()
    print("full set:")

    num_correct = (predictions == y).sum().item()
    print(f"Accuracy: {num_correct/y.shape[0]:0.4f}")

    print("\t0\t1")
    for i in range(2):
        called_0 = torch.sum((predictions == 0) & (y == i)).item()
        called_1 = torch.sum((predictions == 1) & (y == i)).item()
        sums = called_0 + called_1
        called_0 /= sums
        called_1 /= sums
        print(i, f"{called_0:.3f}" , f"{called_1:.3f}", sep='\t')
        
    called_epistasic = (predictions == 1)

    print()
    print("M and T evaluation")

    for i in range(8):
        minim = 0.1 + i/10
        maxum = 0.2 + i/10

        subset_0 = ((C[:,1] >= minim) & (C[:,1] < maxum))

        for j in range(10):

            minim = 100 + j*50
            maxum = 150 + j*50

            subset_1 = ((C[:,2] >= minim) & (C[:,2] < maxum))

            subset = (subset_0 & subset_1 & called_epistasic)

            #Counting proportion of those called epistasis which are correct

            subset_count = subset.sum().item()
            num_correct = ((predictions == y) & subset).sum().item()

            if subset_count == 0:
                print("0",end='')
                continue
            
            ratio = num_correct / subset_count

            if ratio < 0.5:
                print(".",end='')
            elif ratio < 0.66:
                print("-",end='')
            elif ratio < 0.80:
                print("+",end='')
            elif ratio < 0.95:
                print("$",end='')
            else:
                print("#",end='')
        
        print()


    print()

    print("epistrength and normstrength evaluation")

    for i in range(9):
        minim = 0.01 + i * 0.01
        maxum = 0.02 + i * 0.01

        subset_0 = ((C[:,7] >= minim) & (C[:,7] < maxum))

        for j in range(20):

            minim = -0.05  + j * 0.005
            maxum = -0.045 + j * 0.005

            subset_1 = ((C[:,10] >= minim) & (C[:,10] < maxum))

            subset = (subset_0 & subset_1 & called_epistasic)

            subset_count = subset.sum().item()
            num_correct = ((predictions == y) & subset).sum().item()

            if subset_count == 0:
                print("0",end='')
                continue
            
            ratio = num_correct / subset_count

            if ratio < 0.5:
                print(".",end='')
            elif ratio < 0.66:
                print("-",end='')
            elif ratio < 0.80:
                print("+",end='')
            elif ratio < 0.95:
                print("$",end='')
            else:
                print("#",end='')
        
        print()


    print()
