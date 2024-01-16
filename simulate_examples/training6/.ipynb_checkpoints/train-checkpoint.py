from processing import *
from models import *
import multiprocessing
import concurrent.futures
import torch
import torch.nn as nn
from ast import literal_eval
import sys

save_file = sys.argv[1] if sys.argv[1].endswith(".pth") else sys.argv[1] + ".pth"
data_dir = "../Data3"
GPU_available = torch.cuda.is_available()
print(GPU_available)
num_files = 100_000
num_epochs = 30
batch_size = 128
train_prop = 0.9
num_estimate = 5000
lr_start = 3e-4
lr_end = lr_start/100

GetMemory()
X = [convert_files(data_dir + "/sampled_genotypes/sample_" + str(i), data_dir + "/commands/command_" + str(i)) for i in range(num_files)] 

C = [convert_command_file1(data_dir + "/commands/command_" + str(i)) for i in range(num_files)]
C = [c for x,c in zip(X,C) if x is not None]

X = [x for x in X if x is not None]


X = torch.tensor(X) - 1
X = X.reshape(-1, sample_width*num_chrom * 2)

print(X.shape)
C = torch.tensor(C)
C = C.reshape(-1, 11)

GetMemory()

y = torch.tensor([i%2 for i in range(1,X.shape[0] + 1)])

# Scramble data
torch.manual_seed(random_seed)
ind = int(train_prop * X.shape[0]) // 2
idx = torch.randperm(X.shape[0] // 2)
idx = [2*i for i in idx[:ind]] + [2*i + 1 for i in idx[:ind]] + [2*i for i in idx[ind:]] + [2*i + 1 for i in idx[ind:]] #include both True and False example for each sample

X = X[idx]
y = y[idx]
C = C[idx]

# Split data
ind = int(train_prop * X.shape[0])
X_train, X_test = X[:ind], X[ind:]
y_train, y_test = y[:ind], y[ind:]
C_train, C_test = C[:ind], C[ind:]

GetMemory()
GetTime()

# lr variables
lr = lr_start
lr_factor = (lr_end/lr_start)**(1/(num_epochs - 1))

# Define network
model = TransformerModel1()
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Define functions for getting random batch and calculating loss
def get_batch(split, num_samples):
    X, y, C = (X_train, y_train, C_train) if split == "train" else (X_test, y_test, C_test)
    idx = torch.randperm(X.shape[0])[:num_samples]
    X = X[idx]
    y = y[idx]
    C = C[idx]
    if GPU_available:
        return X.to("cuda"), y.to("cuda"), C.to("cuda")
    else:
        return X, y, C

@torch.no_grad()
def estimate_loss(num_samples):
    model.eval()
    for split in ["train", "test"]:
        X, y, C = get_batch(split, num_samples)
        y_pred = torch.zeros(num_samples)
        for i in range(0, num_samples, batch_size):
            try:
                X_batch = X[i:i+batch_size]
                y_pred[i:i+batch_size] = model(X_batch)
            except IndexError:
                X_batch = X[i:]
                y_pred[i:] = model(X_batch)

        if GPU_available:
            y_pred = y_pred.to("cuda")                       


        
        weak_epistasis = (C[:,7] < C[:,10])
        weak_count = weak_epistasis.sum().item()
        
        loss = criterion(y_pred, y.float())
        predictions = (y_pred >= 0.5).int()


        print()
        print(split, "full set:")

        num_correct = (predictions == y).sum().item()
        print(f"Loss {split}: {loss.item():0.5f}, Accuracy {split}: {num_correct/num_samples:0.4f}")

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
        print(split, "M and T evaluation")

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

        print(split, "epistrength and normstrength evaluation")

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

    model.train()


if GPU_available:
    print("GPU is available.")
    model = model.to("cuda")
    criterion = criterion.to("cuda")
#    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<enter-size-here>"
else:
    print("No GPU available. Running on CPU.")

GetMemory()
# estimate_loss(min(num_estimate, X_test.shape[0]))
#Training loop

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb: SIZE"

model.train()
for epoch in range(num_epochs):
    print("-----------------------------------------------------------------")
    print(f"Started training on epoch {epoch + 1} of {num_epochs}, learning rate {lr:0.7f}")
    GetTime()
    # Scramble data
    idx = torch.randperm(X_train.shape[0])
    X_train = X_train[idx]
    y_train = y_train[idx]

    # Shuffle in sampling dimension
    X_train = X_train.reshape(-1,2*sample_width,num_chrom)
    idx = torch.randperm(num_chrom)
    X_train = X_train[:,:,idx]

    # Randomly switch two sites 
    mask = torch.rand(X_train.shape[0]) < 0.5
    shift_indices = torch.fmod(torch.arange(2*sample_width) + sample_width, 2*sample_width)
    X_train[mask] = X_train[mask][:, shift_indices]


    X_train = X_train.reshape(-1,sample_width*num_chrom*2)
    # Randomly muliply each example by 1 or -1
    rand = torch.randint(0,2,size=(X_train.shape[0],1)) * 2 - 1
    X_train *= rand


    for ind in range(0,X_train.shape[0],batch_size):

        #print(torch.cuda.max_memory_allocated(),"and", torch.cuda.memory_allocated())

        try:
            X_batch = X_train[ind:ind+batch_size]
            y_batch = y_train[ind:ind+batch_size]
        except IndexError:
            X_batch = X_train[ind:]
            y_batch = y_train[ind:]

        if GPU_available:
            X_batch = X_batch.to("cuda")
            y_batch = y_batch.to("cuda")

        try:
            y_pred = model(X_batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            y_pred = model(X_batch)


        loss = criterion(y_pred, y_batch.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    lr *= lr_factor
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    estimate_loss(min(num_estimate, X_test.shape[0]))
    # for child in model.children():
    #     if isinstance(child,nn.Linear):
    #         print("next")
    #         print("weight")
    #         print(child.weight)
    #         print("bias")
    #         print(child.bias)

if save_file.lower() != "none.pth":
    torch.save(model.state_dict(), save_file)

"""
-----------------------------------------------------------------
Started training on epoch 1 of 30, learning rate 0.0003000
Total time elapsed: 0h 15m 54s

train full set:
Loss train: 0.26406, Accuracy train: 0.8760
	0	1
0	0.862	0.138
1	0.109	0.891

train M and T evaluation
$#$$$$$$$$
$$$$+$$$$$
$##$$$$$+$
$#$+$$$++$
+$$+$$$$$$
#$$$$$+$$#
$$+$+$$+$$
$$$$+$$#$$

train epistrength and normstrength evaluation
$$+$#$#++00++$$+$+$$
$$+$##++$00$$$$#$-+$
$$$##-$$+00$$$$+$$+$
$$###$$$#00$$$#+$$++
$$$$$$$$$00$+$$$$$+$
#$$$+$+$$00$$+$#$+$$
$+$$#+$$+00$$$#$$$+$
$$$+$#$$$00$$$$##$$+
$$#$$$#$$00$$+$$+$++


test full set:
Loss test: 0.28253, Accuracy test: 0.8646
	0	1
0	0.858	0.142
1	0.129	0.871

test M and T evaluation
+$#$###$##
+$$$$#####
+$##$#####
$$+$$$##$$
-$-+-+$$+$
$$$+$$+++$
$$$$$+$$$$
#$$$$$$$$+

test epistrength and normstrength evaluation
#####$##$00-$$+-$+--
##$###$$#00+++$+$+--
$########00+$#$+++-+
#$##$#$$$00-$#+$+$+$
#$####$$+00+#$$+++-+
$###$#$#-00+$$#$-$$$
$#####$#+00+$##$++++
$$####$#+00$$+$+$$$$
$####$#$$00$$$$$$$++

-----------------------------------------------------------------
Started training on epoch 2 of 30, learning rate 0.0002560
Total time elapsed: 0h 17m 53s

train full set:
Loss train: 0.25561, Accuracy train: 0.8754
	0	1
0	0.844	0.156
1	0.093	0.907

train M and T evaluation
$$$#$$$$$$
$$$$#$$$$+
+++$+$$$$+
$$$#++$$++
$$$$$+$$$$
$$$$$$$$+$
$$+$+$$+$$
+$$$$$$$$$

train epistrength and normstrength evaluation
+#$$$$#++00$$$#$$#+$
++$$++++$00#$$$#+$$$
$$$$$#$#+00+$#+#+$$+
$+$$+$+$$00$#$$$$$$#
$+$+$$$$$00$+$#$$++$
-$$$$-$#+00$$$$+$#$$
#$$$$$$$+00$$+$$$$$+
+++#$++##00$$+$$+$#$
$++-$+#$#00$+$+$$$$$


test full set:
Loss test: 0.26367, Accuracy test: 0.8774
	0	1
0	0.849	0.151
1	0.094	0.906

test M and T evaluation
+$###$####
$$#$#$####
++##$#####
$$+$+##$$$
++$++-+$++
$$$$-+$$$-
$++$$+$$$-
$$++$$+$$$

test epistrength and normstrength evaluation
##$######00++$+++---
#######$#00+$$$$++-$
########$00$$$+$-++-
####$###+00++#$#+$--
#######$$00+#$+$+$--
######$$$00#$#$$++++
######$#$00+$$#$+---
#$#####$$00##$$$--++
$###$$#$+00$$$$$$$+.

-----------------------------------------------------------------
Started training on epoch 3 of 30, learning rate 0.0002184
Total time elapsed: 0h 19m 51s

train full set:
Loss train: 0.22087, Accuracy train: 0.8932
	0	1
0	0.889	0.111
1	0.103	0.897

train M and T evaluation
$$$$#$$$$#
$$$$#$##$$
$$$$#$$#$$
$#$$$$$$$$
$$$$$$+$$$
#$$$$$$$$$
$$#$$#$##$
$$$$$$$$$#

train epistrength and normstrength evaluation
##$$$$$$$00#+$#$#$$$
+$+$$$##+00$$$$$$$$$
$$$$$$$$$00$#$$$$$$$
$+$$$$$+$00#$$#$$+##
$$$$$##$#00#+$+$#$##
$$#$$#$$$00$$$$$##$+
$$$$$$$$$00#$$$#$$+#
$#$$$$$$$00#$$+$$$#$
#$$$#$$#$00$$$$$+$$$


test full set:
Loss test: 0.24324, Accuracy test: 0.8834
	0	1
0	0.890	0.110
1	0.123	0.877

test M and T evaluation
+#$$######
+$$$$$##$#
+$##$#####
+$++$$$$$$
+$$$$+$+++
$##$$$$$$$
#$$##$$$#+
$$$$#$$$$#

test epistrength and normstrength evaluation
#$##$###+00--$+$--$.
######$#$00.$$$$$$-+
########$00$$$$$#$++
######$#$00$$#$$+$+-
######$#$00-$$$$$$$$
#####$+#+00$$#$+#$$+
###$#$+##00$$$$$+$++
#######$+00+$$#++$+$
$#####$$$00####$$$$-

-----------------------------------------------------------------
Started training on epoch 4 of 30, learning rate 0.0001863
Total time elapsed: 0h 21m 49s

train full set:
Loss train: 0.22554, Accuracy train: 0.8920
	0	1
0	0.905	0.095
1	0.121	0.879

train M and T evaluation
$##$$$$#$$
$$##$+#$$$
$$$$$$$$$$
#$$$$$#$$#
$##+#$$$#$
$+$#$$#$$$
$$$$$$##+$
$$+$$#$$$$

train epistrength and normstrength evaluation
$#$$$#$$$00#$#-$$#$#
$$#$$###$00$$$#$$$#$
#$#$+$+$#00#$$+#$$$#
$+#++$$$$00#$$##$$$#
$$$#$$$$$00++#$#$$##
$$$$##$$$00$+#$$#$#$
$#$$$$$$$00$$$$$$$#$
$$$$$$$#$00$$$$$$$$$
$$$+$$#+$00$#$$#+$$$


test full set:
Loss test: 0.25256, Accuracy test: 0.8838
	0	1
0	0.883	0.117
1	0.116	0.884

test M and T evaluation
+$#$#####$
$$$##$$$##
+$$$$#####
+-$+$$$$##
$$$+$++++$
#$$+$$$$++
$#$##$$$$$
$#$$$$$#$$

test epistrength and normstrength evaluation
$########00+$$-$+$-+
######$#$00-+.++$$+-
########$00$$$$$++-.
########+00+$#$#$$$+
########+00+$$$$$+$+
#####$#$+00$$$$$$$.+
######$#$00$+$$$+$++
##$$####+00$$$$$$+++
$######$-00+$$##$$$-

-----------------------------------------------------------------
Started training on epoch 5 of 30, learning rate 0.0001589
Total time elapsed: 0h 23m 49s

train full set:
Loss train: 0.22283, Accuracy train: 0.8908
	0	1
0	0.861	0.139
1	0.079	0.921

train M and T evaluation
$#$$+$$$$+
$$+$$$$$$$
$$$$+$$$$$
$+$$+$$+$$
$$+$+$$$$$
#$$$$$$$#$
$$$$+$$$$$
$#$$$$$$+$

train epistrength and normstrength evaluation
$#$++$$$$00+$+$$++#$
$$$$+$$++00$$$+$$-#$
$$+$$#$$$00$$$$#$$$$
-.$+$$$#$00+#$$$$$#$
$#$$$$#$#00$$$$+$+#$
$$$+$+$+$00$$+$#$$$#
#+$$$#$#$00$$#$$$++#
#+$+$$$$$00+++$$$#$$
$$#+$$$$+00$$+$$++$#


test full set:
Loss test: 0.24767, Accuracy test: 0.8776
	0	1
0	0.849	0.151
1	0.095	0.905

test M and T evaluation
+$$$#####$
$#$$######
$$$####$##
$$++++$$#$
$+++++-+++
#$$++$$$+-
$$$$$$$$$#
$$#+#$#$#$

test epistrength and normstrength evaluation
####$$#$$00--+-+--.-
#######$#00$$$$$$$--
#########00$$+$++-$+
########$00++$+$++$-
#########00++$+$$$$+
######$$+00+$#++$+-+
########-00+$#$+$$+-
##$#####$00$$$$$$-++
#######$-00$#$$$+#$+

-----------------------------------------------------------------
Started training on epoch 6 of 30, learning rate 0.0001356
Total time elapsed: 0h 25m 50s

train full set:
Loss train: 0.22017, Accuracy train: 0.8998
	0	1
0	0.888	0.112
1	0.088	0.912

train M and T evaluation
#$+$$$#$$$
#$$$+$$$$$
$$#$$#$#+$
$$$$#$$$$+
$$#$$+$$$$
$$$$#$$$$$
$##+$$$$$$
$+$$#$+#$#

train epistrength and normstrength evaluation
$$$$$$$#$00$#+#$$##$
+$$$+$##$00$$+$#+$$$
$+$#+$$$$00$$+$$$$$$
$$+#$$$##00$$$#$$$#$
##$$$$$$$00$$$#$$$#$
#$$$$$$##00$$#$$$$$$
+$+$$$$#$00$##$$$$$$
$+$$$$$#+00+$$#+$$#$
#$$+$$+$$00$$$+$$#$$


test full set:
Loss test: 0.23087, Accuracy test: 0.8864
	0	1
0	0.874	0.126
1	0.102	0.898

test M and T evaluation
$$$$######
+$$#$#$###
+$#$$#$###
$+$$$$$###
$$$$$+++++
$$#++$+$$$
#$$$$$$$$$
$$#$$$$$$$

test epistrength and normstrength evaluation
$#$#$###+00-$+-+--.-
#####$$#$00-$$$$$+$-
########$00+$$$+++++
########$00+$$+$$$-$
#######$#00+#$$$$$++
#####$#$+00$#$#$#+--
######$#$00#$##$++++
#######$+00$$+#$$-+$
$#######+00$$#$$$$++

-----------------------------------------------------------------
Started training on epoch 7 of 30, learning rate 0.0001157
Total time elapsed: 0h 27m 50s

train full set:
Loss train: 0.21622, Accuracy train: 0.8976
	0	1
0	0.896	0.104
1	0.101	0.899

train M and T evaluation
$$$$#$$$##
$$$$$$$$$$
$$$$$$$#$$
$#$$+$$#$$
$$$#$$$$$#
##$$#$#$$$
$$$$$+$#$#
$#$$##$$$+

train epistrength and normstrength evaluation
$$+$++$$+00#$+#$#$$$
$$$$$$$$$00$#$$$#$$#
$$$$$$$$-00#$#$+$+$$
#$$+$$+$#00$##$$$###
+$$####$$00$#$$$$$##
$$+$#$$$$00#$$$$$$$$
$#$+#$#$#00$$##$$$$$
$$-$$$#$$00$$$$$$$$$
$#$$$$$$#00$$$#$#$$#


test full set:
Loss test: 0.23267, Accuracy test: 0.8854
	0	1
0	0.877	0.123
1	0.107	0.893

test M and T evaluation
$$#$#$$###
$#$$$#####
+$$#######
$$+$$$$###
$$$+++++$+
$$$$++$$+$
#$$$$$+#$$
$#$+$$+$$$

test epistrength and normstrength evaluation
$$####$#$00+$#--+++$
#######$$00--$+$+$+.
########$00+$$$$-$$$
#$#####$$00+$$$$+$$+
########$00+$$$$++$$
####$$$$$00$#$+++$-+
###$###$-00$$#$$#+$+
##$####$$00$$$$$$+$-
########+00+$###$$++

-----------------------------------------------------------------
Started training on epoch 8 of 30, learning rate 0.0000987
Total time elapsed: 0h 29m 52s

train full set:
Loss train: 0.22415, Accuracy train: 0.8942
	0	1
0	0.887	0.113
1	0.099	0.901

train M and T evaluation
#$$$$+$+$$
$$$$$$#$$#
$$##$$$$$$
$$$#$$+$$$
$$$$#$$$$$
$$#$$++$$$
#$$$$$$$$#
#$$$#$$$$$

train epistrength and normstrength evaluation
$$#$#$$$$00$$$$$$$$$
$$####$+$00##$+$$+#$
#$#$$$+$$00$$$+$+#$+
#$+$$-$-$00$$$$$$$$+
$$$$$$$$#00++$$$#$++
$$$$$+$$$00$+###$$$$
+$##$$#$#00$$$#$#$$$
$$#$+$$$$00$$$$$$#$$
#$$$#$$#$00$#$$#$#$+


test full set:
Loss test: 0.23320, Accuracy test: 0.8872
	0	1
0	0.887	0.113
1	0.113	0.887

test M and T evaluation
$$###$####
$$####$###
+$$$######
+$$#$$#$$$
$$++-+++++
$$#++$$$$$
$$#$$$+$$$
$$#$$#$$$$

test epistrength and normstrength evaluation
#####$##$00$+$+++.++
#######$$00-+$$-$+--
########$00+$$$$+$$+
########$00+-#$$$$++
########$00+$$$$#$$-
####$#$$$00$$$#+$+$$
########$00$$$#$$+++
#$#####$#00$#$$+$+++
#####$$$$00+#$#$$$$-

-----------------------------------------------------------------
Started training on epoch 9 of 30, learning rate 0.0000842
Total time elapsed: 0h 31m 54s

train full set:
Loss train: 0.22531, Accuracy train: 0.8870
	0	1
0	0.889	0.111
1	0.115	0.885

train M and T evaluation
$$##$$$$$$
$++$$$$$$$
$$$##$$$+$
$$$#+$#$$$
$$#$$$$+$$
$$$$$$#$$$
$$$$##$$++
$$$$$$$$$#

train epistrength and normstrength evaluation
#$$$#++$#00$$$$+$$$#
#$$$$#$$$00$$$$$+$$$
$#$$#$$$$00$$$$#$$+#
$#$$$$$$$00###$$$$$#
$$$$+$$$#00$$$#$$$+$
#$##$$$$+00$$$$$##$+
$++###-$$00$#$$+##$#
$$$$+$+$$00+#$++$$$+
$$$#$$#$$00$+$+$$$-#


test full set:
Loss test: 0.23702, Accuracy test: 0.8856
	0	1
0	0.882	0.118
1	0.111	0.889

test M and T evaluation
$$#$#$####
+$##$#####
+$$#$##$##
$++$$$$$$$
$+++++++++
$$$$$+$+$$
$$$$$$$$$#
$#$$#$$$$$

test epistrength and normstrength evaluation
#######$$00--$++++++
########$00$-$$$$$-$
#########00+$$$-$$+-
########$00$$#$$$#++
########$00-$$#$$$+$
########$00$#$$++$-+
###$####$00+$$$$$$--
#######+$00$$+$$$++-
#####$#$$00$$$$#++$+

-----------------------------------------------------------------
Started training on epoch 10 of 30, learning rate 0.0000719
Total time elapsed: 0h 33m 56s

train full set:
Loss train: 0.20837, Accuracy train: 0.9034
	0	1
0	0.872	0.128
1	0.065	0.935

train M and T evaluation
$$$$$$$$+$
$$$$+$$+$$
$$$$$$$$+$
$$$$$$$$$#
$$#$$$$$$$
$$$$$#$$$#
$$$#$$$$+$
$$$$$$$$$$

train epistrength and normstrength evaluation
$$$$$$$$$00$$#+#$#$$
#$$$+#+#$00+$#$+$#$#
-$$#$#$+#00#-$$+$#-$
+$#$#$$+$00$#$$$$$$$
##$#$$$+$00#$$+++#$+
+$$+--$$$00$$$$$+$$#
$$$#$-$#$00+##++$+##
$$+$$$$$#00$#$#$$$$#
$#$#$$$##00$$#$$$$$$


test full set:
Loss test: 0.23493, Accuracy test: 0.8842
	0	1
0	0.859	0.141
1	0.091	0.909

test M and T evaluation
$$#######$
$$$#$#####
+$#$$#####
$$$$$$$$#$
-$++++$+$+
$+$+++$+$+
$$$+$$++$$
$#$+$$$$++

test epistrength and normstrength evaluation
#$##$##$$00+-++-$+.-
######$#$00+$$$-+$.+
#########00+#+#$+$++
######$#$00+$$$$+++-
########$00+#+$$+$++
####$$$$$00##$$$-++-
########$00$#$$$++--
###$###$+00$$$$+$++-
#######$$00$$##$$$$-

-----------------------------------------------------------------
Started training on epoch 11 of 30, learning rate 0.0000613
Total time elapsed: 0h 35m 58s

train full set:
Loss train: 0.20209, Accuracy train: 0.9010
	0	1
0	0.903	0.097
1	0.102	0.898

train M and T evaluation
$$$$$$$##$
$$$$$#$$$$
$$#$$$#$$$
#$$$$$$$$$
$$$$$+$$$$
$$$$$$#$$$
$$$$$##$$$
$$$$$$#$$+

train epistrength and normstrength evaluation
##$$$$#++00+$#$+$$+$
#$$$$$+##00$#$$#$$$#
$##$#$#$$00$+#$#$$##
#+$$$$$#$00$$#$$$$#$
$$$$$$+$$00$+#$$$$$$
$$$$$+#+$00$$$$#$#$$
$$$#$$$$$00###$$$$$#
$$##$$$##00$$+$#$$#$
$$$$$#$$#00$$$#$$#$#


test full set:
Loss test: 0.23583, Accuracy test: 0.8878
	0	1
0	0.894	0.106
1	0.118	0.882

test M and T evaluation
$#$#######
$##$######
+$########
$$$$$$####
$$++$$+$$$
$#$++-#$+$
#$$+$$$$$$
$$$+$$+$$$

test epistrength and normstrength evaluation
#########00$$$++-++.
########$00++#$++++-
#########00+$#$$+-+-
########$00$$#$$$$$-
########$00$$$$$$++$
#####$$$$00$###+$-+$
########$00+#$$$$+-+
###$#####00$#$$#$$++
#####$#$$00$#$##+$$+

-----------------------------------------------------------------
Started training on epoch 12 of 30, learning rate 0.0000523
Total time elapsed: 0h 37m 59s

train full set:
Loss train: 0.21320, Accuracy train: 0.8938
	0	1
0	0.876	0.124
1	0.088	0.912

train M and T evaluation
$$$$$$$$$+
$+$$$$$$$$
$$$$$$$$#$
+$$$$$$$#$
$#$#$$+$$$
$$$$$+$#$$
+$#$$$$$$$
#$$$$##$$$

train epistrength and normstrength evaluation
$+$$$$-$+00#$$##$+$#
$$+$$$#+$00$$$##+$$$
$$$$$#$$$00$$$++$$$+
#$$$#$$+$00###$$$#$$
#$$-#+$$$00#$$+$$$$$
$$$$$#$$$00$$##+$##$
$$$$$$+$#00$##$$$+$$
$+$$$$$#$00$+$$$$$-#
$$+$$$$#$00+$$$$$$$+


test full set:
Loss test: 0.22719, Accuracy test: 0.8952
	0	1
0	0.881	0.119
1	0.090	0.910

test M and T evaluation
$$#$$#$##$
$$##$##$##
$$$#######
+$$$$$$$##
$$$++++++$
$+#$$+$+$$
#$$+$+$$$$
$$$$#$$#$$

test epistrength and normstrength evaluation
$####$###00-+$+$$+$-
#######$#00+++$$-$-+
########$00$$$$$$+++
########$00++$++-$--
#######$$00$$$+$$$$$
######$#$00+$$$+$$++
######$#+00+$$$#$+-+
######$$$00+$$$$$$++
######$$+00+##$#$$-$

-----------------------------------------------------------------
Started training on epoch 13 of 30, learning rate 0.0000446
Total time elapsed: 0h 40m 1s

train full set:
Loss train: 0.21571, Accuracy train: 0.8966
	0	1
0	0.869	0.131
1	0.076	0.924

train M and T evaluation
$$$$$$$$$$
+$$+$$$$$$
#$#$$$$$$$
$$$$$$$#$$
$$$$$$$#$#
#$$$$$$$$$
$$$$$+#$$$
$$$$$$$#$$

train epistrength and normstrength evaluation
$#$$$$++$00$$$+#$$$#
##$$$$$$$00$$$$$$$#$
+$#$##+##00$$+##$$$$
$$+$$$#$$00+$$#$+$#+
#$$$$+$$$00#$$+$+$$$
$$+++$$##00+$$$#$++$
+$$$$#$$$00$$$$$$$$#
++-+$-$$$00$$$$$$$#$
#+#$#$$$$00$$$#$$#$$


test full set:
Loss test: 0.22396, Accuracy test: 0.8912
	0	1
0	0.870	0.130
1	0.087	0.913

test M and T evaluation
$$#$#$###$
$$########
$+########
$$$$$$$$#$
$+-+$+$$+$
$$+$++$+$$
$$$+$$$$+$
$$$+$$$$#$

test epistrength and normstrength evaluation
#$#####$+00-++-$++-+
######$##00$+$$$-++-
#########00$$##$$++-
########+00+$$$#--+-
########$00$$$$$$+$+
#####$#$$00$##$$++-+
######$#$00#$$$$++++
######$$$00$$$$$$$+-
#######$+00+#$$$+$++

-----------------------------------------------------------------
Started training on epoch 14 of 30, learning rate 0.0000381
Total time elapsed: 0h 42m 3s

train full set:
Loss train: 0.21622, Accuracy train: 0.8978
	0	1
0	0.890	0.110
1	0.094	0.906

train M and T evaluation
$$$$$$+$$$
$$$$$$$$$+
$$$$$$$$$$
$$$#$$$$$$
$$$$$$$#$$
#$#$+$$$+$
#$$$$$#+$#
$$$$$$$$$$

train epistrength and normstrength evaluation
##$$#-#$$00$$$$$+$$$
$$$$$$$+$00$##$$$$$+
$#+$+#$$$00$#$$$+$$$
$$$$#$+#+00+$#+$#$$$
$#+$#+$$+00#+$#$#$#$
+$$$$$##$00$$$+$##$$
$$$$+#$$#00#$$#$$$$#
$$$#$$$#+00$#$$#$$$$
$##$$$-$$00$$$$++$##


test full set:
Loss test: 0.23771, Accuracy test: 0.8852
	0	1
0	0.874	0.126
1	0.104	0.896

test M and T evaluation
$$#$#$#$##
$$$#$#$###
-$$$######
++$$$$$$#$
+++$$+$$+-
$$$+$+$$$$
##$+#$$$$$
$#$+$+$$#$

test epistrength and normstrength evaluation
$###$##$$00+$+++$--+
#######$$00$+-++$$--
#########00+++$$$$++
########$00+++$++#$-
########$00++$+$$#-+
#####$+$$00$$$$$$+-+
######$#+00$$$$#$$++
#######$$00$$+$$#+--
#######$+00$###$$$$+

-----------------------------------------------------------------
Started training on epoch 15 of 30, learning rate 0.0000325
Total time elapsed: 0h 44m 4s

train full set:
Loss train: 0.22885, Accuracy train: 0.8850
	0	1
0	0.864	0.136
1	0.094	0.906

train M and T evaluation
$$$$$$$-$$
$$$$+#+$$$
$$$$+$$$$$
+$$$$$$$$$
$$$$$+#$$$
$$$$$#$$$$
$$$$$$$$$$
+$$+$+$#$+

train epistrength and normstrength evaluation
$+$$+$$$$00$#$$$+#$$
$+$$#$$$$00$$+++$$$+
+$$#$$$$#00$$$$$$$$$
$$+##+$$$00#$$$$$$$#
#$+$+$$+$00+$+$$+$$+
$$+$+$+$+00##$$$$-$$
$+$$+$$#$00#$$#$#++$
$#$$#$$$$00$$#$+$$$$
$##$#$#+#00$+$$-$+$$


test full set:
Loss test: 0.23159, Accuracy test: 0.8850
	0	1
0	0.882	0.118
1	0.112	0.888

test M and T evaluation
$$########
#####$####
$$########
$+$$$$$$$$
+++++-$--+
+$$+$++$$+
#$$$$$$$+$
#$$##+#$$$

test epistrength and normstrength evaluation
#########00+++-+$---
#######$#00-$+$+-$--
#########00+$$$++$--
########$00$+#$+++++
########$00$$#$$$$$+
######$#$00$$$#$$+++
#####$###00$$$#$++-+
########$00$$++##+++
#########00$$$$$$+++

-----------------------------------------------------------------
Started training on epoch 16 of 30, learning rate 0.0000277
Total time elapsed: 0h 46m 6s

train full set:
Loss train: 0.20955, Accuracy train: 0.8988
	0	1
0	0.880	0.120
1	0.083	0.917

train M and T evaluation
$$$$##$#+$
$$$$$$$$$$
#$#$+$$#$$
$$$$$$$$$$
$$$$$$#++$
$#+$$+$#$$
#$$$+$#$$$
$$$#$$$$$#

train epistrength and normstrength evaluation
#$$$##$+$00$$$$$#$$+
$$$#$$#$$00$$#$$$$#$
$#$$$##++00+$$$$$$+$
$$$#$$$$#00+$$$$$$##
$$#$#$$$#00$$$#$$$##
$#$$$$$$$00++$+#$$$$
+$$#$$#+$00+#$$$#$#$
$$$$$+$$$00$#$$$$$$$
+$-$$$###00$+$$$$$$$


test full set:
Loss test: 0.23030, Accuracy test: 0.8846
	0	1
0	0.867	0.133
1	0.099	0.901

test M and T evaluation
$$########
#$##$#$###
$$$#######
+$$$$$$$$$
$++$++++$+
#$$$++$+$+
$$$+$++$++
$#$+#$#$$$

test epistrength and normstrength evaluation
#########00++$$-++-+
#########00+#$++++$-
########$00$$##+$++$
########$00$+$+$$-+-
########$00+#+$$$+-$
######$$$00+##$++++-
#####$###00$+#$$$--+
###$###$$00$$$$+$-+$
#######$$00###$#+$-$

-----------------------------------------------------------------
Started training on epoch 17 of 30, learning rate 0.0000236
Total time elapsed: 0h 48m 8s

train full set:
Loss train: 0.20813, Accuracy train: 0.8976
	0	1
0	0.888	0.112
1	0.094	0.906

train M and T evaluation
#$+$$$$$#$
$$$$$$$#$$
$#$$$$$#$$
#$$$#$$++$
$$##$$##$$
$$$$#$#$$$
$$$$$$$$$$
$#+$$$$#$$

train epistrength and normstrength evaluation
###$$$+$+00##$$#$+$#
##$$$$$#$00#$$$$$$#$
$$#+$$$$#00$+$##$#$$
$#$$$$+$$00$$#$#$$$$
$$$#$#$$#00$$#+##$+#
$$$#$$$$$00$$#+$$$$$
+$$$$$$$$00##$$$##$$
$$$##$$$$00$$$$+$-$$
##$#+$$$$00$#$+$#+$#


test full set:
Loss test: 0.22688, Accuracy test: 0.8886
	0	1
0	0.878	0.122
1	0.101	0.899

test M and T evaluation
$$$$#$###$
$$#$######
+$$#######
-+$#$$$$##
$+++$$++$-
#$#+#+$+$$
$$#$#++$$$
$$$$$$$$$$

test epistrength and normstrength evaluation
$#######$00-+++++--+
#######$$00$$$#-++.-
#########00+#$#$$$++
#########00+$#$$$$++
########$00$#$$$$+-+
#####$$#+00$$$#+$++$
###$#$$#$00#$$+$$+++
#######$#00$$$$$$+$+
#######$+00$#$$$+$+$

-----------------------------------------------------------------
Started training on epoch 18 of 30, learning rate 0.0000202
Total time elapsed: 0h 50m 10s

train full set:
Loss train: 0.20934, Accuracy train: 0.8976
	0	1
0	0.884	0.116
1	0.088	0.912

train M and T evaluation
$$$$#$$#$+
$$$$$$$$$$
$$$$$$+$$$
$$$$$$$$$$
$$$$$$+$#$
$$#$$$$$#$
$$$$$$$$$$
++$$$$$$$$

train epistrength and normstrength evaluation
$$+$+$#$$00$$$$#$+#+
+$+$#+$$#00+#++$+#$$
#+$$#$$+#00$$$$$#$$$
$$#$$$$$+00$$$$$$$$$
#$#$$#$+$00$#$$$$$$+
$+$##$$$$00$#+$#++$#
$$$$$$$$+00+#$$+$$$$
$$$$+$$$$00$#$+#$$$#
$$$$$+$$$00$##$$$$+$


test full set:
Loss test: 0.22346, Accuracy test: 0.8914
	0	1
0	0.879	0.121
1	0.096	0.904

test M and T evaluation
$$$######$
$$#$######
$$#$######
+$$$+$$$#$
++++$+++++
#$$$$$$$$$
$$$$#+$$$$
$#$$$+$$$$

test epistrength and normstrength evaluation
#########00-+$++-+$+
######$$#00+$++$+++-
########$00$+$$$+$++
#########00+$$+$$$$-
#########00+##$$$$-$
######$$+00+##+$$$+-
#####$##$00$#$$#++--
########$00$$$###$++
#######$-00$$$$#$$++

-----------------------------------------------------------------
Started training on epoch 19 of 30, learning rate 0.0000172
Total time elapsed: 0h 52m 12s

train full set:
Loss train: 0.20548, Accuracy train: 0.9000
	0	1
0	0.877	0.123
1	0.077	0.923

train M and T evaluation
$$$$$$$$$$
#$+$$$$$$+
$$$$++$$$$
+$#$$$#$$#
#$$#$#$$$$
$$$$$$$$$$
$$$$#$$#$$
$$$$$$$$$+

train epistrength and normstrength evaluation
$$+$$$$+$00$#$$$$$##
$$$$$$$+$00$#++#$$+$
$$$$$+#$$00#+$$$#$$$
++$#+$$$$00$$$$$#$$#
$#+$$##$$00$$$+$####
#$+$$$$$$00++$#$$$+$
$$$$#$$#+00$##+$$$$$
$$$$$+$$$00$+#$+#+$$
#$$$#$#$$00+$##+$$+$


test full set:
Loss test: 0.21664, Accuracy test: 0.8974
	0	1
0	0.883	0.117
1	0.088	0.912

test M and T evaluation
$#########
#$####$###
$$$#######
$$+$$$#$#$
+$+++++++$
$$$$$$+$$$
$$$$$+$$$$
#$$$$+$$$+

test epistrength and normstrength evaluation
$########00++$+-+--+
#######$#00+++$$+++-
#########00$$+$#++$+
########$00$$$$+++++
#########00$$#+$$+$+
########$00$$$#$++++
######$##00$$$$#$+-+
########$00$#$$$$+++
#######$+00$$$$$$$$+

-----------------------------------------------------------------
Started training on epoch 20 of 30, learning rate 0.0000147
Total time elapsed: 0h 54m 14s

train full set:
Loss train: 0.21865, Accuracy train: 0.8942
	0	1
0	0.865	0.135
1	0.077	0.923

train M and T evaluation
$$$$$$+$$$
$#$$$$$$$$
$$$$$$$$+$
$$$$$#$$$$
#$$+$$+#+#
+$#++$$+$#
$$$$$$$$$$
$$-$+##$+#

train epistrength and normstrength evaluation
+$$$$+#$$00#$#+$$$$$
-#$$$$$$+00#$$$$$#+$
$$$#+$$$+00#$$$+##+$
$-$$$+$$#00$$++$$$$$
$-+$$$$$$00$$$$$$###
$+#$$$$#$00+#$$+#+$+
$$#$++$$$00+#$+$#$$$
#$$$$##$$00##$$##$$$
$$+#$++#$00$$#$+$++$


test full set:
Loss test: 0.22963, Accuracy test: 0.8940
	0	1
0	0.870	0.130
1	0.083	0.917

test M and T evaluation
$#########
$$##$###$#
+$#$$#####
$$$$$$$$##
$+++$++++$
$$$$+$#$$+
#$$$$$$$$+
$$$++$$$$$

test epistrength and normstrength evaluation
#$#######00--$+-$+--
#######$#00$++$++++-
########$00$$$$$+$++
########+00+$$+$++++
########$00-$$#$#$++
######$$$00$$$$+$$++
########$00$$$$#$+-+
#######$$00+$$$$++$+
#######$$00$$$#$$$+-

-----------------------------------------------------------------
Started training on epoch 21 of 30, learning rate 0.0000125
Total time elapsed: 0h 56m 16s

train full set:
Loss train: 0.21227, Accuracy train: 0.8936
	0	1
0	0.873	0.127
1	0.086	0.914

train M and T evaluation
$$$$$$$$$$
$$$$$$$$$$
$#$$$$$$$$
$$$$$$$+$$
+$$$$$$$$$
$$+$$$$$$$
$$$$$$#$$#
$$$$$$$$$$

train epistrength and normstrength evaluation
$$$+$$$++00$+$##$##+
$$$$$$#$$00$$$$$$-$#
$$$$$$$#$00$$$#+#$$+
$-#$$$$$#00$$##$+$$$
$+##+$$+$00#$##$$$##
+#$$$$$$$00$#$$$$$+$
#$$$$$$##00+$$-$+$#$
+$##$$$$$00$+$+$++$$
$$$$$$$#$00#$$$$$$#$


test full set:
Loss test: 0.22968, Accuracy test: 0.8892
	0	1
0	0.879	0.121
1	0.101	0.899

test M and T evaluation
$$#$$$####
$#####$###
$$#####$##
+$$$$+#$$#
++++$$++-$
$-$$$$$+$$
$$#$$$$$$$
$#$$#$$$$$

test epistrength and normstrength evaluation
########$00.+-$+++++
#######$$00-$$++$#-+
########$00$$$$++$++
########+00$$+$+$#++
#########00$$$#$+$++
########+00$#$$+$$++
###$#####00$##+$$+-+
########$00$$$$$$$$-
#######$$00+$$$#+$+-

-----------------------------------------------------------------
Started training on epoch 22 of 30, learning rate 0.0000107
Total time elapsed: 0h 58m 18s

train full set:
Loss train: 0.21923, Accuracy train: 0.8912
	0	1
0	0.878	0.122
1	0.095	0.905

train M and T evaluation
$$$+$+$$$$
$$$$$$#$+$
$$#$+$$+$#
$$$$$$$#+$
$$$$$#+$#$
$#$$$$$$#+
$$$$$$$+#$
#$$$$$+$+$

train epistrength and normstrength evaluation
+$$++$#$#00+$#+$$$+$
+$$$$$+$$00+$#$$##$$
$#$$#$$$$00+++#$$+$$
$#$+$$$$#00+$#$$$$$+
$$+$$#$#$00$$$$+$##$
###$$$#$$00#$#$#$+++
$$+$#$#$$00$+#$$$#$#
$$#$$$+$$00#$#+$$$#+
#$$$+$$$$00$#$$$+$#$


test full set:
Loss test: 0.22885, Accuracy test: 0.8914
	0	1
0	0.879	0.121
1	0.096	0.904

test M and T evaluation
$$#$######
+$$#$#####
+#$#######
+$$#$$$$#$
$+++$+++-$
+$$#++$+$$
$$$$$$+$$$
$$$$$++$$$

test epistrength and normstrength evaluation
$$#######00++$$$++-+
######$$$00+$++-+$+-
#########00$$$$$$+--
########$00++#$-$$-+
########$00$$+$$$$+-
########+00+$##+$+++
########$00$$#$$++-.
#######$$00$$$$$$$$-
#####$##$00$##$$$+$+

-----------------------------------------------------------------
Started training on epoch 23 of 30, learning rate 0.0000091
Total time elapsed: 1h 0m 20s

train full set:
Loss train: 0.21680, Accuracy train: 0.8928
	0	1
0	0.887	0.113
1	0.101	0.899

train M and T evaluation
$$$$$##$$$
$$$$+$$$#$
$$$$$$$$$$
#$$#$$$$$$
$$$$$$#$$$
$+#$$$$$#$
+$$$$+$$$$
+$$$$$##$+

train epistrength and normstrength evaluation
##$$#$#$$00#$$#$$$$$
###$+$$$$00++$$$#$$$
++$$#$+++00$$$+$+$$$
$$$$$#$+$00$$$$#$$$$
$$#$+$$$$00$$$$$+#$$
$$$$$$$#$00$$$$#+#$#
$#$$$++#$00#$$$#$$+$
+$#$#$$$$00$$$$+$+-$
$$#+#$$$+00#$$$#$+$+


test full set:
Loss test: 0.23380, Accuracy test: 0.8856
	0	1
0	0.883	0.117
1	0.112	0.888

test M and T evaluation
$$########
###$$#####
+$$#######
+$+++$$$#$
$$+$++++$+
$$$$+$$$$+
$$$##$$$$$
$#$+$$$$$$

test epistrength and normstrength evaluation
$$#######00+-++$-+--
#######-$00-$$$$+$++
########$00-$$$$-$++
########$00-+#+$+$+-
#########00$$$#$+$$+
######$#$00$#$$+++++
######$##00$$###$-.+
###$####$00$$$#$#+-+
#####$#$$00$#$$#++++

-----------------------------------------------------------------
Started training on epoch 24 of 30, learning rate 0.0000078
Total time elapsed: 1h 2m 23s

train full set:
Loss train: 0.21085, Accuracy train: 0.8958
	0	1
0	0.875	0.125
1	0.082	0.918

train M and T evaluation
$$$$#$$$$$
$$$$$$$$$#
$$$$$+##$#
+$$#$$$$+$
$$$$+$$$#$
$+$$$$$+$$
$+$$$$$$$$
$$$$$$$$$$

train epistrength and normstrength evaluation
#+$$#$$$$00+$$##$$$$
#$$$+$$$#00$$$$#$$$$
$$++$$#$+00+$####$-$
+##$$$$+$00##$$$+$#$
$$+#$+$++00++$$+#$+$
$+$$$$##$00#$$$$$$$#
$$$$$$$#+00$$##$$$+$
$$$+#+-$$00$$$$+$$$$
#$$$#$$$$00$$$$$$$$+


test full set:
Loss test: 0.22135, Accuracy test: 0.8898
	0	1
0	0.868	0.132
1	0.089	0.911

test M and T evaluation
$$###$####
$$$$$#####
+$$$######
$$$$+$$$#$
$+++$+++$$
$$$$$+++$$
$$$+$++$$+
$#$$$+$$$$

test epistrength and normstrength evaluation
########+00--$$++-++
#######$$00+$+$+$$++
#########00+$$#-+#++
#######$#00-$$$$+$+-
########$00+#+$$$$-#
######$$$00$$$#++$-$
######$##00$$$$$$$--
#######$$00+#$+$+$+-
########+00+#$##$+$-

-----------------------------------------------------------------
Started training on epoch 25 of 30, learning rate 0.0000066
Total time elapsed: 1h 4m 25s

train full set:
Loss train: 0.20345, Accuracy train: 0.9026
	0	1
0	0.891	0.109
1	0.085	0.915

train M and T evaluation
$#$$$#$$$$
$$$$+$+$$$
$$$$$$$#$$
$##$$#$$+$
$$$#$$+$#+
$$$$$$$$$$
$$$$$$$$$$
$$$+#$#$++

train epistrength and normstrength evaluation
+$$$$$$$$00#$$$+$$#$
$$$+$#$$$00$$$$$#-$$
$#$$$$$##00$$$$$#$#$
$+#$+$$$+00$$$$$$$$$
$$#$$$$$$00#$++$$+$$
$#$$#$$$$00$$++$$$#$
$#$$#$+$$00+$$#+$#+#
$$$$#$$$$00$$#$#$+$$
$$$$#$$$$00$##$$##$#


test full set:
Loss test: 0.23304, Accuracy test: 0.8888
	0	1
0	0.868	0.132
1	0.091	0.909

test M and T evaluation
$$#$$#####
#$#$#$$###
+$$#######
+$$+$+$$#$
$++$$+$+++
$$$$$+$+$+
$#$$$++$$$
$$$$$$$+$$

test epistrength and normstrength evaluation
#######$#00---++---+
#######$$00-$+++++--
#########00$$$#$$$$-
#######$$00$+##$+++-
#########00$$$$+$$+#
#####$$$$00+$+$$++-+
########$00#$$$++++-
########$00$$$$$+$-$
$######$$00+$$$#$$+-

-----------------------------------------------------------------
Started training on epoch 26 of 30, learning rate 0.0000057
Total time elapsed: 1h 6m 27s

train full set:
Loss train: 0.21805, Accuracy train: 0.8958
	0	1
0	0.887	0.113
1	0.096	0.904

train M and T evaluation
$$$#$$$$$$
#$$$$$$$##
$$$$$$$$$#
$$$$$+$$$+
$$$$$$#$$$
$$$$#$$$$+
#+++$$$$#$
$$##$$$$$$

train epistrength and normstrength evaluation
###$+#++$00$+##$$+$$
$+#$$$-$$00$#$$$$$$#
$$$#$$$$+00$++$$$$$$
$$#$+#$+$00$$$$$$$#$
+#$#$#$$$00-+##$$$$$
$-$#$$$#$00+$##+##+$
#$$#$$#$#00$$#$#$$#$
+$++$+$$$00$$#+$#$$#
$$$$##$$$00###$+#$$$


test full set:
Loss test: 0.22860, Accuracy test: 0.8892
	0	1
0	0.881	0.119
1	0.102	0.898

test M and T evaluation
$$#$$##$##
$$$#######
+$$####$##
$++$$$$$#$
$$+++$$-$+
++$+++$$$$
##$$$$$+$$
$$$$#+$$$$

test epistrength and normstrength evaluation
$$#######00-+-$+++--
#######$#00-+$$-+++-
########$00+$$$$+$-$
########$00++$$-$$++
########$00+$$$$$$-+
########$00$$#$+$+++
########$00$$$##$+-$
########$00$$$$#$$++
######$$+00+#++$$$++

-----------------------------------------------------------------
Started training on epoch 27 of 30, learning rate 0.0000048
Total time elapsed: 1h 8m 29s

train full set:
Loss train: 0.20873, Accuracy train: 0.8952
	0	1
0	0.879	0.121
1	0.088	0.912

train M and T evaluation
##$$$$#$$+
$$$$$$$$$$
$$$$$$+$$$
$$$$##$$$$
$$$$$$$$+$
$$$$$#$$$$
$$$#$$$$$$
$$$$$$++#+

train epistrength and normstrength evaluation
$+$$$$$$$00##$$$#$$$
+$+$$$#$$00$$##+#$$#
$#+$$$$$$00$#$$++$+#
$$$$+-$$$00#$#$$$$+$
#$#+$$$$$00$#$$#$#$$
+$$$$$+$$00#$$$$$$+#
$$$-$+$$$00##$$#$$$$
$$$$$$#$#00$+$$$$#.$
#$$$$$$#$00+$$$#+-++


test full set:
Loss test: 0.22710, Accuracy test: 0.8880
	0	1
0	0.873	0.127
1	0.097	0.903

test M and T evaluation
+$#$#$####
$#$#$#####
$$$$######
++$$$$$$$#
$$+++++++$
$$$$++$+++
###+$+$$$$
$$$$#++$$$

test epistrength and normstrength evaluation
$$#####$#00+++++$++-
######$$$00+$+$-$$++
#########00$-$$$$$+#
#######$+00++##++$--
#########00$$$#$+++$
####$##$$00$$#$+++-$
#####$##$00$$$$$+$--
########$00$#+$$$+++
$#####$$+00$#$$$$+-+

-----------------------------------------------------------------
Started training on epoch 28 of 30, learning rate 0.0000041
Total time elapsed: 1h 10m 32s

train full set:
Loss train: 0.21230, Accuracy train: 0.8978
	0	1
0	0.881	0.119
1	0.085	0.915

train M and T evaluation
$$#$++$$$$
#$$$$#$$$$
$$$$$$$$$$
$$$$$$$$+$
$$$$$$$$$$
$$$$$$$#$$
$$#$$$$$$$
$$$$$$$$$$

train epistrength and normstrength evaluation
#$$##$$$#00$$+#$$$#$
$#$$+$$#+00$$$+#$$$$
$$$$#$#$#00$+$$-$$+$
#$$$$#+$+00$$+#$$$#$
#$$+#$$$$00$++$$#$##
$+$#+$$+$00$-##$$$+$
$$$$$#+$$00$$#$$$#$$
$$#$$$$+$00###+$$$$$
$$++#+$$#00$$$#$$+#$


test full set:
Loss test: 0.22082, Accuracy test: 0.8958
	0	1
0	0.884	0.116
1	0.092	0.908

test M and T evaluation
$$#$#$####
$$#$######
$$$#######
+$$$$$$$#$
$++++++$$$
$$$$$+$$$$
$$$+$$$$$$
$$$$$$$$$+

test epistrength and normstrength evaluation
#$#####$$00-+-$$$--+
#######$#00-+$$-$$+-
#########00$$$$$++-$
########$00++$$$$$-+
#########00+$+$++$+$
######$#+00$##$$$++-
########$00$$$$$+$-+
###$###$+00$#$#$++$$
$#####$#+00$##$$$+$+

-----------------------------------------------------------------
Started training on epoch 29 of 30, learning rate 0.0000035
Total time elapsed: 1h 12m 35s

train full set:
Loss train: 0.20513, Accuracy train: 0.9016
	0	1
0	0.884	0.116
1	0.081	0.919

train M and T evaluation
$#$$$$$#$#
+$+$#+$-##
$$+$$$$$+$
$+$$$$+$$$
$$$$$$$$$$
$$$$$$$$$#
$$$$$+$$$#
$+$$$$$$$$

train epistrength and normstrength evaluation
$$$$$$#++00$+$##$$+$
+$$$$+$$$00$$+$#$$#$
++#$$$$$$00$#$+$+$+$
$###$$##$00#$$$##$$+
$$#+$$$$$00$$#####$#
$$$$$$$+$00+$#$$#$$$
$$#$#$$$#00$+$$#+$$#
+#+$#$#$#00$+#$$$$$$
+$$$$$$+$00$$#$$$$$+


test full set:
Loss test: 0.21843, Accuracy test: 0.8938
	0	1
0	0.884	0.116
1	0.096	0.904

test M and T evaluation
$$#$#$####
$$##$#####
$$########
$+$+$$$$$$
$+++++$$$+
#$$$$$$$$+
#$$+$+$$$$
$#$$#$$$$$

test epistrength and normstrength evaluation
$#######+00-+$$$$+-+
#######$#00++$$+-++-
########$00+$$$$-+$+
########$00$$#$++++-
########$00+#$$#+++$
#####$###00+#$++++$+
#########00$$$$$$+++
#######$$00$$$$$$+$+
#####$#$$00$#$#$$$+$

-----------------------------------------------------------------
Started training on epoch 30 of 30, learning rate 0.0000030
Total time elapsed: 1h 14m 37s

train full set:
Loss train: 0.21853, Accuracy train: 0.8900
	0	1
0	0.878	0.122
1	0.098	0.902

train M and T evaluation
$$$$$$$$#$
$$$+#$$$$$
+$+$$$$$$$
$$$$$$#$#$
+$$$$$$+$$
#+$$$$$$#$
$#$$$$$$#$
$$$$$+$+$#

train epistrength and normstrength evaluation
$$$+$-$$#00$$$$$$$$$
$$$$$$$$$00$#$+$+$#$
$$$$+$$$$00$+$$$$##$
$##$#$#-$00##+$$$$#$
+$$#$#$$$00$#+$$$+$$
$$+$$$$$$00#$#$$$$+$
#$$$$+$$+00$##$$$$$-
+$+#$#$#$00$#$+$#+$+
$+$$$$$$+00#$$$$$$$#


test full set:
Loss test: 0.22285, Accuracy test: 0.8922
	0	1
0	0.883	0.117
1	0.099	0.901

test M and T evaluation
$$###$####
$##$#$##$#
$$########
+$$$$$$$$#
$+-$$+++$+
$$$+$+$$$+
##$+$+$$$+
$##+$$$$$$

test epistrength and normstrength evaluation
########$00+$++++$-$
#######$$00-+$$+$++-
########$00$$$#+++$+
#$######$00++#$$$$+-
########$00+$#$$$$++
#####$$#$00$$$$+$$+$
###$#$##$00$#$$$$$-+
#######$$00$#$$$$+$$
######$#$00$$$$$$+$$
"""