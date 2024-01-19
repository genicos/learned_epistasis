from processing import *
from models import *
import multiprocessing



GPU_available = torch.cuda.is_available()
print(GPU_available)
sys.stdout.flush()

num_files = 100_000
num_epochs = 40
batch_size = 256
train_prop = 0.9
num_estimate = 1000
lr = 0.000001 * 10000




saving = True

data_dir = "../../data/batch_6/"


if saving:
    X = [create_input1(data_dir+"sampled_genotypes/sample_" + str(i), data_dir+"commands/command_" + str(i)) for i in range(num_files)]
    X = [x for x in X if x is not None]
    X = torch.tensor(X).float()
    torch.save(X, "X100k_p11")
    print("FINISHED X")
    sys.stdout.flush()

    C = [convert_command_file1(data_dir+"sampled_genotypes/sample_" + str(i), data_dir+"commands/command_" + str(i)) for i in range(num_files)]
    C = [x for x in C if x is not None]
    C = torch.tensor(C).float()
    torch.save(C, "C100k_p11")
    
else:
    X = torch.load("X100k_p5")
    C = torch.load("C100k_p5")



print(X.shape)
print(C.shape)


num_files = X.shape[0]

X = X.reshape(num_files*16, 15, models.num_indv, piece_size)
C = C.reshape(num_files*16, 15)

print(X.shape)
print(C.shape)

####

y = torch.tensor([i%2 for i in range(1,num_files*4 + 1)])
y = y.repeat_interleave(4)





sys.stdout.flush()

GetMemory()







# Scramble data
#ind = int(train_prop * X.shape[0])
idx = torch.randperm(X.shape[0])
# idx = [3*i for i in idx[:ind]] + [3*i + 1 for i in idx[:ind]] + [3*i + 2 for i in idx[:ind]] +  [3*i for i in idx[ind:]] + [3*i + 1 for i in idx[ind:]] + [3*i + 2 for i in idx[ind:]]


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


# Define network
model = EpiModel()
criterion = nn.BCELoss()
criterion = CustomLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)







# Define functions for getting random batch and calculating loss
def get_batch(split, num_samples):
    X, y , C  = (X_train, y_train, C_train) if split == "train" else (X_test, y_test, C_test)
    
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

        X, y , C = get_batch(split, num_samples)

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

        
        """
        

            
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
        """

            

    model.train()


if GPU_available:
    print("GPU is available.")
    model = model.to("cuda")
    criterion = criterion.to("cuda")
#    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<enter-size-here>"
else:
    print("No GPU available. Running on CPU.")

GetMemory()
estimate_loss(min(num_estimate, X_test.shape[0]))
#Training loop

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb: SIZE"





model.train()
for epoch in range(num_epochs):
    print("-----------------------------------------------------------------")
    print(f"Started training on epoch {epoch + 1} of {num_epochs}.")
    GetTime()
    # Scramble data
    idx = torch.randperm(X_train.shape[0])
    X_train = X_train[idx]
    y_train = y_train[idx]
    C_train = C_train[idx]
    
    for ind in range(0,X_train.shape[0],batch_size):

        #print(torch.cuda.max_memory_allocated(),"and", torch.cuda.memory_allocated())

        try:
            X_batch = X_train[ind:ind+batch_size]
            y_batch  =  y_train[ind:ind+batch_size]
        except IndexError:
            X_batch = X_train[ind:]
            y_batch  =  y_train[ind:]

        if GPU_available:
            X_batch  =  X_batch.to("cuda")
            y_batch  =  y_batch.to("cuda")

        try:
            y_pred = model(X_batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            y_pred = model(X_batch)


        loss = criterion(y_pred, y_batch.float())   # get loss

        optimizer.zero_grad()   #
        loss.backward()         #
        optimizer.step()        #loss.backward knows where to go

    estimate_loss(min(num_estimate, X_test.shape[0]))
    sys.stdout.flush()


GetTime()