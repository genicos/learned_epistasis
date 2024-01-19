import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(64)

num_indv = 99
num_chrom = num_indv * 2
piece_size = 5 #21
len_chrom = 500
samples_per_morgan = 1000

n_embd = len_chrom
input_size = num_indv
head_size = 16 #4000#64
num_heads = 4 #head_size must be divisible by num_heads
num_blocks = 2


class Head(nn.Module):
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("communication_matrix", torch.ones(input_size,input_size))

    def forward(self, x1, x2):
        # x1 from output (sampling), x2 from input (site location)

        # Input both (batch, input_size, n_embd)
        # Output (batch, input_size, head_size)
        k = self.key(x1) # (batch, input_size, head_size)
        q = self.query(x2)  # (batch, input_size, head_size)
        W = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (batch, input_size, input_size)
        W = W.masked_fill(self.communication_matrix[:input_size, :input_size] == 0, float('-inf')) # (batch, input_size, input_size)
        W = F.softmax(W, dim=-1)

        v = self.value(x2) # (batch, input_size, head_size)
        out = W @ v # (batch, input_size, head_size)
        return out
    
class MultiHead(nn.Module):

    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.linear = nn.Linear(head_size*num_heads,n_embd)

    def forward(self, x1, x2):
        x = torch.cat([head(x1, x2) for head in self.heads],dim=-1) #(batch,input_size,head_size (global))
        x = self.linear(x) #(batch,input_size,n_embd)
        return x
    
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x) #(batch,input_size,n_embd)
    
class Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.multihead = MultiHead(num_heads, head_size // num_heads)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.ln4 = nn.LayerNorm(n_embd)

    def forward(self, x1, x2):
        # input = output = (batch,input_size,n_embd)
        # add a linear layer and norm for x2
        lnx1 = self.ln1(x1)
        x1 = x1 + self.multihead(lnx1,lnx1)
        x1 = x1 + self.multihead(self.ln2(x1), self.ln3(x2))

        x1 = x1 + self.ffwd(self.ln4(x1))
        return x1

class TransformerModel1(nn.Module):

    def __init__(self):
        super().__init__()
        self.pos_embedding = nn.Embedding(len_chrom, n_embd)
        #self.blocks = nn.Sequential(*[Block() for _ in range(num_blocks)])
        self.blocks = nn.ModuleList([Block() for _ in range(num_blocks)])
        self.multihead = MultiHead(num_heads, head_size // num_heads)
        self.linear = nn.Linear(input_size*n_embd,1) #can change the output size of this
        self.ln1 = nn.LayerNorm(input_size*n_embd)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # x1 (batch, input_size, n_embd) sampling
        # x2 (batch, n_embd) site location
        x2 = x2.unsqueeze(1) # (batch, 1, n_embd)
        x2 = x2.repeat(1, input_size, 1) # (batch, input_size, n_embd)
        for block in self.blocks:
            x1 = block(x1, x2) #(batch, input_size n_embd)
        x1 = x1.reshape(x1.shape[0], input_size*n_embd) #(batch, input_size*n_embd)
        x1 = self.ln1(x1) #(batch, input_size * n_embd)
        x1 = self.linear(x1) #(batch, 1)
        x1 = self.sigmoid(x1) #(batch,1)
        x1 = x1.reshape(-1) #(batch)

        return x1
    
class SimpleModel(nn.Module):
    #Simple model to use as baseline

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2*n_embd, 250)

        self.ln = nn.LayerNorm(250)

        self.relu = nn.Sigmoid()

        self.linear2 = nn.Linear(250,1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x1, x2):

        x1 = x1.mean(dim=1)
        #x2 = x2.unsqueeze(1)
        x = torch.cat((x1, x2), dim=1)
        x = x.reshape(x.shape[0], -1) # (batch, input_size*2*n_embd)



        x = self.linear1(x)
        x = self.relu(x)

        x = self.ln(x)

        x = self.linear2(x)
        x = x.reshape(-1)
        x = self.sigmoid(x)

        return x





class SumModel(nn.Module):
    #Simple model to use as baseline

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 9, (1,piece_size))

        self.linear1 = nn.Linear(2*num_chrom*piece_size, 200)

        self.ln = nn.LayerNorm(200)

        self.relu = nn.Sigmoid()

        self.linear2 = nn.Linear(9,20)
        self.linear3 = nn.Linear(20,1)

        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(p=0.1)


    def forward(self, x1):

        #x2 = x2.unsqueeze(1)
        x = x1

        #x = x.reshape(x.shape[0], -1) # (batch, input_size*2*n_embd)

        x = self.dropout(x)
        x = self.conv1(x)
        #x = self.ln(x)
        x = self.relu(x)

        x = x.mean(dim=2)

        x = x.reshape(x.shape[0], -1) # (batch, input_size*2*n_embd)
        
        x = self.linear2(x)
        x = self.relu(x)
        
        x = self.linear3(x)

        x = x.reshape(-1)
        x = self.sigmoid(x)

        return x




class EpiModel(nn.Module):
    #Simple model to use as baseline

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(15, 20, (1,piece_size))

        self.conv2 = nn.Conv2d(20, 20, (1,1))

        self.linear1 = nn.Linear(2*num_indv*piece_size, 200)

        self.ln = nn.LayerNorm(200)

        self.relu = nn.Sigmoid()

        self.linear2 = nn.Linear(20,20)
        self.linear3 = nn.Linear(20,1)

        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(p=0.1)


    def forward(self, x1):

        #x2 = x2.unsqueeze(1)
        x = x1

        #x = x.reshape(x.shape[0], -1) # (batch, input_size*2*n_embd)

        #x = self.dropout(x)
        x = self.conv1(x)
        x = self.sigmoid(x)


        x = self.conv2(x)
        x = self.sigmoid(x)

        x = x.mean(dim=2)


        x = x.reshape(x.shape[0], -1) # (batch, input_size*2*n_embd)
        
        x = self.linear2(x)
        x = self.sigmoid(x)
        
        x = self.linear3(x)

        x = x.reshape(-1)
        x = self.sigmoid(x)

        return x
    

"""
models:
SimpleModel:       simple 2 layer FFNN, used as baseline test
TransformerModel1: Decoder only transformer with attention mechanism in the len_chrom dimension
                   2 feedfoward layers are applied after
"""




class CustomLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss(reduction='none')

    def forward(self, input, output):
        # Calculate the loss
        #loss = torch.pow(input - output, 2)

        loss = self.loss(input,output)

        #weight = ((output < 0.5).int() + 1)
        weight = ((output * -1) + 2)

        loss = loss * weight
        
        loss = torch.mean(loss)

        return loss