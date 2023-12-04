import torch
import torch.nn as nn
import torch.nn.functional as F

num_chrom = 100
len_chrom = 1000

sample_width = 1

n_embd = len_chrom
input_size = num_chrom

dropout = 0.5


class PairwiseSimpleModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.ln1 = nn.Linear(sample_width * input_size*2, 4*input_size*2)
        self.ln2 = nn.Linear(4*input_size*2, input_size*2)
        self.ln3 = nn.Linear(input_size*2, 1)

        self.norm = nn.LayerNorm(2*input_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # x = F.one_hot(x,num_classes=3)
        x = x.float()
        x = x.reshape(x.shape[0],-1)

        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.ln2(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.ln3(x)
        x = self.sigmoid(x)
        x = x.reshape(-1)

        return x
    


input_size1 = sample_width*2
# output_size1 = 64
n_chrom = 100
n_embd = n_chrom + 2
head_size = 96
num_heads = 8 #head_size must be divisible by num_heads
num_blocks = 3




class Head(nn.Module):
  
   def __init__(self, head_size):
       super().__init__()
       self.key = nn.Linear(n_embd, head_size, bias=False)
       self.query = nn.Linear(n_embd, head_size, bias=False)
       self.value = nn.Linear(n_embd, head_size, bias=False)
       self.register_buffer("communication_matrix", torch.ones(input_size1,input_size1))


   def forward(self, x):
       # Input (batch, 2*num_samples, n_embd)
       # Output (batch, 2*num_samples, head_size)
       k = self.key(x) # (batch, input_size, head_size)
       q = self.query(x)  # (batch, input_size, head_size)
       W = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (batch, input_size, input_size)
       W = W.masked_fill(self.communication_matrix[:input_size1, :input_size1] == 0, float('-inf')) # (batch, input_size, input_size)
       W = F.softmax(W, dim=-1)


       v = self.value(x) # (batch, input_size, head_size)
       out = W @ v # (batch, input_size, head_size)
       return out
  
class MultiHead(nn.Module):


   def __init__(self,num_heads,head_size):
       super().__init__()
       self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) #this can be parallelized
       self.linear = nn.Linear(head_size*num_heads,n_embd)


   def forward(self, x):
       x = torch.cat([head(x) for head in self.heads],dim=-1) #(batch,input_size,head_size (global))
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


   def forward(self, x):
       # input = output = (batch,input_size,n_embd)
       x = x + self.multihead(self.ln1(x))
       x = x + self.ffwd(self.ln2(x))
       return x


class TransformerModel1(nn.Module):


    def __init__(self):
        super().__init__()
        self.pos_embedding = nn.Embedding(sample_width*2, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(num_blocks)])
        self.multihead = MultiHead(num_heads, head_size // num_heads)
        self.linear1 = nn.Linear(2*sample_width*n_embd,sample_width*n_embd//2) #can change the output size of this
        self.ln1 = nn.LayerNorm(2*sample_width*n_embd)


        ##
        self.ln2 = nn.LayerNorm(sample_width*n_embd//2)

        self.ln3 = nn.LayerNorm(100)

        self.linear2 = nn.Linear(sample_width * n_embd//2,100)
        self.linear3 = nn.Linear(100,1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
       
        x = x.reshape(-1, 2*sample_width,n_chrom)
        # X (batch, 2*sample_width, n_chrom)
        #print(x.shape)
        device = "cuda" if torch.cuda.is_available() else "cpu"
    #    pos_embd = self.pos_embedding(torch.arange(sample_width*2).to(device)) # (2*sample_width, n_embd)
    #    x = x + pos_embd #(batch, 2*sample_width, n_embd) # we possibly want to concatenate this instead of adding
        #x = torch.cat((x, torch.zeros(len_chrom)),dim=2)

        ar = torch.arange(2*sample_width)
        site_num = (ar // sample_width).to(device)
        site_pos = ((ar % sample_width)/sample_width).to(device)
        site_num = site_num.unsqueeze(1).unsqueeze(0)
        site_num = site_num.repeat(x.shape[0],1,1)
        site_pos = site_pos.unsqueeze(1).unsqueeze(0)
        site_pos = site_pos.repeat(x.shape[0],1,1)
        x = torch.cat((x,site_num,site_pos),2)

        x = self.blocks(x) #(batch, 2*sample_width, n_embd)
        x = x.reshape(x.shape[0], 2*sample_width*n_embd) #(batch, 2*sample_width*n_embd)
        x = self.ln1(x) #(batch,2*sample_width*n_embd)
        x = self.linear1(x) #(batch, sample_width * n_embd//2)
        x = self.relu(x)

        x = self.ln2(x)
        x = self.linear2(x) #(batch, 100) #add layernorms?
        x = self.relu(x)

        x = self.ln3(x)
        x = self.linear3(x) #(batch, 1)
        x = x.reshape(-1) #(batch)
        x = self.sigmoid(x) #(batch)


        # print(x.shape)
        # x = self.multihead(x) #batch, input_size, head_size
        # x = x.view(batch, input_size*head_size)
        # x = self.linear1(x)
        # x = self.relu(x)
        # x = self.linear2(x)


        return x
    
piece_size = 15

class EpiModel(nn.Module):
    #Simple model to use as baseline

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 20, (1,piece_size))

        self.conv2 = nn.Conv2d(20, 20, (1,1))

        self.linear1 = nn.Linear(2*num_chrom*piece_size, 200)

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

        x = x.reshape(-1, )


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
