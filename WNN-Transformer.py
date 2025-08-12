import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import ao_core as ao
import numpy as np
from datetime import datetime
"""
KEY POINTS:
1. The BigramLanguageModel is a simple next token prediction model that when combined with the transformer blocks can learn complex patterns in the data.
2. B,T,C stands for Batch size, Time steps (sequence length), and Channels (features) = x.shape where x is the input tensor.
3. This model is being tranined on the tiny shakespeare dataset and is try to generate some text based on the patterns it learns.
4. The Transformer architecture is as follows:
   - Attention Heads: These are components that allow the model to focus on different parts of the input sequence.
   - Multi-Head Attention: This combines multiple attention heads to get information about different parts of the input text seqeunce
   - Feed Forward Neural Network: This is a simple neural net that essenetially processes the output of the attention heads and returns an output corresponding to the input sequence. This can be likened to a convolutional layer in a CNN.
   - Transformer Blocks: All of the above components can be stacked together to form a transformer block.
   - Language Model: The BigramLanguageModel is a simple next token prediction model that when combined with the transformer blocks can learn complex patterns in the data.
5. Trainable params in one block: 
   """

# Hyperparameters
block_size = 64 # how many tokens to predict
batch_size = 256 # how many in parralel training examples to run
max_iters = 5000
eval_interval = 300
learning_rate = 3e-4
eval_iters = 200
n_embedd = 64 # embedding dimension for each token
n_layer = 6 # is the number of transformer blocks
n_head = 8 # number of heads in multi-head attention
dropout = 0.2 # dropout is esentially a technique to prevent overfitting by randomly disabling some neural connections during training

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load dataset
import requests, time

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

for attempt in range(5):  # Try up to 5 times
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        break
    except requests.exceptions.RequestException as e:
        print(f"Attempt {attempt+1} failed: {e}")
        time.sleep(2)  # Wait before retry
else:
    raise RuntimeError("Failed to download dataset after multiple attempts.")

with open("tinyshakespeare.txt", "w", encoding="utf-8") as f:
    f.write(response.text)

# Tokenization
chars = sorted(list(set(response.text)))
vocab_size = len(chars) # is 65 for current dataset (tiny shakjespere)

def numToBinary(num):
    binary = format(num, "064b")
    return list(binary)

def binaryToNum(binary):
    return int(str(binary), 2)

stoi = {ch: i for i, ch in enumerate(chars)} # string to index
itos = {i: ch for i, ch in enumerate(chars)} # index to string

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

data = torch.tensor(encode(response.text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, test_data = data[:n], data[n:]

# Batch generation
def get_batch(split):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# Loss estimation
@torch.no_grad() # telling torch that we are not doing backprop so it doesn't need to track gradients and can be more efficent
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "test"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Model components
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embedd, head_size, bias=False)
        self.query = nn.Linear(n_embedd, head_size, bias=False)
        self.value = nn.Linear(n_embedd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        w = q @ k.transpose(-2, -1) * C **-0.5 # comes from the scaled dot-product attention formula 
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        v = self.value(x)
        return w @ v

class weightlessNeuralNetwork():
    def __init__(self):
        self.WNN = ao.Agent(Arch=ao.Arch(arch_i=[64], arch_z=[64]))

    def float_to_bin(self, arr, threshold=0):  # The most basic conversion function. if input is greater than threshold, it will be 1, else 0
        """Converts a float32 embedding to a binary embedding."""
        binary_embedding = np.where(arr > threshold, 1, 0)
        return binary_embedding


    def forward(self, input):
        B,T,C = input.shape
        
        wnn_out = []
        print("input shape: ", input.shape)
        input = torch.flatten(input, start_dim=0, end_dim=1) # gets me to (B*T, C)
        for i, row in enumerate(torch.unbind(input, 0)): # iterate over the B*T dim
            row = row.flatten().cpu().detach().numpy()
            print("row: ", row)
            print("row shape: ", row.shape) # needs to be (64)
            wnn_out.append(self.WNN.next_state(row)) 
        wnn_out = torch.tensor(wnn_out, dtype=torch.float32, device=device)
        wnn_out=wnn_out.reshape(B,T,-1)
        print("wnn out shape: ", wnn_out.shape)
        return wnn_out

    def train(self, x, y):
        print(len(x))

        B,T,C = x.shape
        x = x.reshape(-1, C)  # shape (B*T, C)
        y = y.reshape(-1) # shape (b*T,)
        print("need ", len(x), "iterations")
        print("shape x: ", x.shape)
        print("shape y: ", y.shape)
        print("x subset: ", x[:5])
        print("y subset: ", y[:5])
        #for inp, lbl in zip(x, y):
    #         for input,label in zip(inp,lbl):
                    
    #                 input = input.flatten().cpu().numpy()
    #                 #label = label.flatten().cpu().numpy()
    #                 # for i, l in zip(input, label):
    #                 #     input_flat.append(i)
    #                 #     label_flat.append(l)
    #    # input_flat = self.float_to_bin(np.array(input_flat))

        # input = self.float_to_bin(x.cpu().numpy())
        # label = self.float_to_bin(y.cpu().numpy())
        input=[]
        label=[]
        for inp, lbl in zip(x,y):
            input.append(self.float_to_bin(inp.cpu().numpy()))
            label.append(numToBinary(lbl.cpu().numpy()))         
        # print("len inp ",  len(input))
        # #print("shape inp: ", input.shape)
        # print("inp subset: ", input[:5])

        # print("len lbl: ", len(label))
        # print("lbl subset: ", label[:5])
        #print("shape lbl: ", label.shape)
        self.WNN.next_state_batch(input,label)
        self.WNN.reset_state()
        print("finished wnn train")

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embedd, n_embedd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embedd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embedd, 4 * n_embedd), # 4x projection is a common practice to increase complexity
            nn.ReLU(),
            nn.Linear(4 * n_embedd, n_embedd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# our first transformer block
class Block(nn.Module):
    def __init__(self, n_embedd, n_head, use_wnn=False):
        super().__init__()
        head_size = n_embedd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embedd)
        self.WNN = weightlessNeuralNetwork()  # Using the weightless neural network
        self.ln1 = nn.LayerNorm(n_embedd) # layer normalization
        self.ln2 = nn.LayerNorm(n_embedd)
        self.use_wnn = use_wnn

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        if self.use_wnn: # not using wnn for training only for inference 

            x= self.WNN.forward(x)
            print("output: ", x.shape)
        return x

# Language model

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedd) # build up a token embedding table that maps each token to an embedding vector
        self.position_embedding_table = nn.Embedding(block_size, n_embedd) # positional encoding embedding table to give the model a sense of the position of each token in the sequence
        self.blocks = nn.Sequential(*[Block(n_embedd, n_head=n_head) for _ in range(n_layer)]) # Transformer blocks stacked together
        self.lm_head = nn.Linear(n_embedd, vocab_size) # final linear layer to project the output of the transformer blocks to the vocabulary size for next token prediction


    def forward(self, idx, targets=None):
        B, T = idx.shape # Idx is the input tensor but different from the input to the transformer blocks because 
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x) # logits are the raw scores for each token in the vocab at each pos in the sequence corresponding to the

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            print("generating token: ", _)
            idx_cond = idx[:, -block_size:] # we are making sure to only use the last block as context not the entire sequence
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]# get the last logit prediction for the next token
            probs = F.softmax(logits, dim=1)
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]   # expect (B, vocab_size)
            probs = F.softmax(logits, dim=1)

            print("DEBUG: idx shape:", idx.shape)             # (B, cur_seq_len)
            print("DEBUG: logits shape:", logits.shape)       # should be (B, vocab_size)
            print("DEBUG: probs shape:", probs.shape)         # should be (B, vocab_size)
            print("DEBUG: probs dtype/device:", probs.dtype, probs.device)
            assert probs.dim() == 2, "probs must be 2-D (B, V); got dim=" + str(probs.dim())
            idx_next = torch.multinomial(probs, num_samples=1) # get the next token by sampling from the probabilities
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Training
model = BigramLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {step}: Train loss {losses['train']:.4f}, Test loss {losses['test']:.4f}")

    xb, yb = get_batch("train")
    

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
# save the weights
torch.save(model.state_dict(), "bigram_language_model.pth")

# Train WNN once at the end

xb, yb = get_batch("train")

# addingthe embedding 
emb = model.token_embedding_table(xb) + model.position_embedding_table(torch.arange(xb.size(1), device=device))

# Train WNN on block outputs
with torch.no_grad():  # no gradients in PyTorch
    temp_x = emb
    for block in model.blocks:
        temp_x = temp_x + block.sa(block.ln1(temp_x))
        temp_x = temp_x + block.ffwd(block.ln2(temp_x))
                # WNN have static lookup table: training them multiple times on same data doesnt change outcome
        print("start train")
        now = datetime.now()
        block.WNN.train(temp_x, yb)
        print("finished train")
        print("time for wnn train: ", datetime.now()-now)

# Text generation

for block in model.blocks: # we want to use the wnn forward when generating text
    block.use_wnn=True

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=5)[0].tolist()
print(decode(generated))
