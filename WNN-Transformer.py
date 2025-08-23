import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import ao_core as ao
import numpy as np
from datetime import datetime
from datasets import load_dataset
import time
from sklearn.decomposition import PCA
import ast
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
block_size = 64 # how many tokens to look at in the past to predict the next token
batch_size = 128 # how many in parralel training examples to run
max_iters = 25000 # how many iterations to train for
eval_interval = 300
learning_rate = 3e-4
eval_iters = 200
n_embedd = 128 # embedding dimension for each token
n_layer = 8 # is the number of transformer blocks
n_head = 8 # number of heads in multi-head attention
dropout = 0.2 # dropout is esentially a technique to prevent overfitting by randomly disabling some neural connections during training

num_wnn_blocks = 1
compression = True # disables compression. For small scale testing we would probably want compression to reduce the inference time of the WNN but on larger models it would be ideal to disable compression to get the best performance

wnn_binary_compression_level = 2 # mostly to increase speed of inference- less neurons

if not compression:
    wnn_binary_compression_level = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu" # for debug

print(f"Using device: {device}")

# Load dataset

# url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# for attempt in range(5):  # Try up to 5 times
#     try:
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         break
#     except requests.exceptions.RequestException as e:
#         print(f"Attempt {attempt+1} failed: {e}")
#         time.sleep(2)  # Wait before retry
# else:
#     raise RuntimeError("Failed to download dataset after multiple attempts.")

# with open("tinyshakespeare.txt", "w", encoding="utf-8") as f:
#     f.write(response.text)

#dataset = response.text
with open("tinyshakespeare.txt", "r") as f:
    dataset = f.read()
# dataset = load_dataset("Salesforce/wikitext", 'wikitext-103-raw-v1', split="train")["text"]
# dataset = "\n".join(dataset)

min_freq = 50  # we are filtering out words that appear less than 50 times in the dataset. this brings the vocab size from -6000 to -600
char_counts ={}
for char in dataset:
    char_counts[char] = char_counts.get(char, 0) +1
# Tokenization
chars = set()
for char, count in char_counts.items():
    if count >min_freq:
        chars.add(char)
chars = sorted(chars)
vocab_size = len(chars) # is 65 for current dat   zaset (tiny shakjespere)
print("vocab size: ", vocab_size)




def compress_binary(embedding):
    if not compression:
        return embedding
    embedding = np.array(embedding).reshape(1, -1)
    flat = pca.transform(embedding).flatten().tolist()
    return flat


def decompress_binary(compressed):
    if not compression:
        return compressed
    compressed = np.array(compressed).reshape(1, -1)
    return pca.inverse_transform(compressed).flatten().tolist()

def numToBinary(num):
    binary = format(int(num), f"0{int(n_embedd)}b")
    return list(binary)

def floatToBinary(num, threshold=0):
    if num >= threshold:
        return 1
    else:
        return 0

def binaryToNum(binary):
    return int(str(binary), 2)

def listToBinary(numpy_array):
    return np.where(numpy_array > 0, 1, 0).astype(np.uint8)

def tensorToBinary(tensor):
    return np.where(tensor.cpu().detach().numpy() > 0, 1, 0).astype(np.uint8)

stoi = {ch: i for i, ch in enumerate(chars)} # string to index
itos = {i: ch for i, ch in enumerate(chars)} # index to string

def encode(s): return [stoi[c] for c in s if c in stoi]
def decode(l): return ''.join([itos[i] for i in l])

data = torch.tensor(encode(dataset), dtype=torch.long)
n = int(0.9 * len(data))
train_data, test_data = data[:n], data[n:]

def count_parameters(model):
    parameters = 0
    for param in model.parameters():
        if param.requires_grad:
            parameters += param.numel()
    return parameters

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

        """
            attention = (Query*Key/sqrt(key dim))*Value
        """
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        head_dim = q.size(-1)  # this is head_size
        w = q @ k.transpose(-2, -1) * (head_dim ** -0.5)# comes from the scaled dot-product attention formula 
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        v = self.value(x)
        return w @ v  # final output 

class weightlessNeuralNetwork():
    def __init__(self):
        #self.WNN = ao.Agent(Arch=ao.Arch(arch_i=[block_size], arch_z=[block_size], connector_function="rand_conn", connector_parameters=[int(block_size*0.6), block_size, int(block_size*0.6, block_size)]))
        self.WNN = ao.Agent(Arch=ao.Arch(arch_i=[int(vocab_size/wnn_binary_compression_level)], arch_z=[int(vocab_size/wnn_binary_compression_level)]))
        self.external_state_count = 0 # debug
        self.lookup_map = []

    def float_to_bin(self, arr, threshold=0):  # The most basic conversion function. if input is greater than threshold, it will be 1, else 0
        """Converts a float32 embedding to a binary embedding."""
        binary_embedding = np.where(arr > threshold, 1, 0)
        return binary_embedding

    def forward(self, input):
        B,T,C = input.shape  # here batch size is always 1
        input = input.cpu().detach().numpy() 
        outputs = np.zeros(input.shape, dtype=np.float32)
        for b in range(B):
            for t in range(T):
                inp = np.array(input[b, t])
                inp = compress_binary(self.float_to_bin(inp))
                out = self.WNN.next_state(inp)  # produce compressed or full output depending on agent
                out = decompress_binary(out)
                self.lookup_map.append(out)  # optional: or don't append at all
                outputs[b, t] = out
            self.WNN.reset_state()

        return torch.tensor(outputs, dtype=torch.float32, device=device)

    def train(self, x, y):

        Bx,Tx,Cx = x.shape
        x = x.reshape(Bx*Tx, Cx)  # shape (B*T, C)
        Bt, Tt, Ct = y.shape
        y = y.reshape(Bt* Tt, Ct) # shape (b*T,)

        input=[]
        label=[]
        for inp, lbl in zip(x,y):
            input.append(compress_binary(self.float_to_bin(inp.detach().cpu().numpy())))
            label.append(compress_binary(self.float_to_bin(lbl.detach().cpu().numpy())))         

        self.WNN.next_state_batch(input,label, unsequenced=False) 
        self.external_state_count += len(input) # debug
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
    def __init__(self, n_embedd, n_head, use_wnn=False, wnn_block=False):
        super().__init__()
        head_size = n_embedd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embedd)
        self.wnn_block = wnn_block # we only activate the wnn block in the last transformer block
        self.use_wnn = use_wnn # here as more of a placeholder

        self.WNN = weightlessNeuralNetwork()  # Using the weightless neural network init anyway to reduce error
        self.ln1 = nn.LayerNorm(n_embedd) # layer normalization
        self.ln2 = nn.LayerNorm(n_embedd)
        self.pre_wnn_x = None  #
       

    def forward(self, x):

        x = x + self.sa(self.ln1(x)) 
        self.pre_wnn_x = x + self.ffwd(self.ln2(x))
        if self.wnn_block:
            if self.use_wnn: # not using wnn for training only for inference 
                self.pre_wnn_x.requires_grad_()
                x = self.pre_wnn_x + (self.WNN.forward(self.pre_wnn_x))
            else:
                x = self.pre_wnn_x
        return x   

# Language model

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedd) # build up a token embedding table that maps each token to an embedding vector
        self.position_embedding_table = nn.Embedding(block_size, n_embedd) # positional encoding embedding table to give the model a sense of the position of each token in the sequence
        blocks = []
        for _ in range(n_layer-num_wnn_blocks):
            blocks.append(Block(n_embedd, n_head=n_head))  # the rest are normal transformer blocks
        for _ in range(num_wnn_blocks):
            blocks.append(Block(n_embedd, n_head=n_head, wnn_block=True))
        self.blocks = nn.Sequential(*blocks) # The star unpacks the list so it can be used for the nn.Sequential
        self.lm_head = nn.Linear(n_embedd, vocab_size) # final linear layer to project the output of the transformer blocks to the vocabulary size for next token prediction



    def forward(self, idx, targets=None):
        B, T = idx.shape # Idx is the input tensor but different from the input to the transformer blocks because 
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_emb
        x = self.blocks.forward(x)
        logits = self.lm_head(x) # logits are the raw scores for each token in the vocab at each pos in the sequence corresponding to the

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def trainWNN(self, label):
        """
            Here is the magic. WNNs are continously trainabkle so we can introduce this trainWNN function at any point even after training and apply new labels in realtime.
            This is a much stronger kind of continous learning than RLHF since we are changing a meaningful proption of the underlying weights of the model instead of a small subset.
        """
        # Note in the function below there is alot of handling for if on of the tensors are empty. this is because we intially trained the model on tiny shaspeare dataset which has a very small vocab size and so the WNN is not trained on all the tokens in the vocab.
                # we will now precompute torch.linalg.inv for the lm_head weights to speed up the WNN training
        W = self.lm_head.weight.to(device) # shape is vocab_size, n_embedd
        n_embedd = W.shape[1]
        A = W.T @ W # this is calculating the gram matrix
        A = A + 1e-4 * torch.eye(n_embedd, device=device) # adding a small value to the diagonal for numerical stability
        A_inv = torch.linalg.inv(A) # The goal here is to approximate the inverse of the bigram model to we can get the embedding that would produce the change in logits
        B = A_inv @ W.T
        #with torch.no_grad():  # no gradients in PyTorch
        for i,block in enumerate(self.blocks): # iterate through each wnn transformer block
            if not block.wnn_block:
                continue
            now = datetime.now()
            contexts = []
            true_next_tokens = []
            encoded_label = encode(label)

            for k in range(len(encoded_label) - 1):
                context = encoded_label[max(0, k - block_size + 1):k+1]
                
                # Pad the context
                padded_context = [0] * (block_size - len(context)) + context # TODO for me create a unique padding token
                contexts.append(padded_context)
                true_next_tokens.append(encoded_label[k+1])

            if not contexts:
                continue

            contexts_tensor = torch.tensor(contexts, dtype=torch.long, device=device)
            targets_tensor = torch.tensor(true_next_tokens, dtype=torch.long, device=device)

            # --- Forward and Backward Pass ---
            
            # We need to temporarily set the model to train mode for gradients
            self.train()
            
            x = self.token_embedding_table(contexts_tensor) + self.position_embedding_table(torch.arange(block_size, device=device))

            for block_idx in range(i):
                x = self.blocks[block_idx].forward(x)  # pass through all previous blocks but not the current one


            _ = block.forward(x)  # pass through the
            pre_wnn_x = block.pre_wnn_x  # get the pre WNN x from the current block

            pre_wnn_x.retain_grad()  # we need gradients w.r.t. this tensor

            x = pre_wnn_x

            for block_idx in range(i+1, len(self.blocks)):
                x = self.blocks[block_idx].forward(x)

            logits = self.lm_head(x)

            loss = F.cross_entropy(logits[:, -1, :], targets_tensor)

            self.zero_grad()
            loss.backward()

            wnn_target_resiudal = -pre_wnn_x.grad


            print("time for preparing data: ", datetime.now()-now)

            now1 = datetime.now()
            block.WNN.train(pre_wnn_x.detach(), wnn_target_resiudal)
            print("time for training WNN: ", datetime.now()-now1)

    def generate(self, idx, max_new_tokens):
        for block in self.blocks:
            block.WNN.lookup_map=[]#n between diff runs reset the lookup map
        for _ in range(max_new_tokens):
            print("generating token: ", _)
            print("current generated string: ", decode(idx[0].tolist()))
            idx_cond = idx[:, -block_size:] # we are making sure to only use the last block as context not the entire sequence
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]# get the last logit prediction for the next token
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1) # get the next token by sampling from the probabilities
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Training
model = BigramLanguageModel().to(device)
model.load_state_dict(torch.load("ModelStateSaves/WNNTransformerTinyShakespearDataset128Emed25kIters8Layers.pth", weights_only=False))

if compression:

    pca = PCA(n_components=int(vocab_size/wnn_binary_compression_level))  # reduce to 128 dims
    embs = model.token_embedding_table.weight.detach().cpu().numpy() 
    pca.fit(embs)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# for step in range(max_iters):
#     if step % eval_interval == 0:
#         losses = estimate_loss()
#         print(f"Step {step}: Train loss {losses['train']:.4f}, Test loss {losses['test']:.4f}")

#     xb, yb = get_batch("train")
    

#     logits, loss = model(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

# torch.save(model.state_dict(), "WNNTransformerTinyShakespearDataset128Emed25kIters8Layers.pth")




#Text generation

# max_new_tokens = 5
# context = torch.tensor([encode("The people")], dtype=torch.long, device=device)
# now = datetime.now()
# generated = model.generate(context, max_new_tokens)[0].tolist()
# print(decode(generated))

#print("time to generate ", max_new_tokens , " : ", datetime.now()-now, "s")

for block in model.blocks: # we want to use the wnn forward when generating text
    block.use_wnn=True

# retrain = load_dataset("Salesforce/wikitext", 'wikitext-103-raw-v1', split="train")["text"]
# retrain = "\n".join(retrain)
# print("len retrain: ", len(retrain))

#model.trainWNN(retrain[:1000])  # retrain the WNN on new data

model.trainWNN("""london is a city of rain and neon, old stone and new glass.
a river runs slow, a market hums and a bridge keeps its watch.
the tube thrums beneath and buses seam the day.
art, code, and craft meet in a street where light and shadow pass.
at night the city breathes, a long awake that never sleeps.""")
print("WNN story state after retraining: ", model.blocks[-1].WNN.WNN.state)
print("debug external state count: ", model.blocks[-1].WNN.external_state_count) # debug
max_new_tokens = 120
context = torch.tensor([encode("london is ")], dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens)[0].tolist()
print(decode(generated))

#print("Number of parameters in the model: ", count_parameters(model))