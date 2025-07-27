
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F

block_size = 8
batch_size = 32
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
eval_iters = 200
n_embedd = 32

device = "cuda" if torch.cuda.is_available() else "cpu"

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)

# Save to file (optional)
with open("tinyshakespeare.txt", "w", encoding="utf-8") as f:
    f.write(response.text)


# Load into a variable
text = response.text

# Preview the first 500 characters
#print(text[:500])

chars = sorted(list(set(text)))
vocab_size = len(chars)

encoding = {}
decoding = {}

for i, char in enumerate(chars):
    encoding[char] = i

for i, char in enumerate(chars):
    decoding[i] = char

def encode(text):
    encoded = []
    for c in text:
        encoded.append(encoding[c])

    return encoded

def decode(text):
    decoded = []
    for c in text:
        decoded.append(decoding[c])

    return decoded

data = torch.tensor(encode(text), dtype = torch.long)
print(data.shape, data.dtype)
#print(data[:1000])


n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]



train_data[:block_size+1]

x = train_data[:block_size]
y=train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print("when input is ", context, " the target is ", target)

torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == "train" else test_data
    ix  = torch.randint(len(data) - block_size, (batch_size,)) # since batch size is 4 we generate 4 numbers between 0 and length of data - block size
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # of set of 1 of x
    x, y = x.to(device), y.to(device)
    return x,y

xb, yb = get_batch("train")
print("inputs: ")
print(xb.shape)
print(xb)
print("targets: ")
print(yb.shape)
print(yb)

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        print("when input is ", context.tolist(), " the target is ", target)

# Bigram langauge model

torch.manual_seed(1337)

@torch.no_grad() # makes it more efficent because we are not doing backprop
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "test"]:
        losses = torch.zeros(eval_iters)
        for k in range (eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,  n_embedd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embedd) # positional embedding
        self.sa_heads = MultiHeadAttention(4, n_embedd//4)
        self.ffwd= FeedForward(n_embedd)

        self.lm_head = nn.Linear(n_embedd, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape # B is batch size, T is block size

        token_emb= self.token_embedding_table(idx)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device)) # get the positional embedding for each position in the input sequence
        x = token_emb + pos_emb # add the token and positional embeddings
        x = self.sa_heads(x) # apply self attention head
        x = self.ffwd(x) # apply feed forward network
        logits = self.lm_head(x)
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # calc the cross entropy ( negative log) loss

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # cant take more than block size tokens as input so am clipping it off
            logits, loss = self(idx_cond) # get the predictions
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = 1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    
class FeedForward(nn.Module):
    def __init__(self, n_embedd):
        super().__init__()
        self.net = nn.Sequential(  # basic 
            nn.Linear(n_embedd,  n_embedd),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)
    
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embedd, head_size, bias=False)
        self.query = nn.Linear(n_embedd, head_size, bias=False)
        self.value = nn.Linear(n_embedd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size))) # creating trill parameter which is the lower triangular matrix

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        w = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T) # from the formula of attention: attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V
        w = w.masked_fill(self.tril[:T,:T] == 0, float("-inf")) # mask the upper triangular part ensuring future tokens are not able to communicate with past ones
        w = F.softmax(w, dim=-1) # normalize the weights via the softmax function: 
        v = self.value(x) # (B,T,C)
        out = w @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out
    
model = BigramLanguageModel().to(device)
logits, loss = model(xb, yb)
print(logits.shape)
print(loss)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

for steps in range(10000):
    if steps % eval_interval == 0:
        loss=estimate_loss()
        print(f"step {steps}: train loss {loss['train']:.4f}, test loss {loss['test']:.4f}")
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(idx = context, max_new_tokens=100)[0].tolist()))
