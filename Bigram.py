import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
block_size = 8
batch_size = 32
max_iters = 5000
eval_interval = 300
learning_rate = 3e-4
eval_iters = 200
n_embedd = 32

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)

# Load dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
text = response.text

with open("tinyshakespeare.txt", "w", encoding="utf-8") as f:
    f.write(text)

# Tokenization
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)} # string to index
itos = {i: ch for i, ch in enumerate(chars)} # index to string

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
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
@torch.no_grad()
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

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        w = q @ k.transpose(-2, -1) * C**-0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1)
        v = self.value(x)
        return w @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embedd, n_embedd)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, n_embedd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embedd, 4 * n_embedd),
            nn.ReLU(),
            nn.Linear(4 * n_embedd, n_embedd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embedd, n_head):
        super().__init__()
        head_size = n_embedd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embedd)
        self.ln1 = nn.LayerNorm(n_embedd) # layer normalization
        self.ln2 = nn.LayerNorm(n_embedd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Language model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedd)
        self.position_embedding_table = nn.Embedding(block_size, n_embedd)
        self.blocks = nn.Sequential(
            Block(n_embedd, 4),
            Block(n_embedd, 4),
            Block(n_embedd, 4),
            nn.LayerNorm(n_embedd)  # final layer normalization
        )
        self.lm_head = nn.Linear(n_embedd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
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

# Text generation
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=100)[0].tolist()
print(decode(generated))
