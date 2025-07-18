
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F

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
    encoding[i] = char

def encode(text):
    encoded = []
    for c in text:
        encoded.append(encoding[c])

    return encoded

def decode(text):
    decoded = []
    for c in text:
        decoded.append(encoding[c])

    return decoded

print(encode("hii there"))
print(decode(encode("hii there")))

data = torch.tensor(encode(text), dtype = torch.long)
print(data.shape, data.dtype)
#print(data[:1000])


n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]


block_size = 8
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

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        logits = self.token_embedding_table(idx)
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
            logits, loss = self(idx) # get the predictions
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = 1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim = 1)

        return idx
    
model = BigramLanguageModel(vocab_size)
logits, loss = model(xb, yb)
print(logits.shape)
print(loss)

print(decode(model.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
batch_size = 32
for steps in range(10000):
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())

print(decode(model.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))


