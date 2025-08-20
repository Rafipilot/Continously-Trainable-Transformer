import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import ao_core as ao
import numpy as np
from datetime import datetime
from datasets import load_dataset
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
batch_size = 128 # how many in parralel training examples to run
max_iters = 25000 # how many iterations to train for
eval_interval = 300
learning_rate = 3e-4
eval_iters = 200
n_embedd = 128 # embedding dimension for each token
n_layer = 8 # is the number of transformer blocks
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

dataset = response.text

# dataset = load_dataset("Salesforce/wikitext", 'wikitext-103-raw-v1', split="train")["text"]
# dataset = "\n".join(dataset)

min_freq = 50
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

def numToBinary(num):
    binary = format(int(num), f"0{int(n_embedd)}b")
    return list(binary)

def binaryToNum(binary):
    return int(str(binary), 2)

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

# Batch generation
def get_batch(split):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

def get_batch_for_real_time_wnn_training(text):
    if len(text) < block_size:
        print("ERROR length of labl: ", len(text), " is smaller than len block size: ", block_size, " Please increase amount of label!")
        quit()  # Fatal error so quit 
    ix = torch.randint(len(text) - block_size, (batch_size,))
    x = torch.stack([text[i:i + block_size] for i in ix])
    y = torch.stack([text[i + 1:i + block_size + 1] for i in ix])
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
        head_dim = q.size(-1)  # this is head_size
        w = q @ k.transpose(-2, -1) * (head_dim ** -0.5)# comes from the scaled dot-product attention formula 
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        v = self.value(x)
        return w @ v

class weightlessNeuralNetwork():
    def __init__(self):
        #self.WNN = ao.Agent(Arch=ao.Arch(arch_i=[block_size], arch_z=[block_size], connector_function="rand_conn", connector_parameters=[int(block_size*0.6), block_size, int(block_size*0.6, block_size)]))
        self.WNN = ao.Agent(Arch=ao.Arch(arch_i=[n_embedd], arch_z=[n_embedd]))
        self.lookup_map = []

    def float_to_bin(self, arr, threshold=0):  # The most basic conversion function. if input is greater than threshold, it will be 1, else 0
        """Converts a float32 embedding to a binary embedding."""
        binary_embedding = np.where(arr > threshold, 1, 0)
        return binary_embedding

    def forward(self, input):
        B,T,C = input.shape
        input = input.cpu().detach().numpy() 
        outputs = np.zeros(input.shape, dtype=np.float32)
        if len(self.lookup_map) >= int(block_size):
            self.lookup_map.pop(0) # roll the context window
        now = datetime.now()
        num_next_states = 0
        for b in range(B):
            for t in range(T):

                if t >= len(self.lookup_map):

                    inp = np.array(input[b,t])
                    inp= self.float_to_bin(inp)
                    out = self.WNN.next_state(inp)
                    self.lookup_map.append(out)
                    num_next_states+=1
                else:
                    out = self.lookup_map[t]

                outputs[b,t] = out
            self.WNN.reset_state() # reset the state of the agent in between batches
        print("time for wnn: ", datetime.now()-now)
        outputs = torch.from_numpy(outputs).to(device)
        return outputs
        

    def train(self, x, y):
        B,T,C = x.shape
        x = x.reshape(-1, C)  # shape (B*T, C)
        y = y.reshape(-1) # shape (b*T,)

        input=[]
        label=[]
        for inp, lbl in zip(x,y):
            input.append(self.float_to_bin(inp.detach().cpu().numpy()))
            label.append(numToBinary(lbl.cpu().numpy()))         

        self.WNN.next_state_batch(input,label) # In each batch sequence is vital
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
        if wnn_block:
            self.WNN = weightlessNeuralNetwork()  # Using the weightless neural network
        self.ln1 = nn.LayerNorm(n_embedd) # layer normalization
        self.ln2 = nn.LayerNorm(n_embedd)
       

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) 
        x = x + self.ffwd(self.ln2(x))
        if self.wnn_block:
            if self.use_wnn: # not using wnn for training only for inference 
                x= x + (self.WNN.forward(x)) # potentially scale this value via a hyperparameter
        return x

# Language model

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedd) # build up a token embedding table that maps each token to an embedding vector
        self.position_embedding_table = nn.Embedding(block_size, n_embedd) # positional encoding embedding table to give the model a sense of the position of each token in the sequence
        blocks = []
        for i in range(n_layer):
            if i != n_layer-1:
                blocks.append(Block(n_embedd, n_head=n_head))
            else:
                blocks.append(Block(n_embedd, n_head=n_head, wnn_block=True))  # the last block is a wnn block only
        self.blocks = nn.Sequential(*blocks) # The star unpacks the list so it can be used for the nn.Sequential
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
    
    def trainWNN(self, label):
        """
            Here is the magic. WNNs are continously trainabkle so we can introduce this trainWNN function at any point even after training and apply new labels in realtime.
            This is a much stronger kind of continous learning than RLHF since we are changing a meaningful proption of the underlying weights of the model instead of a small subset.
        """
        with torch.no_grad(): 
            # xs=torch.tensor([]).to(device)
            # ys= torch.tensor([]).to(device)
            
            # for k, token in enumerate(label): # when training on new data there is chance that the token is not in the training vocab so this is error handling
            #     if token not in stoi:
            #         print(f"Token '{token}' not in vocabulary, skipping.")
            #         continue
            #     if k==0: # if we are at the first token we cant really teach it anything since no previous context
            #         continue 
            #     previous_tokens = label[:k]
            #     current_label = token
            #     current_label_encoded = torch.tensor(encode(current_label)).to(device)
            #     previous_tokens_encoded=torch.tensor([]).to(device)
            #     for previous_token in previous_tokens:
            #         previous_tokens_encoded = torch.cat((previous_tokens_encoded, torch.tensor(encode(previous_token)).to(device))).to(device)
            #     position_locations = torch.arange(0, len(current_label), device=device, dtype=torch.long) # position locations for the current label
            #     emb = self.token_embedding_table(current_label_encoded) + self.position_embedding_table(position_locations.to(device))
            #     emb = emb.unsqueeze(0).to(device) # adding the batch dimension ( of one ) so it doesnt break
            #     for i, block in enumerate(self.blocks):
            #         if i == len(self.blocks)-1:
            #             x= emb + block.sa.forward(emb)
            #             x = x + block.ffwd.forward(x)
            #             xs = torch.cat((xs, x))
            #             ys = torch.cat((ys, previous_tokens_encoded))
            xb, yb = get_batch_for_real_time_wnn_training(torch.tensor(encode(label), dtype=torch.long)) # get a batch of data for training the WNN
            emb = model.token_embedding_table(xb) + model.position_embedding_table(torch.arange(xb.size(1), device=device))
            temp_x = emb
            print("shape temp x: ", temp_x.shape)

            for i, block in enumerate(self.blocks):
                if i == len(self.blocks)-1:
                    temp_x = temp_x + block.sa(block.ln1(temp_x))
                    temp_x = temp_x + block.ffwd(block.ln2(temp_x))
                            # WNN have static lookup table: training them multiple times on same data doesnt change outcome
                            # This means I will train them only once seperatly outside of the main training loop
                    print("start train")
                    now = datetime.now()
                    block.WNN.train(temp_x, yb)
                    print("finished train")
                    print("time for wnn train: ", datetime.now()-now)


    def generate(self, idx, max_new_tokens):
        self.blocks[-1].WNN.lookup_map = [] # in between diff runs reset the lookup map
        for _ in range(max_new_tokens):
            print("generating token: ", _)
            idx_cond = idx[:, -block_size:] # we are making sure to only use the last block as context not the entire sequence
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]# get the last logit prediction for the next token
            probs = F.softmax(logits, dim=1)
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]   # expect (B, vocab_size)
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1) # get the next token by sampling from the probabilities
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Training
model = BigramLanguageModel().to(device)
model.load_state_dict(torch.load("ModelStateSaves/WNNTransformerTinyShakespearDataset128Emed25kIters8Layers.pth", weights_only=False))
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

# Train WNN once at the end

xb, yb = get_batch("train")

# addingthe embedding 
emb = model.token_embedding_table(xb) + model.position_embedding_table(torch.arange(xb.size(1), device=device))

# Train WNN on block outputs
with torch.no_grad():  # no gradients in PyTorch
    temp_x = emb
    for i, block in enumerate(model.blocks):
        if i == n_layer-1: # Since only WNN in the last block only need to train the wnn in that layer
            temp_x = temp_x + block.sa(block.ln1(temp_x))
            temp_x = temp_x + block.ffwd(block.ln2(temp_x))
                    # WNN have static lookup table: training them multiple times on same data doesnt change outcome
                    # This means I will train them only once seperatly outside of the main training loop
            print("start train")
            now = datetime.now()
            block.WNN.train(temp_x, yb)
            print("finished train")
            print("time for wnn train: ", datetime.now()-now)
            print("WNN story state: ", block.WNN.WNN.state)
            #print(block.WNN.WNN.activations_global_C)

# Text generation

for block in model.blocks: # we want to use the wnn forward when generating text
    block.use_wnn=True

max_new_tokens = 0
context = torch.tensor([encode("The ")], dtype=torch.long, device=device)
now = datetime.now()
generated = model.generate(context, max_new_tokens)[0].tolist()
print(decode(generated))

print("time to generate ", max_new_tokens , " : ", datetime.now()-now, "s")

#model.trainWNN("The capital of France is Paris.")   # active realtime training!!

model.trainWNN("""Paris is late night light on the river and sound in the street.  
A friend texts you to meet at the bridge and you go at once.  
The tower shines and you see phones rise to catch the glow.  
It feels like a stage where every hour is a show.  

The cafe chairs face the street so all may watch the world pass.  
Some talk fast and laugh, some sit in calm silence with wine.  
You feel both guest and part of the crowd at the same time.  
The city holds you like a story you want to keep reading.  
               
The metro hums and the walls are marked with names.  
A song plays on a small speaker and no one minds.  
You walk up to the street and the air is full of scent.  
Fresh bread, sharp smoke, rain on stone, all at once.  

Paris feels alive in both noise and quiet.  
The day is rush, the night is long talk.  
The river moves slow and you pause to watch it.  
Even the pause feels like a scene worth saving.  

The city is never still.  
Tourists walk with maps, locals move with speed.  
Shops glow with glass, street art blooms on the wall.  
Paris mixes new and old with ease.  

It is both photo and dream.  
You feel the pull of style, the charm of chance.  
Every step is a choice, yet none feels wrong.  
Paris is both plan and flow.  

Work from home is a window and a screen and a small plant on the desk.
An app dings and you smile, then sigh, then type a short reply.
AI helps sort the feed, yet the heart still seeks a human note.

Late night code and cold pizza by the lamp.
The build fails, you fix one line, then push and wait for green.
Debug is both a test and a small victory.

City bus, small talk, earbuds in, a playlist on low.
A perfect song finds you in a grey hour and makes the day move.
We all keep a few songs that feel like friends.

Shop online, add to cart, wait for the box to come.
Unpack, a nice note, a small brand card, a clean wrap.
A purchase is a tiny story that lands on your door.

Climate talk is loud, yet the day is hot and the storm is near.
We plan and we act, the feed shows hope and fear.
Small changes count like coins in a long jar.

Dating is a mix of chat and meet, of photo and real face.
A message can bloom or fade, a short laugh can save the night.
We learn to trust a walk more than a long thread.

Memes fly fast, we share, we laugh, we forget.
A joke can travel miles in a beat and then it is old.
Still, the laugh is true in the quick moment.

Sleep is short, coffee is long, the code must ship.
We trade one hour for one bug fix and keep the pulse.
Rest waits like a soft light at the end of the day.

Travel plans map to a spot on a small screen, then a ticket is bought.
A new city means new steps, a new corner to keep.
Packing is small ritual, a last look at the mirror.

Street food wins: a hot bun, a small spice, a quick smile.
A stall and a narrow table make a world in a street.
Flavors tell a story faster than a long book.

Night is soft with neon and low talk and a slow walk home.
Some nights feel like a film, some nights feel like a note.
We carry both and call them memory.

Friends call and we say yes or no, we meet at a cafe or by the pier.
A plan is a small promise that turns into a laugh or a quiet hour.
We keep the calls even when the feed is loud.

Startups chase a dream and a fund, they hope and they try.
A pitch is a short script, a deck is a plain poem.
The team moves fast and holds a strange calm.

Design is light and calm, with a grid and a clean line.
A good page makes the mind rest and the click come easy.
Minimal is not empty, it is a kind of care.

Privacy is a word we trade for ease, we click and move on.
We set a few walls, then open a few doors, we learn the cost.
A choice is daily, small, and real.

An old song can fix a day, a line can save a mood.
We replay a verse and the past bends into the now.
Music is a map we keep on a short loop.

Kids play with small plastic and big imaginations.
They build a castle and make a rule, then break it with a laugh.
Childhood is loud with small wonders.

A quiet weekend is a rare prize: slow coffee, a long walk, small sun.
We read, we stare, we let the device sleep for one long hour.
It feels like a small repair to the day.

               lol the feed keeps scrolling but you only skim the best part.  
brb making coffee, then back to the meme parade.  
omg that track hits like a friend at midnight.  
idk the plan but the vibe is right.  

push the commit, pray the build, sip the cold brew.  
we chase green tests and small wins like coins.  
dm for meetup, we pick a street and see.  
ghosted? shrug, new chat, new laugh.  

late ride, neon rain, shoes wet, heart light.  
a small light in the window is enough now.  
rent is high, hope is louder, friends stay close.  
weekend hack and a long nap as a reward.  

climate posts flood the feed and you plant a tree.  
an app reminds you to breathe, you set it and forget it.  
privacy is a choice we half make each day.  
we trade a secret for a new ease and move on.  

gaming till four, a funny clip at dawn, then bed.  
we queue the match, we win the small fight.  
the chat reacts with fire and a few heart marks.  
victory is loud for a minute, then we call it.  

stream the set, skip the ad, stay for the drop.  
a song loops and you mark the mood as yours.  
playlists are the map of a week in a row.  
a riff can heal the dull work hour.  

shop cart, one click, box on the door, small joy.  
unbox slow, read the note, keep the card.  
a brand can be kind and you keep the name.  
it is retail, but it is also a small love.  

late night text, a short joke, a long reply.  
we send a snap and wait for the blue dot.  
meet at the cafe near the light, say yes or no.  
a walk can fix a date more than chat can.  

startup energy is eggs and fast code and hope.  
pitch decks are poems with charts and calm fonts.  
we call it hustle and call it craft at once.  
a good team holds and moves like a tide.  

the feed gives trends, we pick one and add color.  
memes age fast, we archive and laugh later.  
a viral clip is the town square for a day.  
we pass jokes like small flags around the net.  

city nights are neon, trains, a small slice of home.  
the corner shop knows your name and your late face.  
street food smells fix hunger and make new friends.  
a single bite can be a story in a minute.  

work from home is slippers and long calls and calm.  
zoom smiles are real in their own small way.  
we share the screen, then share a quiet laugh.  
home is desk and plant and the small sun on wood.  

a good design is empty and warm and easy to read.  
buttons that move like thought, pages that do not shout.  
we favor calm, we prize the clean line and pause.  
less is a choice that asks for care, not fear.  

we post a photo, wait for small praise, feel seen.  
likes are coins, replies are letters, both matter.  
a comment can make a day and a block can heal.  
we learn to give both and to take what helps.  

kids learn fast with small screens and big dreams.  
they make new games with old toys and a loud laugh.  
a plastic car and a song can make a whole world.  
childhood is messy but bright and loud.  

rent day stress, a small plan, a lunch with a friend.  
we split the bill and split the worry and feel better.  
money talk is real talk with a small laugh at the end.  
we promise to save and then spend on a small trip.  

night walks with a friend and a slow talk on the pier.  
we map the city with steps that do not rush.  
some nights are film, some nights a soft note.  
we keep both and call them memory.  

fitness is a small streak and a stubborn mind.  
we run two miles, then one more, then stop and smile.  
progress is the small mark on the week that grows.  
a band on the wrist is a kind of quiet praise.  

books on the shelf and a slow read on the train.  
a line can cut clean and make the day better.  
we keep a few pages for the night and sleep with them.  
reading is a small refuge from the loud feed.  

coffee shops are offices for the soft creative crew.  
laptops open, notes spread, a plan forms in a line.  
we find a socket, claim a chair, and start the day.  
a warm cup and a steady pace make the work nice.  

we learn from failures and add them to the shelf.  
a mistake is a small map to the next good try.  
celebrate the fix, note the bug, then move on.  
process is a friend if you teach it to be.  

street art blooms on a blank wall and brightens the lane.  
someone paints hope with a small spray and a wide smile.  
the city keeps those marks and the short brave words.  
public color is a shout that calms the day.  

we text our folks, we hear a voice, we laugh and cry.  
family is the old compass that still points home.  
we share a quick clip and get a loud reaction.  
love is small acts and a saved call history.  

a quiet morning is a rare and held prize.  
slow coffee, sun on the shelf, a soft plan for the day.  
turn off the feed for one long hour and breathe.  
this is a tiny repair we owe our heads.  

we learn slang and lose it and learn it again.  
language is a live thing that moves with the group.  
say a new word, watch it spread, then call it old.  
we all write the slang that will be next years dust.  

packing is ritual, a soft fold, a last check in the mirror.  
tickets as small proof of a long choice, a plan begins.  
travel is steps that make new maps in the heart.  
we bring home small souvenirs that keep the trip close.  

a tiny garden on the balcony is green proof of care.  
pots and soil, a small hand to water every few days.  
plants hold light and teach quiet, they do not rush.  
a leaf can make the room a new bright place.  

we set alarms, we snooze, we start again and try.  
habits are small acts that turn into a new self.  
one day at a time is a soft plan that works.  
we keep a small list and cross it like a win.  

threads of code link like sentences in a long play.  
a clean function is a small stanza that sings.  
we refactor like a poet trims a line for the better.  
good code is calm in shape and kind to the mind.  

we watch old films and learn a soft way to move.  
a scene can teach the way to stand at a corner.  
heroes are small people with a loud choice at once.  
we borrow moves from the reels in the night.  

an apology is short and true and it heals fast.  
say it clean, mean it, then add a small act to show.  
repair is a chain of steps not a single line.  
we all can mend, we just must start small.  

sleep early one night and the whole week shifts.  
rest is not weak, it is a quiet power we earn.  
dreams come soft and give maps to the next day.  
sleep is the small reset that keeps us kind.  

                 """)


print("WNN story state after retraining: ", model.blocks[-1].WNN.WNN.state)
max_new_tokens = 20
context = torch.tensor([encode("The ")], dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens)[0].tolist()
print(decode(generated))

print("Number of parameters in the model: ", count_parameters(model))