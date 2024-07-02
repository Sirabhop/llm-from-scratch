# Databricks notebook source
import torch

from torch import nn
from torch.nn import functional as F

block_size:int = 8
step_size:int = 1
batch_size:int = 4

max_iter:int = 10000

eval_iters:int = 2000
dropout:float = 0.2

# COMMAND ----------

with open('./data/wizard_of_oz.txt', 'r') as file:
    text = file.read()

vocab = sorted(list(set(text)))
str_to_int = {j:i for i, j in enumerate(vocab)}
int_to_str = {str_to_int[i]:i for i in vocab}

encoder = lambda x: [str_to_int[v] for v in x]
decoder = lambda x: [int_to_str[v] for v in x]

# COMMAND ----------

n = int(0.8 * len(text))

text = torch.tensor(encoder(text), dtype=torch.long)

train = text[:n]
test = text[n:]

# COMMAND ----------

def get_batch(split):
    data = train if split == 'train' else test
    ix = torch.randint(len(data) -block_size, (batch_size, ))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+step_size:i+block_size+step_size] for i in ix])

    return x, y

# COMMAND ----------

x, y = get_batch('train')

# COMMAND ----------

@torch.no_grad()
def estimate_loss(model, eval_iters):
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out

# COMMAND ----------

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, index, targets=None):
        logits = self.token_emb_table(index)

        if targets != None:
            B, T, C = logits.shape
            # T = time = sequence of integer [1, 0, 0, 0] where 1 is known and 0 is unknown
            # C = channel = vocab size
            # B = batch
            n = B*T
            logits = logits.view(n, C) # C is important, B & T are insig so merge them together
            targets = targets.view(n) 

            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
    
        return logits, loss
    
    def generate(self, index, max_new_tokens):

        for _ in range(max_new_tokens):

            logits, loss = self.forward(index)
            logits = logits[:, -1, :] # (B, C)

            probs = F.softmax(logits, dim=-1) # (B, C) -> Focus on last dimension

            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            index = torch.cat((index, index_next), dim=1) # (B, T+1)

        return index

# COMMAND ----------

model = BigramLanguageModel(vocab_size=len(vocab))

def get_model_output(num_text_to_generate):
    context = torch.zeros((1,1), dtype=torch.long)
    generated_char = model.generate(context, max_new_tokens=num_text_to_generate)
    decoded_char = decoder(generated_char[0].tolist())

    print(''.join(decoded_char))

# COMMAND ----------

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for _iter in range(max_iter):
    if _iter % eval_iters == 0:
        print(f'step: {_iter} with loss {estimate_loss(model, eval_iters)}')

    xb, yb = get_batch('train')

    logits, loss = model.forward(xb, yb)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()
    optimizer.step()

# COMMAND ----------

get_model_output(100)

# COMMAND ----------

block_size:int = 8
step_size:int = 1
batch_size:int = 4

max_iter:int = 10000

eval_iters:int = 2000
dropout:float = 0.2

learning_rate = 3e-3
n_embd = 384
n_layer = 4
n_head = 4

# COMMAND ----------

device='cuda'

# COMMAND ----------

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head

        self.sa = MultiheadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x+y)
        y = self.ffwd(x)
        x = self.ln(x + y)
        return x    

# COMMAND ----------

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

# COMMAND ----------

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.values = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.trill[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v

        return out

# COMMAND ----------

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, F) -> (B, T, [h1, h1, h1, h2, h2, h2, h3, h3, h3])
        out = self.dropout(self.proj(out))

        return out

# COMMAND ----------

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed) #n_embed new param -> use in token embedding
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = b_head) for _ i range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd) # Help model converge better
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)
    
    def _init_weights(self, module):

        if isintance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        

    def forward(self, index, targets=None):

        logits = self.token_embedding_table(index)

        tok_emd = self.token_embedding_table(index)
        pos_emd = self.position_embedding_table(torch.arange(T, device=device))

        x = tok_emd + pos_emd
        x = self.blocks(x) #(B,T,C)
        x = self.ln_f(x) #(B,T,C)
        logits = self.lm_head(x) #(B,T,Vocab siz)

        if targets is None:
            loss = None:
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    
    
    
