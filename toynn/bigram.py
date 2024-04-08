import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# HYPERPARAMETERS
block_size = 8
batch_size = 32
n_embd = 32
learning_rate = 1e-3
max_tokens = 300
device = "cuda" if torch.cuda.is_available() else "cpu"
max_iters = 10000
eval_iters = 300
eval_interval = 300


with open("../notebooks/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Getting the vocab size
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Tokenizing, encoding, decoding
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda i: "".join([itos[t] for t in i])

# Encoding the entire tiny shakespeare
data = torch.tensor(encode(text), dtype=torch.long)

# Let's now split up the data into train and validation sets
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split="train"):
    data = train_data if split == "train" else val_data
    # Gets for random start_ix
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize the word-frequency or weight matrix
        # self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        # Rewriting this to incorporate word embeddings C instead
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

    def forward(self, idx, targets=None):
        # idx = index of Xb, targets = Yb
        # This is equivalent to one-hot(xenc) @ W, which is a lookup table.
        logits = self.token_embedding_table(idx)  # B, T, C
        if targets is None:
            loss = None
        else:
            # We need to reshape this to (B, C, T) as thats what pytorch uses.
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is a (B, T) and you want to take this and extend to (B, T + max_new_tokens)
        for _ in range(max_new_tokens):
            # Get logits and loss is None because targets are None.
            logits, loss = self(idx)
            # Get the logits at the Tth time dimension to predict T+1.
            logits = logits[:, -1, :]
            # Get the probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # B, 1
            # Shifting to the T+1th token.
            idx = torch.cat((idx, idx_next), dim=1)  # B, T+1
        return idx


# Initializing model
model = BigramLanguageModel()
m = model.to(device)

# Setting an optimizer object
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step: {iter} | train loss: {losses['train']: .4f} | val loss: {losses['val']: .4f}"
        )
    # Get sample data
    xb, yb = get_batch("train")

    # Forward pass
    logits, loss = model(xb, yb)

    # Set grad to zero
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\n--\n".join([decode(i) for i in m.generate(context, max_tokens).tolist()]))
