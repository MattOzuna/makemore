import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

words = open("names.txt", "r").read().splitlines()

# build vocab of char to and from integers
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

# ===================================================================================================#
# Tokenization of entire dataset with X being inputs and Y being correct predictions
# block_size = 3
# X, Y = [], []

# for w in words:
#     # print(w)
#     context = [0] * block_size
#     for ch in w + ".":
#         ix = stoi[ch]
#         X.append(context)
#         Y.append(ix)
#         # print("".join(itos[i] for i in context), "--->", itos[ix])
#         context = context[1:] + [ix]

# X = torch.tensor(X)
# Y = torch.tensor(Y)

# ===================================================================================================#
# Single training loop further iterated on down below in training loop

# C = torch.randn((27, 2))

# emb = C[X]

# W1 = torch.randn((6, 100))
# b1 = torch.randn(100)

# # torch.cat([emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]], 1)
# # torch.cat(torch.unbind(emb, 1), 1)
# # torch.view([32,6])
# # emb.view([32, 6]) == torch.cat(torch.unbind(emb, 1), 1)

# # look and the blogpost at 27min for more details on .view

# emb.view(emb.shape[0], 6) @ W1 + b1
# # can also do the following because of the way that pytoch interprets -1
# h = torch.tanh(emb.view(-1, 6) @ W1 + b1)

# W2 = torch.randn((100, 27))
# b2 = torch.randn(27)

# logits = h @ W2 + b2
# counts = logits.exp()
# prob = (counts / counts.sum(1, keepdim=True))

# loss = -prob[torch.arange(32), Y].log().mean()

# ===================================================================================================#
# training split, dev/validation split, test split
# 80%, 10%, 10%


# build the dataset
def build_dataset(words):
    block_size = 3
    X, Y = [], []
    for w in words:
        # print(w)
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print("".join(itos[i] for i in context), "--->", itos[ix])
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

# ===================================================================================================#
# Cleaned up version

# Create Weights and biases
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 10), generator=g)  # Embedding layer of the 27 characters
W1 = torch.randn((30, 200), generator=g)
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

# Used to find the correct learning rate
# lre = torch.linspace(-3, 0, 1000)
# lrs = 10**lre

# lri = []
stepi = []
lossi = []
loss_tracking = 0


for i in range(50000):
    # mini-batch construct
    ix = torch.randint(0, Xtr.shape[0], (32,))

    # forward pass
    emb = C[Xtr[ix]]  # pulling mini-batch from X
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(
        logits, Ytr[ix]
    )  # more effecient that our manual calculations from a space and time perspective

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    loss_tracking = loss.item()

    # update
    # lr = lrs[i] # Used when finding the learning rate
    lr = 0.1
    for p in parameters:
        p.data += -lr * p.grad

    # track learning rate stats
    # lri.append(lre[i])
    stepi.append(i)
    lossi.append(loss.log10().item())

print(loss_tracking)

# Shows how the different learning rates perform
# plt.plot(lri, lossi)
# plt.show()

# shows how the loss performs accros each step
# plt.plot(stepi, lossi)
# plt.show()

emb = C[Xdev]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print("Dev loss:", loss.item())

# ===================================================================================================#
# Sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)
block_size = 3

for _ in range(20):
    out = []
    context = [0] * block_size # initialized with all ...
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print("".join(itos[i] for i in out))
