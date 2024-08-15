import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random


# Improvements from video2
# 1. initialization of hidden layer weights and biases set all to 0,
# so all the network start learning earlier in the training cycle
# 2. Beware of the extreme values in you hidden layer,
# because after activation the neuron could become unable to learn
#  if the output is in the flat region of the actiavtion function
# 3. Batch normalization is used to control the activation stats.
# It creates a gausian disttribution around 0 with a 1 standard deviation
# at initialization, which solves the issues from the 1st point.
# Scaling allows the network to adjust as it sees fit once training has begun.
# This also regularizes the NN, because all the example in a batch are couple mathematically.
# This introduces some 'noise' in the outputs of the hidden layer,
# which helps battle against overfitting.

# ===================================================================================================#

words = open("names.txt", "r").read().splitlines()

# build vocab of char to and from integers
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(itos)

# ===================================================================================================#
# build the dataset

block_size = 3


def build_dataset(words):
    X, Y = [], []
    for w in words:
        # print(w)
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # crop and append

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])  # 80%
Xdev, Ydev = build_dataset(words[n1:n2])  # 10%
Xte, Yte = build_dataset(words[n2:])  # 10%

# ===================================================================================================#
# Initialization

n_embed = 10  # the dimensionality of the character embedding vectors
n_hidden = 200  # the number of nuerons in the hidden layer of MLP

# initialization scales
W1i = (5 / 3) / ((n_embed * block_size) ** 0.5)

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embed), generator=g)
W1 = torch.randn((n_embed * block_size, n_hidden), generator=g) * W1i
# hidden layer bias becomes uneccesary with batch normalization
# b1 = torch.randn(n_hidden, generator=g) * 0.01
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
b2 = torch.randn(vocab_size, generator=g) * 0

bngain = torch.ones((1, n_hidden))  # batch normilzation gain
bnbias = torch.zeros((1, n_hidden))  # batch normalization bias

# calculate mean and std over the entirety of training for use in eval and sampling
bnmean_running = torch.zeroes((1, n_hidden))
bnstd_running = torch.zeroes((1, n_hidden))

parameters = [C, W1, W2, b2, bngain, bnbias]
print("Number of parameters:", sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True

# ===================================================================================================#
# Optimization

max_steps = 200000
batch_size = 32
lossi = []
epsilon = 0.00001  # used to prevent division by zero in the batch normilzation step

for i in range(max_steps):
    # mini-batch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]  # batch X,Y

    # forward pass
    emb = C[Xb]  # embed the chars into vectors
    embcat = emb.view(emb.shape[0], -1)  # Concatenate of the context embed vectors (Flatten)
    hpreact = embcat @ W1  # hidden layer pre activation

    # batchNorm layer
    bnmeani = hpreact.mean(0, keepdim=True)  # batch mean for the i-th step
    bnstdi = hpreact.std(0, keepdim=True)  # batch std for the i-th step
    # Normalization and scaling of the batch
    hpreact = (bngain * bnmeani) / bnstdi + bnbias + epsilon
    # The running mean and std will be mostly what they used to be,
    # but will receive small changes after each mini-batch
    # our momentum is set to 0.001 because the batch size is relatively small
    # if the batch size were bigger a larger momentum would be acceptable
    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi

    # Non-linearity
    h = torch.tanh(hpreact)  # activation function of hidden layer
    logits = h @ W2 + b2  # output layer
    loss = F.cross_entropy(logits, Yb)  # loss function

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < 100000 else 0.01  # step with learning rate decay
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    if i % 10000 == 0:
        print(f"{i:7d}/{max_steps:7d}: {loss.item():4f}")
    lossi.append(loss.log10().item())

    # Used to show the initialization of weight and biases to avoid extreme values in hidden layer
    # plt.figure(figsize=(20,10))
    # # plt.imshow(h.abs() > .99, cmap='gray', interpolation='nearest')
    # plt.hist(h.view(-1).tolist(), 50)
    # plt.show()

# ===================================================================================================#
# calibrate the batch norm at the end of traing for use in evaluation and sampling
# Done in traingin loop instead

# with torch.no_grad():
#     # pass the training set through
#     emb = C[Xtr]
#     embcat = emb.view(emb.shape[0], -1)
#     hpreact = embcat @ W1 + b1
#     # measure the mean/std over the entire traingin set
#     bnmean = hpreact.mean(0, keepdim=True)
#     bnstd = hpreact.std(0, keepdim=True)


# ===================================================================================================#
# Evaluation
@torch.no_grad()  # this disables gradient tracking, increase efficiency
def split_loss(split):
    # gives different datasets depending on what you pass in
    x, y = {"train": (Xtr, Ytr), "val": (Xdev, Ydev), "test": (Xte, Yte)}[split]
    emb = C[x]
    embcat = emb.view(emb.shape[0], -1)  # concat into (N, block_size * n_emb)
    hpreact = embcat @ W1
    hpreact = (
        bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
    )  # Use batch mean and std
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2  # (N, vocab_size)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())


print("====Evaluation====")
split_loss("train")
split_loss("val")


# ===================================================================================================#
# Sample from the model

g = torch.Generator().manual_seed(2147483647 + 10)

print("====Sample====")
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        # sample from the distribution
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        # shift the context
        context = context[1:] + [ix]
        out.append(ix)
        # if we sample the special '.' token break
        if ix == 0:
            break
    print("".join(itos[i] for i in out))
