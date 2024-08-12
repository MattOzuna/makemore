import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

words = open("names.txt").read().splitlines()

N = torch.zeros((27, 27), dtype=torch.int32)

chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

# ===================================================================================================#
# Counting based probabilty distribution instead of gradient based

# for w in words:
#     chs = ["."] + list(w) + ["."]
#     for ch1, ch2 in zip(chs, chs[1:]):
#         ix1 = stoi[ch1]
#         ix2 = stoi[ch2]
#         N[ix1, ix2] += 1


# ===================================================================================================#
# uncomment to get a visualliation of all bigrams(pairs of chars) from names dataset.
# plt.figure(figsize=(16,16))
# plt.imshow(N, cmap='Blues')
# for i in range(27):
#     for j in range(27):
#         chstr = itos[i] + itos[j]
#         plt.text(j,i, chstr, ha="center", va="bottom", color='gray')
#         plt.text(j,i, N[i, j].item(), ha="center", va="top", color='gray')
# plt.axis('off')
# plt.show()
# ===================================================================================================#

# P = (N+1).float()
# P /= P.sum(1, keepdim=True)
# g = torch.Generator().manual_seed(2147483647)

# for i in range(5):
#     out=[]
#     idx = 0
#     while True:
#         p = P[idx]
#         idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
#         out.append(itos[idx])
#         if idx == 0:
#             break
#     print(''.join(out))

# ===================================================================================================#
# Gradient Dissent based Model

# GOAL FOR A MODEL: maximize likelihood of the data w.r.t model params (statistical modeling)
# equivalent to maximizing the log likelihood (because log is monotonic)
# equivalent to minimizing the negative log likelihood
# equivalent to minimizing the average negative log likelihood

# log(a*b*c) = log(a) +log(b) + log(c)

xs, ys = [], []

for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement() # returns total number of elements in the input tensor

# dont forget to make a float so the model and more finely manipulate


g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

# Moved the following commented lines into the training loop
# xenc = F.one_hot(xs, num_classes=27).float()

# # @ is a matrix multiplacation operator in pytorch
# logits = xenc @ W  # predict log-counts

# # SoftMax Activation functions
# counts = logits.exp()  # get a fake count of the probs
# probs = counts / counts.sum(1, keepdim=True)  # probabilities for next char
# loss =  -probs[torch.arange(5), ys].log().mean()

# ===================================================================================================#

for k in range(100):
    #Forward Pass
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()  
    probs = counts / counts.sum(1, keepdim=True)  
    loss =  -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
    # print(loss.item())

    # Backward Pass
    W.grad = None #set the gradient to zero
    loss.backward()

    #update
    W.data += -50 * W.grad # silly learning rate for this simple example

# Sample fromt he NN
for i in range(5):
    out=[]
    idx = 0
    while True:
        xenc = F.one_hot(torch.tensor([idx]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdim=True)

        idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[idx])
        if idx == 0:
            break
    print(''.join(out))