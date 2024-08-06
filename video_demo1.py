import torch
import matplotlib.pyplot as plt

words = open("names.txt").read().splitlines()

N = torch.zeros((27, 27), dtype=torch.int32)

chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0

for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

itos = {i:s for s,i in stoi.items()}

#===================================================================================================#
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
#===================================================================================================#

P = N.float()
P /= P.sum(1, keepdim=True)
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
    out=[]
    idx = 0
    while True:
        p = P[idx]
        idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[idx])
        if idx == 0:
            break
    print(''.join(out))

#===================================================================================================#
#GOAL: maximize likelihood of the data w.r.t model params (statistical modeling)
#equivalent to maximizing the log likelihood (because log is monotonic)
#equivalent to minimizing the negative log likelihood
#equivalent to minimizing the average negative log likelihood

# log(a*b*c) = log(a) +log(b) + log(c)
#===================================================================================================#


