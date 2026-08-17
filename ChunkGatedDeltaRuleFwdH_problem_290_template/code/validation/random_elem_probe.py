import torch

torch.manual_seed(0)
g = torch.Generator().manual_seed(1007)
a = torch.randn((128, 61), dtype=torch.float32, generator=g).to(torch.bfloat16)
b = torch.randn((61, 128), dtype=torch.float32, generator=g).to(torch.bfloat16)
ref = (a @ b).float()

af = a.to(torch.float32); bf = b.to(torch.float32)
M, K = a.shape; N = b.shape[1]

# seq
acc = torch.zeros((M, N), dtype=torch.float32)
for k in range(K):
    acc += af[:, k:k+1] * bf[k:k+1, :]
seq = acc.to(torch.bfloat16).float()

# p4
accs = [torch.zeros((M, N), dtype=torch.float32) for _ in range(4)]
for k in range(K):
    accs[k % 4] += af[:, k:k+1] * bf[k:k+1, :]
p4 = ((accs[0] + accs[1]) + (accs[2] + accs[3])).to(torch.bfloat16).float()

print("seq mism:", (seq != ref).sum().item(), " p4 mism:", (p4 != ref).sum().item())

# examine seq mismatch elements
idx = (seq != ref).nonzero()
for i in idx[:3]:
    r, c = int(i[0]), int(i[1])
    products = af[r, :] * bf[:, c]
    # several reduction orders
    s = torch.zeros((), dtype=torch.float32)
    for k in range(K):
        s += products[k]
    accs2 = [torch.zeros((), dtype=torch.float32) for _ in range(4)]
    for k in range(K):
        accs2[k % 4] += products[k]
    p4v = ((accs2[0]+accs2[1])+(accs2[2]+accs2[3]))
    # pairwise tree
    tree = products.clone()
    n = K
    while n > 1:
        half = n // 2
        for j in range(half):
            tree[j] = tree[j] + tree[n - 1 - j]
        n = (n + 1) // 2
    # blocks of 4 sequential
    b4 = torch.zeros((), dtype=torch.float32)
    for kb in range(0, K, 4):
        t = products[kb:kb+4]
        tsum = t[0]
        for j in range(1, t.numel()):
            tsum = tsum + t[j]
        b4 = b4 + tsum
    def bf16v(x):
        return float(torch.tensor(float(x), dtype=torch.float32).to(torch.bfloat16).float().item())
    print(f"elem [{r},{c}] torch={float(ref[r,c])} ({bf16v(ref[r,c])}) seq={float(s)} ({bf16v(s)}) p4={float(p4v)} ({bf16v(p4v)}) tree={float(tree[0])} ({bf16v(tree[0])}) b4={float(b4)} ({bf16v(b4)})")
