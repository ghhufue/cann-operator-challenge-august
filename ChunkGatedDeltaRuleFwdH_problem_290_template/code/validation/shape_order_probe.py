import torch

def mm_orders(a, b):
    M, K = a.shape
    N = b.shape[1]
    af = a.to(torch.float32); bf = b.to(torch.float32)
    results = {}
    # sequential
    acc = torch.zeros((M, N), dtype=torch.float32)
    for k in range(K):
        acc += af[:, k:k+1] * bf[k:k+1, :]
    results["seq"] = acc.to(torch.bfloat16)
    # p4 interleaved, combine orders
    for order in ["p4_0123", "p4_pair", "p4_pair02", "p4_0321"]:
        accs = [torch.zeros((M, N), dtype=torch.float32) for _ in range(4)]
        for k in range(K):
            accs[k % 4] += af[:, k:k+1] * bf[k:k+1, :]
        if order == "p4_0123":
            out = accs[0] + accs[1] + accs[2] + accs[3]
        elif order == "p4_pair":
            out = (accs[0] + accs[1]) + (accs[2] + accs[3])
        elif order == "p4_pair02":
            out = (accs[0] + accs[2]) + (accs[1] + accs[3])
        else:
            out = accs[0] + accs[3] + accs[2] + accs[1]
        results[order] = out.to(torch.bfloat16)
    return results

shapes = {
    "MM1_partial (61,128)x(128,128)": ((61, 128), (128, 128)),
    "MM1_full   (64,128)x(128,128)": ((64, 128), (128, 128)),
    "MM2_partial (128,61)x(61,128)": ((128, 61), (61, 128)),
    "MM2_full   (128,64)x(64,128)": ((128, 64), (64, 128)),
    "MM2_K64_v64 (64,64)x(64,64)": ((64, 64), (64, 64)),
}

torch.manual_seed(0)
for sname, (sa, sb) in shapes.items():
    for seed in range(4):
        g = torch.Generator().manual_seed(seed * 1000 + 7)
        a = torch.randn(sa, dtype=torch.float32, generator=g).to(torch.bfloat16)
        b = torch.randn(sb, dtype=torch.float32, generator=g).to(torch.bfloat16)
        ref = (a @ b).float()
        cands = mm_orders(a, b)
        line = []
        for name, val in cands.items():
            mism = (val.float() != ref).sum().item()
            line.append(f"{name}={mism}/{ref.numel()}")
        print(f"{sname} seed={seed}: " + "  ".join(line))
