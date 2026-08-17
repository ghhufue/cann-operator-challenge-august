import torch
import struct

def to_bf16(x):
    # round fp32 to bf16 exactly like torch cast
    return x.to(torch.bfloat16).float()

def f32(v):
    return struct.unpack('f', struct.pack('f', v))[0]

def acc_p4_orders(a, b, k):
    """a: (M,K) bf16, b: (K,N) bf16. Returns dict of candidate outputs fp32."""
    M, K = a.shape
    N = b.shape[1]
    af = a.float().numpy()
    bf = b.float().numpy()
    # sequential
    seq = torch.zeros((M, N), dtype=torch.float32)
    # p4 with 4 accumulators, different combine orders
    accs = [torch.zeros((M, N), dtype=torch.float32) for _ in range(4)]
    for kk in range(K):
        accs[kk % 4] += af[:, kk:kk+1] * bf[kk:kk+1, :]
    p4_0123 = accs[0] + accs[1] + accs[2] + accs[3]
    p4_pair01 = (accs[0] + accs[1]) + (accs[2] + accs[3])
    p4_pair02 = (accs[0] + accs[2]) + (accs[1] + accs[3])
    p4_0321 = accs[0] + accs[3] + accs[2] + accs[1]
    p4_0231 = (accs[0] + accs[2]) + (accs[3] + accs[1])
    return {"seq": seq, "p4_0123": p4_0123, "p4_pair01": p4_pair01,
            "p4_pair02": p4_pair02, "p4_0321": p4_0321, "p4_0231": p4_0231}

torch.manual_seed(0)
best = None
for K in [32, 64, 96, 128, 160, 192, 224, 256]:
    for seed in range(6):
        g = torch.Generator().manual_seed(seed * 100 + K)
        a = torch.randn((4, K), dtype=torch.float32, generator=g)
        b = torch.randn((K, 4), dtype=torch.float32, generator=g)
        a = a.to(torch.bfloat16)
        b = b.to(torch.bfloat16)
        ref = (a @ b).float()  # torch bf16 matmul, then widen
        cands = acc_p4_orders(a, b, K)
        for name, val in cands.items():
            # compare at bf16 precision: torch output is bf16, so compare val rounded to bf16
            diff = (val.to(torch.bfloat16).float() != ref).sum().item()
            print(f"K={K:4d} seed={seed} {name:10s} mismatches={diff}")
print("done")
