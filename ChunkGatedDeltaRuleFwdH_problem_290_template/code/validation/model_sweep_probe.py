import numpy as np
import torch

def models_vec(products):
    M, N, K = products.shape
    out = {}
    acc = np.zeros((M, N), dtype=np.float32)
    for k in range(K):
        acc = acc + products[:, :, k]
    out["seq"] = acc
    acc = np.zeros((M, N), dtype=np.float32)
    for k in range(0, K - 1, 2):
        acc = acc + (products[:, :, k] + products[:, :, k + 1])
    if K % 2 == 1:
        acc = acc + products[:, :, K - 1]
    out["pair_acc"] = acc
    accs = [np.zeros((M, N), dtype=np.float32) for _ in range(4)]
    for k in range(K):
        accs[k % 4] = accs[k % 4] + products[:, :, k]
    out["p4_pair"] = (accs[0] + accs[1]) + (accs[2] + accs[3])
    out["p4_0123"] = ((accs[0] + accs[1]) + accs[2]) + accs[3]
    arr = products.copy()
    n = K
    while n > 1:
        half = n // 2
        arr[:, :, :half] = arr[:, :, :half] + arr[:, :, n - half:n][:, :, ::-1]
        n = (n + 1) // 2
    out["tree"] = arr[:, :, 0]
    return out

def bf16_round(x):
    t = torch.from_numpy(np.ascontiguousarray(x.astype(np.float32)))
    return t.to(torch.bfloat16).float().numpy()

for tokens in [61, 64]:
    agg = {}
    for seed in range(8):
        g = torch.Generator().manual_seed(seed * 100 + 3)
        kk = torch.randn((tokens, 128), dtype=torch.float32, generator=g).to(torch.bfloat16)
        vv = torch.randn((tokens, 128), dtype=torch.float32, generator=g).to(torch.bfloat16)
        kt = kk.transpose(-1, -2)
        ref = (kt @ vv).float()
        af = kt.to(torch.float32).numpy()
        bf = vv.to(torch.float32).numpy()
        products = np.einsum('ik,kj->ijk', af, bf, optimize=False).astype(np.float32)
        cands = models_vec(products)
        refb = ref.numpy()
        for name, val in cands.items():
            mism = int((bf16_round(val) != refb).sum())
            agg[name] = agg.get(name, 0) + mism
    print(f"transposed-A tokens={tokens}: {agg}")

agg = {}
for seed in range(8):
    g = torch.Generator().manual_seed(seed * 100 + 5)
    ww = torch.randn((61, 128), dtype=torch.float32, generator=g).to(torch.bfloat16)
    st = torch.randn((128, 128), dtype=torch.float32, generator=g).to(torch.bfloat16)
    ref = (ww @ st).float()
    af = ww.to(torch.float32).numpy()
    bf = st.to(torch.float32).numpy()
    products = np.einsum('ik,kj->ijk', af, bf, optimize=False).astype(np.float32)
    cands = models_vec(products)
    refb = ref.numpy()
    for name, val in cands.items():
        mism = int((bf16_round(val) != refb).sum())
        agg[name] = agg.get(name, 0) + mism
print(f"contiguous-A (MM1): {agg}")
