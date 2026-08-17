import json
from pathlib import Path
import numpy as np
import torch

CASE_DIR = Path(r"D:\code\tmp_golden_probe\official_example")
CHUNK_SIZE = 64
DTYPE_MAP = {"bfloat16": (torch.bfloat16, "<u2"), "float32": (torch.float32, "<f4"), "int64": (torch.int64, "<i8")}

def load_bin(name, shape, dtype):
    tdt, ndt = DTYPE_MAP[dtype]
    arr = np.fromfile(CASE_DIR / name, dtype=np.dtype(ndt))
    ten = torch.from_numpy(arr.reshape(shape))
    if dtype == "bfloat16":
        ten = ten.view(torch.bfloat16)
    return ten

manifest = json.loads((CASE_DIR / "manifest.json").read_text())
t = {}
for name, meta in manifest["tensors"].items():
    if name.startswith("input"):
        t[name] = load_bin(meta["file"], tuple(meta["shape"]), meta["dtype"])

k = t["input_k"]; w = t["input_w"]; u = t["input_u"]; g = t["input_g"]; init = t["input_initial_state"]
cu = t["input_cu_seqlens"]; ci = t["input_chunk_indices"]

k_by_head = k.transpose(1, 2).contiguous()
# sequence 0: bos=0 eos=61; head 0
state = init[0, 0].to(torch.float32).to(k.dtype)
w_sel = w[0, 0, 0:61, :]
k_sel = k_by_head[0, 0, 0:61, :]

def mm_p4(a, b):
    M, K = a.shape
    N = b.shape[1]
    af = a.to(torch.float32); bf = b.to(torch.float32)
    accs = [torch.zeros((M, N), dtype=torch.float32) for _ in range(4)]
    for kk in range(K):
        accs[kk % 4] += af[:, kk:kk+1] * bf[kk:kk+1, :]
    out = (accs[0] + accs[1]) + (accs[2] + accs[3])
    return out.to(torch.bfloat16)

ws_torch = w_sel @ state
ws_p4 = mm_p4(w_sel, state)
print("MM1 torch vs P4:", (ws_torch != ws_p4).sum().item(), "/", ws_torch.numel())
idx = (ws_torch != ws_p4).nonzero()
if idx.numel():
    i = idx[0]
    print("first mm1 mismatch at", i.tolist(), float(ws_torch[tuple(i)]), float(ws_p4[tuple(i)]))
    K = w_sel.shape[1]
    af = w_sel.to(torch.float32); bf = state.to(torch.float32)
    exact = (af[:, :, None] * bf[None, :, :])  # [61, K, 128]
    m, n = int(i[0]), int(i[1])
    products = exact[m, :, n]
    seq_sum = products.float().sum()
    p4_accs = [products[kkk::4].float().sum() for kkk in range(4)]
    p4sum = ((p4_accs[0] + p4_accs[1]) + (p4_accs[2] + p4_accs[3]))
    print("seq fp32 sum:", float(seq_sum), "p4 fp32 sum:", float(p4sum), "diff:", float(seq_sum - p4sum))
    print("torch bf16 result:", float(ws_torch[m, n]), "p4 bf16:", float(ws_p4[m, n]))
