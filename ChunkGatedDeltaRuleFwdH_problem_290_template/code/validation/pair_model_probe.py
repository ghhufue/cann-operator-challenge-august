import json
from pathlib import Path
import numpy as np
import torch

CASE_DIR = Path(r"D:\code\tmp_golden_probe\official_example")
DTYPE_MAP = {"bfloat16": (torch.bfloat16, "<u2"), "float32": (torch.float32, "<f4"), "int64": (torch.int64, "<i8")}

def load_bin(name, shape, dtype):
    tdt, ndt = DTYPE_MAP[dtype]
    arr = np.fromfile(CASE_DIR / name, dtype=np.dtype(ndt))
    ten = torch.from_numpy(arr.reshape(shape))
    if dtype == "bfloat16":
        ten = ten.view(torch.bfloat16)
    return ten

def cast_to_float16(t):
    return t.to(torch.float32).to(torch.float16)

manifest = json.loads((CASE_DIR / "manifest.json").read_text())
t = {}
for name, meta in manifest["tensors"].items():
    if name.startswith("input"):
        t[name] = load_bin(meta["file"], tuple(meta["shape"]), meta["dtype"])

k = t["input_k"]; w = t["input_w"]; u = t["input_u"]; g = t["input_g"]; init = t["input_initial_state"]
k_by_head = k.transpose(1, 2).contiguous()

seq = 0; head = 5
state = init[seq, head].to(torch.float32).to(k.dtype)
w_sel = w[0, head, 0:61, :]
u_sel = u[0, head, 0:61, :]
g_sel = g[0, head, 0:61]
k_sel = k_by_head[0, head // 8, 0:61, :]

ws = w_sel @ state
ws_fp16 = torch.nan_to_num(ws.to(torch.float16), nan=0.0, posinf=torch.inf, neginf=-torch.inf)
v_new = cast_to_float16(u_sel).float() - ws_fp16.float()
v_new_fp16 = cast_to_float16(v_new)
g_last = g_sel[60:61]
gate = cast_to_float16((g_last - g_sel).exp().float())
v_decay = v_new_fp16.float() * gate[..., None]
v_decay_bf16 = cast_to_float16(v_decay).to(u.dtype)

update_torch = k_sel.transpose(-1, -2) @ v_decay_bf16
r, c = 53, 70
print("torch:", float(update_torch[r, c]), hex(int(update_torch[r, c].view(torch.int16)) & 0xFFFF))

af = k_sel.transpose(-1, -2).to(torch.float32)
bf = v_decay_bf16.to(torch.float32)
products = af[r, :] * bf[:, c]  # (61,)

def bf16(x):
    return float(torch.tensor(float(x), dtype=torch.float32).to(torch.bfloat16).float().item())

def eval_model(name, fn):
    v = fn()
    print(f"{name}: fp32={float(v)} bf16={bf16(v)} bits={hex(int(torch.tensor(float(v),dtype=torch.float32).to(torch.bfloat16).view(torch.int16)) & 0xFFFF)}")

# model: pairs (k,k+1) summed, then sequential acc over pairs
def pair_acc():
    acc = torch.zeros((), dtype=torch.float32)
    K = products.numel()
    for k in range(0, K - 1, 2):
        acc = acc + (products[k] + products[k+1])
    if K % 2 == 1:
        acc = acc + products[K-1]
    return acc
eval_model("pair_acc", pair_acc)

# model: vdpbf16ps style - pair sum exact then acc
def pair_acc2():
    acc = torch.zeros((), dtype=torch.float32)
    K = products.numel()
    for k in range(0, K, 2):
        s = products[k:k+2].double().sum()
        acc = acc + s.float()
    return acc
eval_model("pair_acc_exact", pair_acc2)

# model: 4 accs but with pair-sums
def p4_pairs():
    accs = [torch.zeros((), dtype=torch.float32) for _ in range(4)]
    K = products.numel()
    for k in range(0, K - 1, 2):
        s = (products[k] + products[k+1])
        accs[(k//2) % 4] = accs[(k//2) % 4] + s
    if K % 2 == 1:
        accs[(K//2) % 4] = accs[(K//2) % 4] + products[K-1]
    return (accs[0]+accs[1])+(accs[2]+accs[3])
eval_model("p4_of_pairs", p4_pairs)

# seq exact
def seq():
    acc = torch.zeros((), dtype=torch.float32)
    for k in range(products.numel()):
        acc = acc + products[k]
    return acc
eval_model("seq", seq)

# p4 of single products
def p4_single():
    accs = [torch.zeros((), dtype=torch.float32) for _ in range(4)]
    for k in range(products.numel()):
        accs[k % 4] = accs[k % 4] + products[k]
    return (accs[0]+accs[1])+(accs[2]+accs[3])
eval_model("p4_single", p4_single)

# pairwise tree
def tree():
    arr = products.clone()
    n = arr.numel()
    while n > 1:
        half = n // 2
        for j in range(half):
            arr[j] = arr[j] + arr[n-1-j]
        n = (n + 1) // 2
    return arr[0]
eval_model("tree", tree)
