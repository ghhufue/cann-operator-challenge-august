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

def cast_to_float16(t):
    return t.to(torch.float32).to(torch.float16)

manifest = json.loads((CASE_DIR / "manifest.json").read_text())
t = {}
for name, meta in manifest["tensors"].items():
    if name.startswith("input") or name.startswith("golden"):
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

def mm_p4(a, b, combine="pair"):
    M, K = a.shape
    N = b.shape[1]
    af = a.to(torch.float32); bf = b.to(torch.float32)
    accs = [torch.zeros((M, N), dtype=torch.float32) for _ in range(4)]
    for kk in range(K):
        accs[kk % 4] += af[:, kk:kk+1] * bf[kk:kk+1, :]
    if combine == "pair":
        return (accs[0] + accs[1]) + (accs[2] + accs[3])
    if combine == "0123":
        return accs[0] + accs[1] + accs[2] + accs[3]
    return accs[0] + accs[3] + accs[2] + accs[1]

mm2_p4 = mm_p4(k_sel.transpose(-1, -2).contiguous(), v_decay_bf16, "pair").to(torch.bfloat16)
print("MM2 torch vs p4_pair total:", (update_torch != mm2_p4).sum().item())

r, c = 53, 70
print("torch update:", float(update_torch[r, c]), hex(int(update_torch[r, c].view(torch.int16)) & 0xFFFF))
print("p4_pair     :", float(mm2_p4[r, c]), hex(int(mm2_p4[r, c].view(torch.int16)) & 0xFFFF))

af = k_sel.transpose(-1, -2).to(torch.float32)
bf = v_decay_bf16.to(torch.float32)
products = af[r, :] * bf[:, c]
print("exact fp64 sum:", float(products.double().sum()))
acc = torch.zeros((), dtype=torch.float32)
for kk in range(61):
    acc += products[kk]
print("seq fp32:", float(acc))
accs = [torch.zeros((), dtype=torch.float32) for _ in range(4)]
for kk in range(61):
    accs[kk % 4] += products[kk]
print("p4 accs:", [float(a) for a in accs])
print("p4 pair:", float((accs[0]+accs[1])+(accs[2]+accs[3])))
print("p4 0123:", float(accs[0]+accs[1]+accs[2]+accs[3]))
print("p4 0321:", float(accs[0]+accs[3]+accs[2]+accs[1]))

def bf16(x):
    return float(torch.tensor(float(x), dtype=torch.float32).to(torch.bfloat16).float().item())
print("bf16(seq):", bf16(acc), "bf16(p4pair):", bf16(float((accs[0]+accs[1])+(accs[2]+accs[3]))))

# also check what torch actually returns as bf16 before the fp16 cast
up = update_torch[r, c]
print("torch bf16 value:", float(up), "bits:", hex(int(up.view(torch.int16)) & 0xFFFF))
