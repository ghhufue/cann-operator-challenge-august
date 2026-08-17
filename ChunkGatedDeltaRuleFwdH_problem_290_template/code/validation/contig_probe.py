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

r, c = 53, 70

kt = k_sel.transpose(-1, -2)           # non-contiguous view
ktc = kt.contiguous()                   # contiguous copy
ktm = k_sel.t().contiguous()            # same as ktc

u1 = kt @ v_decay_bf16
u2 = ktc @ v_decay_bf16
u3 = ktm @ v_decay_bf16
u4 = v_decay_bf16.t() @ k_sel  # alternative orientation [V,61]@[61,K] -> .t() -> [K,V]
u4 = u4.t().contiguous()

print("non-contig kt @ v:", float(u1[r, c]), hex(int(u1[r, c].view(torch.int16)) & 0xFFFF))
print("contig     ktc@ v:", float(u2[r, c]), hex(int(u2[r, c].view(torch.int16)) & 0xFFFF))
print("t+contig   ktm@ v:", float(u3[r, c]), hex(int(u3[r, c].view(torch.int16)) & 0xFFFF))
print("v.t@k .t():", float(u4[r, c]), hex(int(u4[r, c].view(torch.int16)) & 0xFFFF))
print("all equal:", torch.equal(u1, u2), torch.equal(u1, u3), torch.equal(u1, u4))
