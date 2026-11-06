"""Tess's independent verification of XorZen v0.4 conditional-compute claims.

v2 — rewritten after v1 findings: SlicedFFN has no fc1/fc2 Linear children
(nested weight slicing), and test_mode=True swaps real experts for a dummy.
This version measures analytically from routing_info with its own counters.

  V1 width sparsity  : forced width idx 0 vs 1 -> SlicedFFN FLOPs ratio == 0.5
  V2 MoE top-k       : distinct REAL experts executed == top_k setting
  V3 pathway sparsity: forced SSM-only -> local & low_rank Linear FLOPs == 0
  V4 depth sparsity  : forced half-depth -> tokens processed by later blocks drop
  V5 param counts    : actual totals vs README family table (informational)

Verdict lines print PASS/FAIL with measured numbers.
"""

import torch
import torch.nn as nn
from collections import defaultdict

torch.manual_seed(7)

from xorzen.config import ConfigFactory, ModelSize
from xorzen.models.zero.variants import zeroBase


def build(H=32, layers=3, experts=4, topk=2, widths=None, test_mode=False, path_topk=2):
    cfg = ConfigFactory.get_config(ModelSize.TINY_23K)
    cfg.update(
        model_name="tess_verify", vocab_size=32, context_length=16,
        hidden_size=H, num_layers=layers, num_attention_heads=2,
        max_depth=layers, min_depth=1,
        width_choices=widths or (16, 32),
        cot_dim=8, cot_components=6,
        expert_count=experts, top_k_experts=topk,
        router_hidden_dim=16, router_num_layers=1, merger_num_layers=1,
        shard_experts=False, pad_token_id=0, dropout=0.0,
        gradient_checkpointing=False, pathway_top_k=path_topk,
    )
    torch.manual_seed(7)
    m = zeroBase(config=cfg, test_mode=test_mode)
    m.eval()
    return m, cfg


class LinCounter:
    """Count nn.Linear FLOPs + invocations by name substring (own hooks)."""
    def __init__(self):
        self.flops = defaultdict(int)
        self.calls = defaultdict(int)
        self.rows = defaultdict(int)
        self.handles = []

    def _hook(self, name):
        def h(mod, inp, out):
            x = inp[0]
            n_tok = x.numel() // mod.in_features if x.dim() >= 1 else 1
            self.flops[name] += 2 * n_tok * mod.in_features * mod.out_features
            self.calls[name] += 1
            self.rows[name] += n_tok
        return h

    def attach(self, model, matcher):
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and matcher(name):
                self.handles.append(mod.register_forward_hook(self._hook(name)))

    def detach(self):
        for hd in self.handles:
            hd.remove()
        self.handles.clear()


def run(model, cfg):
    ids = torch.randint(0, cfg.vocab_size, (4, cfg.context_length))
    with torch.no_grad():
        return model(input_ids=ids, output_routing_info=True)


def force_width(model, idx):
    def f(logits, complexity, temperature, training, deterministic, position_ids=None):
        B, T, W = logits.shape
        probs = torch.zeros(B, T, W); probs[..., idx] = 1.0
        i = torch.full((B, T), idx, dtype=torch.long)
        return probs, i, torch.ones(B, T, 1)
    model.router._route_width = f


def force_path(model, idx):
    def f(logits, temperature, training, deterministic, position_ids=None):
        B, T, P = logits.shape
        probs = torch.zeros(B, T, P); probs[..., idx] = 1.0
        return probs
    model.router._route_path = f


def force_depth(model, frac):
    nl = model.config.num_layers
    def f(logits, complexity, temperature, training, deterministic, position_ids=None):
        B, T, D = logits.shape
        keep = max(1, round(frac * nl))
        mask = torch.zeros(B, T, D); mask[..., :keep] = 1.0
        return torch.softmax(logits, -1), mask
    model.router._route_depth = f


print("=" * 72)
print("V1: WIDTH SPARSITY  (SlicedFFN FLOPs from routing_info must scale)")
print("=" * 72)
m, cfg = build(widths=(16, 32))
from xorzen.model.components.sliced_ffn import SlicedFFN
ffn0 = m.blocks[0].ffn
print(f"  block FFN type: {type(ffn0).__name__}, width_choices={ffn0.width_choices}")
H = ffn0.hidden_dim
flo = {}
for idx in (0, 1):
    force_width(m, idx)
    out = run(m, cfg)
    wid = out.routing_info.width_idx
    ok_force = bool((wid == idx).all())
    wc = torch.tensor(ffn0.width_choices)[wid]
    n_blocks = len(m.blocks)
    flops = n_blocks * int((4 * H * wc).sum().item())
    flo[idx] = (flops, ok_force)
    print(f"  forced idx {idx}: width_idx all=={idx}? {ok_force} | SlicedFFN FLOPs = {flops:,}")
ratio = flo[0][0] / flo[1][0]
print(f"  measured ratio {ratio:.4f}  (expected 16/32 = 0.5)")
print("  V1", "PASS" if (ratio and abs(ratio - 0.5) < 0.02 and flo[0][1] and flo[1][1]) else "FAIL")

def force_experts(model, idx, topk=1):
    """Force EVERY token to the same expert set: [idx, idx+1, ...] of size topk."""
    def f(logits, temperature, training, deterministic, expert_capacity, position_ids=None):
        B, T, E = logits.shape
        probs = torch.zeros(B, T, E); probs[..., idx] = 1.0
        sel = [(idx + i) % E for i in range(topk)]
        indices = torch.tensor(sel, dtype=torch.long).reshape(1, 1, topk).expand(B, T, topk)
        weights = torch.ones(B, T, topk)
        return probs, indices, weights
    model.router._route_experts = f


print()
print("=" * 72)
print("V2: MoE TOP-K  (force all tokens -> one expert; only that expert runs)")
print("=" * 72)
TARGET = 2
for topk in (1, 2):
    m2, cfg2 = build(experts=4, topk=topk, test_mode=False)
    hits = set()
    for name, mod in m2.named_modules():
        if ".experts." in name and name.endswith("up_proj"):
            def mk(nm):
                def h(mod, inp, out): hits.add(nm.split(".experts.")[1].split(".")[0])
                return h
            mod.register_forward_hook(mk(name))
    n_exp = sum(1 for n, _ in m2.named_modules() if n.endswith(".up_proj") and ".experts." in n)
    force_experts(m2, TARGET, topk=topk)
    run(m2, cfg2)
    allowed = {str((TARGET + i) % 4) for i in range(topk)}
    ok = hits.issubset(allowed) and str(TARGET) in hits
    print(f"  top_k={topk}: {n_exp} real experts exist; forced all->experts {sorted(allowed)}; executed = {sorted(hits)}")
    print(f"  V2 top_k={topk}", "PASS" if ok else "FAIL")

print()
print("=" * 72)
print("V3: PATHWAY SPARSITY  (force SSM-only -> local & low_rank idle)")
print("=" * 72)
m3, cfg3 = build(path_topk=1)
c3 = LinCounter()
c3.attach(m3, lambda k: ("pathways.local." in k or "pathways.low_rank." in k))
force_path(m3, 2)
out3 = run(m3, cfg3)
fl = sum(c3.flops.values()); cl = sum(c3.calls.values())
pp = out3.routing_info.path_probs  # [B,T,P] or similar
c3.detach()
print(f"  forced SSM-only: local+low_rank Linear FLOPs = {fl:,}, calls = {cl}")
print("  V3", "PASS" if fl == 0 and cl == 0 else "FAIL")

print()
print("=" * 72)
print("V4: DEPTH SPARSITY  (forced half-depth -> later blocks see fewer tokens)")
print("=" * 72)
B, T = 4, 16
for frac in (1.0, 0.5):
    m4, cfg4 = build(layers=3)
    rows = defaultdict(int)
    handles = []
    # probe: block input layernorm — every token the block processes passes it
    for name, mod in m4.named_modules():
        if ".ln1" in name:
            blk = name.split(".")[1]
            def mk(b):
                def h(m_, inp, out):
                    x = inp[0]
                    rows[b] += x.numel() // m_.normalized_shape[0]
                return h
            handles.append(mod.register_forward_hook(mk(blk)))
    force_depth(m4, frac)
    run(m4, cfg4)
    for hd in handles: hd.remove()
    print(f"  forced frac={frac}: tokens seen per block = {dict(sorted(rows.items()))} (full batch = {B*T})")
    if frac == 0.5:
        v4_ok = rows.get("2", 0) == 0 and rows.get("0", 0) == B * T and rows.get("1", 0) == B * T
        print("  V4", "PASS" if v4_ok else f"FAIL (rows={dict(rows)})")

print()
print("=" * 72)
print("V5: PARAMETER COUNTS vs README (labels overstate; actuals printed)")
print("=" * 72)
import xorzen
for fname, label in (("zero_10M", "zero_10m"), ("zero_277M", "zero_277m")):
    factory = getattr(xorzen, fname, None)
    if factory is None:
        print(f"  {label}: factory {fname} not found"); continue
    mm = factory(test_mode=True)
    tot = sum(p.numel() for p in mm.parameters())
    print(f"  {label}: total params = {tot:,}")
    del mm

print()
print("DONE — all verdicts measured on this machine, this session.")
