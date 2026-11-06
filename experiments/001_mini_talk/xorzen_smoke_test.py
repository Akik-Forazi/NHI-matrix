#!/usr/bin/env python3
"""XORZEN zero_1M smoke test — real execution, real monitoring.
Checks: import, config, param count, forward/backward timing, RSS memory.
"""
import time
import resource
import torch

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

print(f"torch={torch.__version__} threads={torch.get_num_threads()} rss_start={rss_mb():.0f}MB")

from xorzen.models.zero import zero_1M
from xorzen.models.zero.variants import zero_tiny_23k

t0 = time.time()
model = zero_1M()
print(f"zero_1M instantiated in {time.time()-t0:.1f}s, rss={rss_mb():.0f}MB")

c = model.config
for k in ["hidden_size", "num_layers", "num_attention_heads", "vocab_size",
          "context_length", "num_experts", "num_widths", "num_pathways"]:
    print(f"  config.{k} = {getattr(c, k, '?')}")

actual = sum(p.numel() for p in model.parameters())
claimed = zero_1M.PARAM_COUNT
print(f"params actual={actual:,} claimed={claimed:,} delta={actual-claimed:+,}")

# forward + backward timing on dummy batch
B, T = 4, 128
x = torch.randint(0, c.vocab_size, (B, T))
y = torch.randint(0, c.vocab_size, (B, T))
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i in range(3):
    t0 = time.time()
    out = model(x, labels=y)
    opt.zero_grad(set_to_none=True)
    out.loss.backward()
    opt.step()
    print(f"step {i}: loss={out.loss.item():.4f} fwd+bwd+step={time.time()-t0:.2f}s rss={rss_mb():.0f}MB")

# generation smoke (model is untrained — output should be gibberish; that is EXPECTED)
# NOTE: generate() recomputes the full sequence each step; prompt+new must stay <= context_length
t0 = time.time()
prompt = torch.randint(0, c.vocab_size, (1, 8))
gen = model.generate(prompt, max_new_tokens=100, do_sample=True, temperature=0.9)
print(f"generate 100 tokens: {time.time()-t0:.1f}s total_len={gen.shape[1]}")

print("SMOKE TEST PASS")
