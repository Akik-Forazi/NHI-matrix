#!/usr/bin/env python3
"""NHI-matrix first mini model: XorZen zero_1M trained on Gutenberg mini-corpus.
Real execution + real monitoring: loss curve, tok/s, RSS, eval loss, raw samples.
"""
import json, os, re, sys, time, resource
from collections import Counter
import torch

RUN_DIR = "/home/z/my-project/XorZen-fresh/runs/mini_talk"
os.makedirs(RUN_DIR, exist_ok=True)

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1800
BATCH, SEQ = 8, 128
LR, WARMUP = 3e-4, 100
VOCAB = 4096  # our BPE vocab; model vocab_size=6765, we keep ids < 4096

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

t_start = time.time()
log_lines = []
def log(msg):
    line = f"[{time.time()-t_start:7.1f}s] {msg}"
    print(line, flush=True)
    log_lines.append(line)

# ---------- 1. corpus ----------
from pathlib import Path
data_dir = Path("/home/z/my-project/XorZen-fresh/data/gutenberg/txt")
texts = []
for f in sorted(data_dir.glob("book_*.txt")):
    raw = f.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"\*\*\* START OF.*?\*\*\*(.*?)\*\*\* END OF", raw, re.S)
    body = m.group(1) if m else raw
    texts.append(body.strip())
    log(f"corpus: {f.name} -> {len(body):,} chars (cleaned)")
corpus = "\n\n".join(texts)
log(f"corpus total: {len(corpus):,} chars")

# ---------- 2. word-level tokenizer (tokenizers lib is pathologically slow in this
# sandbox: 3.6s for a 10k-char tiny BPE. Word-level is instant and fits vocab 6765.) ----------
SPECIALS = ["<pad>", "<unk>", "<s>", "</s>"]
MAX_WORDS = 6000  # 4 specials + 6000 words = 6004 <= 6765 model vocab
words = re.findall(r"\w+|[^\w\s]", corpus)
freq = Counter(words)
vocab = SPECIALS + [w for w, _ in freq.most_common(MAX_WORDS)]
stoi = {w: i for i, w in enumerate(vocab)}
ids = [stoi.get(w, 1) for w in words]  # 1 = <unk>
unk_rate = sum(1 for i in ids if i == 1) / len(ids)
log(f"tokenizer: word-level vocab={len(vocab)} (model allows 6765), stream={len(ids):,} tokens, unk_rate={unk_rate:.1%}")
with open(f"{RUN_DIR}/vocab.json", "w") as f:
    json.dump(vocab, f)

def decode(tok_ids):
    out = []
    for i in tok_ids:
        if i < len(vocab):
            w = vocab[i]
            if w in SPECIALS:
                continue
            # glue punctuation to previous word
            if out and re.match(r"^[^\w\s]", w):
                out[-1] = out[-1] + w
            else:
                out.append(w)
    return " ".join(out)

# split: 95% train / 5% val
n_val = int(len(ids) * 0.05)
val_ids = torch.tensor(ids[:n_val], dtype=torch.long)
train_ids = torch.tensor(ids[n_val:], dtype=torch.long)
log(f"split: train={len(train_ids):,} val={len(val_ids):,}")

def get_batch(buf):
    ix = torch.randint(0, len(buf) - SEQ - 1, (BATCH,))
    x = torch.stack([buf[i:i+SEQ] for i in ix])
    y = torch.stack([buf[i+1:i+SEQ+1] for i in ix])
    return x, y

# ---------- 3. model ----------
from xorzen.models.zero import zero_1M
model = zero_1M()
nparam = sum(p.numel() for p in model.parameters())
ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"model: zero_1M params={nparam:,} trainable={ntrain:,} vocab={model.config.vocab_size} ctx={model.config.context_length}")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s+1)/WARMUP, 1.0) * max(0.05, 0.5*(1+__import__('math').cos(__import__('math').pi*(s-WARMUP)/max(STEPS-WARMUP,1)))) if s >= WARMUP else (s+1)/WARMUP)

PROMPTS = ["Alice was ", "The Time Traveller ", "Once upon a "]
def encode(text):
    return [stoi.get(w, 1) for w in re.findall(r"\w+|[^\w\s]", text)]

def sample(prompt_str, max_new=60, temperature=0.8, seed=None):
    """Custom sampling loop: slice logits to live vocab (ids >= len(vocab) are dead rows)."""
    if seed is not None:
        torch.manual_seed(seed)
    ids_in = torch.tensor([[2] + encode(prompt_str)], dtype=torch.long)  # <s> prefix
    with torch.no_grad():
        for _ in range(max_new):
            if ids_in.shape[1] >= 127:  # stay under context_length=128
                break
            out = model(ids_in[:, -127:], return_dict=True)
            logits = out.logits[:, -1, :len(vocab)] / temperature
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            ids_in = torch.cat([ids_in, nxt], dim=1)
    return decode(ids_in[0].tolist()[1:])

@torch.no_grad()
def eval_loss():
    model.eval()
    losses = []
    for _ in range(20):
        x, y = get_batch(val_ids)
        out = model(x, labels=y, return_dict=True)
        losses.append(out.loss.item())
    model.train()
    return sum(losses)/len(losses)

metrics = {"loss_curve": [], "eval_curve": [], "samples": [], "env": {
    "torch": torch.__version__, "device": "cpu", "threads": torch.get_num_threads(),
    "steps": STEPS, "batch": BATCH, "seq": SEQ, "lr": LR, "tokenizer": "word-level",
    "vocab_size": len(vocab),
    "corpus_chars": len(corpus), "corpus_tokens": len(ids)}}

# ---------- 4. training with monitoring ----------
log("=" * 70)
log(f"TRAINING START: {STEPS} steps x B{BATCH} x T{SEQ}")
step_times = []
model.train()
for step in range(1, STEPS + 1):
    t0 = time.time()
    x, y = get_batch(train_ids)
    out = model(x, labels=y, return_dict=True)
    opt.zero_grad(set_to_none=True)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    step_times.append(time.time() - t0)

    if step % 50 == 0 or step == 1:
        tps = BATCH * SEQ / (sum(step_times[-50:]) / len(step_times[-50:]))
        rss = rss_mb()
        log(f"step {step:5d}/{STEPS} | loss={out.loss.item():.4f} | {tps:,.0f} tok/s | rss={rss:.0f}MB | lr={sched.get_last_lr()[0]:.2e}")
        metrics["loss_curve"].append({"step": step, "loss": round(out.loss.item(), 4)})
        with open(f"{RUN_DIR}/metrics.json", "w") as f:  # incremental dump — survives kill
            json.dump(metrics, f, indent=2)

    if step in (1, 600, 1200, 1800):
        el = eval_loss()
        metrics["eval_curve"].append({"step": step, "eval_loss": round(el, 4)})
        log(f"  eval_loss @ step {step}: {el:.4f}")
        s = sample(PROMPTS[0], max_new=40, seed=42)
        metrics["samples"].append({"step": step, "prompt": PROMPTS[0], "text": s})
        log(f"  SAMPLE @ {step} | prompt='Alice was ' -> {s!r}")

    if step % 600 == 0:
        torch.save({"model": model.state_dict(), "step": step,
                    "config": {k: getattr(model.config, k) for k in ["hidden_size", "num_layers", "vocab_size", "context_length"]}},
                   f"{RUN_DIR}/ckpt_step{step}.pt")
        log(f"  checkpoint saved: ckpt_step{step}.pt")

# ---------- 5. final TALK: multiple prompts, multiple temperatures ----------
el_final = eval_loss()
metrics["eval_curve"].append({"step": STEPS, "eval_loss": round(el_final, 4)})
log(f"eval_loss final @ step {STEPS}: {el_final:.4f}")
log("=" * 70)
log("FINAL TALK — raw generations (unfiltered, decoded from BPE ids < 4096)")
for p in PROMPTS:
    for temp in (0.5, 0.8, 1.1):
        s = sample(p, max_new=80, temperature=temp, seed=7)
        metrics["samples"].append({"step": STEPS, "prompt": p, "temp": temp, "text": s})
        log(f"TALK [{temp}] '{p}' -> {s!r}")

torch.save({"model": model.state_dict(), "step": STEPS}, f"{RUN_DIR}/ckpt_final.pt")
metrics["wall_minutes"] = round((time.time()-t_start)/60, 1)
metrics["tok_per_s_final"] = round(BATCH*SEQ/(sum(step_times[-100:])/len(step_times[-100:])))
metrics["log"] = log_lines
with open(f"{RUN_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
log(f"DONE in {metrics['wall_minutes']} min | metrics.json + ckpt_final.pt saved | final tok/s={metrics['tok_per_s_final']}")
