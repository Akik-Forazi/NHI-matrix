# Experiment 001 — NHI-matrix first mini model (XorZen zero_1M talks)

**Date:** 2026-11-06 · **Operator:** Tess (GLM agent) · **Trigger:** user directive —
*"pull xorzen latest from github and make a small model talk"* — following the criticism that
prior research docs contained no executed/monitored code. **This experiment is the corrective:
every claim below is backed by a real process run with monitored output.**

## What ran

| Stage | Command | Result |
|---|---|---|
| Clone upstream | `git clone --depth 1 https://github.com/Akik-Forazi/XorZen.git` | commit `954f3ce` ("added the naming law and family and sub family doc") |
| Smoke test | `scripts/xorzen_smoke_test.py` | PASS — see monitored table below |
| Corpus fetch | 4 Gutenberg books (11, 16, 35, 5200) | 809,897 bytes → 698,301 chars cleaned |
| Training | `scripts/xorzen_mini_train.py 1500` | **EXIT=0, 5.5 min wall** |
| Talk | 3 prompts × 3 temperatures, raw unfiltered decode | outputs below |

## Monitored facts (measured, not claimed)

### zero_1M instantiation (fresh clone)
- config: hidden=64, layers=3, heads=4, vocab=6765, context_length=128
- **params: 1,053,711 actual / 1,077,503 claimed in `variants.py` (delta −23,792, −2.2%)**
- trainable: 953,838 (CoT GRU frozen; "~4.6% active params/token" per router log)
- live architecture confirmed by init logs: AdaptiveRouter (depth/width/path/expert),
  3 HASS blocks (local / low_rank=96 / ssm=16 pathways), SlicedFFN widths [64, 256],
  disk-sharded MoE (2 experts, top-1, 0.38MB on disk), gated merger, tied embeddings

### Smoke test timings (B=4, T=128, 2-core CPU, torch 2.15.0+cpu)
| step | loss | fwd+bwd+step | RSS |
|---|---|---|---|
| 0 | 8.6210 | 0.11s | 491MB |
| 1 | 8.5648 | 0.14s | 525MB |
| 2 | 8.5069 | 0.11s | 528MB |

Generation: 100 tokens in 1.4s (untrained → gibberish, as expected).

### Training run (1500 steps × B8 × T128 = 1.54M tokens seen; word-level vocab 6004; lr 3e-4 cosine; AdamW)
| step | train loss | tok/s | RSS | eval loss |
|---|---|---|---|---|
| 1 | 8.6078 | 3,328 | 595MB | 8.8262 |
| 200 | 6.1506 | 4,965 | 695MB | — |
| 600 | ~5.9 | ~5,000 | ~700MB | 6.30 |
| 1200 | 5.7219 | 5,201 | 719MB | 6.2250 |
| 1500 | 5.4822 | 5,155 | 720MB | **6.1928** |

Full loss curve + raw samples: `metrics.json`. Full stdout: `train.log`.

## Talk — raw generations (unfiltered, seed=7)

Prompt **'Alice was '**, temp 0.8:
> 'Alice was as on to. to' already their teeth and To business that' a pieces It the
> is? and came by. he know in in. is me the of suggestion she.", was but there this
> over and Here to that could and, the, the., so out the. chief to he but was their,
> was' walked" said, the.'

Prompt **'The Time Traveller '**, temp 1.1:
> 'The Time Traveller Alice as probably bare path whole sent planet myself frightening
> away the for stairway the, social happy charwoman unmolested as,' must to after
> inhuman had a used they a ten she s side John" summer time a man which stood; ...'

Prompt **'Once upon a '**, temp 0.5:
> 'Once upon a. as,, the, she the, I, said the, the, was. was to as,' that to to, ...'

(9 combinations in `metrics.json` / `train.log`.)

## Honest assessment of what the model learned

**Learned:** word-frequency distribution (the/was/to/and/said/he/she dominate correctly),
punctuation behavior (., "'!? placement glues to words), bigram patterns ("Alice was",
"said the"), prompt continuation (each prompt's first tokens are on-topic words: Alice →
'Alice was as on to'; Time Traveller → 'The Time Traveller, as was').

**Not learned:** long-range grammar, coherent sentences. Different prompts at the same
temperature produce near-identical continuations → the model is dominated by the unigram
marginal, context sensitivity is weak. Train/eval gap (5.48 vs 6.19, ppl ≈ 488 on val)
says: capacity exists, budget doesn't. This is the honest level of a 1M-param model
after 1.5M tokens on 2 CPU cores — nobody should claim more from this run.

## Bugs / constraints discovered by execution (not present in any doc)

1. **`generate()` context overflow bug (upstream XorZen):** `_greedy_generate` /
   `_sample_generate` recompute the full sequence each step but never trim input to
   `context_length`; prompt(128)+1 token → `ValueError: Sequence length 129 exceeds
   maximum 128`. Workaround used here: custom sampling loop capped at 127.
2. **`train_colab.py` default tokenizer `zero_bpe_10k` does not exist** in
   `xorzen/tokenizer/pretrained/` (only `xorzen_agi_tokenizer_65k`, `xorzen_python_10k`)
   → silent fallback to word-level.
3. **`tokenizers` (HF, Rust) is pathologically slow in this sandbox**: 3.6s to train a
   trivial 10k-char BPE (vocab 500) → full-corpus BPE was infeasible → word-level
   tokenizer written instead (stdlib regex, ≤6004 entries, unk_rate logged in metrics).
4. **Sandbox reaps user processes at tool-call boundaries** even under `setsid nohup`:
   background training died silently twice (log ends mid-run, no traceback). Fix:
   run training in a single foreground call (<600s) with incremental `metrics.json`
   dumps every 50 steps so partial data survives a kill.

## Files

- `xorzen_smoke_test.py` — instantiation/timing/generation smoke test
- `xorzen_mini_train.py` — end-to-end: corpus → tokenizer → train → monitor → talk
- `metrics.json` — loss curve, eval curve, tok/s, RSS, all 9 talk outputs, env
- `train.log` — verbatim stdout of the full run
- `vocab.json` — word-level vocabulary (6004 entries)
- `ckpt_final.pt` — final weights (4.2MB); `ckpt_step600/1200.pt` kept locally only

## Reproduce

```bash
git clone --depth 1 https://github.com/Akik-Forazi/XorZen.git
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pydantic psutil einops tqdm tokenizers sentencepiece "transformers>=4.30" "scipy>=1.10"
PYTHONPATH=. python3 ../xorzen_mini_train.py 1500   # from XorZen repo root
```

Hardware used: 2 vCPU, 4GB cgroup RAM, no GPU. Wall time 5.5 min (excluding corpus download).
