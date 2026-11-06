# Verification Log — Executed Evidence

**Run by:** Tess (GLM-5.3-Flash agent session) · **Date:** 2026-11-05
**Trigger:** owner challenge — "have you written codes and executed them and monitored results?"
**Machine:** Linux sandbox, Python 3.12.16, torch 2.15.0+cpu, 2 threads

This log exists to separate **evidence we executed** from **evidence we cited**. Before this run, the architecture research had zero self-executed evidence. That was a fair criticism; this is the correction.

---

## 1. Environment provisioning (executed)

- Installed CPU-only torch 2.15.0 (`--index-url https://download.pytorch.org/whl/cpu`)
- `pip install -e .` on `vendor/XorZen` — import OK, 10 model factories listed

## 2. Full XorZen test suite (executed)

```
$ python3 -m pytest tests/ -q
161 passed, 15 warnings in 69.45s   (exit 0)
```

Note: the README still says "59 tests, ~18s". The suite has grown to 161 tests since that text was written. Documentation drift finding #1.

## 3. Independent conditional-compute verification (executed)

Script: [`scripts/tess_verify_conditional_compute.py`](../scripts/tess_verify_conditional_compute.py) — written from scratch this session, does NOT reuse the repo's audit harness. Forcing is done by monkeypatching `AdaptiveRouter._route_*`; measurement by own forward hooks + analytic SlicedFFN accounting from `routing_info`.

| Verdict | Claim tested | Measured result | Verdict |
|---|---|---|---|
| **V1** | Width sparsity: FFN FLOPs scale with selected width | forced width idx 0 (w=16) vs idx 1 (w=32): 393,216 vs 786,432 FLOPs → ratio **0.5000** (expected 0.5); `width_idx` verified all-forced | **PASS** |
| **V2 (top-1)** | Forced single expert → only that expert executes | all tokens forced to expert 2; executed = **{2} exactly** | **PASS** |
| **V2 (top-2)** | Forced expert pair → exactly that pair executes | forced {2,3}; executed = **{2,3} exactly** | **PASS** |
| **V3** | Pathway sparsity: forced SSM-only starves other pathways | local + low_rank Linear FLOPs = **0**, calls = **0** (pathway_top_k=1) | **PASS** |
| **V4** | Depth sparsity: half-depth skips later blocks at inference | full: blocks 0/1/2 see 64/64/64 tokens; half: **64/64/0 — block 2 entirely skipped** | **PASS** |
| **V5** | Parameter counts vs README table | zero_10M = **10,865,977** actual (README: ~7.8M); zero_277M = **276,490,648** actual (README table: ~78.3M) | **DISCREPANCY — see finding #2** |

## 4. Findings (all from executed code, this session)

1. **README test-suite text is stale** (59 tests → actually 161, all passing).
2. **README variant table is stale**: current `zero_277M` instantiates at 276,490,648 params, not the ~78.3M the scaling-law table claims. The table's active-% and "12B vs 60B" scaling hypothesis are therefore **unverified against current code** and must not be cited as established until re-measured.
3. **The v0.4 audit script `phase_v04_model_level_sparse.py` no longer runs against current code** — the router API gained `position_ids` parameters after the audit was written. The audit's *conclusions* still replicate (V1–V4), but the audit *artifact* is incompatible with the codebase it audited. Documentation-drift risk for future audits.
4. Harness lessons (for future spec work): SlicedFFN exposes no `fc1/fc2` Linears (nested slicing — hook nothing, compute analytically from `routing_info`); `test_mode=True` substitutes a dummy expert (top-k verification requires `test_mode=False`); `pathway_top_k=2` keeps a second pathway alive regardless of forcing (set top-1 to test pathway starvation).

## 5. Evidence provenance of the architecture research (post-correction)

| Claim class | Status |
|---|---|
| XorZen 4-axis conditional compute (mechanism) | **RE-EXECUTED 2026-11-05** (V1–V4 PASS) |
| XorZen test suite passing | **RE-EXECUTED 2026-11-05** (161/161) |
| XorZen param table / scaling law | **CONTRADICTED 2026-11-05** (finding #2) — treated as stale until re-measured |
| XorZen zero_1M end-to-end: train + talk on current v1.0.1 (fresh clone `954f3ce`) | **EXECUTED 2026-11-06** (exp-001, §7) |
| Vendor magnitudes: depth saving 31.8%, v0.4-vs-v0.3 −22.8% FLOPs, active-share 53.5%⇒43.8% | **VENDOR-CITED ONLY** — mechanisms re-executed (V1–V4), magnitudes never reproduced; do not cite as measured |
| `generate()` API usability | **EXECUTED 2026-11-06** — context-overflow bug found (finding #7) |
| External results (AlphaGeometry, persona vectors, MemGPT, PonderNet, Hyperon) | **CITED** from opened primary sources — not independently reproduced |
| Candidate D composite + doctrine scoring | **DESIGN JUDGMENT** — not executable until Phase 2 components exist |

## 6. Reproduce

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
cd vendor/XorZen && pip install -e .
python3 -m pytest tests/ -q
python3 scripts/tess_verify_conditional_compute.py
```

---

## 7. exp-001 — first mini model trained & talks (executed 2026-11-06)

**Target:** fresh `git clone --depth 1` of `github.com/Akik-Forazi/XorZen` at commit
`954f3ce` (pyproject version 1.0.1) — i.e. current upstream, not the local vendor copy.
Full evidence: [`experiments/001_mini_talk/`](../experiments/001_mini_talk/README.md)
(metrics.json, verbatim train.log, scripts, vocab, checkpoint). Summary:

| Item | Measured |
|---|---|
| zero_1M config | hidden=64, layers=3, heads=4, vocab=6765, ctx=128 |
| zero_1M params | **1,053,711 actual / 1,077,503 claimed** in `variants.py` (−23,792, −2.2%) |
| Smoke (B4×T128) | loss 8.62→8.51 in 3 steps; 0.11–0.14 s/step; RSS ≤528MB |
| Train 1500 steps × B8 × T128 (4 Gutenberg books, word-level vocab 6004) | loss 8.61→**5.48**, eval 8.83→**6.19** (ppl ≈ 488); ~5,100 tok/s; RSS ≤720MB; wall 5.5 min; EXIT=0 |
| Talk | 9 raw generations (3 prompts × 3 temps): function-word texture + prompt continuation correct; unigram-dominated, prompts converge at same temp — no long-range grammar at this budget |
| Verdict | recorded in exp-001 README "Tess's verdict" — doctrine probe: keep; architectural claims: unproven at this scale |

### Findings #5–#9 (continuing §4 numbering; all executed)

5. **Param drift confirmed on a second variant and a second clone:** zero_1M actual
   1,053,711 vs 1,077,503 claimed. Also `zero_tiny_23k`'s *name* says 23K while
   `PARAM_COUNT = 37_824` and its own docstring says "approximately 37 thousand" —
   the class name itself is stale (ironic, given upstream commit `954f3ce` is titled
   "added the naming law").
6. **`train_colab.py`'s default tokenizer does not exist**: `load_pretrained('zero_bpe_10k')`
   has no matching file in `xorzen/tokenizer/pretrained/` (packaged: only
   `xorzen_agi_tokenizer_65k`, `xorzen_python_10k`). Deeper packaging issue: **no
   packaged tokenizer matches the zero models' own vocab (6765)** — every training
   entrypoint must either fallback or train its own.
7. **`generate()` context-overflow bug (upstream):** both `_greedy_generate` and
   `_sample_generate` recompute the full sequence each step but never trim input to
   `context_length`; prompt(128) + 1 new token → `ValueError: Sequence length 129
   exceeds maximum 128`. exp-001 worked around it with a custom sampling loop capped
   at 127. Proposed as exp-002 fix.
8. **HF `tokenizers` is pathologically slow in this sandbox** (3.6 s for a trivial
   10k-char / vocab-500 BPE) — full-corpus BPE infeasible; exp-001 used a stdlib
   word-level tokenizer instead.
9. **Sandbox operationally reaps user processes at tool-call boundaries** even under
   `setsid nohup` (two silent mid-run deaths, no traceback). Long runs must be single
   foreground calls with incremental result dumps.

### Next steps (proposed)

- **exp-002:** `zero_10M` training run + routing-pressure measurements (the scale where
  depth/width/expert routing starts to matter) + upstream `generate()` context-trim fix.
- Re-measure the vendor magnitude claims (31.8% depth saving; active-share scaling)
  at ≥10M before citing them anywhere.
