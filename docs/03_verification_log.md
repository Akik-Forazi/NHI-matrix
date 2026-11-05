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
| XorZen 4-axis conditional compute | **RE-EXECUTED today** (V1–V4 PASS) |
| XorZen test suite passing | **RE-EXECUTED today** (161/161) |
| XorZen param table / scaling law | **CONTRADICTED today** (finding #2) — treated as stale until re-measured |
| External results (AlphaGeometry, persona vectors, MemGPT, PonderNet, Hyperon) | **CITED** from opened primary sources — not independently reproduced |
| Candidate D composite + doctrine scoring | **DESIGN JUDGMENT** — not executable until Phase 2 components exist |

## 6. Reproduce

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
cd vendor/XorZen && pip install -e .
python3 -m pytest tests/ -q
python3 scripts/tess_verify_conditional_compute.py
```
