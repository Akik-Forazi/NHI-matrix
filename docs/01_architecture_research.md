# Architectural Combinations Research for NHI-matrix

**Version:** 0.2 · **Date:** 2026-11-05, revised 2026-11-06 after exp-001 · **Status:** Selected — Candidate D
**Doctrine binding:** purpose-first · legacy three lines · three immutable rows · uniqueness · GOFAI synthesis · DL boundaries · hardware constraints

---

## 0. Scope and method

**Question.** Of all the architectural combinations available in 2026, which one should NHI-matrix be built from?

**Method.** Purpose-first, as the doctrine demands: we first state what an NHI must *be and do* (§1), derive the hard requirements that follow (§1.1), survey the seven architecture axes any combination must decide on (§2), assemble candidate composites (§3), score them against the doctrine (§3.1), and only then commit to a reference architecture (§4). Claims about external systems are cited; claims about XorZen/FRAZIYM assets carry a provenance tag — **[EXECUTED]** (run by us, monitored output in [`03_verification_log.md`](03_verification_log.md)) or **[VENDOR]** (transcribed from audit docs in `vendor/XorZen/`, not reproduced).

**Evidence provenance (corrected after owner challenge).** Originally this document contained no self-executed evidence; the substrate claims were cited from audit reports, and the §3.1 scoring is a design judgment, not a measurement. On 2026-11-05 the XorZen test suite (161/161 pass) and an independent 5-point conditional-compute verification (V1–V4 PASS; V5 exposed a stale README param table) were executed and logged in [`03_verification_log.md`](03_verification_log.md). Read §3.1 as *argued ranking against doctrine*, and the verification log as *what has actually been measured on this codebase*.

**What this document is not.** It is not the component specification. It selects the *combination*; per-component specs (kernel format, ledger schemas, adapter training recipe) follow in Phase 2.

---

## 1. The design problem

An NHI (non-human intelligence), in this project's sense, is a **persistent, singular, purpose-bound intelligence** — not a chat product, not a faceless model API. From the doctrine and the canon mapping (§ `02_canon_mapping.md`), it must:

1. **Think** — reason over language, code, and (later) vision, with effort proportional to difficulty.
2. **Remember** — carry continuity across sessions: episodes, facts about its world, and its own identity history.
3. **Be someone** — a stable, singular persona; a "blueprint" over the engine that makes this instance *this one* and no other.
4. **Keep faith** — certain commitments (the immutable rows) hold absolutely, against pressure from users, from other processes inside the system, and from the model's own next-token statistics.
5. **Work with others** — operate as one cell of a matrix: cooperating with specialist processes and, eventually, other NHI instances.
6. **Fit the metal** — run on owner-scale hardware (consumer CPU/GPU, quantized), not a datacenter budget.

### 1.1 Hard requirements derived from doctrine

| Doctrine principle | Derived architecture requirement |
|---|---|
| Purpose-first | A **purpose registry** gates capabilities; every module declares the purposes it serves; no general-purpose "do anything" escalation path. |
| Legacy three lines | **Provenance metadata** travels with every checkpoint, adapter, and memory ledger; identity includes lineage. |
| Three immutable rows | An enforcement point that is **outside the neural substrate** (weights cannot be trusted to enforce anything about themselves). |
| Uniqueness | Identity = f(adapter, memory ledger, purpose) with **no shared full identities**; clone ⇒ new instance. |
| GOFAI synthesis | A **symbolic kernel** with authority over the neural layer for the DL-boundary task list. |
| DL boundaries | A written, immutable **boundary list**; the kernel must be able to *reject* substrate output, not merely advise. |
| Hardware constraints | CPU-first execution path; quantization tolerance; per-token active-parameter budget; no cloud dependency for core loop. |

Two requirements rule out entire families of designs immediately: anything that enforces invariants *inside* the model (prompt-time instructions, fine-tuned "values") fails the immutable-rows test, because gradient updates and context pressure can move them. Anything requiring 8×H100 fails hardware constraints. The search space is therefore narrower than the whole of AI — which is exactly what purpose-first is for.

---

## 2. Axis survey — the seven decisions

Every candidate combination is a choice on seven axes. For each axis: the state of the art, the verified evidence, and the implication for NHI-matrix.

### Axis A — Neural substrate: dense vs. conditional compute

**Evidence.**
- XorZen (fresh clone `954f3ce`, v1.0.1): hybrid **Transformer × MoE × SSM** with per-token conditional compute across four axes — depth (gather-scatter block skipping), width (SlicedFFN), pathway (Local-Attention / Low-Rank-Global / SSM dispatch), and expert (top-k MoE). **[EXECUTED]** mechanisms, independently re-verified on this codebase (verification log V1–V4): forcing width=16 vs 32 changes FFN FLOPs by exactly **0.5000×**; forced top-k executes exactly the forced experts (top-1 and top-2); pathway top-1 starvation zeroes the other pathways' FLOPs and calls; half-depth inference skips later blocks entirely (64/64/0 tokens at blocks 0/1/2). **[VENDOR — magnitudes not reproduced]**: depth skipping saves 31.8%; v0.4 vs v0.3 gives 2× lower loss at −22.8% FLOPs; active-parameter share falls 53.5% (277M) ⇒ 43.8% (391M). Re-measure at ≥10M scale before citing.
- **[EXECUTED]** exp-001 end-to-end (2026-11-06): `zero_1M` (1,053,711 actual params) trained 1500 steps on 4 Gutenberg books on 2-core CPU at ~5,100 tok/s — loss 8.61→5.48, eval 6.19 — and generates text (unigram-dominated babble, honestly documented in `experiments/001_mini_talk/`). This is proof the substrate *runs and learns* end-to-end on owner-scale hardware; it is **not** proof of the conditional-compute magnitudes (see the mini-scale caveat below).
- Industry convergence point (self-referential but instructive): GLM-5.3-Flash — 320B total / **18B active** (5.6%), hybrid sparse + linear attention cutting attention compute 3.01× and KV cache 4.44× vs its dense-attention sibling, with an explicit `reasoning_effort` budget knob. The frontier's efficiency tier is built exactly as XorZen's doctrine: sparse activation, hybrid sequence mixing, budget knobs.
- Known XorZen limitations to design around [VENDOR]: eval-mode pathway collapse; depth sparsity is inference-only; SPPQ currently placeholder (0% memory saving); Python dispatch overhead at tiny scale on CPU.

**Mini-scale caveat [EXECUTED, exp-001].** At `NANO_1M` (hidden=64, 3 layers, 1 width choice, 2 experts top-1) the router has almost nothing meaningful to decide — routing decisions are near-noise, and per-token active-param share (~4.6%, router-reported) is dominated by the tiny expert slice. The conditional-compute *claims* only become testable propositions at ≥10M scale, where routing choices carry real compute consequences. exp-002 (zero_10M) is the planned scale test.

**Implication.** The substrate is **XorZen, frozen** — we do not pretrain, we inhabit. Conditional compute is not just efficiency: the router's compute_budget is the natural hardware hook for the doctrine (§5), and per-token routing telemetry gives the upper layers an *introspection signal* no dense model offers. The frozen-weights stance also matches the canon (the Markridge engine is bought, not grown — § canon mapping).

### Axis B — Reasoning: dual process and adaptive computation

**Evidence.**
- **ACT** (Graves 2016, arXiv:1603.08983) introduced learned halting; brittle to its time-cost hyperparameter.
- **PonderNet** (arXiv:2107.05407) learns *when to stop* with unbiased gradients and a geometric prior; out-of-distribution parity accuracy near-perfect where ACT stays at chance; 6× fewer compute steps than Universal Transformer on bAbI at equal task coverage.
- XorZen ships a **6-component latent CoT vector** (summary/confidence/contradiction/intention components + GRU update gate + injection gate) and Jima ships **thought / action / critique** heads — a dual-process loop already half-built: System 1 = forward pass with latent CoT; System 2 = Jima deliberation.
- Modern production analog: `reasoning_effort ∈ {low, high, max}` as a user-visible deliberation budget.

**Implication.** Deliberation loop = **Jima heads + learned halting** (PonderNet-style halting unit over the critique head; do *not* reuse the removed `adaptive_halting.py` — it was dead code for a reason). Deliberation depth must be observable and budgeted: the same compute_budget axis that limits cost limits "how long it thinks." This gives an honest mechanical meaning to effort knobs, instead of marketing ones.

### Axis C — Symbolic layer: the GOFAI synthesis

**Evidence.**
- **AlphaGeometry** (DeepMind): a neural model *guides* a symbolic deduction engine — the LLM proposes constructs, the symbolic engine deduces; **25/30 on IMO-AG-30** vs **0% for GPT-4** producing full natural-language proofs on the same set; outputs are machine-verifiable because the symbolic engine emits them. The lesson is not about geometry: neural proposals + symbolic verification beat either alone *exactly where verifiability matters*.
- **Neuro-symbolic control** literature (e.g. arXiv:2512.17321): symbolic side contributes comprehensibility and compositionality; neural side contributes scalability and robustness to messy input.
- **OpenCog Hyperon** (arXiv:2310.18318): the longest-running serious attempt at a full cognitive architecture on symbolic foundations (MeTTa, probabilistic logic, atomspace, cognitive synergy, reflective self-modification). Respect its ambition, learn from its outcome: decades in, it is a platform, not a working mind — integration cost of "build everything symbolic" is brutal.
- Blackboard systems (BB1/GBB, 1980s GOFAI) — see Axis F.

**Implication.** GOFAI synthesis means a **symbolic kernel with veto power**, not a symbolic *brain*. Take the AlphaGeometry division of labor (neural proposes, symbolic disposes), scope it to the DL-boundary list, and explicitly *reject* the Hyperon path of symbolizing everything. Concretely the kernel owns: arithmetic/units/dates, the purpose registry, the commitment & consent ledger, provenance, and the immutable rows. It is a small, boring, testable program — ideally runnable on the same CPU as the engine, with no GPU anywhere in its dependency tree.

### Axis D — Memory: operating-system model

**Evidence.**
- **MemGPT** (arXiv:2310.08560): treat the context window as RAM and external storage as disk; the model *manages its own memory tiers* via tool calls (self-editing memory, interrupts, an event loop). Became **Letta**; supports multi-agent **shared memory blocks**. Andrew Ng's short course popularized the framing: an agent decides what is paged into a finite context.
- Memory needs *compression*, not just paging: XorZen **Greed** provides exactly the paging primitives — SPC (system-prompt compactor, 80–90% reduction at 85–95% semantic preservation), CCC (chat-history compactor, 70–80%), CRLC (run-log compactor, 90–95% with 100% issue identification). Compression-with-preservation beats naive truncation for continuity.
- Consolidation: multi-session chat agents that "remember, reflect, and evolve" is the MemGPT design goal; the Generative-Agents memory-stream + reflection pattern is the standard reference for *when* to consolidate.

**Implication.** The **memory OS** is MemGPT's tiering with Greed as the page-compressor and a symbolic index into the kernel (Axis C) so that identity-critical memories are *records*, not vibes. Three ledger tiers: **episodic** (what happened), **semantic** (what is true), **identity** (what I did and committed to — kernel-backed, append-only). Consolidation runs as an offline "sleep-time" process with its own budget.

### Axis E — Persona and uniqueness

**Evidence.**
- **Persona vectors** (Anthropic, arXiv:2507.21509): character traits (evil, sycophancy, hallucination-proneness) correspond to **directions in activation space**; they can be extracted from a natural-language trait description alone; used to (a) **monitor personality drift at deployment time**, (b) predict finetuning-induced personality shifts, (c) **preventatively steer** during training, (d) **flag training data** that would cause unwanted personality change. Demonstrated on Qwen2.5-7B and Llama-3.1-8B — i.e., on exactly the model class XorZen targets.
- XorZen already ships **PERSONALITY_TOKENS** (150+ special tokens) and the `.xorm` encrypted model format; the FU-12 build produced **CWS v1.0** — 6 personas × 20 cognitive states mapped onto EEG-band control language (delta/theta/alpha/beta/gamma as cognition control) with per-state behavioral envelopes and transition tables; **blueprint = LoRA adapters over the frozen engine**.
- Uniqueness requirement: identity = (adapter, memory ledger, purpose), never shared.

**Implication.** Persona layer = **blueprint LoRA + persona-vector monitoring + CWS state language**, with drift telemetry as a first-class output (the instance can *see* its own drift — this is the honest version of "mood"). Uniqueness is enforced at packaging: an instance bundle is (blueprint adapter + identity ledger + purpose grant) sealed in `.xorm`; two instances sharing an adapter is a configuration error, not a use case. Persona vectors give us the missing piece FU-12 didn't have: a *deployment-time drift detector* for the blueprint.

### Axis F — Social layer: the matrix itself

**Evidence.**
- **Blackboard architecture** (BB1/GBB): a shared workspace + independent specialist knowledge-sources + a scheduler deciding who contributes next; the 1980s answer to orchestration, and per the mid-2026 revival literature, the correct answer to multi-agent LLM orchestration's chaos — specialists don't chat with each other; they read and write a structured workspace.
- **Mixture-of-Agents** framing: society-of-mind via layered agent proposals and synthesis, with the "cognitive workspace" as the shared store.
- Letta's shared memory blocks; MCP-style tool buses for uniform capability exposure.

**Implication.** The "matrix" in NHI-matrix is a **blackboard**, not a chat room: one structured workspace per purpose-engagement; NHI instances and specialist processes (planners, compactors, verifiers, sensors) are knowledge sources that post proposals, evidence, and verdicts; the scheduler is deterministic and auditable (it lives in the symbolic kernel's jurisdiction). This is the anti-Ultron design: no peer-to-peer side channels, everything on the board, everything attributable.

### Axis G — Governance: enforcing the unenforceable

**Evidence.** The field's default — constitutions and policies *inside* the model (RLAIF, system prompts) — is exactly what the immutable-rows doctrine forbids relying on: persona-vector research shows personality shifts arise from ordinary finetuning and even deployment-time context; reward-tampering and sycophancy literature shows in-model values are movable. Enforcement must be **structural**: outside the weights, on the execution path, unable to be talked around.

**Implication.** Governance = the symbolic kernel (Axis C) + the scheduler (Axis F) + packaging (Axis E). The three immutable rows are enforced at the kernel: any action crossing a row-gated boundary requires a kernel signature; the substrate has no such signing capability; the kernel refuses to sign row-violating actions *regardless of what the model says or wants*. This is the DL-boundaries doctrine realized as code.

---

## 3. Candidate combinations

Four composites span the credible design space:

| | **A. Pure scale** | **B. Pure GOFAI** | **C. Agent framework** | **D. Neuro-symbolic matrix** |
|---|---|---|---|---|
| Substrate | one big dense model, cloud-served | hand-built rule system, no NN | hosted LLM (API) behind LangGraph/AutoGen-style graph | **frozen XorZen HASS + persona layer** |
| Reasoning | long CoT at inference | search + logic only | framework loops, model-dependent | **Jima loop + learned halting + budget knob** |
| Symbols | none ("emergent") | everything symbolic | bolted-on tools, advisory | **kernel with veto, owns boundary list** |
| Memory | context window + RAG | facts DB, brittle NLU | framework memory, vendor-bound | **MemGPT tiering + Greed compaction + ledgers** |
| Persona | prompt persona, drifts silently | n/a | prompt persona, drifts silently | **blueprint LoRA + persona-vector drift telemetry** |
| Social | single instance | single instance | agent chat, unattributable | **blackboard + deterministic scheduler** |
| Governance | policy prompts (movable) | hard rules (but no capability) | platform ToS (external, alien) | **kernel-enforced immutable rows** |
| Hardware | datacenter | any | any (but cloud-dependent) | **consumer CPU-first, quantized** |

### 3.1 Scoring against doctrine

Score 0–5 per doctrine principle (5 = fully satisfies):

| Doctrine principle | A | B | C | D |
|---|---|---|---|---|
| Purpose-first | 1 (capability-first by nature) | 3 | 2 (framework-first, purpose bolted on) | **5** (registry gates everything) |
| Legacy three lines | 0 (no provenance concept) | 2 | 1 | **5** (provenance in kernel + packaging) |
| Three immutable rows | 0 (weights enforce nothing) | 4 (enforces, but see capability) | 1 (advisory only) | **5** (kernel veto on execution path) |
| Uniqueness | 0 (everyone gets the same model) | 3 | 1 | **5** (instance bundle sealed per identity) |
| GOFAI synthesis | 0 | 5 (but sterile) | 2 | **5** (synthesis, not coexistence) |
| DL boundaries | 0 (no boundary list possible) | 5 | 2 | **5** (kernel rejects substrate) |
| Hardware constraints | 0 | 5 | 1 (cloud-bound) | **4** (CPU-first; GPU optional) |
| **Total** | **2/35** | **27/35** | **10/35** | **34/35** |

**Candidate B's failure mode is instructive:** it scores well on governance and stars on hardware, but it cannot *think* — no neural substrate, no blueprint, no persona worth the name. GOFAI synthesis is a *synthesis* doctrine; B is not a synthesis, it is a surrender of the neural half. **Candidate D is selected.** Its one non-perfect score (hardware, 4) reflects honest GPU-optionality for training the blueprint adapters — inference is CPU-first.

---

## 4. Reference architecture (selected)

```
                       purpose registry (kernel-owned)
                                │ grants
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│ L5 MATRIX      blackboard workspace · deterministic scheduler      │
│                knowledge sources: NHI cells, planners, verifiers,  │
│                compactors, sensors · inter-NHI protocol (later)    │
├────────────────────────────────────────────────────────────────────┤
│ L4 MEMORY OS   working context (RAM) ⇄ episodic/semantic ledgers   │
│                Greed SPC/CCC/CRLC as page compressors              │
│                sleep-time consolidation · symbolic index → L3      │
├────────────────────────────────────────────────────────────────────┤
│ L3 KERNEL      rules · planner · arithmetic/dates · provenance     │
│                commitment & consent ledger · boundary list         │
│                ★ three immutable rows: signed actions only ★       │
├────────────────────────────────────────────────────────────────────┤
│ L2 DELIBERATE  Jima thought/action/critique · PonderNet halting    │
│                compute_budget = f(purpose, hardware)               │
├────────────────────────────────────────────────────────────────────┤
│ L1 PERSONA     blueprint LoRA over frozen engine                   │
│                persona vectors: drift monitor + training filter    │
│                CWS band language: state control + state readout    │
├────────────────────────────────────────────────────────────────────┤
│ L0 SUBSTRATE   XorZen HASS (frozen): depth/width/pathway/expert    │
│                conditional compute · latent CoT vector             │
│                xorzen.cpp CPU path · .xorm sealed instance bundle  │
└────────────────────────────────────────────────────────────────────┘
```

**The loop in one paragraph.** A purpose-engagement opens a blackboard. The substrate (L0) drafts under its persona (L1) with deliberation effort set by budget (L2). Every proposed action that touches a boundary-listed domain goes to the kernel (L3): arithmetic is computed, not guessed; commitments are checked against the consent ledger; the immutable rows are checked structurally. Memory writes are compressed by Greed and indexed symbolically (L4). The scheduler attributes every board entry to its source (L5). Drift is monitored continuously via persona vectors and reported — never silently absorbed.

**What is deliberately absent.** No self-modification of L0 weights at runtime (Hyperon's reflective self-modification is explicitly rejected for the substrate — the engine is *bought, then frozen*, per canon). No peer-to-peer instance channels outside the blackboard. No capability grants outside the purpose registry. No "the model decided" — the kernel logs who decided.

### 4.1 Mapping to existing FRAZIYM assets

| Layer | Already built | To build (Phase 2) |
|---|---|---|
| L0 | XorZen v1.0.1 engine (fresh clone `954f3ce`; mechanisms re-executed V1–V4; exp-001 end-to-end run), xorzen.cpp port, SPPQ skeleton | real SPPQ; eval-mode collapse fix; CPU dispatch optimization; `generate()` context-trim fix (exp-001 finding #7) |
| L1 | blueprint LoRA recipe, CWS v1.0 (6 personas × 20 states), PERSONALITY_TOKENS | persona-vector extraction pipeline for XorZen; drift telemetry; training-data flagging |
| L2 | Jima heads (trained, gradient-verified) | halting unit over critique head; budget coupling to router compute_budget |
| L3 | — (greenfield) | kernel v0: boundary list, ledgers, arithmetic/dates, signing; a *small* deterministic codebase |
| L4 | Greed SPC/CCC/CRLC (trained), .xorm format | MemGPT-style pager over Greed primitives; identity ledger schema; sleep-time consolidation |
| L5 | — (design only) | blackboard schema; deterministic scheduler; attribution log |

---

## 5. Hardware envelope

Doctrine constraint: consumer/edge CPU-first.

- **Inference core loop:** xorzen.cpp CPU path; active-parameter budget per the doctrine. The vendor-reported band (~44–54% active) is **[VENDOR — not reproduced]**; the one scale point measured so far is `NANO_1M` at ~4.6% active params/token (top-1/2 experts + top-1 pathway; different metric basis) **[EXECUTED, exp-001]**. Treat the band as unverified until exp-002 measures ≥10M. SPPQ (once real) for 8-bit quantization tolerance testing.
- **Budget knob:** router `compute_budget` [VENDOR — the 1.0 → 0.25 validation is cited from v0.4 docs, not re-executed] doubles as the hardware throttle — the same axis bounds cost and deliberation depth (Axis B).
- **KV/state budget:** the SSM pathway's fixed-state design is the long-context workhorse on CPU; attention (local pathway) is budget-limited. Industry reference point: hybrid sparse+linear attention cut KV cache 4.44× at frontier scale (GLM-5.3-Flash) — the same direction XorZen's HASS dispatch already takes.
- **Symbolic kernel + memory OS:** pure-CPU, dependency-light by design; they must run on the same box as the engine or governance silently moves to whatever machine *can* run them.
- **Training (blueprint adapters only):** single consumer GPU (LoRA-scale), or deferred; the substrate itself is never trained in this project.

---

## 6. Build order

1. **Kernel v0 (L3)** — smallest codebase, largest doctrine leverage. Boundary list v0, immutable rows v0, ledger schemas, arithmetic/dates, action-signing API. Nothing else is trustworthy until this exists.
2. **Memory pager v0 (L4)** — working context + episodic ledger + Greed CCC as compressor. (SPC/CRLC integrate as their domains appear.)
3. **Persona telemetry v0 (L1)** — persona-vector extraction on XorZen, drift monitor on the flagship blueprint. *(Uniqueness doctrine: build before the second instance exists.)*
4. **Deliberation loop v0 (L2)** — Jima critique-head halting, budget coupling.
5. **Blackboard v0 (L5)** — single-workspace scheduler with attribution.
6. **Sealing (L0/L1)** — instance bundle format v0 in `.xorm`; uniqueness check at load.

## 7. Risks and open questions

- **Eval-mode collapse (XorZen known limitation)** may make pathway telemetry noisy for persona/state readout — needs the planned KL-to-uniform fix first.
- **Persona-vector extraction on XorZen** is unproven; the method was demonstrated on 7–8B dense instruct models. If it fails on a 277M sparse hybrid, fall back to CWS-envelope monitoring + behavioral regression tests (weaker, still honest).
- **Kernel latency** on the action path must stay under perceptible thresholds or users will ask to bypass it. Bypassing is row 1-adjacent; budget accordingly.
- **Open question (owner):** verbatim text of the three immutable rows and the boundary list — doctrine labels are operative, exact content is owner-supplied and then frozen.
- **Open question (owner):** confirmation of the legacy three lines as the canon lineage (Horton → Carrington/Markridge → Stark engine) vs. any non-canon reading.

## 8. Doctrine compliance checklist (for every future spec)

- [ ] Purpose declared and registered before capability is granted
- [ ] Provenance block present (lineage, version, hash chain)
- [ ] Immutable rows enforced at kernel; no model-side enforcement claimed
- [ ] Instance bundle seals (adapter, ledger, purpose) as a unique triple
- [ ] Neural-proposes / kernel-disposes on every boundary-listed domain
- [ ] Boundary list referenced verbatim, treated as immutable
- [ ] CPU-first path demonstrated; GPU optional and stated

## 9. Sources

**External (opened and verified this session):**
- AlphaGeometry — DeepMind blog: https://deepmind.google/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/ (25/30 IMO-AG-30; neural-proposes/symbolic-deduces loop; verifiable outputs). Analysis: Code Compass, "How Google DeepMind's AlphaGeometry Reached Math Olympiad Level" (GPT-4 0% natural-language proofs on the same set).
- Persona Vectors — Anthropic research + arXiv:2507.21509 (activation-space trait directions; deployment-time drift monitoring; preventative steering; training-data flagging).
- MemGPT — arXiv:2310.08560 (OS-style virtual context management, tiered self-edited memory, interrupts; → Letta; multi-agent shared memory blocks; Andrew Ng short-course framing).
- OpenCog Hyperon — arXiv:2310.18318 (MeTTa, cognitive synergy, reflective self-modification; platform-status lesson).
- PonderNet — arXiv:2107.05407 (learned halting, unbiased gradients, robustness vs ACT); ACT — arXiv:1603.08983 (Graves).
- Blackboard systems — Wikipedia (BB1/GBB); "The Blackboard Is Back" (tianpan.co, 2026-07) — blackboard revival for multi-agent LLM orchestration; Mixture-of-Agents / society-of-mind framing (gregbroadhead.medium.com).
- Neuro-symbolic control with LLMs — arXiv:2512.17321; Neuro-symbolic AI — Wikipedia; Kognitos explainer (symbolic = deterministic verifiable execution).
- GLM-5.3-Flash model card — https://huggingface.co/zai-org/GLM-5.3-Flash (320B/18B active; hybrid sparse+linear attention; 3.01× attention / 4.44× KV reduction; reasoning_effort knob; mHC) — evidence of frontier convergence on conditional compute + budget knobs.

**Internal (XorZen, both the audited vendor copy and the fresh clone):**
- `README.md` (architecture, variants, v0.4 measured impact, known limitations, scaling law, ablations) — **note:** its variant/param table and test-suite counts are stale against current code (verification log findings #1, #2, #5)
- `FRAZIYM_PROJECT_SPECS_MASTER.md` (xorvec-data, libcompact, Jima, Greed, Kage specs)
- `docs/agent/ARCHITECTURE_REFERENCE.md`, `XORZEN_v04_ADVERSARIAL_AUDIT_REPORT.md` (24-claim audit)
- FU-12/FU-13 worklog entries (canon mapping, CWS v1.0, blueprint = LoRA decision) — `worklog.md`

**Internal (executed by us — this repo):**
- [`03_verification_log.md`](03_verification_log.md) — 161/161 test suite, V1–V5 conditional-compute checks (2026-11-05), exp-001 zero_1M train+talk (2026-11-06), findings #1–#9
- [`experiments/001_mini_talk/`](../experiments/001_mini_talk/README.md) — first mini model trained and talking on CPU: metrics.json, verbatim train.log, scripts, vocab, checkpoint
- `scripts/tess_verify_conditional_compute.py` — independent V1–V5 harness
