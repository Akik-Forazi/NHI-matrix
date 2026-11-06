# NHI-matrix

**A purpose-first architecture for Non-Human Intelligence, built on the XorZen substrate.**

NHI-matrix is the specification and research program for a matrix-structured non-human intelligence: a frozen, hardware-constrained neural engine wrapped in a persona layer, a symbolic verifiable kernel, a memory operating system, and a blackboard-style social workspace — engineered so that *what the system is for* decides *how it is built*, and so that certain rows of the matrix simply cannot be rewritten by anything inside the system.

This repository is the founding canon and research record. It currently contains the **architectural combinations research** that selects the reference architecture, the **executed verification log** (test suite, conditional-compute checks, findings), and **experiments/001_mini_talk** — the first executed experiment: an XorZen `zero_1M` mini model trained 1500 steps on CPU and made to talk.

---

## Founding doctrine

Seven principles govern every design decision in this project. They were dictated by the owner and are binding on all parts, agents, and future revisions.

| # | Principle | Binding consequence |
|---|-----------|---------------------|
| 1 | **Purpose-first** | No component exists without a declared purpose in the purpose registry; capabilities are granted *per purpose*, never in bulk. Architecture is derived from purpose downward, never from capability upward. |
| 2 | **Legacy three lines** | The system must know and honor the lineage it descends from — the three canonical lines established in the founding canon (see `docs/02_canon_mapping.md`). Provenance is part of identity. |
| 3 | **Three immutable rows** | Three rows of the matrix are constitutionally immutable: no process at any layer — including the symbolic kernel itself — may rewrite them. Enforcement is structural, not advisory. |
| 4 | **Uniqueness (独一性)** | Each NHI instance is singular: a unique (adapter, memory ledger, purpose) triple. Instances never share a full identity. Cloning produces a new instance, not a copy of an existing one. |
| 5 | **GOFAI synthesis** | Symbolic reasoning is a first-class citizen, not a plugin. The neural system proposes; the symbolic kernel verifies and commits. Identity-critical state lives in symbols, not in weights. |
| 6 | **DL boundaries** | A written list of tasks the deep-learning substrate must *never* be trusted with alone (arithmetic, identity, consent, commitments, law). The boundary list is itself immutable. |
| 7 | **Hardware constraints** | The engine must run on consumer/edge hardware (CPU-first, quantization-tolerant, active-parameter budgeted). Efficiency is a moral constraint of the design, not an optimization afterthought. |

The verbatim founding-canon text of principles 2, 3, and 6 is being re-incorporated from the owner's dictation sessions; the doctrine table above is the operative summary. Owner corrections override this document.

---

## The reference architecture (selected combination)

The research in `docs/01_architecture_research.md` surveys seven architecture axes and scores four composite candidates against the doctrine. The selected combination ("Candidate D — the Neuro-Symbolic Matrix") is, top to bottom:

```
┌─────────────────────────────────────────────────────────────┐
│  L5  MATRIX / SOCIAL      blackboard workspace + scheduler  │
│      specialist knowledge-sources; inter-NHI protocol       │
├─────────────────────────────────────────────────────────────┤
│  L4  MEMORY OS            tiered paging (MemGPT pattern)    │
│      Greed compactors as page-compression primitives        │
│      episodic / semantic / identity ledgers; consolidation  │
├─────────────────────────────────────────────────────────────┤
│  L3  SYMBOLIC KERNEL      rules, planner, arithmetic,       │
│      commitment + consent ledger, provenance log            │
│      ★ three immutable rows live HERE, kernel-enforced ★    │
├─────────────────────────────────────────────────────────────┤
│  L2  DELIBERATION         thought / action / critique loop  │
│      learned halting; compute_budget knob (System 2)        │
├─────────────────────────────────────────────────────────────┤
│  L1  PERSONA              blueprint LoRA + persona vectors  │
│      CWS control bands; drift monitored, never silent       │
├─────────────────────────────────────────────────────────────┤
│  L0  SUBSTRATE            XorZen HASS engine (frozen)       │
│      Transformer × MoE × SSM, 4-axis conditional compute    │
│      CPU-first via xorzen.cpp; .xorm; SPPQ quantization     │
└─────────────────────────────────────────────────────────────┘
```

The one-sentence form: **a frozen conditional-compute engine, made singular by a monitored persona layer, kept honest by a symbolic kernel it cannot overrule, given continuity by a memory OS, and set to work with others on a blackboard.**

---

## Repository map

```
NHI-matrix/
├── README.md                        ← you are here (charter + doctrine)
├── docs/
│   ├── 01_architecture_research.md  ← axis survey, candidate scoring, reference architecture
│   ├── 02_canon_mapping.md          ← VisionQuest canon → real architecture mapping
│   └── 03_verification_log.md       ← EXECUTED evidence: 161/161 suite, V1–V5 checks, exp-001, findings #1–#9
├── experiments/
│   └── 001_mini_talk/               ← EXECUTED: XorZen zero_1M trained 1500 steps, talks
│                                       (metrics.json, train.log, scripts, vocab, ckpt)
└── scripts/
    └── tess_verify_conditional_compute.py  ← independent V1–V5 harness
```

Planned (not yet written — do not treat as existing):

```
├── canon/            founding canon parts (doctrine verbatim, legacy lines, immutable rows)
├── specs/            component specifications (kernel, memory OS, persona layer, blackboard)
├── purpose/          purpose registry format + governance rules
└── bench/            doctrine compliance tests ("can the substrate override row 1? must fail")
```

---

## Relationship to the FRAZIYM ecosystem

NHI-matrix is designed *on* XorZen and *with* its sibling projects:

| Asset | Role in NHI-matrix |
|-------|--------------------|
| **XorZen Core** (HASS, SlicedFFN, AdaptiveRouter, CoT vector) | L0 substrate; conditional compute is the hardware-constraints doctrine made real |
| **Greed** (SPC / CCC / CRLC compactors) | L4 paging primitives — the memory OS compresses, never truncates |
| **Jima** (thought / action / critique heads) | L2 deliberation loop |
| **xorvec-data** | corpus pipeline for blueprint training material |
| **libcompact / xorzen.cpp** | CPU-first deployment of L0 and L4 |
| **.xorm + PERSONALITY_TOKENS** | encrypted instance packaging; persona tokens feed L1 |
| **XorZen `/NHI/` handoff kit** | agent-facing bridge maintained inside the upstream XorZen repo (`NHI/` at `Akik-Forazi/XorZen`): engine-as-measured notes, doctrine summary, links back here — so a fresh agent can pick up the program from inside the engine repo |

---

## Status

- **Phase 0 — Architecture research: complete.** Reference architecture selected (Candidate D, 34/35).
- **Verification: executed.** 2026-11-05: 161/161 upstream test suite + independent V1–V5 conditional-compute checks on the vendor copy (findings #1–#4). 2026-11-06: exp-001 — fresh upstream clone (`954f3ce`), zero_1M smoke + 1500-step CPU training + 9 raw talk generations (findings #5–#9). Everything measured is logged in `docs/03_verification_log.md`; everything not yet measured is tagged `[VENDOR]` in `docs/01`.
- **exp-002 (proposed):** zero_10M routing-pressure run + upstream `generate()` context-trim fix.
- **Agent handoff (2026-11-06):** upstream XorZen now carries the `NHI/` kit (commit pending push) — XORZEN_NOTES (engine as measured, findings #1–#9), AGENTS contract, NHI_MATRIX program summary — discoverable from `AGENT_HANDOFF.md` §9.
- **Phase 1 — Founding canon:** in progress; verbatim doctrine text from owner dictation being incorporated.
- **Phase 2 — Component specs:** not started. Kernel v0 (L3) remains the priority build item.

## License

TBD — specification repository. XorZen and FRAZIYM components remain under their existing licenses (XORZENX Proprietary for XorZen Core). Nothing in this repo grants rights to the engine.
