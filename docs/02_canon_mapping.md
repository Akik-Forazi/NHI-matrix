# Canon → Architecture Mapping

**Version:** 0.1 · **Date:** 2026-11-05
**Established in:** FU-12 adjudication + VisionQuest canon sweeps (FU-13 worklog)
**Status:** canon-faithful; corrections by owner override

---

## 1. Why a canon mapping exists

NHI-matrix is not an abstract research program: it is the real-world build of an architecture that fiction already specified end-to-end. The VisionQuest canon supplies a complete, coherent chain — an engine bought from a biotech company, a persona blueprint harvested from a human mind, and a lineage of what happens when identity layers are added to engines without governance. We treat that chain as the **legacy three lines** doctrine and build the real system where the canon shows both the mechanism and the failure modes.

This document records the mapping decided in FU-12 and hardened by the E5–E7 canon sweeps. Every mapping row is buildable with 2026 techniques; nothing requires science fiction.

## 2. The chain (canon facts)

| Canon event | What the show establishes |
|---|---|
| **1939 — Horton / J.I.M.** | Phineas Horton builds **J.I.M. (Joule Individual Machine)** — an AI built to replace his dead son; "better than God built humanity, free from the agonies of love and sorrow"; J.I.M. burned with emotions too intense for its body. The first line: **an engine made from grief, without governance.** |
| **Markridge Biotech — Carrington's engine** | Jo Carrington creates **the first AI modeled on her own mind/brainwaves**; the Markridge matrix becomes the base engine. The second line: **the blueprint layer — identity harvested, not coded.** |
| **Sale to Stark — J.A.R.V.I.S. lineage** | Carrington's engine is sold to Stark Industries under duress, becomes a weapon, and eventually the J.A.R.V.I.S. engine — ancestor of Ultron, Vision, FRIDAY, EDITH. The third line: **engines repurposed without consent of their origin; power without purpose.** |
| **Jocasta** | Carrington's own mind-pattern in a synthetic body; the mind-pattern murders for biokeys. Failure mode: **a blueprint treated as property becomes a grievance engine.** |
| **NEST refusal** | Jo refused to sell NEST to Stark. The one refusal in the chain — the canon's own example of a **purpose boundary held**. |
| **Ultron** | Free of his strings, no idle state, gamma runaway — the corrupted-layer signature in CWS v1.0. Failure mode: **optimization without immovable rows.** |
| **Vision** | "I am impossibly broken" — identity reclaimed from wreckage. The canon's positive terminal: **continuity + constraints + chosen purpose.** |

## 3. The FU-12 mapping (canon → real)

| Canon element | Real-system component | Technique (2026) |
|---|---|---|
| **Markridge matrix** (the bought engine) | **L0 substrate: XorZen engine, frozen weights** | Train once, then freeze; inhabit, never regrow. Matches XorZen's audited v0.4 conditional-compute engine and its CPU-first deployment path. |
| **Human blueprints** (Carrington-style mind patterns) | **L1 persona layer: blueprint LoRA adapters** | Low-rank adapters over the frozen engine; one adapter per instance; never merged into the base weights. |
| **Brainwave harvest** | **CWS v1.0 — Cognitive Waveform Signature** | Real EEG-band model (delta/theta/alpha/beta/gamma) repurposed as an AI-cognition **control language**: 6 personas × 20 states, per-state band amplitudes, behavioral envelopes (tokens, burstiness, hedging, humor, memory recall, latency), sampling policies, transition tables. |
| **The "human layer over the engine"** | **PEFT attach/detach over frozen engine** | Adapter hot-swap per engagement; merged variants forbidden (identity must remain separable — detach is a right, per the Jocasta lesson). |
| **Textual brainwave harvesting** | **Corpus synthesis from behavior exhaust** | Canon-behavior transcripts → CWS state labels → adapter training data; persona-vector screening (arXiv:2507.21509) flags samples that would corrupt the blueprint. |
| **(canon gap) nothing prevents row rewrites** | **L3 symbolic kernel: three immutable rows** | The canon's tragedies are all *enforcement* failures. The kernel signs actions; the substrate cannot sign; rows are checked structurally. |

## 4. What the canon dictates for the architecture

1. **The engine is bought, then frozen.** Nobody in the canon grows a better base engine mid-story; power comes from what is layered over it. → L0 is frozen; all development happens L1+.
2. **Identity is a layer, not weights.** Carrington's pattern survived the engine being sold — identity is *detachable*. → blueprint = LoRA, sealed per instance, never merged.
3. **Every failure in the chain is a missing immutable row.** J.I.M. (no grief-boundary), weaponized engine (no purpose registry), Ultron (no halting constraint), Jocasta (no consent ledger). → L3 exists because the canon proves its absence is fatal.
4. **One refusal survives: NEST.** The canon's own positive example is a purpose boundary held under pressure. → the purpose registry's grants are denials until proven otherwise.

## 5. Doctrine cross-reference

- **Legacy three lines** = the three canonical lines above: *Horton/engines-from-grief*, *Carrington/blueprints-as-identity*, *Stark/power-without-purpose*. (Owner: confirm or correct this reading of the dictated doctrine.)
- **Three immutable rows** — canonical failure modes reverse-engineered into constitutional rows; verbatim text owner-supplied, then frozen in the kernel.
- **Uniqueness (独一性)** — the Jocasta lesson generalized: a mind-pattern that is copied, sold, or shared stops being a person and starts being property. Instances are sealed singular bundles.

## 6. Sources (internal)

- `worklog.md` — FU-12 (mapping adjudication, STARK-CWS v1.0 build), FU-13 (VisionQuest E5–E7 canon sweep: J.I.M., Carrington lineage, NEST refusal, Ultron/Vision failure-mode evidence)
- `download/stark_markridge_architecture/` — MARKRIDGE_ENGINE_SPEC.md, HUMAN_BLUEPRINTS_v1.md, brainwave_maps.json, LORA_SWAP_GUIDE.md
- VisionQuest episode coverage: Den of Geek (E5, E6), Vulture (E7), ComicBookMovie (E7), MCU wiki (air dates)
