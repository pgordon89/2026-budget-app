# Fiscus — state of the project

*A snapshot of what is built, what is measured, and what is known to be wrong.*

Repository: `github.com/pgordon89/2026-budget-app` · TypeScript · Node 20/22

---

## What this is

An AI-native personal finance app whose real purpose is to demonstrate defensible AI engineering. The central problem: a bank hands you `SQ *BLUE BOTTLE #4432 SANFRAN CA` and you need `food.coffee`, reliably, across 54 categories, for pennies.

The thesis is that a language model should never decide what your balance is — only what a transaction *means*, and only after the cheap deterministic paths have failed.

---

## Headline numbers

All reproducible from a clean checkout with **no API key and no network** — the model tier replays a committed response cache.

```bash
npm install && npm run eval
```

| | |
|---|---|
| Coverage | 89.0% of holdout transactions categorized |
| Precision | 98.1% of those correct |
| Resolved | 87.3% end to end |
| Human review queue | 11.0% |
| **Cost** | **$0.61 per 1,000 transactions** |
| Cost if every transaction hit the model | $3.34 per 1,000 (extrapolated) |
| Transfers misclassified as spend | **0 of 67** |
| Tier 0 key collisions | **0** |

Per tier:

| tier | share | precision |
|---|---|---|
| 1 — merchant memory | 73.7% | 99.6% |
| 2 — nearest-neighbour vote | 8.0% | 89.3% |
| 3 — Claude Haiku 4.5 | 7.3% | 91.9% |

**82% of the model bill is removed by the tiers that run before it.** That is the architectural argument, priced.

---

## Architecture

Four tiers, cheapest first. Each either answers confidently or passes the transaction on. Nothing guesses.

**Tier 0 — normalization.** Strips processor prefixes, store numbers, city/state blobs and reference codes to a stable merchant key. 2,460 raw descriptors collapse to 647 keys with **zero collisions**. Deliberately conservative: over-stripping produces confident wrong answers, which cost far more than cache misses.

**Tier 1 — merchant memory.** Exact key → category tallies from the user's own confirmed history. Two gates, because one number cannot do both jobs (see below). Postgres-backed, with the in-memory and SQL implementations proven equivalent by a conformance suite.

**Tier 2 — nearest-neighbour vote.** For merchants history has never seen: the most lexically similar labelled merchants vote in proportion to how near they are, each spreading its weight across its own label distribution so ambiguity is inherited rather than laundered. A key is excluded from its own neighbourhood — otherwise the tier re-answers what Tier 1 declined, routing around its own abstention.

**Tier 3 — Claude Haiku 4.5.** Forced tool use with a strict schema, Zod validation, a bounded repair loop, and per-call token/cost/latency recording. Few-shot examples come from Tier 2's neighbours.

**Correction loop.** A user fixing a category writes back to the merchant store atomically — ledger row, immutable correction log, and reweighted tally in one transaction.

---

## Engineering decisions worth defending

**Every threshold is selected on a validation split, never on the holdout.** History is split again; the golden set is scored once at settings already fixed. Selection also runs on a replay with corrections *writing back*, because that is the population a deployed store actually has.

**Abstention is the feature.** The costs are asymmetric: an escalation costs a fraction of a cent and is recoverable; a wrong answer is silent and lands in a budget total. Selection rule, stated in code: *maximise coverage subject to a 97% precision floor*.

**Confidence claims are calibrated, not asserted.** Tier 3's self-reported confidence is overconfident in the middle band (states ~0.75, right 43% of the time) and reliable above 0.90. The gate landed at 0.90 by sweep *before* that table was read.

**The eval is free, so it gates every push.** Deterministic tiers compute live; the model tier replays a committed cache. The usual reason model evals are excluded from CI is per-run spend, and the cache removes it.

**The regression gate is directional and exact.** Ten metrics, each with its own direction — higher cost and more transfer errors are regressions. Zero tolerance, because everything is deterministic. Verified by deliberately regressing the system and confirming exit code 1.

**Structured output is layered.** Forced `tool_choice` + `strict: true` makes an out-of-taxonomy category *unrepresentable*. Zod covers what strict schemas cannot express (no numeric bounds, so `confidence: 4.2` validates). A direction check catches `income.refund` on a −$6.75 charge. The repair loop fires for the last two — 0 times in 182 calls, which is the intended result.

**Money is integers.** Cents as bigint, prediction cost as micro-dollars. The one part of this system that must not be probabilistic is the arithmetic.

**The taxonomy lives in TypeScript.** It is the model's label space *and* the eval's label space. The `categories` table is seeded from it so foreign keys work without creating a second definition that drifts.

---

## Three findings the measurement produced

### 1. A Wilson gate is not a precision policy

Tier 1 scored confidence as a Wilson lower bound — sound for rejecting `1/1 = 100%`. But Wilson moves with agreement *and* sample size, so it scores "seen 4 times, always groceries" identically to "seen 100 times, groceries 39% of the time".

| Wilson gate | admits agreement at n=10 | n=30 | n=100 |
|---|---|---|---|
| 0.30 | 0.59 | 0.47 | **0.39** |

The shipped 0.30 gate was, on well-observed merchants, a **39%-precision policy**. It had never been asked, because a static store has few merchants near the line — and the correction loop's entire effect is tipping merchants across it.

Two hypotheses were measured and discarded first: a support floor (83% → 91%, never reached target, cost 25 points of coverage) and raising Wilson to 0.50 (65% → 71%, +45% on the model bill). The fix was a separate **agreement** gate. Overall precision 98.5% → 99.4%, hard slice 93.8% → 97.3%.

### 2. The correction loop is real but modest, and it cost precision

Replaying the holdout one transaction at a time in two arms — corrections write back, or are discarded — the loop is worth **+0.7 points of coverage and ~$0.05 per 1,000**. Smaller than "the system gets cheaper as it is used" implies.

It also, before the agreement gate, shipped its marginal answers at 64.7% precision. That defect is what led to finding #1.

### 3. The fixture was flattering Tier 2

Every merchant used to be present for all 30 months, so **no merchant was ever new** — and Tier 2 exists precisely to answer merchants history has not seen. It was evaluated against a population containing none of its own use case.

Merchants now have lifecycles and arrive in near-miss families (`PRESIDIO DENTAL` beside `PRESIDIO VETERINARY`). The result: **Tier 2's precision is 89.3%, not the 98.9% it previously reported.** Nine points of that was the absence of contested neighbourhoods.

---

## Known problems

**Tier 2 ships at 89.3% precision, below the project's own 97% floor.** Now visible because the fixture is realistic. Tightening it pushes traffic to the model tier or the review queue, so it is a cost/precision product call rather than a purely technical one. *This is the top open item.*

**Neither tier's gate can reach 97% on the answers the correction loop creates.** Best achievable is 88.7% (Tier 1) and 95.2% (Tier 2) over populations now large enough to believe. On a corpus with genuine churn those answers are the hard residue by construction; no threshold makes them 97% precise — only a different mechanism would.

**The cost-aware routing claim is unproven.** "Haiku for bulk, Sonnet for reasoning" needs both models scored on identical transactions, and Sonnet 5 and Opus 5 both return HTTP 429 from the development environment. Confirmed as an access restriction, not rate limiting.

**No hosted embedding model is wired up.** Tier 2's vectors are a fitted TF-IDF over word and character-trigram features — offline, deterministic, zero cost. Deliberate ordering: this is the number a paid model must beat. The `Embedder` interface is the seam.

**Prompt caching is declared and does nothing.** The shared system prompt is ~2,900 tokens against Haiku 4.5's 4,096-token minimum cacheable prefix. `cache_control` is accepted, no error raised, `cache_creation_input_tokens` stays 0. Recorded per-model in the price table because the failure is otherwise silent.

**A coverage-maximising selection rule drifts permissive.** It has twice selected settings that cleared the floors on validation and regressed the holdout. When the floors cannot bind, the objective is wrong rather than merely loose.

**Persona savings rate is 1.0%**, down from 3.2%, after adding churn merchants. Low but solvent, and flagged rather than tuned to a target.

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Taxonomy, seeded fixture corpus, temporal split | ✅ |
| 2 | Four-tier pipeline, eval harness, CI accuracy gate | ✅ |
| 3 | Ledger, correction loop, import, UI, budgets, reporting | ledger + correction loop done |
| 4 | NL → constrained SQL, insight agent, cash-flow forecasting | |
| 5 | Multimodal statement ingestion, cost/observability dashboard | |

**Next up in Phase 3:** CSV/OFX import, then the Next.js UI over this ledger, then budgets and reporting.

---

## Commands

```bash
npm run eval                 # scored run + regression gate (this is what CI enforces)
npm run eval -- --write-baseline

npm run analyze:normalizer   # Tier 0: collapse ratio, collisions
npm run analyze:memory       # Tier 1: coverage/precision sweep
npm run analyze:knn          # Tier 2: k and gate selection, calibration
npm run analyze:llm          # Tier 3: cost, latency, calibration, routing
npm run analyze:gate         # gate selection against a write-back replay (--tier=2)
npm run analyze:learning     # what the correction loop is actually worth

npm run generate:data        # deterministic — regeneration is byte-identical
npm test                     # 106 tests
npm run typecheck
```

Local dev and CI run Postgres via PGlite — real Postgres semantics, no server, no credentials, same migrations that deploy to Neon.
