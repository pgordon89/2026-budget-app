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
| Coverage | 88.4% of holdout transactions categorized |
| Precision | 98.8% of those correct |
| Resolved | 87.4% end to end |
| Human review queue | 11.6% |
| Counted in budget totals | 85.4%, at **99.7%** precision |
| **Cost** | **$0.59 per 1,000 transactions** |
| Cost if every transaction hit the model | $3.34 per 1,000 (extrapolated) |
| Transfers misclassified as spend | **0 of 68** |
| Tier 0 key collisions | **0** |

Per tier:

| tier | share | precision |
|---|---|---|
| 1 — merchant memory | 73.7% | 99.6% |
| 2 — nearest-neighbour vote | 8.5% | 97.7% |
| 3 — Claude Haiku 4.5 | 6.2% | 90.4% |

**82% of the model bill is removed by the tiers that run before it.** That is the architectural argument, priced.

---

## Architecture

Four tiers, cheapest first. Each either answers confidently or passes the transaction on. Nothing guesses.

**Tier 0 — normalization.** Strips processor prefixes, store numbers, city/state blobs and reference codes to a stable merchant key. 2,460 raw descriptors collapse to 647 keys with **zero collisions**. Deliberately conservative: over-stripping produces confident wrong answers, which cost far more than cache misses.

**Tier 1 — merchant memory.** Exact key → category tallies from the user's own confirmed history. Two gates, because one number cannot do both jobs (see below). Postgres-backed, with the in-memory and SQL implementations proven equivalent by a conformance suite.

**Tier 2 — nearest-neighbour vote.** For merchants history has never seen: the most lexically similar labelled merchants vote in proportion to how near they are, each spreading its weight across its own label distribution so ambiguity is inherited rather than laundered. A key is excluded from its own neighbourhood — otherwise the tier re-answers what Tier 1 declined, routing around its own abstention.

**Tier 3 — Claude Haiku 4.5.** Forced tool use with a strict schema, Zod validation, a bounded repair loop, and per-call token/cost/latency recording. Few-shot examples come from Tier 2's neighbours.

**Correction loop.** A user fixing a category writes back to the merchant store atomically — ledger row, immutable correction log, and reweighted tally in one transaction.

**Label trust — `provisional` or `confirmed`.** A category being *shown* and a category being *summed* are different claims, and the ledger stores them separately. Labels from similarity or from the model are displayed and pre-fill a one-tap confirmation; they are excluded from every budget total until independent evidence backs them. Totals are built from exact merchant matches and human confirmations only.

---

## Engineering decisions worth defending

**Every threshold is selected on a validation split, never on the holdout.** History is split again; the golden set is scored once at settings already fixed. Selection also runs on a replay with corrections *writing back*, because that is the population a deployed store actually has.

**Abstention is the feature.** The costs are asymmetric: an escalation costs a fraction of a cent and is recoverable; a wrong answer is silent and lands in a budget total. Selection rule, stated in code: *maximise coverage subject to a 97% precision floor*.

**Escalation is priced, not assumed.** "Send it to the model instead" is only cheap if the model answers it. `analyze:tier2` puts a real Haiku response on file for every transaction Tier 1 escalates, so each candidate gate's cost is measured rather than extrapolated from a mean call price — and the thing that measurement exposes is that Tier 3 answers about half of what it receives. Surrendering traffic does not move work to a model; it splits it between a model and a person, and only the priced version of that sentence is decision-grade.

**Confidence claims are calibrated, not asserted.** Tier 3's self-reported confidence is overconfident in the middle band (states ~0.75, right 43% of the time) and reliable above 0.90. The gate landed at 0.90 by sweep *before* that table was read.

**The eval is free, so it gates every push.** Deterministic tiers compute live; the model tier replays a committed cache. The usual reason model evals are excluded from CI is per-run spend, and the cache removes it.

**The regression gate is directional and exact.** Ten metrics, each with its own direction — higher cost and more transfer errors are regressions. Zero tolerance, because everything is deterministic. Verified by deliberately regressing the system and confirming exit code 1.

**Structured output is layered.** Forced `tool_choice` + `strict: true` makes an out-of-taxonomy category *unrepresentable*. Zod covers what strict schemas cannot express (no numeric bounds, so `confidence: 4.2` validates). A direction check catches `income.refund` on a −$6.75 charge. The repair loop fires for the last two — 0 times in 182 calls, which is the intended result.

**Money is integers.** Cents as bigint, prediction cost as micro-dollars. The one part of this system that must not be probabilistic is the arithmetic.

**The taxonomy lives in TypeScript.** It is the model's label space *and* the eval's label space. The `categories` table is seeded from it so foreign keys work without creating a second definition that drifts.

---

## Six findings the measurement produced

### 1. A Wilson gate is not a precision policy

Tier 1 scored confidence as a Wilson lower bound — sound for rejecting `1/1 = 100%`. But Wilson moves with agreement *and* sample size, so it scores "seen 4 times, always groceries" identically to "seen 100 times, groceries 39% of the time".

| Wilson gate | admits agreement at n=10 | n=30 | n=100 |
|---|---|---|---|
| 0.30 | 0.59 | 0.47 | **0.39** |

The shipped 0.30 gate was, on well-observed merchants, a **39%-precision policy**. It had never been asked, because a static store has few merchants near the line — and the correction loop's entire effect is tipping merchants across it.

Two hypotheses were measured and discarded first: a support floor (83% → 91%, never reached target, cost 25 points of coverage) and raising Wilson to 0.50 (65% → 71%, +45% on the model bill). The fix was a separate **agreement** gate. Overall precision 98.5% → 99.4%, hard slice 93.8% → 97.3%.

### 2. The correction loop is real but modest, and it cost precision

Replaying the holdout one transaction at a time in two arms — corrections write back, or are discarded — the loop is worth **+3.4 points of coverage and $0.12 per 1,000** over the second half, from 134 user corrections teaching 51 merchants the store had never seen.

It also, before the agreement gate, shipped its marginal answers at 64.7% precision. That defect is what led to finding #1; they now run at 95.3% over 43 answers.

The honest caveat: this number moved a lot. It was **+0.7 points** before finding #4 retuned Tier 2, because a tier that answers contested neighbourhoods confidently is a tier that denies the correction loop the corrections it learns from. The loop's value is not a property of the loop alone — it is a property of how much the tiers above it abstain, which makes it a number to re-measure after any gate change rather than to quote once.

### 3. The fixture was flattering Tier 2

Every merchant used to be present for all 30 months, so **no merchant was ever new** — and Tier 2 exists precisely to answer merchants history has not seen. It was evaluated against a population containing none of its own use case.

Merchants now have lifecycles and arrive in near-miss families (`PRESIDIO DENTAL` beside `PRESIDIO VETERINARY`). The result: **Tier 2's precision is 89.3%, not the 98.9% it previously reported.** Nine points of that was the absence of contested neighbourhoods.

### 4. Tier 2's gate was tightened in the wrong direction, and the frontier ran backwards

The obvious fix for a tier at 89.3% against a 97% floor is to raise its gate. `npm run analyze:tier2` priced the whole grid — putting a real Haiku response on file for every transaction Tier 1 escalates, so each setting's cost is measured rather than extrapolated from a mean call price — and the frontier ran the *wrong way*: every increment above the shipped `minConfidence` cost precision **as well as** coverage, 85.7% down to 63.6%.

Bucketing the two factors separately said why:

| bucket | 0.00–0.50 | 0.50–0.70 | 0.70–0.90 | 0.90+ |
|---|---|---|---|---|
| by **agreement** | 27.8% | 57.1% | 65.4% | **87.7%** |
| by **nearest similarity** | 53.1% | 78.0% | 53.8% | **42.9%** |

Agreement climbs and never reverses. Similarity does not rise at all and its *top* bucket is its worst — near-miss families share a token, so the closest lexical neighbour of `PRESIDIO VETERINARY` is `PRESIDIO DENTAL`, and similarity near 1.0 is evidence of a shared name fragment, not a shared category. Their product peaks mid-range, so raising a threshold on it deletes the well-behaved middle band before it touches the badly-behaved top one.

Moving the weight — `minConfidence` 0.50 → 0.30, `minAgreement` 0.51 → 0.90 — gave **Tier 2 8.0% → 8.5% share, 89.3% → 97.7% precision, $0.6115 → $0.5937 per 1,000.** Better on all three at once.

Net effect on the whole cascade: **10 wrong answers removed, 1 right answer gained.** Whole-cascade coverage falls 89.0% → 88.4% because the gate reshapes which transactions Tier 2 takes rather than taking more; resolved rises anyway, which is the same fact from the other side.

### 5. Handing Tier 2's vote to the model was worth nothing

The plan's highest-value item: when Tier 2's neighbourhood is too split to publish an answer, it discards a full weighted distribution and escalates with only the three merchant keys that produced it. The model is left re-deriving a distribution the previous tier already computed exactly. Passing it costs a few dozen tokens.

Two arms, 269 holdout transactions, differing only by that block. Compared **at matched coverage** rather than at a fixed gate — the prior shifts the whole confidence distribution, so holding the accept gate at 0.90 would compare two different operating points and read the coverage gap as a quality gap:

**96 of 109 correct without the prior. 95 of 109 with it.**

One answer, in the wrong direction. The likely reason is that the information was already present: the neighbour keys and their labels were in the prompt before the change, and aggregating three of them is not work the model needed help with. A prior pays only when it carries something the evidence does not, and a summary of the evidence carries nothing.

Reverted. Two things the method is worth more for than the result:

- **The version bump changed two things, so both were priced.** v2 added the vote *and* reworded the system block. A third arm, free from the committed v1 responses, isolated the rewording — it cost **11 points of validation precision** (96.7% → 85.7%), so it was reverted too. Bundled changes get separated before either is judged.
- **The planned control did not exist, and it is reported rather than dropped.** The design assumed a slice of keys with no neighbours, whose prompts are identical across arms, as a free read on model nondeterminism. That slice is empty here — a character-trigram embedder shares a feature with nearly anything — so the delta has no error bar. A difference of 1 in 109 needs none, but the gap is stated.

### 6. The correction residue needed a third state, not a better threshold

Findings #1 and #4 both ended at the same wall: the answers the correction loop creates top out at 92.3% (Tier 1) and 95.2% (Tier 2) whatever the gate, against a 97% floor, and Tier 3 runs at 90.4% on its own traffic. A system whose only states are *answered* and *sent to a human* must then either sum known-bad numbers into monthly totals or queue a quarter of the ledger.

`provisional` is the third state. The label is written, shown, and pre-fills a one-tap confirmation; it is excluded from every total until something independent backs it. `npm run analyze:provisional` swept twelve rules over two dimensions — a floor on independent sightings, and which tiers may certify their own answers — and selected on validation, scored once on the holdout:

| | counted in totals | precision |
|---|---|---|
| sum every pipeline answer (the old boolean) | 100% | 98.4% |
| **selected rule** | **85.4%** | **99.7%** |

**14.6 points of coverage in the totals, for 1.3 points of precision on the numbers a user actually reads.**

Three things the measurement settled that the design got wrong:

- **The tier matters; the count barely does.** Moving the support floor from 1 to 2 changed the counted set by one row out of 546. Almost the whole effect came from *which tiers may self-certify* — a dimension a support floor cannot express, and a support floor was the obvious first design. The rule reduces to: exact merchant matches and human confirmations count, similarity and the model do not.
- **The 97% floor decides nothing here, so the objective had to change.** All twelve candidates clear it. A coverage-maximising rule would take the loosest and book 7 wrong totals to gain 3.8 points — the same slack-floor failure that twice selected a tier gate which regressed the holdout. Errors are ordered first instead, with coverage as the tiebreak, because the fallback is one tap rather than a paid escalation.
- **Automatic promotion barely works, and that was the plan's premise.** The design assumed provisional rows would be lifted out as evidence accumulated. Measured: **4 rows of 221.** The reason is structural — a merchant only produces provisional rows when the pipeline *answers* it, and answering it is exactly why the user is never asked about it, so evidence never arrives on its own. The backlog is drained by taps, not by waiting. Because promotion is per merchant-and-category, clearing 221 rows costs 170 taps rather than 221 — a real but modest 1.3 rows per tap.

---

## Known problems

**Tier 2's 97.7% is a point estimate over 129 answers.** The 95% lower bound is 93.4%, and the same setting scored 89.1% on the validation split. It clears the floor as an observation, not as a guarantee — the honest width of these estimates at this corpus size is several points, and a bigger holdout is what would settle it.

**Tier 3's precision is denominator-sensitive and the gate can false-alarm on it.** Retuning Tier 2 dropped `tiers.llm.precisionPct` from 91.89% to 90.43% while the model's absolute error count stayed at exactly 9 — Tier 2 had absorbed 17 transactions it was answering, all 17 correctly. Any commit that moves a gate needs error *counts* checked before a per-tier alarm is believed. Documented in `evals/gate.ts` rather than fixed, because the alarm is worth keeping.

**Neither tier's gate can reach 97% on the answers the correction loop creates.** Best achievable is 92.3% (Tier 1, over 182 answers) and 95.2% (Tier 2, over 83). On a corpus with genuine churn those answers are the hard residue by construction; no threshold makes them 97% precise — only a different mechanism would. That mechanism is plan item B.

**The cost-aware routing claim is unproven.** "Haiku for bulk, Sonnet for reasoning" needs both models scored on identical transactions, and Sonnet 5 and Opus 5 both return HTTP 429 from the development environment. Confirmed as an access restriction, not rate limiting.

**No hosted embedding model is wired up.** Tier 2's vectors are a fitted TF-IDF over word and character-trigram features — offline, deterministic, zero cost. Deliberate ordering: this is the number a paid model must beat. The `Embedder` interface is the seam.

**Prompt caching is declared and does nothing.** The shared system prompt is ~2,900 tokens against Haiku 4.5's 4,096-token minimum cacheable prefix. `cache_control` is accepted, no error raised, `cache_creation_input_tokens` stays 0. Recorded per-model in the price table because the failure is otherwise silent.

**A coverage-maximising selection rule drifts permissive.** It has twice selected settings that cleared the floors on validation and regressed the holdout. When the floors cannot bind, the objective is wrong rather than merely loose.

**Promotion-by-evidence is nearly inert: 4 rows of 221.** The provisional backlog is cleared by the user tapping confirm, not by the system settling it — so the one-tap UI is the mechanism, not a convenience on top of it. Until that UI exists, 14.6% of transactions stay out of totals with no way to clear them.

**The eval does not score the budget-facing numbers.** `npm run eval` reports cascade coverage and precision; counted-in-totals and its precision come from `analyze:provisional` and are not in the regression gate. The number closest to what a user experiences is the one CI does not protect.

**The review queue is 11.6% and nothing has moved it.** Tier 3 receives 269 escalations and answers 94; the rest fail its 0.90 confidence gate. This is now the largest single bucket of unresolved work, and finding #5 closed off the cheapest idea for reducing it.

**Persona savings rate is 1.0%**, down from 3.2%, after adding churn merchants. Low but solvent, and flagged rather than tuned to a target.

---

## Plan — the next four work items

Ordered by effort-to-evidence ratio. Each has an acceptance test stated up front, because the selection-rule episode showed what happens when it isn't.

### A. Tier 2 precision (the top open item), three angles in cost order

1. ~~**Tighten the gate and price it.**~~ ✅ **Done — see finding #4.** The answer was that tightening is strictly counterproductive on this score, and reshaping the gate onto the factor that is actually ordered took Tier 2 from 89.3% to **97.7% precision at 8.5% share for $0.5937 per 1,000** — better on all three axes. Delivered `npm run analyze:tier2`, which prices the frontier against real model responses rather than a mean call cost.
2. ~~**Contested-neighbourhood hybrid.**~~ ❌ **Done — measured and rejected, see finding #5.** Handing Tier 2's vote to the model as an explicit prior was worth 95 correct answers out of 109 against the control's 96, at matched coverage on the holdout. The neighbour keys were already in the prompt; summarising them added nothing. Delivered `npm run analyze:prior` and the committed responses for both arms, so the negative result is reproducible rather than asserted.

3. **Hosted embedding model.** TF-IDF over character trigrams is exactly what confuses `PRESIDIO DENTAL` with `PRESIDIO VETERINARY` — the `Embedder` interface and disk cache are the prepared seam. **Blocked on access:** the dev environment's proxy injects Anthropic credentials only; Voyage/OpenAI need their own keys and egress. Built behind the cache when unblocked so the eval stays free. *Accept if:* it beats the lexical baseline on the same corpus — and note the bar moved, from 89.3% to **97.7%**, because the gate fix took most of what a better embedder was expected to deliver. The remaining case for it is coverage, not precision.

### B. ~~The correction-loop residue gets a mechanism, not a threshold~~ ✅ Done — see finding #6

Shipped: a `category_status` column (`provisional` | `confirmed`), a `confirmed_count` on the merchant store that only independent evidence can move, promotion scoped to merchant-and-category, and `totalsByCategory` filtering to confirmed rows so no caller has to remember to. Selected by `npm run analyze:provisional`, and the selected rule is re-run through the real `Ledger` against Postgres and asserted to match the sweep — a rule chosen against a simulation is only as good as the simulation.

**Result: 85.4% of transactions counted in totals at 99.7% precision, against 98.4% for summing everything.** The acceptance test (≥97% on counted rows, provisional excluded) passes.

Carried forward: the one-tap confirm surface. Finding #6 showed automatic promotion clears 4 rows of 221, so that UI is the mechanism rather than a convenience — it belongs with the Next.js work, and the ledger query it needs (`reviewQueue`, which now distinguishes *questions* from *proposals*) is already there.

### C. Fix the selection objective, not just its floors

"Maximise coverage subject to floors" has twice selected settings that passed validation and regressed the holdout — classic overfitting to the split. Two changes, together:

- **Margin:** select against 97.5% on validation to target 97% deployed.
- **Objective:** switch to *expected cost per resolved transaction*, with a wrong answer priced at a large synthetic multiple of an escalation. The system already believes errors are expensive; the formula should too — permissive settings then penalise themselves.

*Accept if:* the new objective, run against history, rejects both settings that previously cleared the floors and regressed the holdout. If it wouldn't have caught them, it isn't fixed.

Item #4 supplies a third test case, and an easier one: the objective must *accept* the 0.30/0.90 gate, which beat the incumbent on coverage, precision and cost simultaneously. Any objective monotone in the three does. A rule that rejects it is broken in the opposite direction, and checking both signs is cheaper than discovering the second failure later.

### D. Unblock prompt caching by crossing the 4,096 floor — with static tokens only

The system prompt is ~2,900 tokens against Haiku 4.5's 4,096-token minimum cacheable prefix, so `cache_control` is silently inert. Padding it over the line could improve cost *and* accuracy together — the rare non-tradeoff — but the padding must be **static**: caching is prefix-match, so the per-transaction neighbour examples cannot move into the prefix without destroying the cache they're meant to fill. The padding is per-category canonical exemplars, fixed across every call; neighbour few-shots stay in the volatile suffix. Ships as `PROMPT_VERSION = 'v2'`.

*Accept if:* `cache_creation_input_tokens` goes non-zero, `cache_read_input_tokens` dominates on the second call onward, and Tier 3 precision does not regress (improvement is the hope, non-regression is the gate).

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Taxonomy, seeded fixture corpus, temporal split | ✅ |
| 2 | Four-tier pipeline, eval harness, CI accuracy gate | ✅ |
| 3 | Ledger, correction loop, import, UI, budgets, reporting | ledger, correction loop, label-trust states done |
| 4 | NL → constrained SQL, insight agent, cash-flow forecasting | |
| 5 | Multimodal statement ingestion, cost/observability dashboard | |

**Next up:** the four plan items above (Tier 2 precision first), then CSV/OFX import, the Next.js UI over this ledger, budgets and reporting.

---

## Commands

```bash
npm run eval                 # scored run + regression gate (this is what CI enforces)
npm run eval -- --write-baseline

npm run analyze:normalizer   # Tier 0: collapse ratio, collisions
npm run analyze:memory       # Tier 1: coverage/precision sweep
npm run analyze:knn          # Tier 2: k and gate selection, calibration
npm run analyze:tier2        # Tier 2: the gate frontier, priced in real model calls
npm run analyze:prior        # Tier 2's vote as a Tier 3 prior — the negative result
npm run analyze:provisional  # which labels a budget total may include
npm run analyze:llm          # Tier 3: cost, latency, calibration, routing
npm run analyze:gate         # gate selection against a write-back replay (--tier=2)
npm run analyze:learning     # what the correction loop is actually worth

npm run generate:data        # deterministic — regeneration is byte-identical
npm test                     # 106 tests
npm run typecheck
```

Local dev and CI run Postgres via PGlite — real Postgres semantics, no server, no credentials, same migrations that deploy to Neon.
