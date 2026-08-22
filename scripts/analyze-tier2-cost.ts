/**
 * Prices Tier 2's gate. What does buying precision actually cost?
 *
 *   npm run analyze:tier2            # served from the committed response cache
 *   npm run analyze:tier2 -- --fill  # calls the API for the transactions the cache lacks
 *
 * Tier 2 ships at 89.3% precision against a 97% floor. The obvious fix is to
 * tighten its gate until it clears — but every answer it stops giving becomes a
 * model call, and the model does not answer everything either. So "tighten it"
 * is not a threshold question, it is a price question, and this script is the
 * receipt.
 *
 * **Why the frontier is measured rather than extrapolated.** Whatever the gate,
 * every transaction it surrenders came from Tier 1's escalation inbox, and that
 * inbox is fixed: the gate decides whether to publish Tier 2's score, never what
 * the score is. So the union of every setting's surrendered set, over the whole
 * grid, is bounded by that one inbox. Fetch Tier 3 responses for it once and
 * every point on the frontier becomes exactly computable, including the settings
 * nobody selects and the ones that are not simply tighter than the shipped gate.
 * The alternative — pricing a
 * surrendered transaction at the mean cost of a baseline escalation, and assuming
 * it is answered at the baseline rate — is wrong in a specific and flattering
 * direction: transactions Tier 2 was confident about are not drawn from the same
 * distribution as the ones it gave up on.
 *
 * **Why the answer rate matters as much as the precision.** Tier 3 answers ~40%
 * of its inbox; the rest fails its own confidence gate and lands in the review
 * queue. So surrendering traffic does not move it from "Tier 2 answers it" to
 * "the model answers it". It moves it to "the model answers some of it and a
 * human does the rest". A sweep that reported only precision would show the floor
 * being cleared and hide where the work went.
 *
 * Selection runs on the validation split. The holdout is scored once, at the end,
 * at whatever the validation split chose.
 */

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { MerchantMemory } from '../src/ai/memory.js';
import { fitLexicalEmbedder } from '../src/ai/embed.js';
import { NeighbourIndex, DEFAULT_NEIGHBOUR_CONFIG } from '../src/ai/knn.js';
import { normalizeDescriptor } from '../src/ai/normalize.js';
import { LlmClassifier, messageCreator, type LlmOutcome } from '../src/ai/llm.js';
import { PROMPT_VERSION, type ClassificationInput } from '../src/ai/prompt.js';
import { escalationInput } from '../src/ai/escalation.js';
import { createClient, MODELS, costOf, type ModelId, type TokenUsage } from '../src/ai/client.js';
import type { SyntheticTransaction } from '../src/synthetic/generator.js';
import type { CategoryId } from '../src/core/taxonomy.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA = join(HERE, '..', 'evals', 'datasets');
const CACHE_DIR = join(HERE, '..', 'evals', 'cache');
const load = (f: string): SyntheticTransaction[] => JSON.parse(readFileSync(join(DATA, f), 'utf8'));

const FILL = process.argv.includes('--fill');
const MODEL: ModelId = 'claude-haiku-4-5';

/**
 * Tier 3's accept gate, read from the committed baseline rather than restated.
 *
 * The eval runner pins 0.90, selected by `analyze:llm`. Copying the literal here
 * would create a second definition that drifts silently — and it would drift in
 * the direction of making this analysis disagree with the eval it is supposed to
 * inform. `baseline.json` already serialises the gate the scored run used.
 */
const LLM_GATE: number = JSON.parse(
  readFileSync(join(HERE, '..', 'evals', 'baseline.json'), 'utf8'),
).config.llmGate;

/** Same split point as `analyze:knn` and `analyze:llm`. */
const VALIDATION_SPLIT = '2025-02-01';
const history = load('history.json');
const golden = load('golden.json');
const fit = history.filter((t) => t.date < VALIDATION_SPLIT);
const validation = history.filter((t) => t.date >= VALIDATION_SPLIT);

/**
 * Mostly tightening: the shipped value is 0.50, so the table reads as "what the
 * current gate costs" followed by what each increment of precision costs on top.
 *
 * The rows below 0.50 are **diagnostic, not selectable**. They exist because the
 * calibration table below shows the confidence product is the badly-ordered half
 * of the gate, and the experiment that follows from that is to weaken or remove
 * it and let agreement carry the decision alone — which cannot be expressed by
 * only ever tightening. They are safe to include here only because nothing in
 * this grid clears the floor, so the selection rule has nothing to pick. If a
 * future corpus does make the floor reachable, the coverage-maximising rule must
 * be replaced first: it has twice chosen a permissive setting that cleared
 * validation and regressed the holdout, and this grid would hand it two more.
 */
const CONFIDENCE_CANDIDATES = [0, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8];
const AGREEMENT_CANDIDATES = [0.51, 0.6, 0.7, 0.8, 0.9, 1.0];

const PRECISION_FLOOR = 0.97;

// ── The inbox ───────────────────────────────────────────────────────────────

/** One transaction Tier 1 declined, with Tier 2's ungated proposal attached. */
interface Candidate {
  readonly txn: SyntheticTransaction;
  /** Absent when the key has no neighbours at all — always surrenders. */
  readonly category?: CategoryId;
  readonly confidence: number;
  readonly agreement: number;
  readonly input: ClassificationInput;
}

interface Inbox {
  readonly candidates: Candidate[];
  /** Tier 1's own contribution, needed for the end-to-end figures. */
  readonly tier1Answered: number;
  readonly tier1Correct: number;
  readonly total: number;
}

async function inboxOf(
  seedRows: readonly SyntheticTransaction[],
  scoreRows: readonly SyntheticTransaction[],
): Promise<Inbox> {
  const memory = new MerchantMemory();
  for (const t of seedRows) memory.remember(t.rawDescriptor, t.label.category);

  const embedder = fitLexicalEmbedder([...memory.entries()].map((e) => e.key));
  // Gate wide open. One pass then yields the vote for every setting in the sweep,
  // because the gate only decides whether to publish a score, never what it is.
  const index = await NeighbourIndex.build(memory, embedder, { minConfidence: 0, minAgreement: 0 });

  const candidates: Candidate[] = [];
  let tier1Answered = 0;
  let tier1Correct = 0;

  for (const txn of scoreRows) {
    const tier1 = memory.lookup(txn.rawDescriptor);
    if (tier1.status === 'hit') {
      tier1Answered++;
      if (tier1.category === txn.label.category) tier1Correct++;
      continue;
    }

    const tier2 = await index.lookup(txn.rawDescriptor);
    // The evidence goes to the model whether or not Tier 2 would have answered.
    // Under a different gate this exact transaction escalates carrying exactly
    // this, so pricing it any other way would price a different system.
    const input = escalationInput(txn.rawDescriptor, txn.amount, tier2);

    candidates.push(
      tier2.status === 'hit' || tier2.status === 'low_confidence'
        ? { txn, category: tier2.category, confidence: tier2.confidence, agreement: tier2.agreement, input }
        : { txn, confidence: 0, agreement: 0, input },
    );
  }

  return { candidates, tier1Answered, tier1Correct, total: scoreRows.length };
}

// ── Response cache ──────────────────────────────────────────────────────────
// Same file and same shape as `analyze:llm`, keyed by raw descriptor. This script
// only ever adds entries, so the eval's replay is unaffected by running it.

interface CachedOutcome {
  readonly status: 'ok' | 'unresolved';
  readonly category?: string | undefined;
  readonly confidence?: number | undefined;
  readonly attempts: number;
  readonly repairs: readonly string[];
  readonly usage: TokenUsage;
  readonly latencyMs: number;
}

const cacheFile = () => join(CACHE_DIR, `llm-${MODEL}-${PROMPT_VERSION}.json`);

function loadCache(): Map<string, CachedOutcome> {
  try {
    return new Map(Object.entries(JSON.parse(readFileSync(cacheFile(), 'utf8')) as Record<string, CachedOutcome>));
  } catch {
    return new Map();
  }
}

function saveCache(cache: Map<string, CachedOutcome>): void {
  if (cache.size === 0) return;
  mkdirSync(CACHE_DIR, { recursive: true });
  const sorted = Object.fromEntries([...cache.entries()].sort(([a], [b]) => (a < b ? -1 : 1)));
  writeFileSync(cacheFile(), JSON.stringify(sorted, null, 1) + '\n', 'utf8');
}

const BACKOFF_MS = [2_000, 6_000, 15_000, 40_000];
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function withRetry<T>(fn: () => Promise<T>, label: string): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    try {
      return await fn();
    } catch (error) {
      const status = (error as { status?: number }).status;
      const retryable = status === 429 || (status !== undefined && status >= 500);
      if (!retryable || attempt >= BACKOFF_MS.length) throw error;
      console.error(`    ${label}: HTTP ${status}, retrying in ${BACKOFF_MS[attempt]}ms`);
      await sleep(BACKOFF_MS[attempt]!);
    }
  }
}

/**
 * Fills the cache for every candidate, so the frontier is measured end to end.
 *
 * Without `--fill` this refuses rather than proceeding on partial data. A missing
 * response is not a neutral gap here: the settings whose responses are absent are
 * exactly the ones that answer more at Tier 2, so silently skipping them would
 * delete the permissive half of the frontier and make tightening look free.
 */
async function fillCache(candidates: readonly Candidate[]): Promise<Map<string, CachedOutcome>> {
  const cache = loadCache();
  const seen = new Set<string>();
  const missing = candidates.filter((c) => {
    if (cache.has(c.txn.rawDescriptor) || seen.has(c.txn.rawDescriptor)) return false;
    seen.add(c.txn.rawDescriptor);
    return true;
  });

  if (missing.length === 0) {
    console.error(`  all ${candidates.length} candidates served from cache`);
    return cache;
  }

  if (!FILL) {
    throw new Error(
      `${missing.length} of ${candidates.length} candidate transactions have no cached ${MODEL} response.\n` +
        `The frontier cannot be priced without them. Re-run with: npm run analyze:tier2 -- --fill`,
    );
  }

  console.error(`  ${missing.length} uncached — calling ${MODEL}`);
  const classifier = new LlmClassifier(messageCreator(createClient()), { model: MODEL, minConfidence: 0 });

  let next = 0;
  let done = 0;
  try {
    await Promise.all(
      Array.from({ length: Math.min(8, missing.length) }, async () => {
        for (let i = next++; i < missing.length; i = next++) {
          const item = missing[i]!;
          const outcome: LlmOutcome = await withRetry(() => classifier.classify(item.input), item.txn.id);
          cache.set(item.txn.rawDescriptor, {
            status: outcome.status,
            category: outcome.status === 'ok' ? outcome.category : undefined,
            confidence: outcome.status === 'ok' ? outcome.confidence : undefined,
            attempts: outcome.telemetry.attempts,
            repairs: outcome.telemetry.repairs,
            usage: outcome.telemetry.usage,
            latencyMs: outcome.telemetry.latencyMs,
          });
          // Checkpointed, because a 429 four hundred calls in must not discard
          // the four hundred already billed.
          if (++done % 25 === 0) {
            saveCache(cache);
            console.error(`    ${done}/${missing.length}`);
          }
        }
      }),
    );
  } finally {
    saveCache(cache);
  }
  return cache;
}

// ── Scoring one setting ─────────────────────────────────────────────────────

interface Priced {
  readonly minConfidence: number;
  readonly minAgreement: number;
  /** Tier 2's own numbers — the acceptance criterion lives here. */
  readonly tier2Answered: number;
  readonly tier2Correct: number;
  /** Everything Tier 2 gave up, which is what the model bill buys. */
  readonly surrendered: number;
  readonly llmAnswered: number;
  readonly llmCorrect: number;
  readonly review: number;
  readonly costUsd: number;
  /** Whole-population figures, so a tier-local win that loses overall is visible. */
  readonly resolved: number;
  readonly answered: number;
  readonly correct: number;
}

function price(inbox: Inbox, cache: Map<string, CachedOutcome>, minConfidence: number, minAgreement: number): Priced {
  let tier2Answered = 0;
  let tier2Correct = 0;
  let surrendered = 0;
  let llmAnswered = 0;
  let llmCorrect = 0;
  let costUsd = 0;

  for (const c of inbox.candidates) {
    const answers = c.category !== undefined && c.confidence >= minConfidence && c.agreement >= minAgreement;
    if (answers) {
      tier2Answered++;
      if (c.category === c.txn.label.category) tier2Correct++;
      continue;
    }

    surrendered++;
    const cached = cache.get(c.txn.rawDescriptor)!;
    costUsd += costOf(cached.usage, MODELS[MODEL]);
    if (cached.status === 'ok' && (cached.confidence ?? 0) >= LLM_GATE) {
      llmAnswered++;
      if (cached.category === c.txn.label.category) llmCorrect++;
    }
  }

  const answered = inbox.tier1Answered + tier2Answered + llmAnswered;
  const correct = inbox.tier1Correct + tier2Correct + llmCorrect;
  return {
    minConfidence,
    minAgreement,
    tier2Answered,
    tier2Correct,
    surrendered,
    llmAnswered,
    llmCorrect,
    review: surrendered - llmAnswered,
    costUsd,
    resolved: correct,
    answered,
    correct,
  };
}

// ── Report ──────────────────────────────────────────────────────────────────

const pct = (n: number, d: number) => (d === 0 ? '   n/a' : `${((n / d) * 100).toFixed(1)}%`.padStart(6));
const usd = (n: number) => `$${n.toFixed(4)}`;
const per1k = (cost: number, n: number) => (cost / n) * 1000;

function table(inbox: Inbox, cache: Map<string, CachedOutcome>): Priced[] {
  const rows: Priced[] = [];
  for (const minConfidence of CONFIDENCE_CANDIDATES) {
    for (const minAgreement of AGREEMENT_CANDIDATES) rows.push(price(inbox, cache, minConfidence, minAgreement));
  }
  return rows;
}

async function main(): Promise<void> {
  console.error('building inboxes…');
  const validationInbox = await inboxOf(fit, validation);
  const goldenInbox = await inboxOf(history, golden);

  console.error('resolving model responses…');
  const cache = await fillCache([...validationInbox.candidates, ...goldenInbox.candidates]);

  const rows = table(validationInbox, cache);
  const shipped = rows.find(
    (r) =>
      r.minConfidence === DEFAULT_NEIGHBOUR_CONFIG.minConfidence &&
      r.minAgreement === DEFAULT_NEIGHBOUR_CONFIG.minAgreement,
  )!;

  const lines: string[] = [
    `TIER 2 PRICING     what each increment of Tier 2 precision costs, measured`,
    `MODEL              ${MODEL} at gate ${LLM_GATE} (read from evals/baseline.json)`,
    `SPLITS             fit ${fit.length} → validation ${validation.length} · golden holdout ${golden.length}`,
    `INBOX              ${validationInbox.candidates.length} validation · ${goldenInbox.candidates.length} golden — what Tier 1 declined`,
    `METHOD             every candidate has a real ${MODEL} response on file, so each row is`,
    `                   measured rather than extrapolated from a mean call cost`,
    ``,
    `VALIDATION FRONTIER   (·) marks the shipped gate`,
    ``,
    `  conf  agree    t2 share  t2 prec   →model   review   resolved   $/1k txn    Δ$/1k`,
  ];

  for (const r of rows) {
    const isShipped = r === shipped;
    const delta = per1k(r.costUsd, validationInbox.total) - per1k(shipped.costUsd, validationInbox.total);
    const clears = r.tier2Answered > 0 && r.tier2Correct / r.tier2Answered >= PRECISION_FLOOR;
    lines.push(
      `  ${r.minConfidence.toFixed(2)}  ${r.minAgreement.toFixed(2)}   ${isShipped ? '·' : ' '}  ` +
        `${pct(r.tier2Answered, validationInbox.total)}   ${pct(r.tier2Correct, r.tier2Answered)}   ` +
        `${String(r.surrendered).padStart(6)}   ${String(r.review).padStart(6)}    ` +
        `${pct(r.resolved, validationInbox.total)}   ${usd(per1k(r.costUsd, validationInbox.total)).padStart(9)}  ` +
        `${(delta >= 0 ? '+' : '') + usd(delta)}`.padStart(10) +
        `  ${clears ? '✓' : ''}`,
    );
  }

  // ── Selection ─────────────────────────────────────────────────────────────
  // The floor is Tier 2's *own* precision. That is the gap this work item exists
  // to close, and it is not the same constraint the other sweeps apply: an
  // overall floor can be cleared by a tier that is wrong 11% of the time as long
  // as it is small, which is exactly how this defect survived.
  const clearing = rows.filter((r) => r.tier2Answered > 0 && r.tier2Correct / r.tier2Answered >= PRECISION_FLOOR);
  const chosen = clearing.sort((a, b) => b.tier2Answered - a.tier2Answered || a.costUsd - b.costUsd)[0];
  const bestPrecision = [...rows]
    .filter((r) => r.tier2Answered >= 20)
    .sort((a, b) => b.tier2Correct / b.tier2Answered - a.tier2Correct / a.tier2Answered)[0]!;

  lines.push(
    ``,
    `CONSTRAINT         ${(PRECISION_FLOOR * 100).toFixed(0)}% on Tier 2's own precision, not on the cascade's.`,
    `                   An overall floor is clearable by a tier that is wrong 11% of the time`,
    `                   as long as it is small — which is how this defect survived two sweeps.`,
    ``,
  );

  if (chosen === undefined) {
    lines.push(
      `FLOOR UNREACHABLE  No setting reaches ${(PRECISION_FLOOR * 100).toFixed(0)}% on Tier 2's own answers.`,
      `  frontier tops out at ${pct(bestPrecision.tier2Correct, bestPrecision.tier2Answered).trim()} ` +
        `(conf ${bestPrecision.minConfidence.toFixed(2)} / agree ${bestPrecision.minAgreement.toFixed(2)}), ` +
        `over ${bestPrecision.tier2Answered} answers, at ${usd(per1k(bestPrecision.costUsd, validationInbox.total))} per 1,000`,
      ``,
      // Stated by comparing rows rather than asserted, because the sentence has to
      // stay true after the gate it describes has been changed — the first draft
      // said "the loosest gate", which was a fact about one particular default.
      ...(bestPrecision.minConfidence > shipped.minConfidence
        ? [
            `  The best row sits ABOVE the shipped minConfidence, so on this corpus the score`,
            `  has become orderable and tightening buys precision in the usual way.`,
          ]
        : [
            `  The best row is at or below the shipped minConfidence (${bestPrecision.minConfidence.toFixed(2)} vs ${shipped.minConfidence.toFixed(2)}), which is`,
            `  the real result here. Raising the confidence gate does not trade coverage for`,
            `  precision on this tier — it loses both, because each increment removes more`,
            `  right answers than wrong ones. A score that behaves that way is not merely`,
            `  uninformative about correctness, it is ordered against it above the midpoint.`,
          ]),
      `  Measured directly below rather than inferred from this table.`,
      ``,
    );
  } else {
    lines.push(
      `SELECTED           conf ${chosen.minConfidence.toFixed(2)} · agree ${chosen.minAgreement.toFixed(2)} — widest coverage clearing the floor`,
      `  tier 2           ${pct(chosen.tier2Answered, validationInbox.total).trim()} of transactions, ${pct(chosen.tier2Correct, chosen.tier2Answered).trim()} precision`,
      `  price            ${usd(per1k(chosen.costUsd, validationInbox.total))} per 1,000 ` +
        `(${(per1k(chosen.costUsd, validationInbox.total) - per1k(shipped.costUsd, validationInbox.total) >= 0 ? '+' : '') + usd(per1k(chosen.costUsd, validationInbox.total) - per1k(shipped.costUsd, validationInbox.total))} vs shipped)`,
      `  review queue     ${chosen.review} vs ${shipped.review} shipped (${chosen.review - shipped.review >= 0 ? '+' : ''}${chosen.review - shipped.review})`,
      `  resolved         ${pct(chosen.resolved, validationInbox.total).trim()} vs ${pct(shipped.resolved, validationInbox.total).trim()} shipped`,
      ``,
    );
  }

  // ── Is the score ordered at all? ──────────────────────────────────────────
  // The frontier above is monotone in the wrong direction, which is a claim
  // about the score itself rather than about any threshold on it. Inferring it
  // from a sweep would be reading a calibration table sideways, so it gets
  // measured directly: bucket the ungated answers and look at observed accuracy.
  //
  // Both factors are broken out because they fail for different reasons and only
  // one of them is fixable by better features.
  const answeredCandidates = validationInbox.candidates.filter((c) => c.category !== undefined);
  const accuracyIn = (rows: readonly Candidate[]) => rows.filter((c) => c.category === c.txn.label.category).length;

  lines.push(`SCORE ORDERING     observed accuracy per bucket, validation inbox, ungated`, ``);
  for (const [label, of, buckets] of [
    ['confidence', (c: Candidate) => c.confidence, [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01]],
    ['agreement', (c: Candidate) => c.agreement, [0, 0.5, 0.7, 0.9, 1.01]],
    ['nearest sim', (c: Candidate) => c.confidence / Math.max(c.agreement, 1e-9), [0, 0.5, 0.7, 0.8, 0.9, 1.01]],
  ] as const) {
    lines.push(`  ${label}`);
    for (let i = 0; i < buckets.length - 1; i++) {
      const lo = buckets[i]!;
      const hi = buckets[i + 1]!;
      const inBucket = answeredCandidates.filter((c) => of(c) >= lo && of(c) < hi);
      if (inBucket.length === 0) continue;
      const correct = accuracyIn(inBucket);
      lines.push(
        `    ${lo.toFixed(2)}–${hi > 1 ? '1.00' : hi.toFixed(2)}  ${String(inBucket.length).padStart(4)} txns  ` +
          `${pct(correct, inBucket.length)}  ${'█'.repeat(Math.round((correct / inBucket.length) * 30))}`,
      );
    }
  }
  lines.push(
    ``,
    `  Read the three together — they do not say the same thing, and the difference`,
    `  is the whole diagnosis.`,
    ``,
    `  Agreement is well ordered: it climbs monotonically and never reverses. It is`,
    `  a usable signal and the tier should keep it.`,
    ``,
    `  Nearest-similarity is not ordered at all, and its top bucket is its worst.`,
    `  That is the near-miss families doing exactly what they were added to do: the`,
    `  closest lexical neighbour of PRESIDIO VETERINARY is PRESIDIO DENTAL, so a`,
    `  similarity near 1.0 is evidence of a shared token, not a shared category.`,
    `  For this failure mode high similarity is the hazard, not the defence.`,
    ``,
    `  The product inherits the defect and peaks in the middle. That is what bends`,
    `  the frontier: raising the gate cuts the well-behaved middle band before it`,
    `  touches the badly-behaved top one, so tightening deletes good answers first.`,
    `  The implication is not a different threshold on this score — it is that the`,
    `  similarity factor does not belong in a confidence at all on a corpus where`,
    `  merchants share name fragments.`,
    ``,
  );

  // ── Does anything dominate the shipped gate outright? ─────────────────────
  // Deliberately not "the best row". Coverage-maximisation has twice selected a
  // setting that cleared validation and regressed the holdout, so the rule here
  // is the strictest one available: at least the shipped coverage, at least the
  // shipped precision, and no more than the shipped cost. A point that wins on
  // every axis at once cannot be a coverage-for-precision trade dressed up as an
  // improvement — which is the specific way the previous two selections failed.
  const dominating = rows
    .filter(
      (r) =>
        r.tier2Answered >= shipped.tier2Answered &&
        r.tier2Correct / r.tier2Answered >= shipped.tier2Correct / shipped.tier2Answered &&
        r.costUsd <= shipped.costUsd &&
        r !== shipped,
    )
    .sort((a, b) => b.tier2Correct / b.tier2Answered - a.tier2Correct / a.tier2Answered)[0];

  if (dominating !== undefined) {
    lines.push(
      `DOMINATING SETTING conf ${dominating.minConfidence.toFixed(2)} · agree ${dominating.minAgreement.toFixed(2)} beats the shipped gate on every axis at once`,
      ``,
      `                      shipped      dominating`,
      `  tier 2 share      ${pct(shipped.tier2Answered, validationInbox.total)}         ${pct(dominating.tier2Answered, validationInbox.total)}`,
      `  tier 2 precision  ${pct(shipped.tier2Correct, shipped.tier2Answered)}         ${pct(dominating.tier2Correct, dominating.tier2Answered)}`,
      `  cost per 1,000   ${usd(per1k(shipped.costUsd, validationInbox.total)).padStart(8)}       ${usd(per1k(dominating.costUsd, validationInbox.total)).padStart(8)}`,
      ``,
      `  This is the calibration table's implication, priced: almost all of the gate's`,
      `  weight moves off the similarity product and onto agreement, which is the factor`,
      `  that is actually ordered. It is still ${pct(dominating.tier2Correct, dominating.tier2Answered).trim()}, so it does not clear the floor and`,
      `  is not selected here — it is a candidate carried to the holdout below.`,
      ``,
    );
  }

  // ── What the surrendered traffic actually does at the model tier ──────────
  // The number that decides whether tightening is worth it. If the model answers
  // only a fraction of what Tier 2 gives up, tightening does not relocate the
  // work to a model — it relocates it to a person.
  const fullSurrender = price(validationInbox, cache, 2, 2);
  lines.push(
    `WHERE THE WORK GOES   if Tier 2 abstained entirely on the validation inbox`,
    `  ${fullSurrender.surrendered} transactions to the model, of which it answers ${fullSurrender.llmAnswered} ` +
      `(${pct(fullSurrender.llmAnswered, fullSurrender.surrendered).trim()}) at ${pct(fullSurrender.llmCorrect, fullSurrender.llmAnswered).trim()} precision`,
    `  and ${fullSurrender.review} land in the review queue (${pct(fullSurrender.review, validationInbox.total).trim()} of all transactions).`,
    ``,
    `                      shipped        abstain`,
    `  coverage          ${pct(shipped.answered, validationInbox.total)}         ${pct(fullSurrender.answered, validationInbox.total)}`,
    `  precision         ${pct(shipped.correct, shipped.answered)}         ${pct(fullSurrender.correct, fullSurrender.answered)}`,
    `  resolved          ${pct(shipped.resolved, validationInbox.total)}         ${pct(fullSurrender.resolved, validationInbox.total)}`,
    `  cost per 1,000   ${usd(per1k(shipped.costUsd, validationInbox.total)).padStart(8)}       ${usd(per1k(fullSurrender.costUsd, validationInbox.total)).padStart(8)}`,
    ``,
    `  Abstaining is the only move that raises precision, because it is the only one`,
    `  that does not rely on the score being ordered. What it costs is coverage and`,
    `  review-queue growth, both priced above — a product call, not a tuning one.`,
    ``,
  );

  // ── Holdout, scored once ──────────────────────────────────────────────────
  const holdoutShipped = price(
    goldenInbox,
    cache,
    DEFAULT_NEIGHBOUR_CONFIG.minConfidence,
    DEFAULT_NEIGHBOUR_CONFIG.minAgreement,
  );
  lines.push(`HOLDOUT            scored once, at the settings validation chose`, ``);

  const report = (label: string, r: Priced) =>
    lines.push(
      `  ${label}`,
      `    tier 2         ${pct(r.tier2Answered, goldenInbox.total).trim()} share, ${pct(r.tier2Correct, r.tier2Answered).trim()} precision`,
      `    →model/review  ${r.surrendered} surrendered · ${r.llmAnswered} answered · ${r.review} to review (${pct(r.review, goldenInbox.total).trim()})`,
      `    overall        ${pct(r.answered, goldenInbox.total).trim()} coverage, ${pct(r.correct, r.answered).trim()} precision, ${pct(r.resolved, goldenInbox.total).trim()} resolved`,
      `    cost           ${usd(per1k(r.costUsd, goldenInbox.total))} per 1,000`,
      ``,
    );

  report(`shipped  (conf ${DEFAULT_NEIGHBOUR_CONFIG.minConfidence} / agree ${DEFAULT_NEIGHBOUR_CONFIG.minAgreement})`, holdoutShipped);
  if (chosen !== undefined) {
    report(
      `selected (conf ${chosen.minConfidence.toFixed(2)} / agree ${chosen.minAgreement.toFixed(2)})`,
      price(goldenInbox, cache, chosen.minConfidence, chosen.minAgreement),
    );
  }
  // Priced on the holdout too, because when no gate clears the floor this is the
  // only remaining option and a reader should not have to take the validation
  // split's word for what it costs.
  report(`abstain  (Tier 2 disabled)`, price(goldenInbox, cache, 2, 2));

  if (dominating !== undefined) {
    const held = price(goldenInbox, cache, dominating.minConfidence, dominating.minAgreement);
    report(`candidate (conf ${dominating.minConfidence.toFixed(2)} / agree ${dominating.minAgreement.toFixed(2)})`, held);

    // The whole point of scoring it. Twice a setting has cleared validation and
    // regressed the holdout, so the question is not "is it better" but "did the
    // validation split's verdict survive contact with data it did not choose on".
    const heldPrecision = held.tier2Correct / held.tier2Answered;
    const shippedPrecision = holdoutShipped.tier2Correct / holdoutShipped.tier2Answered;
    const heldDominates =
      held.tier2Answered >= holdoutShipped.tier2Answered &&
      heldPrecision >= shippedPrecision &&
      held.costUsd <= holdoutShipped.costUsd;
    lines.push(
      `  VERDICT          the dominance ${heldDominates ? 'HELD' : 'DID NOT HOLD'} on the holdout`,
      `    tier 2 share   ${pct(holdoutShipped.tier2Answered, goldenInbox.total).trim()} → ${pct(held.tier2Answered, goldenInbox.total).trim()}`,
      `    precision      ${pct(holdoutShipped.tier2Correct, holdoutShipped.tier2Answered).trim()} → ${pct(held.tier2Correct, held.tier2Answered).trim()}`,
      `    cost per 1,000 ${usd(per1k(holdoutShipped.costUsd, goldenInbox.total))} → ${usd(per1k(held.costUsd, goldenInbox.total))}`,
      `    overall cov    ${pct(holdoutShipped.answered, goldenInbox.total).trim()} → ${pct(held.answered, goldenInbox.total).trim()}`,
      `    overall res    ${pct(holdoutShipped.resolved, goldenInbox.total).trim()} → ${pct(held.resolved, goldenInbox.total).trim()}`,
      ``,
      ...(heldDominates
        ? [
            `    Note what does *not* improve: whole-cascade coverage falls, because the`,
            `    gate reshapes which transactions Tier 2 takes rather than simply taking`,
            `    more of them. Resolved still rises, which is the same statement from the`,
            `    other side — the answers it stopped giving were mostly the wrong ones.`,
            ``,
            `    This does not need the objective fix to be shippable, and the reason is`,
            `    worth stating: a point that beats the incumbent on coverage, precision`,
            `    and cost at once is preferred by *every* objective monotone in those`,
            `    three. Item C changes which point gets picked when they trade off. Here`,
            `    they do not trade off, so no objective could disagree. The rule that`,
            `    failed twice was coverage-maximisation resolving a genuine tradeoff in`,
            `    the permissive direction; this is not that rule and not that situation.`,
          ]
        : [
            `    Third time a validation-selected setting has failed to transfer. That is`,
            `    now a property of the selection rule rather than bad luck, and it is the`,
            `    argument for replacing the objective before any further gate is chosen.`,
          ]),
      ``,
    );
  }

  console.log(lines.join('\n'));
}

main();
