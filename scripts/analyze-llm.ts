/**
 * Measures Tier 3 (model categorisation) and prices the cost-aware routing claim.
 *
 *   npm run analyze:llm              # served from the committed response cache
 *   npm run analyze:llm -- --refresh # re-calls the API and rewrites the cache
 *
 * Two things are being measured, and they are different questions.
 *
 * **Is the tier accurate on its own traffic?** Tier 3 only ever sees what Tiers
 * 1 and 2 declined — the residue after exact match and nearest-neighbour have
 * both given up. That population is harder than the corpus average by
 * construction, so scoring it against the whole holdout would flatter it.
 *
 * **Is the cheap model good enough?** The stack claims cost-aware routing —
 * Haiku for bulk work, Sonnet for reasoning. That is an assertion until the same
 * transactions run through both and the accuracy difference is set against the
 * price difference. Both are scored here, on identical inputs.
 *
 * Responses are cached to disk keyed by model and prompt version, and the cache
 * is committed. An eval that costs money every time it runs is an eval that
 * stops being run; this one reproduces offline, with no API key, from a clean
 * checkout. `--refresh` is the escape hatch when the prompt changes.
 */

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { MerchantMemory } from '../src/ai/memory.js';
import { fitLexicalEmbedder } from '../src/ai/embed.js';
import { NeighbourIndex } from '../src/ai/knn.js';
import { normalizeDescriptor } from '../src/ai/normalize.js';
import { LlmClassifier, messageCreator, type LlmOutcome } from '../src/ai/llm.js';
import { PROMPT_VERSION, type ClassificationInput, type NeighbourExample } from '../src/ai/prompt.js';
import { createClient, MODELS, costOf, addUsage, EMPTY_USAGE, type ModelId, type TokenUsage } from '../src/ai/client.js';
import type { SyntheticTransaction } from '../src/synthetic/generator.js';
import type { CategoryId } from '../src/core/taxonomy.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA = join(HERE, '..', 'evals', 'datasets');
const CACHE_DIR = join(HERE, '..', 'evals', 'cache');
const load = (f: string): SyntheticTransaction[] => JSON.parse(readFileSync(join(DATA, f), 'utf8'));

const REFRESH = process.argv.includes('--refresh');
const MODEL_IDS: ModelId[] = ['claude-haiku-4-5', 'claude-sonnet-5'];

/**
 * Per-model, because the rate limits are per-model. Haiku sustains a wide fan-out;
 * Sonnet 5 starts returning 429 well below it.
 */
const CONCURRENCY: Record<ModelId, number> = {
  'claude-haiku-4-5': 8,
  'claude-sonnet-5': 3,
};

/** Backoff schedule for 429 and 5xx. Fixed rather than jittered so runs are reproducible. */
const BACKOFF_MS = [2_000, 6_000, 15_000, 40_000];

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Retries the transient failures.
 *
 * Without this the first 429 kills the process — and because the cache only
 * persisted at the end, it also discarded every call already paid for. An eval
 * that throws away paid work on a transient error is an eval that gets run once
 * and then quietly abandoned.
 */
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

/** Same split points as Tiers 1 and 2 — the gate is chosen on validation, scored on golden. */
const VALIDATION_SPLIT = '2025-02-01';
const CONFIDENCE_GATE_CANDIDATES = [0, 0.5, 0.6, 0.7, 0.8, 0.9];
const PRECISION_FLOOR = 0.97;

const history = load('history.json');
const golden = load('golden.json');
const fit = history.filter((t) => t.date < VALIDATION_SPLIT);
const validation = history.filter((t) => t.date >= VALIDATION_SPLIT);

const isTransfer = (id: string) => id.startsWith('transfer.');

// ── Cascade: everything Tiers 1 and 2 declined is Tier 3's inbox ─────────────

interface Escalation {
  readonly txn: SyntheticTransaction;
  readonly input: ClassificationInput;
}

interface Cascade {
  readonly escalations: Escalation[];
  /** Answered correctly by Tiers 1+2, for the end-to-end figure. */
  readonly resolved: number;
  readonly resolvedCorrect: number;
}

async function runCascade(
  seedRows: readonly SyntheticTransaction[],
  scoreRows: readonly SyntheticTransaction[],
): Promise<Cascade> {
  const memory = new MerchantMemory();
  for (const t of seedRows) memory.remember(t.rawDescriptor, t.label.category);

  const embedder = fitLexicalEmbedder([...memory.entries()].map((e) => e.key));
  const index = await NeighbourIndex.build(memory, embedder);

  const escalations: Escalation[] = [];
  let resolved = 0;
  let resolvedCorrect = 0;

  for (const txn of scoreRows) {
    const tier1 = memory.lookup(txn.rawDescriptor);
    if (tier1.status === 'hit') {
      resolved++;
      if (tier1.category === txn.label.category) resolvedCorrect++;
      continue;
    }

    const tier2 = await index.lookup(txn.rawDescriptor);
    if (tier2.status === 'hit') {
      resolved++;
      if (tier2.category === txn.label.category) resolvedCorrect++;
      continue;
    }

    const neighbours: NeighbourExample[] =
      tier2.status === 'low_confidence'
        ? tier2.neighbours.map((n) => ({ key: n.key, category: n.category, similarity: n.similarity }))
        : [];

    escalations.push({
      txn,
      input: {
        rawDescriptor: txn.rawDescriptor,
        normalizedKey: normalizeDescriptor(txn.rawDescriptor).key,
        amount: txn.amount,
        neighbours,
      },
    });
  }

  return { escalations, resolved, resolvedCorrect };
}

// ── Response cache ──────────────────────────────────────────────────────────

interface CachedOutcome {
  readonly status: 'ok' | 'unresolved';
  readonly category?: string | undefined;
  readonly confidence?: number | undefined;
  readonly attempts: number;
  readonly repairs: readonly string[];
  readonly usage: TokenUsage;
  readonly latencyMs: number;
}

function cacheFile(model: ModelId): string {
  return join(CACHE_DIR, `llm-${model}-${PROMPT_VERSION}.json`);
}

function loadCache(model: ModelId): Map<string, CachedOutcome> {
  if (REFRESH) return new Map();
  try {
    return new Map(Object.entries(JSON.parse(readFileSync(cacheFile(model), 'utf8')) as Record<string, CachedOutcome>));
  } catch {
    return new Map();
  }
}

function saveCache(model: ModelId, cache: Map<string, CachedOutcome>): void {
  // A model that was never reachable has nothing to record; writing an empty
  // file would commit a cache that looks populated until you open it.
  if (cache.size === 0) return;
  mkdirSync(CACHE_DIR, { recursive: true });
  // Sorted so the committed file has a stable diff.
  const sorted = Object.fromEntries([...cache.entries()].sort(([a], [b]) => (a < b ? -1 : 1)));
  writeFileSync(cacheFile(model), JSON.stringify(sorted, null, 1) + '\n', 'utf8');
}

const toCached = (outcome: LlmOutcome): CachedOutcome => ({
  status: outcome.status,
  category: outcome.status === 'ok' ? outcome.category : undefined,
  confidence: outcome.status === 'ok' ? outcome.confidence : undefined,
  attempts: outcome.telemetry.attempts,
  repairs: outcome.telemetry.repairs,
  usage: outcome.telemetry.usage,
  latencyMs: outcome.telemetry.latencyMs,
});

/** Runs the inbox through one model, filling cache misses with real calls. */
async function classifyAll(model: ModelId, escalations: readonly Escalation[]): Promise<CachedOutcome[]> {
  const cache = loadCache(model);
  const missing = escalations.filter((e) => !cache.has(e.txn.rawDescriptor));

  if (missing.length > 0) {
    console.error(`  ${model}: ${missing.length} uncached (${escalations.length - missing.length} from cache) — calling API`);
    const classifier = new LlmClassifier(messageCreator(createClient()), { model, minConfidence: 0 });

    let next = 0;
    let done = 0;
    try {
      await Promise.all(
        Array.from({ length: Math.min(CONCURRENCY[model], missing.length) }, async () => {
          for (let i = next++; i < missing.length; i = next++) {
            const item = missing[i]!;
            const outcome = await withRetry(() => classifier.classify(item.input), item.txn.id);
            cache.set(item.txn.rawDescriptor, toCached(outcome));
            // Checkpoint, so a later failure cannot discard calls already billed.
            if (++done % 25 === 0) {
              saveCache(model, cache);
              console.error(`    ${done}/${missing.length}`);
            }
          }
        }),
      );
    } finally {
      saveCache(model, cache);
    }
  } else {
    console.error(`  ${model}: all ${escalations.length} served from cache`);
  }

  const resolved = escalations.map((e) => cache.get(e.txn.rawDescriptor));
  const gaps = resolved.filter((o) => o === undefined).length;
  if (gaps > 0) throw new Error(`${model}: ${gaps} transactions never classified — re-run to fill the cache`);
  return resolved as CachedOutcome[];
}

// ── Scoring ─────────────────────────────────────────────────────────────────

interface Score {
  answered: number;
  correct: number;
  transferAsSpend: number;
}

function scoreAt(gate: number, escalations: readonly Escalation[], outcomes: readonly CachedOutcome[]): Score {
  const score: Score = { answered: 0, correct: 0, transferAsSpend: 0 };
  outcomes.forEach((outcome, i) => {
    if (outcome.status !== 'ok' || (outcome.confidence ?? 0) < gate) return;
    const truth = escalations[i]!.txn.label.category;
    score.answered++;
    if (outcome.category === truth) score.correct++;
    if (isTransfer(truth) && !isTransfer(outcome.category!)) score.transferAsSpend++;
  });
  return score;
}

const pct = (n: number, d: number) => (d === 0 ? '   n/a' : `${((n / d) * 100).toFixed(1)}%`.padStart(6));
const usd = (n: number) => `$${n.toFixed(4)}`;

function percentile(values: readonly number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))]!;
}

function totalUsage(outcomes: readonly CachedOutcome[]): TokenUsage {
  return outcomes.reduce((acc, o) => addUsage(acc, o.usage), EMPTY_USAGE);
}

async function main(): Promise<void> {
  console.error('building cascade…');
  const validationCascade = await runCascade(fit, validation);
  const goldenCascade = await runCascade(history, golden);

  const lines: string[] = [
    `PROMPT             ${PROMPT_VERSION}`,
    `SPLITS             fit ${fit.length} · validation ${validation.length} · golden holdout ${golden.length}`,
    `INBOX              ${validationCascade.escalations.length} validation · ${goldenCascade.escalations.length} golden — what Tiers 1+2 declined`,
    ``,
  ];

  const results = new Map<ModelId, { validation: CachedOutcome[]; golden: CachedOutcome[] }>();
  const unavailable = new Map<ModelId, string>();

  for (const model of MODEL_IDS) {
    console.error(`classifying with ${model}…`);
    try {
      results.set(model, {
        validation: await classifyAll(model, validationCascade.escalations),
        golden: await classifyAll(model, goldenCascade.escalations),
      });
    } catch (error) {
      // A model this environment cannot reach is a gap in the report, not a
      // reason to discard the models it can. Recorded by name so the missing
      // comparison is visible rather than quietly absent.
      const status = (error as { status?: number }).status;
      unavailable.set(model, status ? `HTTP ${status}` : (error as Error).message);
      console.error(`  ${model}: unavailable (${unavailable.get(model)}) — skipping`);
    }
  }

  const available = MODEL_IDS.filter((m) => results.has(m));
  if (available.length === 0) throw new Error('no model was reachable; nothing to report');
  if (unavailable.size > 0) {
    lines.push(
      `UNAVAILABLE        ${[...unavailable.entries()].map(([m, why]) => `${m} (${why})`).join(', ')}`,
      `                   not reachable from this environment — any comparison against it is unmeasured`,
      ``,
    );
  }

  // ── Gate selection, on validation only ────────────────────────────────────
  lines.push(
    `GATE SELECTION     confidence gate swept on the validation inbox, never on the holdout`,
    ``,
    `  gate   ` + available.map((m) => m.replace('claude-', '').padStart(20)).join(''),
  );

  const chosen = new Map<ModelId, number>();
  for (const gate of CONFIDENCE_GATE_CANDIDATES) {
    const cells = available.map((model) => {
      const escalations = validationCascade.escalations;
      const s = scoreAt(gate, escalations, results.get(model)!.validation);
      return `${pct(s.answered, escalations.length)}/${pct(s.correct, s.answered)}`.padStart(20);
    });
    lines.push(`  ${gate.toFixed(2)}  ` + cells.join(''));
  }

  for (const model of available) {
    const escalations = validationCascade.escalations;
    const passing = CONFIDENCE_GATE_CANDIDATES.map((gate) => ({ gate, s: scoreAt(gate, escalations, results.get(model)!.validation) }))
      .filter(({ s }) => s.answered > 0 && s.correct / s.answered >= PRECISION_FLOOR)
      .sort((a, b) => b.s.answered - a.s.answered);
    chosen.set(model, passing[0]?.gate ?? 1.1);
  }

  lines.push(
    ``,
    `SELECTED           ` +
      available.map((m) => `${m}=${(chosen.get(m) ?? 1.1) > 1 ? 'abstain' : chosen.get(m)!.toFixed(2)}`).join('  ') +
      `  (widest coverage clearing ${(PRECISION_FLOOR * 100).toFixed(0)}%)`,
    ``,
    `HOLDOUT RESULT     scored once, at the gate chosen above`,
    ``,
  );

  // ── Holdout, per model ────────────────────────────────────────────────────
  for (const model of available) {
    const spec = MODELS[model];
    const outcomes = results.get(model)!.golden;
    const escalations = goldenCascade.escalations;
    const gate = chosen.get(model)!;
    const s = scoreAt(gate, escalations, outcomes);

    const hardIdx = escalations.map((e, i) => (e.txn.label.hard ? i : -1)).filter((i) => i >= 0);
    const hard = scoreAt(gate, hardIdx.map((i) => escalations[i]!), hardIdx.map((i) => outcomes[i]!));

    const usage = totalUsage(outcomes);
    const cost = costOf(usage, spec);
    const latencies = outcomes.map((o) => o.latencyMs);
    const repaired = outcomes.filter((o) => o.attempts > 1).length;
    const unresolved = outcomes.filter((o) => o.status !== 'ok').length;

    lines.push(
      `  ${model}`,
      `    gate           ${gate > 1 ? 'abstain' : gate.toFixed(2)}`,
      `    coverage       ${pct(s.answered, escalations.length).trim()} of the inbox (${s.answered}/${escalations.length})`,
      `    precision      ${pct(s.correct, s.answered).trim()}`,
      `    hard slice     ${pct(hard.answered, hardIdx.length).trim()} coverage, ${pct(hard.correct, hard.answered).trim()} precision`,
      `    transfer→spend ${s.transferAsSpend}`,
      `    repairs        ${repaired}/${outcomes.length} needed a second attempt · ${unresolved} never validated`,
      `    latency        p50 ${percentile(latencies, 50)}ms · p95 ${percentile(latencies, 95)}ms`,
      `    tokens         ${usage.inputTokens} in · ${usage.outputTokens} out · ${usage.cacheReadTokens} cache-read · ${usage.cacheCreationTokens} cache-write`,
      `    cost           ${usd(cost)} for ${outcomes.length} calls (${usd((cost / outcomes.length) * 1000)} per 1k calls)`,
      ``,
    );
  }

  // ── Cascade economics ─────────────────────────────────────────────────────
  const primary: ModelId = 'claude-haiku-4-5';
  const primarySpec = MODELS[primary];
  const primaryOutcomes = results.get(primary)!.golden;
  const primaryGate = chosen.get(primary)!;
  const primaryScore = scoreAt(primaryGate, goldenCascade.escalations, primaryOutcomes);
  const primaryCost = costOf(totalUsage(primaryOutcomes), primarySpec);

  const endToEndAnswered = goldenCascade.resolved + primaryScore.answered;
  const endToEndCorrect = goldenCascade.resolvedCorrect + primaryScore.correct;
  const perCall = primaryCost / primaryOutcomes.length;

  lines.push(
    `CASCADE            all ${golden.length} holdout transactions, Tier 3 = ${primary}`,
    `  tiers 0+1+2      ${pct(goldenCascade.resolved, golden.length).trim()} coverage, ${pct(goldenCascade.resolvedCorrect, goldenCascade.resolved).trim()} precision`,
    `  + tier 3         ${pct(endToEndAnswered, golden.length).trim()} coverage, ${pct(endToEndCorrect, endToEndAnswered).trim()} precision, ${pct(endToEndCorrect, golden.length).trim()} resolved`,
    `  → human review   ${golden.length - endToEndAnswered} (${pct(golden.length - endToEndAnswered, golden.length).trim()})`,
    ``,
    `COST PER 1,000 TRANSACTIONS`,
    `  cascade          ${usd((primaryCost / golden.length) * 1000)}  — measured; only ${pct(primaryOutcomes.length, golden.length).trim()} reach a model`,
    `  model on all     ${usd(perCall * 1000)}  — extrapolated from measured mean call cost, if every transaction were sent`,
    `  saving           ${(100 - (primaryCost / golden.length / perCall) * 100).toFixed(1)}% of the model bill, removed by the deterministic tiers`,
    ``,
  );

  // ── Calibration ───────────────────────────────────────────────────────────
  lines.push(
    `CALIBRATION        ${primary}: stated confidence vs observed accuracy, ungated`,
    `                   a self-reported confidence is worth nothing until this is checked`,
    ``,
  );
  const BUCKETS = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.01];
  for (let i = 0; i < BUCKETS.length - 1; i++) {
    const lo = BUCKETS[i]!;
    const hi = BUCKETS[i + 1]!;
    const idx = primaryOutcomes
      .map((o, j) => (o.status === 'ok' && (o.confidence ?? 0) >= lo && (o.confidence ?? 0) < hi ? j : -1))
      .filter((j) => j >= 0);
    if (idx.length === 0) continue;
    const correct = idx.filter((j) => primaryOutcomes[j]!.category === goldenCascade.escalations[j]!.txn.label.category).length;
    lines.push(
      `  ${lo.toFixed(2)}–${hi > 1 ? '1.00' : hi.toFixed(2)}   ${String(idx.length).padStart(4)} txns   ${pct(correct, idx.length)}  ${'█'.repeat(Math.round((correct / idx.length) * 30))}`,
    );
  }

  // ── Routing verdict ───────────────────────────────────────────────────────
  // "Cheap model for bulk, expensive model for reasoning" is a claim about a
  // tradeoff. Without both sides measured on the same transactions there is no
  // tradeoff to report, and asserting one anyway would be the exact failure this
  // project is built to avoid.
  lines.push(``, `ROUTING            does the expensive model earn its price on this traffic?`);
  if (available.length < 2) {
    lines.push(
      `  UNMEASURED       only ${available.join(', ')} was reachable.`,
      `                   The routing claim stays unproven until a second tier of model can be`,
      `                   scored on these same ${goldenCascade.escalations.length} transactions.`,
    );
  } else {
    const [cheap, dear] = available as [ModelId, ModelId];
    const cheapScore = scoreAt(chosen.get(cheap)!, goldenCascade.escalations, results.get(cheap)!.golden);
    const dearScore = scoreAt(chosen.get(dear)!, goldenCascade.escalations, results.get(dear)!.golden);
    const cheapCost = costOf(totalUsage(results.get(cheap)!.golden), MODELS[cheap]);
    const dearCost = costOf(totalUsage(results.get(dear)!.golden), MODELS[dear]);
    const delta = dearScore.correct - cheapScore.correct;
    lines.push(
      `  ${cheap}    ${pct(cheapScore.correct, cheapScore.answered).trim()} precision at ${pct(cheapScore.answered, goldenCascade.escalations.length).trim()} coverage, ${usd(cheapCost)}`,
      `  ${dear}     ${pct(dearScore.correct, dearScore.answered).trim()} precision at ${pct(dearScore.answered, goldenCascade.escalations.length).trim()} coverage, ${usd(dearCost)}`,
      `  ratio            ${(dearCost / cheapCost).toFixed(1)}x the cost for ${delta >= 0 ? '+' : ''}${delta} additional correct answers`,
    );
  }

  console.log(lines.join('\n'));
}

main();
