/**
 * Measures Tier 2 (nearest-neighbour vote) and picks its k and confidence gate.
 *
 *   npm run analyze:knn
 *
 * Scored on the job it actually has. Tier 2 never sees a transaction Tier 1
 * answered, so measuring it over the whole holdout would blend in a population
 * it never encounters and report a number that describes nothing. Everything
 * below runs on Tier 1's escalations only, split by *why* they escalated:
 *
 *   unseen key       — the merchant is absent from history. This is the tier's
 *                      reason to exist, and the only place it can add coverage.
 *   low confidence   — the merchant is known but genuinely ambiguous. Expected
 *                      to stay hard: if the exact key could not settle it, a
 *                      lexical neighbourhood is unlikely to.
 *
 * Same three-split discipline as Tier 1: fit on the first 12 months, choose k
 * and the gate on the next 6, score the golden holdout once at the chosen
 * settings. The embedder's vocabulary and IDF are fitted on history keys only.
 *
 * The confidence score here is a heuristic — `agreement × nearest-similarity`,
 * not a statistical bound like Tier 1's — so it gets checked the only way a
 * heuristic can be: a calibration table of observed accuracy per confidence
 * bucket. A confidence that does not rise with accuracy is not a confidence.
 */

import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { MerchantMemory, DEFAULT_MEMORY_CONFIG } from '../src/ai/memory.js';
import { fitLexicalEmbedder } from '../src/ai/embed.js';
import { NeighbourIndex } from '../src/ai/knn.js';
import type { SyntheticTransaction } from '../src/synthetic/generator.js';

const DATA = join(dirname(fileURLToPath(import.meta.url)), '..', 'evals', 'datasets');
const load = (f: string): SyntheticTransaction[] => JSON.parse(readFileSync(join(DATA, f), 'utf8'));

const history = load('history.json');
const golden = load('golden.json');

const VALIDATION_SPLIT = '2025-02-01';
const fit = history.filter((t) => t.date < VALIDATION_SPLIT);
const validation = history.filter((t) => t.date >= VALIDATION_SPLIT);

/** Same floor as Tier 1. An answer this tier is not sure of is worth less than
 *  the fraction of a cent it costs to ask the model tier instead. */
const PRECISION_FLOOR = 0.97;
const K_VALUES = [1, 3, 5, 10];
const THRESHOLDS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

const isTransfer = (id: string) => id.startsWith('transfer.');

interface Escalation {
  readonly txn: SyntheticTransaction;
  readonly reason: 'unseen' | 'low_confidence' | 'degenerate';
}

/** Everything Tier 1 declined, which is exactly Tier 2's inbox. */
function escalationsFrom(memory: MerchantMemory, rows: readonly SyntheticTransaction[]): Escalation[] {
  const out: Escalation[] = [];
  for (const txn of rows) {
    const outcome = memory.lookup(txn.rawDescriptor);
    if (outcome.status === 'hit') continue;
    out.push({ txn, reason: outcome.status === 'degenerate' ? 'degenerate' : outcome.status === 'unseen' ? 'unseen' : 'low_confidence' });
  }
  return out;
}

interface Answer {
  readonly truth: string;
  readonly hard: boolean;
  readonly reason: Escalation['reason'];
  readonly predicted?: string;
  readonly confidence: number;
}

async function answerAll(
  seedRows: readonly SyntheticTransaction[],
  escalations: readonly Escalation[],
  k: number,
): Promise<Answer[]> {
  const memory = new MerchantMemory();
  for (const t of seedRows) memory.remember(t.rawDescriptor, t.label.category);

  const keys = [...memory.entries()].map((e) => e.key);
  const embedder = fitLexicalEmbedder(keys);
  // Gate wide open, so one pass yields every threshold in the sweep.
  const index = await NeighbourIndex.build(memory, embedder, { k, minConfidence: 0 });

  const out: Answer[] = [];
  for (const { txn, reason } of escalations) {
    const outcome = await index.lookup(txn.rawDescriptor);
    const base = { truth: txn.label.category, hard: txn.label.hard, reason };
    out.push(
      outcome.status === 'hit' || outcome.status === 'low_confidence'
        ? { ...base, predicted: outcome.category, confidence: outcome.confidence }
        : { ...base, confidence: 0 },
    );
  }
  return out;
}

interface Row {
  answered: number;
  correct: number;
  transferAsSpend: number;
}

function scoreAt(threshold: number, rows: readonly Answer[]): Row {
  const row: Row = { answered: 0, correct: 0, transferAsSpend: 0 };
  for (const a of rows) {
    if (a.predicted === undefined || a.confidence < threshold) continue;
    row.answered++;
    if (a.predicted === a.truth) row.correct++;
    if (isTransfer(a.truth) && !isTransfer(a.predicted)) row.transferAsSpend++;
  }
  return row;
}

const pct = (n: number, d: number) => (d === 0 ? '   n/a' : `${((n / d) * 100).toFixed(1)}%`.padStart(6));

async function main(): Promise<void> {
  // ── 1. Select k and the gate on the validation split ──────────────────────
  const fitMemory = new MerchantMemory();
  for (const t of fit) fitMemory.remember(t.rawDescriptor, t.label.category);
  const validationEscalations = escalationsFrom(fitMemory, validation);

  const byK = new Map<number, Answer[]>();
  for (const k of K_VALUES) byK.set(k, await answerAll(fit, validationEscalations, k));

  const lines: string[] = [
    `SPLITS             fit ${fit.length} · validation ${validation.length} · golden holdout ${golden.length}`,
    `TIER 1 GATE        ${DEFAULT_MEMORY_CONFIG.minConfidence.toFixed(2)} — Tier 2 only ever sees what that declined`,
    `INBOX (validation) ${validationEscalations.length} escalations`,
    ``,
    `SELECTION GRID     coverage / precision over the validation inbox`,
    ``,
    `  gate    ` + K_VALUES.map((k) => `k=${k}`.padStart(15)).join(''),
  ];

  for (const threshold of THRESHOLDS) {
    const cells = K_VALUES.map((k) => {
      const rows = byK.get(k)!;
      const r = scoreAt(threshold, rows);
      return `${pct(r.answered, rows.length)}/${pct(r.correct, r.answered)}`.padStart(15);
    });
    lines.push(`  ${threshold.toFixed(2)}  ` + cells.join(''));
  }

  let best: { k: number; threshold: number; answered: number } | undefined;
  for (const k of K_VALUES) {
    const rows = byK.get(k)!;
    for (const threshold of THRESHOLDS) {
      const r = scoreAt(threshold, rows);
      if (r.answered === 0 || r.correct / r.answered < PRECISION_FLOOR) continue;
      if (best === undefined || r.answered > best.answered) best = { k, threshold, answered: r.answered };
    }
  }

  // ── 2. Score the golden holdout once, at the chosen settings ──────────────
  const memory = new MerchantMemory();
  for (const t of history) memory.remember(t.rawDescriptor, t.label.category);
  const goldenEscalations = escalationsFrom(memory, golden);

  if (best === undefined) {
    lines.push(
      ``,
      `NO SETTING clears the ${(PRECISION_FLOOR * 100).toFixed(0)}% precision floor on the validation inbox.`,
      `Tier 2 should abstain wholesale and let the model tier take this traffic.`,
      `The calibration table below still says whether the score is worth anything.`,
    );
  } else {
    lines.push(
      ``,
      `SELECTED           k=${best.k}, gate ${best.threshold.toFixed(2)} — widest validation coverage clearing the ${(PRECISION_FLOOR * 100).toFixed(0)}% floor`,
    );
  }

  const k = best?.k ?? 5;
  const gate = best?.threshold ?? 1.1; // 1.1 = answer nothing
  const held = await answerAll(history, goldenEscalations, k);
  const overall = scoreAt(gate, held);

  const bucketise = (reason: Escalation['reason']) => held.filter((a) => a.reason === reason);
  const unseen = bucketise('unseen');
  const lowConfidence = bucketise('low_confidence');

  lines.push(
    ``,
    `HOLDOUT INBOX      ${goldenEscalations.length} escalations from Tier 1 (${unseen.length} unseen key, ${lowConfidence.length} low confidence, ${bucketise('degenerate').length} degenerate)`,
    ``,
    `HOLDOUT RESULT     at k=${k}, gate ${gate > 1 ? '— (abstaining)' : gate.toFixed(2)}`,
    `  coverage         ${overall.answered} / ${held.length} (${pct(overall.answered, held.length).trim()}) of the inbox`,
    `  precision        ${pct(overall.correct, overall.answered).trim()}`,
    `  transfer→spend   ${overall.transferAsSpend}`,
  );

  for (const [label, rows] of [['unseen key', unseen], ['low confidence', lowConfidence]] as const) {
    const r = scoreAt(gate, rows);
    lines.push(
      `  ${label.padEnd(16)} ${pct(r.answered, rows.length).trim()} coverage, ${pct(r.correct, r.answered).trim()} precision  (${rows.length} txns)`,
    );
  }

  // ── 3. Calibration — does the score mean anything? ────────────────────────
  lines.push(
    ``,
    `CALIBRATION        observed accuracy per confidence bucket, holdout inbox, ungated`,
    `                   a score that does not rise with accuracy is not a confidence`,
    ``,
  );
  const BUCKETS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.01];
  for (let i = 0; i < BUCKETS.length - 1; i++) {
    const lo = BUCKETS[i]!;
    const hi = BUCKETS[i + 1]!;
    const inBucket = held.filter((a) => a.predicted !== undefined && a.confidence >= lo && a.confidence < hi);
    if (inBucket.length === 0) continue;
    const correct = inBucket.filter((a) => a.predicted === a.truth).length;
    const bar = '█'.repeat(Math.round((correct / inBucket.length) * 30));
    lines.push(`  ${lo.toFixed(2)}–${hi > 1 ? '1.00' : hi.toFixed(2)}   ${String(inBucket.length).padStart(4)} txns   ${pct(correct, inBucket.length)}  ${bar}`);
  }

  // ── 4. What the cascade adds up to ────────────────────────────────────────
  const tier1 = { answered: 0, correct: 0 };
  for (const t of golden) {
    const outcome = memory.lookup(t.rawDescriptor);
    if (outcome.status !== 'hit') continue;
    tier1.answered++;
    if (outcome.category === t.label.category) tier1.correct++;
  }

  const combinedAnswered = tier1.answered + overall.answered;
  const combinedCorrect = tier1.correct + overall.correct;

  lines.push(
    ``,
    `CASCADE            over all ${golden.length} holdout transactions`,
    `  tier 0+1         ${pct(tier1.answered, golden.length).trim()} coverage, ${pct(tier1.correct, tier1.answered).trim()} precision, ${pct(tier1.correct, golden.length).trim()} resolved`,
    `  tier 0+1+2       ${pct(combinedAnswered, golden.length).trim()} coverage, ${pct(combinedCorrect, combinedAnswered).trim()} precision, ${pct(combinedCorrect, golden.length).trim()} resolved`,
    `  → LLM tier       ${golden.length - combinedAnswered} transactions (${pct(golden.length - combinedAnswered, golden.length).trim()})`,
  );

  // ── 5. What it still cannot reach ─────────────────────────────────────────
  const stillOpen = new Map<string, number>();
  for (let i = 0; i < held.length; i++) {
    const a = held[i]!;
    if (a.predicted !== undefined && a.confidence >= gate) continue;
    const key = goldenEscalations[i]!.txn.rawDescriptor;
    stillOpen.set(key, (stillOpen.get(key) ?? 0) + 1);
  }
  lines.push(
    ``,
    `STILL OPEN         top raw descriptors reaching the LLM tier`,
    ...[...stillOpen.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([descriptor, count]) => `  ${String(count).padStart(3)}×  ${descriptor}`),
  );

  console.log(lines.join('\n'));
}

main();
