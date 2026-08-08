/**
 * Measures Tier 1 (merchant memory) on the temporal holdout, and picks its
 * abstention threshold from data rather than taste.
 *
 *   npm run analyze:memory
 *
 * The tier has exactly one tuning decision — how confident it must be before it
 * answers instead of escalating — and that decision is a trade, not an
 * optimum: every point of precision is bought with escalations, and every
 * escalation is an embedding or LLM call someone pays for. So the output is a
 * curve, plus an explicitly stated selection rule applied to it.
 *
 * The rule: **maximise coverage subject to a precision floor.** Precision and
 * coverage are not symmetric costs. A miss is priced in fractions of a cent at
 * the next tier and is recoverable. A wrong answer is silent, lands in a budget
 * total, and is only ever found by a user who happens to look. Coverage is
 * money; precision is trust.
 *
 * Three splits, not two. The corpus already separates history from a temporal
 * golden holdout, but choosing a threshold by reading the holdout is fitting a
 * hyperparameter to the test set — the resulting number is not an estimate of
 * anything. So history is itself split temporally: the earlier part seeds the
 * store, the later part selects the gate, and the golden holdout is scored once
 * at the already-chosen gate and never consulted to pick it.
 *
 * Every split is temporal. The cheap tiers learn from the user's own labeled
 * past, so a random split would leak future labels backwards and report a
 * number production can never reach.
 */

import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { MerchantMemory } from '../src/ai/memory.js';
import type { SyntheticTransaction } from '../src/synthetic/generator.js';

const DATA = join(dirname(fileURLToPath(import.meta.url)), '..', 'evals', 'datasets');
const load = (f: string): SyntheticTransaction[] => JSON.parse(readFileSync(join(DATA, f), 'utf8'));

const history = load('history.json');
const golden = load('golden.json');

/** Answers below this are not worth having. Stated as policy, not tuned to the fixture. */
const PRECISION_FLOOR = 0.97;

/**
 * Deliberately coarse. A gate picked to two decimal places is fitted to this
 * fixture's noise; the curve below is flat enough between grid points that the
 * extra precision would buy nothing real.
 */
const THRESHOLDS = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

/** Splits the 18 months of history into 12 months to fit on, 6 to select on. */
const VALIDATION_SPLIT = '2025-02-01';
const fit = history.filter((t) => t.date < VALIDATION_SPLIT);
const validation = history.filter((t) => t.date >= VALIDATION_SPLIT);

const isTransfer = (id: string) => id.startsWith('transfer.');

interface Scored {
  readonly truth: string;
  readonly hard: boolean;
  /** undefined when memory had nothing to offer at any threshold. */
  readonly predicted?: string;
  readonly confidence: number;
  readonly escalation?: 'unseen' | 'degenerate';
}

function seed(rows: readonly SyntheticTransaction[]): MerchantMemory {
  // Gate wide open: `lookup` then reports the confidence it computed for every
  // known key, and the whole sweep falls out of a single pass.
  const memory = new MerchantMemory({ minConfidence: 0 });
  for (const t of rows) memory.remember(t.rawDescriptor, t.label.category);
  return memory;
}

function scoreAll(memory: MerchantMemory, rows: readonly SyntheticTransaction[]): Scored[] {
  return rows.map((t) => {
    const outcome = memory.lookup(t.rawDescriptor);
    const base = { truth: t.label.category, hard: t.label.hard };
    if (outcome.status === 'unseen' || outcome.status === 'degenerate') {
      return { ...base, confidence: 0, escalation: outcome.status };
    }
    return { ...base, predicted: outcome.category, confidence: outcome.confidence };
  });
}

interface Row {
  threshold: number;
  answered: number;
  correct: number;
  /** True transfers answered as spend or income. The most expensive error class. */
  transferAsSpend: number;
  transfersAnswered: number;
}

function scoreAt(threshold: number, rows: readonly Scored[]): Row {
  const row: Row = { threshold, answered: 0, correct: 0, transferAsSpend: 0, transfersAnswered: 0 };
  for (const s of rows) {
    if (s.predicted === undefined || s.confidence < threshold) continue;
    row.answered++;
    if (s.predicted === s.truth) row.correct++;
    if (isTransfer(s.truth)) {
      row.transfersAnswered++;
      if (!isTransfer(s.predicted)) row.transferAsSpend++;
    }
  }
  return row;
}

const pct = (n: number, d: number) => (d === 0 ? '   n/a' : `${((n / d) * 100).toFixed(1)}%`.padStart(6));

function sweepTable(rows: readonly Scored[]): string[] {
  const hardRows = rows.filter((s) => s.hard);
  const out = [`  gate   coverage  precision  resolved   escalates/1k   hard cov  hard prec   xfer→spend`];
  for (const threshold of THRESHOLDS) {
    const r = scoreAt(threshold, rows);
    const hard = scoreAt(threshold, hardRows);
    const escalations = Math.round(((rows.length - r.answered) / rows.length) * 1000);
    out.push(
      `  ${threshold.toFixed(2)}    ${pct(r.answered, rows.length)}    ${pct(r.correct, r.answered)}    ` +
        `${pct(r.correct, rows.length)}      ${String(escalations).padStart(6)}     ` +
        `${pct(hard.answered, hardRows.length)}   ${pct(hard.correct, hard.answered)}     ` +
        `${String(r.transferAsSpend).padStart(3)}/${r.transfersAnswered}`,
    );
  }
  return out;
}

// ── 1. Select the gate on the validation split ──────────────────────────────
const validationScored = scoreAll(seed(fit), validation);
const passing = THRESHOLDS.map((t) => scoreAt(t, validationScored)).filter(
  (r) => r.answered > 0 && r.correct / r.answered >= PRECISION_FLOOR,
);
const chosen = passing.sort((a, b) => b.answered - a.answered)[0];

// ── 2. Score the golden holdout once, at the already-chosen gate ────────────
const memory = seed(history);
const scored = scoreAll(memory, golden);
const hardRows = scored.filter((s) => s.hard);
const easyRows = scored.filter((s) => !s.hard);
const total = scored.length;

const unseen = scored.filter((s) => s.escalation === 'unseen').length;
const degenerate = scored.filter((s) => s.escalation === 'degenerate').length;

const lines: string[] = [
  `SPLITS             fit ${fit.length} (→ ${VALIDATION_SPLIT}) · validation ${validation.length} · golden holdout ${total}`,
  `STORE              ${memory.size} merchant keys from ${memory.observations} labeled history rows`,
  ``,
  `SELECTION SWEEP    on the validation split — the only table the gate is chosen from`,
  ``,
  ...sweepTable(validationScored),
];

if (chosen === undefined) {
  lines.push(``, `NO GATE clears the ${(PRECISION_FLOOR * 100).toFixed(0)}% precision floor. Tier 1 should not answer at all.`);
  console.log(lines.join('\n'));
  process.exit(0);
}

const gate = chosen.threshold;
const held = scoreAt(gate, scored);
const hard = scoreAt(gate, hardRows);
const easy = scoreAt(gate, easyRows);

lines.push(
  ``,
  `GATE               ${gate.toFixed(2)} — widest validation coverage clearing the ${(PRECISION_FLOOR * 100).toFixed(0)}% precision floor`,
  ``,
  `HOLDOUT RESULT     golden set, scored once at gate ${gate.toFixed(2)}`,
  `  coverage         ${held.answered} / ${total} (${pct(held.answered, total).trim()}) answered at zero marginal cost`,
  `  precision        ${pct(held.correct, held.answered).trim()}`,
  `  resolved         ${pct(held.correct, total).trim()} of all holdout transactions`,
  `  easy slice       ${pct(easy.answered, easyRows.length).trim()} coverage, ${pct(easy.correct, easy.answered).trim()} precision`,
  `  hard slice       ${pct(hard.answered, hardRows.length).trim()} coverage, ${pct(hard.correct, hard.answered).trim()} precision`,
  `  transfer→spend   ${held.transferAsSpend} of ${held.transfersAnswered} answered transfers`,
  ``,
  `ESCALATIONS        ${total - held.answered} (${pct(total - held.answered, total).trim()}) → embedding + LLM tiers`,
  `  unseen key       ${unseen} — merchant absent from history, irreducible at this tier`,
  `  degenerate key   ${degenerate} — Tier 0 could not produce a usable key`,
  `  low confidence   ${total - held.answered - unseen - degenerate} — key known, evidence split or thin`,
  ``,
  `HOLDOUT SWEEP      diagnostic only — NOT used to choose the gate above`,
  ``,
  ...sweepTable(scored),
);

// ── 3. What the tier declines to answer, and why ────────────────────────────
// The low-confidence bucket is the interesting one: it is the tier naming its
// own blind spots, and it is the work list for Tiers 2 and 3.
const declined = new Map<string, { count: number; confidence: number; picked: string }>();
for (const t of golden) {
  const outcome = memory.lookup(t.rawDescriptor);
  if (outcome.status === 'unseen' || outcome.status === 'degenerate') continue;
  if (outcome.confidence >= gate) continue;
  const entry = declined.get(outcome.key) ?? { count: 0, confidence: outcome.confidence, picked: outcome.category };
  entry.count++;
  declined.set(outcome.key, entry);
}

lines.push(
  ``,
  `TOP DECLINED KEYS  known merchants the tier refuses to answer for`,
  ...[...declined.entries()]
    .sort((a, b) => b[1].count - a[1].count)
    .slice(0, 10)
    .map(([key, e]) => `  ${key.padEnd(28)} ${String(e.count).padStart(4)} txns  conf ${e.confidence.toFixed(2)}  would have said ${e.picked}`),
);

console.log(lines.join('\n'));
