/**
 * Measures Tier 0 in the only terms that matter downstream:
 *
 *   coverage  — what share of holdout transactions the normalizer alone makes
 *               answerable from history, for zero marginal cost
 *   collisions — keys that map to more than one merchant. These are the dangerous
 *               failure: a collision does not cause a cache miss, it causes a
 *               confident wrong answer, and it is invisible in aggregate accuracy.
 *
 *   npm run analyze:normalizer
 */

import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { normalizeDescriptor } from '../src/ai/normalize.js';
import type { SyntheticTransaction } from '../src/synthetic/generator.js';

const DATA = join(dirname(fileURLToPath(import.meta.url)), '..', 'evals', 'datasets');
const load = (f: string): SyntheticTransaction[] => JSON.parse(readFileSync(join(DATA, f), 'utf8'));

const history = load('history.json');
const golden = load('golden.json');

// ── Collapse: how many raw descriptor variants fold onto one key ─────────────
const rawSet = new Set<string>();
const keySet = new Set<string>();
let degenerate = 0;
for (const t of [...history, ...golden]) {
  const n = normalizeDescriptor(t.rawDescriptor);
  rawSet.add(t.rawDescriptor);
  keySet.add(n.key);
  if (n.degenerate) degenerate++;
}

// ── Collisions: one key, multiple ground-truth merchants ─────────────────────
const keyToMerchants = new Map<string, Map<string, number>>();
for (const t of [...history, ...golden]) {
  const { key } = normalizeDescriptor(t.rawDescriptor);
  const m = keyToMerchants.get(key) ?? new Map<string, number>();
  m.set(t.label.merchant, (m.get(t.label.merchant) ?? 0) + 1);
  keyToMerchants.set(key, m);
}
const collisions = [...keyToMerchants.entries()].filter(([, m]) => m.size > 1);

// ── Coverage: holdout keys already seen, and labeled unambiguously ───────────
const historyKeyLabels = new Map<string, Map<string, number>>();
for (const t of history) {
  const { key } = normalizeDescriptor(t.rawDescriptor);
  const m = historyKeyLabels.get(key) ?? new Map<string, number>();
  m.set(t.label.category, (m.get(t.label.category) ?? 0) + 1);
  historyKeyLabels.set(key, m);
}

let hit = 0;
let hitCorrect = 0;
let miss = 0;
for (const t of golden) {
  const { key } = normalizeDescriptor(t.rawDescriptor);
  const labels = historyKeyLabels.get(key);
  if (!labels) {
    miss++;
    continue;
  }
  hit++;
  // Plurality vote, matching what the memory tier will do.
  const best = [...labels.entries()].sort((a, b) => b[1] - a[1])[0]![0];
  if (best === t.label.category) hitCorrect++;
}

const pct = (n: number, d: number) => `${((n / d) * 100).toFixed(1)}%`;

console.log(
  [
    `descriptors        ${rawSet.size} distinct raw → ${keySet.size} distinct keys  (${(rawSet.size / keySet.size).toFixed(1)}x collapse)`,
    `degenerate         ${degenerate} (${pct(degenerate, history.length + golden.length)})`,
    ``,
    `TIER 1 PROJECTION  over ${golden.length} holdout transactions`,
    `  key hit          ${hit} (${pct(hit, golden.length)})`,
    `  hit & correct    ${hitCorrect} (${pct(hitCorrect, golden.length)} of all, ${pct(hitCorrect, hit)} precision)`,
    `  escalates        ${miss} (${pct(miss, golden.length)}) → embedding + LLM tiers`,
    ``,
    `COLLISIONS         ${collisions.length} keys map to >1 merchant`,
    ...collisions
      .sort((a, b) => sum(b[1]) - sum(a[1]))
      .slice(0, 10)
      .map(([k, m]) => `  ${k.padEnd(30)} ${[...m.entries()].map(([n, c]) => `${n}(${c})`).join(', ')}`),
  ].join('\n'),
);

function sum(m: Map<string, number>): number {
  let s = 0;
  for (const v of m.values()) s += v;
  return s;
}
