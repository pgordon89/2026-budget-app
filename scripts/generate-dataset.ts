/**
 * Builds the committed fixture corpus.
 *
 *   npm run generate:data
 *
 * Emits a *temporal* split, not a random one. The pipeline's cheap tiers learn from
 * the user's own labeled history, so a random split would leak future labels into
 * the memory store and inflate accuracy. Train on the past, score on the future —
 * same as production.
 */

import { writeFileSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { generateTransactions, ACCOUNTS, type SyntheticTransaction } from '../src/synthetic/generator.js';
import { CATEGORIES } from '../src/core/taxonomy.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = join(HERE, '..', 'evals', 'datasets');

const SEED = 20260101;
const START = '2024-02-01';
const SPLIT = '2025-08-01'; // 18 months history / 12 months holdout
const END = '2026-07-31';

function main(): void {
  const all = generateTransactions({ seed: SEED, startDate: START, endDate: END });

  const history = all.filter((t) => t.date < SPLIT);
  const golden = all.filter((t) => t.date >= SPLIT);

  mkdirSync(OUT_DIR, { recursive: true });
  write('accounts.json', ACCOUNTS);
  write('history.json', history);
  write('golden.json', golden);

  report(all, history, golden);
}

function write(name: string, data: unknown): void {
  writeFileSync(join(OUT_DIR, name), JSON.stringify(data, null, 2) + '\n', 'utf8');
}

function report(all: SyntheticTransaction[], history: SyntheticTransaction[], golden: SyntheticTransaction[]): void {
  const byCategory = new Map<string, number>();
  for (const t of golden) byCategory.set(t.label.category, (byCategory.get(t.label.category) ?? 0) + 1);

  const unseen = CATEGORIES.filter((c) => !byCategory.has(c.id)).map((c) => c.id);
  const hard = golden.filter((t) => t.label.hard).length;

  // Transfers move money between the user's own accounts and investment
  // contributions move money into the user's own net worth — neither is spending.
  // Counting them as spend is what made the first fixture look insolvent.
  const NON_SPEND = new Set(['financial.investment']);
  const isTransfer = (id: string) => id.startsWith('transfer.');
  const real = all.filter((t) => !isTransfer(t.label.category));
  const income = real.filter((t) => t.amount > 0).reduce((s, t) => s + t.amount, 0);
  const spend = -real
    .filter((t) => t.amount < 0 && !NON_SPEND.has(t.label.category))
    .reduce((s, t) => s + t.amount, 0);
  const months = (toMonthIndex(END) - toMonthIndex(START)) + 1;
  const usd = (n: number) => n.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });

  const lines = [
    `seed              ${SEED}`,
    `range             ${START} → ${END}  (${months} months, split ${SPLIT})`,
    `total             ${all.length}`,
    `  history         ${history.length}`,
    `  golden holdout  ${golden.length}  (${hard} hard, ${((hard / golden.length) * 100).toFixed(1)}%)`,
    `categories seen   ${byCategory.size}/${CATEGORIES.length}`,
    ``,
    `income            ${usd(income)}   (${usd(income / months)}/mo)`,
    `spend             ${usd(spend)}   (${usd(spend / months)}/mo)`,
    `savings rate      ${(((income - spend) / income) * 100).toFixed(1)}%`,
    ``,
    `top spend categories (per month)`,
    ...topSpend(real.filter((t) => !NON_SPEND.has(t.label.category)), months, 12),
  ];
  if (unseen.length) lines.push(``, `WARNING no golden examples for: ${unseen.join(', ')}`);

  console.log(lines.join('\n'));
}

/** Sanity check on the fixture's plausibility — a persona whose spend profile is
 *  lopsided produces misleading budget and forecasting demos downstream. */
function topSpend(txns: SyntheticTransaction[], months: number, limit: number): string[] {
  const totals = new Map<string, number>();
  for (const t of txns) {
    if (t.amount >= 0) continue;
    totals.set(t.label.category, (totals.get(t.label.category) ?? 0) - t.amount);
  }
  return [...totals.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([id, total]) => `  ${id.padEnd(26)} $${(total / months).toFixed(0).padStart(6)}`);
}

function toMonthIndex(iso: string): number {
  const [y, m] = iso.split('-').map(Number) as [number, number];
  return y * 12 + (m - 1);
}

main();
