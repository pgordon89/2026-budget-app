/**
 * Does the pipeline actually get cheaper as it is used?
 *
 *   npm run analyze:learning
 *
 * The README has claimed since the first commit that "the system gets cheaper as
 * it is used" — user corrections write back to the merchant store, so tomorrow's
 * transactions are answered by the free tier instead of the paid one. That is a
 * pleasant thing to assert and an unmeasured one, which by this project's own
 * third rule means it is not done.
 *
 * The experiment: walk the 12-month holdout in date order, one transaction at a
 * time, as if it were arriving. Anything the cascade declines goes to a review
 * queue, and a simulated user corrects it using the ground-truth label. Two arms,
 * identical but for one thing:
 *
 *   learning — corrections write back through the real `Ledger.correct()` path,
 *              so Tier 1 sees them on the very next transaction
 *   control  — corrections are counted but never reach the store
 *
 * Everything else is held constant: same seed history, same gates, same replayed
 * model responses. The gap between the arms is the value of the feedback loop,
 * and nothing else.
 *
 * Two deliberate conservatisms, both understating the effect:
 *
 *   - The user only ever corrects the review queue. A real user also notices
 *     confident wrong answers in their ledger; modelling that would teach the
 *     store faster.
 *   - Tier 2's index rebuilds monthly rather than per correction, because
 *     refitting an embedding index on every write is not something anyone ships.
 *     Tier 1 updates immediately, which is what a row upsert actually costs.
 */

import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { createMemoryDatabase } from '../src/db/client.js';
import { Ledger } from '../src/db/ledger.js';
import { MerchantMemoryRepository } from '../src/db/merchant-memory-repository.js';
import { fitLexicalEmbedder } from '../src/ai/embed.js';
import { NeighbourIndex } from '../src/ai/knn.js';
import { MODELS, costOf, type TokenUsage } from '../src/ai/client.js';
import { DEFAULT_LLM_CONFIG } from '../src/ai/llm.js';
import { PROMPT_VERSION } from '../src/ai/prompt.js';
import { accounts } from '../src/db/schema.js';
import { ACCOUNTS, type SyntheticTransaction } from '../src/synthetic/generator.js';
import type { CategoryId } from '../src/core/taxonomy.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA = join(HERE, '..', 'evals', 'datasets');
const load = (f: string): SyntheticTransaction[] => JSON.parse(readFileSync(join(DATA, f), 'utf8'));

const LLM_MODEL = DEFAULT_LLM_CONFIG.model;
const LLM_GATE = 0.9;

const history = load('history.json');
const golden = load('golden.json');

interface CachedLlm {
  status: 'ok' | 'unresolved';
  category?: string;
  confidence?: number;
  usage: TokenUsage;
}
const llmCache: Map<string, CachedLlm> = (() => {
  try {
    const file = join(HERE, '..', 'evals', 'cache', `llm-${LLM_MODEL}-${PROMPT_VERSION}.json`);
    return new Map(Object.entries(JSON.parse(readFileSync(file, 'utf8')) as Record<string, CachedLlm>));
  } catch {
    return new Map();
  }
})();

const monthOf = (iso: string) => iso.slice(0, 7);
const cents = (dollars: number) => Math.round(dollars * 100);

interface MonthBucket {
  month: string;
  transactions: number;
  answered: number;
  correct: number;
  costUsd: number;
  byTier: Record<string, number>;
  corrections: number;
}

interface Outcome {
  readonly answered: boolean;
  readonly correct: boolean;
  readonly tier: string | undefined;
  readonly hard: boolean;
}

interface ArmResult {
  readonly name: string;
  readonly months: MonthBucket[];
  readonly totalCorrections: number;
  readonly llmCacheMisses: number;
  readonly merchantsLearned: number;
  /** Per transaction, so the two arms can be diffed rather than only totalled. */
  readonly outcomes: Map<string, Outcome>;
}

async function runArm(name: string, learns: boolean): Promise<ArmResult> {
  const handle = await createMemoryDatabase();
  try {
    await handle.db.insert(accounts).values(ACCOUNTS.map((a) => ({ ...a })));

    const memory = new MerchantMemoryRepository(handle.db);
    const ledger = new Ledger(handle.db);

    // Seed from the history split, exactly as every other eval does.
    for (const t of history) await memory.remember(t.rawDescriptor, t.label.category);
    const seededKeys = new Set((await hydrateKeys(memory)));

    await ledger.importTransactions(
      golden.map((t) => ({
        id: t.id,
        accountId: t.accountId,
        postedOn: t.date,
        amountCents: cents(t.amount),
        rawDescriptor: t.rawDescriptor,
      })),
    );

    let index = await buildIndex(memory);
    let indexMonth = monthOf(golden[0]!.date);

    const months = new Map<string, MonthBucket>();
    const outcomes = new Map<string, Outcome>();
    let totalCorrections = 0;
    let llmCacheMisses = 0;

    for (const txn of golden) {
      const month = monthOf(txn.date);
      if (month !== indexMonth) {
        // The nightly job: Tier 2's corpus catches up with what Tier 1 learned.
        index = await buildIndex(memory);
        indexMonth = month;
      }

      const bucket =
        months.get(month) ??
        months.set(month, { month, transactions: 0, answered: 0, correct: 0, costUsd: 0, byTier: {}, corrections: 0 }).get(month)!;
      bucket.transactions++;

      let predicted: string | undefined;
      let tier: string | undefined;
      let confidence = 0;
      let cost = 0;

      const tier1 = await memory.lookup(txn.rawDescriptor);
      if (tier1.status === 'hit') {
        predicted = tier1.category;
        tier = 'memory';
        confidence = tier1.confidence;
      } else {
        const tier2 = await index.lookup(txn.rawDescriptor);
        if (tier2.status === 'hit') {
          predicted = tier2.category;
          tier = 'embedding';
          confidence = tier2.confidence;
        } else {
          const tier3 = llmCache.get(txn.rawDescriptor);
          if (tier3 === undefined) {
            // The cascade shifted enough that this transaction now reaches the
            // model when the cached run never sent it. Counted, not guessed at.
            llmCacheMisses++;
          } else {
            cost = costOf(tier3.usage, MODELS[LLM_MODEL]);
            if (tier3.status === 'ok' && (tier3.confidence ?? 0) >= LLM_GATE) {
              predicted = tier3.category;
              tier = 'llm';
              confidence = tier3.confidence ?? 0;
            }
          }
        }
      }

      bucket.costUsd += cost;

      outcomes.set(txn.id, {
        answered: predicted !== undefined,
        correct: predicted === txn.label.category,
        tier,
        hard: txn.label.hard,
      });

      if (predicted !== undefined && tier !== undefined) {
        bucket.answered++;
        bucket.byTier[tier] = (bucket.byTier[tier] ?? 0) + 1;
        if (predicted === txn.label.category) bucket.correct++;
        await ledger.recordPrediction(txn.id, {
          categoryId: predicted as CategoryId,
          source: tier,
          confidence,
          costMicroUsd: Math.round(cost * 1_000_000),
        });
        continue;
      }

      // Declined: it lands in the review queue, and the user resolves it.
      bucket.corrections++;
      totalCorrections++;
      if (learns) {
        await ledger.correct(txn.id, txn.label.category);
      }
    }

    const finalKeys = new Set(await hydrateKeys(memory));
    return {
      name,
      months: [...months.values()],
      totalCorrections,
      llmCacheMisses,
      merchantsLearned: [...finalKeys].filter((k) => !seededKeys.has(k)).length,
      outcomes,
    };
  } finally {
    await handle.close();
  }
}

async function hydrateKeys(memory: MerchantMemoryRepository): Promise<string[]> {
  return [...(await memory.hydrate()).entries()].map((e) => e.key);
}

async function buildIndex(memory: MerchantMemoryRepository): Promise<NeighbourIndex> {
  const store = await memory.hydrate();
  const embedder = fitLexicalEmbedder([...store.entries()].map((e) => e.key));
  return NeighbourIndex.build(store, embedder);
}

// ── Report ──────────────────────────────────────────────────────────────────

const pct = (n: number, d: number) => (d === 0 ? '   n/a' : `${((n / d) * 100).toFixed(1)}%`.padStart(6));
const per1k = (usd: number, n: number) => (n === 0 ? '$0.000' : `$${((usd / n) * 1000).toFixed(3)}`);

function half(months: readonly MonthBucket[], which: 'first' | 'last') {
  const mid = Math.floor(months.length / 2);
  const slice = which === 'first' ? months.slice(0, mid) : months.slice(mid);
  const sum = (f: (b: MonthBucket) => number) => slice.reduce((a, b) => a + f(b), 0);
  const transactions = sum((b) => b.transactions);
  return {
    transactions,
    answered: sum((b) => b.answered),
    correct: sum((b) => b.correct),
    costUsd: sum((b) => b.costUsd),
    memory: sum((b) => b.byTier.memory ?? 0),
    llm: sum((b) => b.byTier.llm ?? 0),
  };
}

async function main(): Promise<void> {
  console.error('running learning arm…');
  const learning = await runArm('learning', true);
  console.error('running control arm…');
  const control = await runArm('control', false);

  const lines: string[] = [
    `LEARNING CURVE     ${golden.length} holdout transactions replayed in date order`,
    `ARMS               learning = corrections write back · control = corrections discarded`,
    `                   everything else held constant: same seed, gates, and replayed model responses`,
    ``,
    `                     ─────────── learning ───────────    ─────────── control ────────────`,
    `  month     txns    cov    prec    $/1k   tier1     cov    prec    $/1k   tier1`,
  ];

  for (let i = 0; i < learning.months.length; i++) {
    const l = learning.months[i]!;
    const c = control.months[i]!;
    lines.push(
      `  ${l.month}  ${String(l.transactions).padStart(5)}  ` +
        `${pct(l.answered, l.transactions)} ${pct(l.correct, l.answered)} ${per1k(l.costUsd, l.transactions).padStart(7)} ${pct(l.byTier.memory ?? 0, l.transactions)}   ` +
        `${pct(c.answered, c.transactions)} ${pct(c.correct, c.answered)} ${per1k(c.costUsd, c.transactions).padStart(7)} ${pct(c.byTier.memory ?? 0, c.transactions)}`,
    );
  }

  // The sharpest question the two arms can answer. An aggregate precision shift
  // of a few tenths hides whether the answers learning newly took on are good
  // ones or barely better than coin flips — and that is the whole question of
  // whether the feedback loop is worth having.
  const marginal = [...learning.outcomes.entries()].filter(
    ([id, l]) => l.answered && !(control.outcomes.get(id)?.answered ?? false),
  );
  const marginalCorrect = marginal.filter(([, l]) => l.correct).length;
  const marginalHard = marginal.filter(([, l]) => l.hard).length;
  const flipped = [...learning.outcomes.entries()].filter(([id, l]) => {
    const c = control.outcomes.get(id);
    return l.answered && (c?.answered ?? false) && l.correct !== c!.correct;
  });
  const byTier = (rows: typeof marginal) => {
    const counts = new Map<string, number>();
    for (const [, l] of rows) counts.set(l.tier ?? '?', (counts.get(l.tier ?? '?') ?? 0) + 1);
    return [...counts].map(([t, n]) => `${t} ${n}`).join(' · ') || 'none';
  };

  const lf = half(learning.months, 'first');
  const ll = half(learning.months, 'last');
  const cf = half(control.months, 'first');
  const cl = half(control.months, 'last');

  const delta = (a: number, b: number) => `${a >= b ? '+' : ''}${(a - b).toFixed(1)}`;

  lines.push(
    ``,
    `FIRST HALF → SECOND HALF`,
    ``,
    `                      learning              control`,
    `  coverage         ${pct(lf.answered, lf.transactions)} → ${pct(ll.answered, ll.transactions)}      ${pct(cf.answered, cf.transactions)} → ${pct(cl.answered, cl.transactions)}`,
    `  precision        ${pct(lf.correct, lf.answered)} → ${pct(ll.correct, ll.answered)}      ${pct(cf.correct, cf.answered)} → ${pct(cl.correct, cl.answered)}`,
    `  answered free    ${pct(lf.memory, lf.transactions)} → ${pct(ll.memory, ll.transactions)}      ${pct(cf.memory, cf.transactions)} → ${pct(cl.memory, cl.transactions)}`,
    `  cost per 1k      ${per1k(lf.costUsd, lf.transactions).padStart(6)} → ${per1k(ll.costUsd, ll.transactions).padStart(6)}      ${per1k(cf.costUsd, cf.transactions).padStart(6)} → ${per1k(cl.costUsd, cl.transactions).padStart(6)}`,
    ``,
    `EFFECT OF LEARNING second half, learning minus control`,
    `  coverage         ${delta((ll.answered / ll.transactions) * 100, (cl.answered / cl.transactions) * 100)} points`,
    `  answered free    ${delta((ll.memory / ll.transactions) * 100, (cl.memory / cl.transactions) * 100)} points`,
    `  cost per 1k      $${((ll.costUsd / ll.transactions) * 1000 - (cl.costUsd / cl.transactions) * 1000).toFixed(3)}`,
    ``,
    `MARGINAL ANSWERS   the transactions learning answered and control did not`,
    `  count            ${marginal.length} (${marginalHard} flagged hard)`,
    `  precision        ${pct(marginalCorrect, marginal.length).trim()} — against the 97% floor every gate in this project is held to`,
    `  by tier          ${byTier(marginal)}`,
    `  flipped          ${flipped.length} answered by both arms but scored differently`,
    ``,
    `CORRECTIONS        ${learning.totalCorrections} reviewed by the user (${pct(learning.totalCorrections, golden.length).trim()} of all transactions)`,
    `MERCHANTS LEARNED  ${learning.merchantsLearned} keys the store had never seen before the holdout began`,
  );

  if (learning.llmCacheMisses > 0 || control.llmCacheMisses > 0) {
    lines.push(
      ``,
      `CACHE MISSES       learning ${learning.llmCacheMisses} · control ${control.llmCacheMisses} transactions reached the model tier`,
      `                   with no cached response; counted as unanswered rather than guessed`,
    );
  }

  console.log(lines.join('\n'));
}

main();
