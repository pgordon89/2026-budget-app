/**
 * The scored eval. One run of the full cascade over the golden holdout, every
 * metric the project gates on, and a JSON baseline.
 *
 *   npm run eval                    # score, compare to the committed baseline
 *   npm run eval -- --write-baseline  # accept the current numbers as the new baseline
 *
 * This is not the same thing as the `analyze:*` scripts. Those sweep parameters
 * and choose settings — they are how a threshold gets picked. This scores the
 * chosen configuration once and decides whether the build is allowed to pass. A
 * script that both tunes and grades would be marking its own homework.
 *
 * It runs offline and for free. The deterministic tiers compute live; the model
 * tier replays the committed response cache. That combination is what makes the
 * whole cascade gateable on every push rather than only on a nightly job — the
 * usual reason LLM evals get excluded from CI is cost, and a committed cache
 * removes it. `analyze:llm -- --refresh` is what re-earns the cache when the
 * prompt version changes.
 *
 * The report carries no timestamp and no run id on purpose. Same commit, same
 * bytes — so a baseline diff shows only what actually moved, and the gate can be
 * exact instead of fuzzy.
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { MerchantMemory, DEFAULT_MEMORY_CONFIG } from '../src/ai/memory.js';
import { fitLexicalEmbedder } from '../src/ai/embed.js';
import { NeighbourIndex, DEFAULT_NEIGHBOUR_CONFIG } from '../src/ai/knn.js';
import { normalizeDescriptor } from '../src/ai/normalize.js';
import { DEFAULT_LLM_CONFIG } from '../src/ai/llm.js';
import { PROMPT_VERSION } from '../src/ai/prompt.js';
import { MODELS, costOf, addUsage, EMPTY_USAGE, type TokenUsage } from '../src/ai/client.js';
import { compareToBaseline, GATED_METRICS } from './gate.js';
import type { SyntheticTransaction } from '../src/synthetic/generator.js';
import type { Tier } from '../src/ai/types.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA = join(HERE, 'datasets');
const BASELINE = join(HERE, 'baseline.json');
const load = (f: string): SyntheticTransaction[] => JSON.parse(readFileSync(join(DATA, f), 'utf8'));

const WRITE_BASELINE = process.argv.includes('--write-baseline');
const LLM_MODEL = DEFAULT_LLM_CONFIG.model;

const history = load('history.json');
const golden = load('golden.json');

const isTransfer = (id: string) => id.startsWith('transfer.');

interface CachedLlm {
  status: 'ok' | 'unresolved';
  category?: string;
  confidence?: number;
  attempts: number;
  usage: TokenUsage;
  latencyMs: number;
}

function loadLlmCache(): Map<string, CachedLlm> {
  try {
    const file = join(HERE, 'cache', `llm-${LLM_MODEL}-${PROMPT_VERSION}.json`);
    return new Map(Object.entries(JSON.parse(readFileSync(file, 'utf8')) as Record<string, CachedLlm>));
  } catch {
    return new Map();
  }
}

/** One scored transaction: which tier answered, what it said, what it cost. */
interface Scored {
  readonly truth: string;
  readonly hard: boolean;
  readonly tier: Tier | 'review';
  readonly predicted?: string;
  readonly correct: boolean;
  readonly costUsd: number;
  readonly latencyMs?: number;
  readonly usage?: TokenUsage;
}

async function run(): Promise<Scored[]> {
  const memory = new MerchantMemory();
  for (const t of history) memory.remember(t.rawDescriptor, t.label.category);

  const embedder = fitLexicalEmbedder([...memory.entries()].map((e) => e.key));
  const index = await NeighbourIndex.build(memory, embedder);
  const llm = loadLlmCache();

  const out: Scored[] = [];
  for (const txn of golden) {
    const truth = txn.label.category;
    const base = { truth, hard: txn.label.hard };

    const tier1 = memory.lookup(txn.rawDescriptor);
    if (tier1.status === 'hit') {
      out.push({ ...base, tier: 'memory', predicted: tier1.category, correct: tier1.category === truth, costUsd: 0 });
      continue;
    }

    const tier2 = await index.lookup(txn.rawDescriptor);
    if (tier2.status === 'hit') {
      out.push({ ...base, tier: 'embedding', predicted: tier2.category, correct: tier2.category === truth, costUsd: 0 });
      continue;
    }

    const tier3 = llm.get(txn.rawDescriptor);
    if (tier3 === undefined) {
      // No cached response and no live call: this transaction was never scored
      // at the model tier. Counting it as a review-queue item would quietly turn
      // a missing measurement into a flattering one, so it is fatal.
      throw new Error(
        `no cached ${LLM_MODEL} response for "${txn.rawDescriptor}" at prompt ${PROMPT_VERSION}. ` +
          `Run: npm run analyze:llm`,
      );
    }

    const answered = tier3.status === 'ok' && (tier3.confidence ?? 0) >= LLM_GATE;
    const cost = costOf(tier3.usage, MODELS[LLM_MODEL]);
    out.push({
      ...base,
      tier: answered ? 'llm' : 'review',
      ...(answered ? { predicted: tier3.category } : {}),
      correct: answered && tier3.category === truth,
      costUsd: cost,
      latencyMs: tier3.latencyMs,
      usage: tier3.usage,
    });
  }
  return out;
}

/**
 * The model tier's accept threshold, chosen on the validation split by
 * `analyze:llm`. Pinned here rather than re-derived, because the runner grades
 * a fixed configuration — recomputing it would let the eval move its own goalposts.
 */
const LLM_GATE = 0.9;

// ── Metrics ─────────────────────────────────────────────────────────────────

const rate = (n: number, d: number) => (d === 0 ? 0 : Number(((n / d) * 100).toFixed(2)));

function percentile(values: readonly number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))]!;
}

function sliceStats(rows: readonly Scored[]) {
  const answered = rows.filter((r) => r.tier !== 'review');
  const correct = answered.filter((r) => r.correct);
  return {
    transactions: rows.length,
    answered: answered.length,
    correct: correct.length,
    coveragePct: rate(answered.length, rows.length),
    precisionPct: rate(correct.length, answered.length),
    resolvedPct: rate(correct.length, rows.length),
  };
}

function normalizerStats() {
  const all = [...history, ...golden];
  const raw = new Set(all.map((t) => t.rawDescriptor));
  const keys = new Set(all.map((t) => normalizeDescriptor(t.rawDescriptor).key));
  const keyToMerchants = new Map<string, Set<string>>();
  for (const t of all) {
    const { key } = normalizeDescriptor(t.rawDescriptor);
    (keyToMerchants.get(key) ?? keyToMerchants.set(key, new Set()).get(key)!).add(t.label.merchant);
  }
  return {
    distinctRawDescriptors: raw.size,
    distinctKeys: keys.size,
    collapseRatio: Number((raw.size / keys.size).toFixed(2)),
    // A key covering two merchants does not cause a miss, it causes a confident
    // wrong answer — invisible in accuracy, so it gets its own gated counter.
    collisions: [...keyToMerchants.values()].filter((m) => m.size > 1).length,
  };
}

function buildReport(scored: readonly Scored[]) {
  const tiers = {} as Record<Tier, ReturnType<typeof sliceStats> & { hitRatePct: number }>;
  for (const tier of ['memory', 'embedding', 'llm'] as const) {
    const rows = scored.filter((r) => r.tier === tier);
    const correct = rows.filter((r) => r.correct).length;
    tiers[tier] = {
      transactions: rows.length,
      answered: rows.length,
      correct,
      hitRatePct: rate(rows.length, scored.length),
      coveragePct: rate(rows.length, scored.length),
      precisionPct: rate(correct, rows.length),
      resolvedPct: rate(correct, scored.length),
    };
  }

  const transfers = scored.filter((r) => isTransfer(r.truth));
  const transfersAnswered = transfers.filter((r) => r.predicted !== undefined);
  const asSpend = transfersAnswered.filter((r) => !isTransfer(r.predicted!)).length;
  const spendAsTransfer = scored.filter(
    (r) => !isTransfer(r.truth) && r.predicted !== undefined && isTransfer(r.predicted),
  ).length;

  // Worst confusions: which specific pairs the system actually gets wrong. An
  // aggregate accuracy number says how much is broken; this says what.
  const pairs = new Map<string, number>();
  for (const r of scored) {
    if (r.predicted === undefined || r.correct) continue;
    const key = `${r.truth} → ${r.predicted}`;
    pairs.set(key, (pairs.get(key) ?? 0) + 1);
  }

  const llmRows = scored.filter((r) => r.tier === 'llm' || r.tier === 'review');
  const llmUsage = llmRows.reduce((acc, r) => (r.usage ? addUsage(acc, r.usage) : acc), EMPTY_USAGE);
  const totalCost = llmRows.reduce((sum, r) => sum + r.costUsd, 0);
  const latencies = llmRows.map((r) => r.latencyMs ?? 0);

  return {
    promptVersion: PROMPT_VERSION,
    config: {
      memoryGate: DEFAULT_MEMORY_CONFIG.minConfidence,
      neighbourGate: DEFAULT_NEIGHBOUR_CONFIG.minConfidence,
      neighbourK: DEFAULT_NEIGHBOUR_CONFIG.k,
      llmGate: LLM_GATE,
      llmModel: LLM_MODEL,
    },
    dataset: {
      historyTransactions: history.length,
      holdoutTransactions: golden.length,
      hardTransactions: golden.filter((t) => t.label.hard).length,
    },
    normalizer: normalizerStats(),
    overall: sliceStats(scored),
    slices: {
      easy: sliceStats(scored.filter((r) => !r.hard)),
      hard: sliceStats(scored.filter((r) => r.hard)),
    },
    tiers,
    review: {
      transactions: scored.filter((r) => r.tier === 'review').length,
      pct: rate(scored.filter((r) => r.tier === 'review').length, scored.length),
    },
    transfers: {
      inHoldout: transfers.length,
      answered: transfersAnswered.length,
      // The costliest error in the product: a transfer booked as spending
      // corrupts every report built on top of it.
      misclassifiedAsSpend: asSpend,
      misclassifiedAsSpendPct: rate(asSpend, transfersAnswered.length),
      spendMisclassifiedAsTransfer: spendAsTransfer,
    },
    worstConfusions: [...pairs.entries()]
      .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
      .slice(0, 10)
      .map(([pair, count]) => ({ pair, count })),
    cost: {
      totalUsd: Number(totalCost.toFixed(6)),
      perThousandTransactionsUsd: Number(((totalCost / golden.length) * 1000).toFixed(4)),
      perThousandModelCallsUsd: Number(((totalCost / Math.max(llmRows.length, 1)) * 1000).toFixed(4)),
      modelTokens: llmUsage,
    },
    // Captured when the response was recorded, not on replay — replaying the
    // cache takes milliseconds and would report a meaningless number.
    llmLatencyAtCaptureMs: {
      p50: percentile(latencies, 50),
      p95: percentile(latencies, 95),
    },
  };
}

// ── Output ──────────────────────────────────────────────────────────────────

function render(report: ReturnType<typeof buildReport>): string {
  const pct = (n: number) => `${n.toFixed(1)}%`.padStart(7);
  const lines = [
    `FISCUS EVAL        prompt ${report.promptVersion} · model ${report.config.llmModel}`,
    `GATES              memory ${report.config.memoryGate} · neighbour ${report.config.neighbourGate} (k=${report.config.neighbourK}) · llm ${report.config.llmGate}`,
    `DATASET            ${report.dataset.holdoutTransactions} holdout (${report.dataset.hardTransactions} hard) from ${report.dataset.historyTransactions} history`,
    ``,
    `TIER 0             ${report.normalizer.distinctRawDescriptors} raw → ${report.normalizer.distinctKeys} keys (${report.normalizer.collapseRatio}x), ${report.normalizer.collisions} collisions`,
    ``,
    `                     share   precision   resolved`,
    ...(['memory', 'embedding', 'llm'] as const).map(
      (t) =>
        `  tier ${t.padEnd(12)} ${pct(report.tiers[t].hitRatePct)}    ${pct(report.tiers[t].precisionPct)}   ${pct(report.tiers[t].resolvedPct)}`,
    ),
    `  human review     ${pct(report.review.pct)}          —         —`,
    ``,
    `OVERALL            ${pct(report.overall.coveragePct)} coverage · ${pct(report.overall.precisionPct)} precision · ${pct(report.overall.resolvedPct)} resolved`,
    `  easy slice       ${pct(report.slices.easy.coveragePct)} coverage · ${pct(report.slices.easy.precisionPct)} precision`,
    `  hard slice       ${pct(report.slices.hard.coveragePct)} coverage · ${pct(report.slices.hard.precisionPct)} precision`,
    ``,
    `TRANSFERS          ${report.transfers.misclassifiedAsSpend} of ${report.transfers.answered} answered booked as spend (${pct(report.transfers.misclassifiedAsSpendPct).trim()})`,
    `                   ${report.transfers.spendMisclassifiedAsTransfer} non-transfers booked as transfers`,
    ``,
    `COST               $${report.cost.perThousandTransactionsUsd.toFixed(4)} per 1,000 transactions · $${report.cost.totalUsd.toFixed(4)} for this run`,
    `LATENCY            model tier p50 ${report.llmLatencyAtCaptureMs.p50}ms · p95 ${report.llmLatencyAtCaptureMs.p95}ms (at capture)`,
    ``,
    `WORST CONFUSIONS`,
    ...report.worstConfusions.map((c) => `  ${String(c.count).padStart(3)}×  ${c.pair}`),
  ];
  return lines.join('\n');
}

async function main(): Promise<void> {
  const report = buildReport(await run());
  console.log(render(report));

  const serialized = JSON.stringify(report, null, 2) + '\n';

  if (WRITE_BASELINE) {
    writeFileSync(BASELINE, serialized, 'utf8');
    console.log(`\nBASELINE           written to evals/baseline.json`);
    return;
  }

  let baseline: unknown;
  try {
    baseline = JSON.parse(readFileSync(BASELINE, 'utf8'));
  } catch {
    console.log(`\nNO BASELINE        run: npm run eval -- --write-baseline`);
    return;
  }

  const gate = compareToBaseline(report, baseline, GATED_METRICS);
  console.log(`\nREGRESSION GATE    ${GATED_METRICS.length} metrics vs evals/baseline.json`);

  for (const change of gate.changes) {
    if (change.verdict === 'unchanged') continue;
    const arrow = change.verdict === 'regressed' ? 'REGRESSED' : 'improved ';
    console.log(`  ${arrow}  ${change.path}: ${change.baseline} → ${change.current} (${change.delta > 0 ? '+' : ''}${Number(change.delta.toFixed(4))})`);
  }
  for (const path of gate.incomparable) console.log(`  new       ${path}: not in baseline`);

  if (gate.regressions.length > 0) {
    console.log(`\nFAIL               ${gate.regressions.length} metric(s) regressed`);
    process.exit(1);
  }
  if (gate.improvements.length > 0) {
    console.log(`\nPASS               ${gate.improvements.length} improved — accept with: npm run eval -- --write-baseline`);
    return;
  }
  console.log(`\nPASS               no change`);
}

main();
