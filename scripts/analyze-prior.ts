/**
 * Is Tier 2's discarded vote worth anything to Tier 3? Measured: no.
 *
 *   npm run analyze:prior
 *
 * When Tier 2 cannot clear its agreement floor it throws away a full weighted
 * distribution over categories and escalates with nothing but the three merchant
 * keys that produced it. The model is then asked to re-derive, from those keys, a
 * distribution the previous tier had already computed exactly. Handing it over
 * costs a few dozen tokens in the volatile half of the prompt, so the hypothesis
 * was cheap enough to be worth buying an answer to.
 *
 * **The verdict, at matched coverage on the holdout: 96 of 109 correct without
 * the prior, 95 of 109 with it.** Nothing, in the wrong direction. The prior was
 * reverted; this script is the receipt.
 *
 * Three things about the method are worth more than the result.
 *
 * **The arms differ by the vote and nothing else.** Both ran prompt v2, so both
 * got the same system block, and the control passed `withVote: false` — keeping
 * the neighbour examples and dropping only the distribution. The neighbours were
 * already in v1, so comparing against v1 instead would have credited the prior
 * with whatever the reworded system block did. A third, free arm reads the
 * committed v1 responses to price that rewording separately, and it is the reason
 * the rewording was reverted too: it cost 11 points of validation precision.
 *
 * **The comparison is at matched coverage, not at a fixed gate.** The prior moves
 * the whole confidence distribution — the v2 system block told the model a split
 * neighbourhood was a reason to stay unsure, and it complied — so holding the
 * accept gate at 0.90 compares two different operating points and reads the
 * coverage gap as a quality gap. Ranking each arm by its own confidence and
 * taking the same number of answers from each asks the only question that
 * matters about a confidence: is it ordered better than the other one.
 *
 * **The planned noise floor did not exist, and that is reported rather than
 * quietly dropped.** The design assumed a slice of transactions with no
 * neighbours at all, whose prompts would be byte-identical across arms, giving a
 * free read on model nondeterminism. On this corpus that slice is empty: a
 * character-trigram embedder finds some shared feature with almost anything, so
 * every escalation carries a vote. The one-answer difference is therefore
 * reported without a noise floor to compare it against — which is a limitation of
 * this measurement, though a delta of 1 in 109 needs no error bar to read as null.
 */

import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { MerchantMemory } from '../src/ai/memory.js';
import { fitLexicalEmbedder } from '../src/ai/embed.js';
import { NeighbourIndex } from '../src/ai/knn.js';
import { escalationInput } from '../src/ai/escalation.js';
import { type ClassificationInput } from '../src/ai/prompt.js';
import { MODELS, costOf, type ModelId, type TokenUsage } from '../src/ai/client.js';
import type { SyntheticTransaction } from '../src/synthetic/generator.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA = join(HERE, '..', 'evals', 'datasets');
const CACHE_DIR = join(HERE, '..', 'evals', 'cache');
const load = (f: string): SyntheticTransaction[] => JSON.parse(readFileSync(join(DATA, f), 'utf8'));

const MODEL: ModelId = 'claude-haiku-4-5';

/**
 * The prompt version the experiment ran under, pinned rather than read from
 * `PROMPT_VERSION`.
 *
 * v2 existed only for this measurement and was reverted when it lost, so the
 * tree no longer contains the prompt that produced these responses. The finding
 * stays reproducible from the committed caches — the same way the eval itself
 * reproduces without an API key — but it can no longer be re-called against the
 * model. Reading the live `PROMPT_VERSION` here would silently repoint this
 * script at the shipped cache the day someone bumps it.
 */
const EXPERIMENT_VERSION = 'v2';

/** As in `analyze:tier2`: read from the scored run, never restated. */
const LLM_GATE: number = JSON.parse(
  readFileSync(join(HERE, '..', 'evals', 'baseline.json'), 'utf8'),
).config.llmGate;

const VALIDATION_SPLIT = '2025-02-01';
const history = load('history.json');
const golden = load('golden.json');
const fit = history.filter((t) => t.date < VALIDATION_SPLIT);
const validation = history.filter((t) => t.date >= VALIDATION_SPLIT);

const GATE_CANDIDATES = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95];

type Arm = 'vote' | 'novote';
const ARMS: readonly Arm[] = ['novote', 'vote'];

// ── The inbox ───────────────────────────────────────────────────────────────

interface Escalation {
  readonly txn: SyntheticTransaction;
  /** Whether Tier 2 had a vote to hand over. Defines the two slices. */
  readonly contested: boolean;
  readonly inputs: Record<Arm, ClassificationInput>;
}

/** Everything the shipped cascade escalates to the model, at the shipped gates. */
async function inboxOf(
  seedRows: readonly SyntheticTransaction[],
  scoreRows: readonly SyntheticTransaction[],
): Promise<Escalation[]> {
  const memory = new MerchantMemory();
  for (const t of seedRows) memory.remember(t.rawDescriptor, t.label.category);

  const embedder = fitLexicalEmbedder([...memory.entries()].map((e) => e.key));
  const index = await NeighbourIndex.build(memory, embedder);

  const out: Escalation[] = [];
  for (const txn of scoreRows) {
    if (memory.lookup(txn.rawDescriptor).status === 'hit') continue;
    const tier2 = await index.lookup(txn.rawDescriptor);
    if (tier2.status === 'hit') continue;

    out.push({
      txn,
      contested: tier2.status === 'low_confidence',
      inputs: {
        vote: escalationInput(txn.rawDescriptor, txn.amount, tier2, true),
        novote: escalationInput(txn.rawDescriptor, txn.amount, tier2, false),
      },
    });
  }
  return out;
}

// ── Caches, one per arm ─────────────────────────────────────────────────────
// Both arms keep their own file, and neither is the one the eval runner replays.
// The treatment arm briefly was, while the prior looked like it might ship; it
// was renamed on the revert so that nothing which scores the product can read a
// response generated under a prompt the product no longer sends.

interface CachedOutcome {
  readonly status: 'ok' | 'unresolved';
  readonly category?: string | undefined;
  readonly confidence?: number | undefined;
  readonly attempts: number;
  readonly repairs: readonly string[];
  readonly usage: TokenUsage;
  readonly latencyMs: number;
}

const cacheFile = (arm: Arm) =>
  join(CACHE_DIR, `llm-${MODEL}-${EXPERIMENT_VERSION}${arm === 'vote' ? '-vote' : '-novote'}.json`);

function loadCache(arm: Arm): Map<string, CachedOutcome> {
  try {
    return new Map(Object.entries(JSON.parse(readFileSync(cacheFile(arm), 'utf8')) as Record<string, CachedOutcome>));
  } catch {
    return new Map();
  }
}

/**
 * Loads one arm, refusing to proceed on a partial cache.
 *
 * There is no `--fill`. Prompt v2 existed only for this experiment and was
 * reverted when it lost, so the tree can no longer produce these responses —
 * generating v1 responses into a file labelled v2 would compare two different
 * prompts and call the difference a result. If this genuinely needs re-running,
 * restore the v2 prompt first and say so in the commit.
 */
function armCache(arm: Arm, escalations: readonly Escalation[]): Map<string, CachedOutcome> {
  const cache = loadCache(arm);
  const missing = escalations.filter((e) => !cache.has(e.txn.rawDescriptor));
  if (missing.length > 0) {
    throw new Error(
      `arm "${arm}" is missing ${missing.length} of ${escalations.length} responses at prompt ` +
        `${EXPERIMENT_VERSION}. Both arms must be complete or the comparison is between two ` +
        `different populations. See the note on armCache: this cannot be refilled from the ` +
        `current tree.`,
    );
  }
  console.error(`  ${arm}: ${escalations.length} served from cache`);
  return cache;
}

// ── Scoring ─────────────────────────────────────────────────────────────────

interface Score {
  inbox: number;
  answered: number;
  correct: number;
  review: number;
  costUsd: number;
  confidenceSum: number;
}

function score(
  escalations: readonly Escalation[],
  cache: Map<string, CachedOutcome>,
  gate: number = LLM_GATE,
): Score {
  const s: Score = { inbox: 0, answered: 0, correct: 0, review: 0, costUsd: 0, confidenceSum: 0 };
  for (const e of escalations) {
    const o = cache.get(e.txn.rawDescriptor)!;
    s.inbox++;
    s.costUsd += costOf(o.usage, MODELS[MODEL]);
    s.confidenceSum += o.confidence ?? 0;
    if (o.status === 'ok' && (o.confidence ?? 0) >= gate) {
      s.answered++;
      if (o.category === e.txn.label.category) s.correct++;
    } else {
      s.review++;
    }
  }
  return s;
}

const pct = (n: number, d: number) => (d === 0 ? '   n/a' : `${((n / d) * 100).toFixed(1)}%`.padStart(6));
const usd = (n: number) => `$${n.toFixed(4)}`;

function reportSlice(
  label: string,
  rows: readonly Escalation[],
  caches: Record<Arm, Map<string, CachedOutcome>>,
): string[] {
  if (rows.length === 0) return [`  ${label}: none`, ``];
  const a = score(rows, caches.novote);
  const b = score(rows, caches.vote);
  const delta = (x: number, y: number, digits = 1) => {
    const d = y - x;
    return `${d >= 0 ? '+' : ''}${d.toFixed(digits)}`;
  };

  return [
    `  ${label}  (${rows.length} transactions)`,
    `                      no vote        vote        delta`,
    `    answered        ${pct(a.answered, a.inbox)}      ${pct(b.answered, b.inbox)}      ${delta((a.answered / a.inbox) * 100, (b.answered / b.inbox) * 100)} pts`,
    `    precision       ${pct(a.correct, a.answered)}      ${pct(b.correct, b.answered)}      ${delta((a.correct / Math.max(a.answered, 1)) * 100, (b.correct / Math.max(b.answered, 1)) * 100)} pts`,
    `    resolved        ${pct(a.correct, a.inbox)}      ${pct(b.correct, b.inbox)}      ${delta((a.correct / a.inbox) * 100, (b.correct / b.inbox) * 100)} pts`,
    `    to review       ${String(a.review).padStart(6)}      ${String(b.review).padStart(6)}      ${b.review - a.review >= 0 ? '+' : ''}${b.review - a.review}`,
    `    mean confidence ${(a.confidenceSum / a.inbox).toFixed(3).padStart(6)}      ${(b.confidenceSum / b.inbox).toFixed(3).padStart(6)}      ${delta(a.confidenceSum / a.inbox, b.confidenceSum / b.inbox, 3)}`,
    `    cost            ${usd(a.costUsd).padStart(6)}      ${usd(b.costUsd).padStart(6)}      ${delta(a.costUsd, b.costUsd, 4)}`,
    ``,
  ];
}

async function main(): Promise<void> {
  console.error('building inboxes…');
  const validationInbox = await inboxOf(fit, validation);
  const goldenInbox = await inboxOf(history, golden);
  const all = [...validationInbox, ...goldenInbox];

  console.error('resolving both arms…');
  const caches = {} as Record<Arm, Map<string, CachedOutcome>>;
  for (const arm of ARMS) caches[arm] = armCache(arm, all);

  const contested = (rows: readonly Escalation[]) => rows.filter((e) => e.contested);
  const uncontested = (rows: readonly Escalation[]) => rows.filter((e) => !e.contested);

  const lines: string[] = [
    `TIER 2 VOTE AS A PRIOR   what Tier 3 does with the distribution Tier 2 discards`,
    `MODEL              ${MODEL} · prompt ${EXPERIMENT_VERSION} (reverted; replayed from cache) · gate ${LLM_GATE}`,
    `ARMS               identical prompts except the vote block; neighbours present in both`,
    `INBOX              ${validationInbox.length} validation · ${goldenInbox.length} golden`,
    `                   ${contested(goldenInbox).length} of the golden inbox carry a contested vote`,
    ``,
    `NOISE FLOOR        PLANNED AND UNAVAILABLE. The design counted on a slice of keys with`,
    `                   no neighbours, whose prompts are byte-identical across arms, to read`,
    `                   the model's own nondeterminism for free. That slice is empty here — a`,
    `                   character-trigram embedder shares some feature with nearly anything,`,
    `                   so every escalation carries a vote. Deltas below have no error bar.`,
    ``,
    `VALIDATION`,
    ``,
    ...reportSlice('contested (vote handed over)', contested(validationInbox), caches),
    ...reportSlice('no vote (control — prompts identical)', uncontested(validationInbox), caches),
    `HOLDOUT`,
    ``,
    ...reportSlice('contested (vote handed over)', contested(goldenInbox), caches),
    ...reportSlice('no vote (control — prompts identical)', uncontested(goldenInbox), caches),
  ];

  // ── Precision at matched coverage ─────────────────────────────────────────
  // A fixed gate is the wrong comparison here and the tables above show why: the
  // prior moves the whole confidence distribution down (the system block tells
  // the model a split neighbourhood is a reason to stay unsure, and it complies).
  // Holding the gate at 0.90 therefore compares two different operating points
  // and reads the coverage difference as a quality difference.
  //
  // So: rank each arm by its own confidence, take the same number of answers from
  // each, and compare precision. Threshold-free, and it asks the only question
  // that matters about a confidence — is it ordered better than the other one.
  const matched = (rows: readonly Escalation[]) => {
    const ranked = (arm: Arm) =>
      rows
        .map((e) => ({ o: caches[arm].get(e.txn.rawDescriptor)!, truth: e.txn.label.category }))
        .filter((r) => r.o.status === 'ok')
        .sort((a, b) => (b.o.confidence ?? 0) - (a.o.confidence ?? 0));

    const control = ranked('novote');
    const treatment = ranked('vote');
    // The budget is what the shipped configuration answers today.
    const n = control.filter((r) => (r.o.confidence ?? 0) >= LLM_GATE).length;
    const hits = (list: typeof control) => list.slice(0, n).filter((r) => r.o.category === r.truth).length;
    return { n, control: hits(control), treatment: hits(treatment) };
  };

  // ── Third arm, free: the previous prompt version ──────────────────────────
  // v2 changed two things at once — it added the vote block *and* reworded the
  // system block (the note that similarity is lexical, and how to read a split).
  // The two arms above isolate the vote. This isolates the rewording, using the
  // v1 responses that are already committed, so the cost is zero. Without it a
  // null result on the vote would still leave the version bump unexplained.
  const v1 = new Map(
    Object.entries(
      JSON.parse(readFileSync(join(CACHE_DIR, `llm-${MODEL}-v1.json`), 'utf8')) as Record<string, CachedOutcome>,
    ),
  );
  const v1Covers = (rows: readonly Escalation[]) => rows.every((e) => v1.has(e.txn.rawDescriptor));

  lines.push(`PROMPT VERSION     v1 vs v2-without-vote, same transactions — isolates the rewording`, ``);
  for (const [label, rows] of [
    ['validation', contested(validationInbox)],
    ['holdout', contested(goldenInbox)],
  ] as const) {
    if (!v1Covers(rows)) {
      lines.push(`  ${label}: v1 cache does not cover this inbox — skipped`, ``);
      continue;
    }
    const a = score(rows, v1);
    const b = score(rows, caches.novote);
    lines.push(
      `  ${label} (${rows.length})`,
      `    v1            ${pct(a.answered, a.inbox)} answered, ${pct(a.correct, a.answered)} precision, ${pct(a.correct, a.inbox)} resolved`,
      `    v2 (no vote)  ${pct(b.answered, b.inbox)} answered, ${pct(b.correct, b.answered)} precision, ${pct(b.correct, b.inbox)} resolved`,
      ``,
    );
  }

  lines.push(`PRECISION AT MATCHED COVERAGE`, ``);
  for (const [label, rows] of [
    ['validation contested', contested(validationInbox)],
    ['holdout contested', contested(goldenInbox)],
  ] as const) {
    const m = matched(rows);
    lines.push(
      `  ${label}: top ${m.n} answers by each arm's own confidence`,
      `    no vote   ${m.control}/${m.n} = ${pct(m.control, m.n).trim()}`,
      `    vote      ${m.treatment}/${m.n} = ${pct(m.treatment, m.n).trim()}   ` +
        `(${m.treatment - m.control >= 0 ? '+' : ''}${m.treatment - m.control} answers)`,
      ``,
    );
  }

  // ── Did it anchor or inform? ──────────────────────────────────────────────
  // The failure mode worth naming: a prior can improve a metric by making the
  // model defer to a vote that Tier 2 itself judged too split to trust. If the
  // model simply adopts the vote's winner, the tier has not gained a
  // second opinion, it has laundered the first one through a more expensive
  // component. So the question is not only "is it more accurate" but "is it
  // still disagreeing, and is it right when it does".
  const contestedGolden = contested(goldenInbox);
  let followed = 0;
  let followedCorrect = 0;
  let deviated = 0;
  let deviatedCorrect = 0;
  let voteItselfCorrect = 0;
  for (const e of contestedGolden) {
    const o = caches.vote.get(e.txn.rawDescriptor)!;
    if (o.status !== 'ok') continue;
    const prior = e.inputs.vote.vote!;
    if (prior.category === e.txn.label.category) voteItselfCorrect++;
    if (o.category === prior.category) {
      followed++;
      if (o.category === e.txn.label.category) followedCorrect++;
    } else {
      deviated++;
      if (o.category === e.txn.label.category) deviatedCorrect++;
    }
  }

  lines.push(
    `ANCHORING CHECK    holdout contested slice, ungated`,
    `  the vote alone would have been right ${pct(voteItselfCorrect, contestedGolden.length).trim()} of the time`,
    `  model followed it   ${String(followed).padStart(4)} times, right ${pct(followedCorrect, followed).trim()}`,
    `  model overrode it   ${String(deviated).padStart(4)} times, right ${pct(deviatedCorrect, deviated).trim()}`,
    ``,
    `  A model that follows a contested vote every time has not added a second opinion,`,
    `  it has relabelled the first one at Tier 3 prices. Overriding, and being right more`,
    `  often than the vote was, is the behaviour that justifies the handover.`,
    ``,
  );

  // ── Does the prior let the accept gate move? ──────────────────────────────
  // On validation only. The gate is a selected parameter and the holdout does
  // not get a vote on it.
  lines.push(
    `GATE SWEEP         validation contested slice — does a better-informed confidence`,
    `                   let the accept gate move?`,
    ``,
    `  gate        no vote: cov/prec        vote: cov/prec`,
  );
  const vc = contested(validationInbox);
  for (const gate of GATE_CANDIDATES) {
    const a = score(vc, caches.novote, gate);
    const b = score(vc, caches.vote, gate);
    lines.push(
      `  ${gate.toFixed(2)}       ${pct(a.answered, a.inbox)} / ${pct(a.correct, a.answered)}        ${pct(b.answered, b.inbox)} / ${pct(b.correct, b.answered)}`,
    );
  }

  console.log(lines.join('\n'));
}

main();
