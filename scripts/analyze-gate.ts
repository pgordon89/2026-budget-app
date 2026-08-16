/**
 * Selects Tier 1's gate on a population that contains freshly-learned merchants.
 *
 *   npm run analyze:gate
 *
 * `analyze:memory` chose `minConfidence: 0.30` by sweeping a store seeded with 18
 * months of history and scoring the next 6. In that population every key that
 * clears the gate has dozens of observations behind it, so the gate was never
 * asked what happens when a merchant is known from two corrections. The answer,
 * measured afterwards by `analyze:learning`, was 64.7% precision on exactly those
 * answers — against the 97% floor everything else here is held to.
 *
 * The defect is not the threshold value. It is that the threshold was selected
 * against a population the deployed system does not have: once the correction
 * loop is running, thin merchants exist continuously. So this re-runs selection
 * on a replay that *creates* them — corrections write back mid-walk, exactly as
 * production does — and sweeps `minConfidence` against `minAgreement` together,
 * because they trade against each other and picking one at a time would find a
 * local answer.
 *
 * Scored on Tiers 1 and 2 only. Tier 3 is excluded deliberately: changing the
 * gate changes which transactions escalate, and the committed response cache
 * covers the escalations of the *baseline* configuration. A tighter gate would
 * escalate transactions with no cached response, and counting those as
 * unanswered would penalise tight gates for a bookkeeping artefact rather than
 * for anything about their behaviour. The defect being fixed is Tier 1 answering
 * on thin evidence, and Tiers 1+2 are where that is visible.
 *
 * The sweep runs against the in-memory store rather than Postgres. They are the
 * same scoring code, and there is a conformance suite asserting the two agree on
 * every Tier 1 behaviour; this is 30 replays and the round trips would dominate.
 */

import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { MerchantMemory, DEFAULT_MEMORY_CONFIG, wilsonLowerBound } from '../src/ai/memory.js';
import { fitLexicalEmbedder } from '../src/ai/embed.js';
import { NeighbourIndex, DEFAULT_NEIGHBOUR_CONFIG } from '../src/ai/knn.js';
import { normalizeDescriptor } from '../src/ai/normalize.js';
import type { SyntheticTransaction } from '../src/synthetic/generator.js';

const DATA = join(dirname(fileURLToPath(import.meta.url)), '..', 'evals', 'datasets');
const load = (f: string): SyntheticTransaction[] => JSON.parse(readFileSync(join(DATA, f), 'utf8'));

/**
 * A thinner seed and a longer walk than the other analyses use.
 *
 * The first attempt at this reused the 12-month-seed / 6-month-walk split from
 * `analyze:memory`, and the defect did not reproduce: freshly-learned merchants
 * were answered at 100% precision under every setting, including a support floor
 * of 10 that makes it arithmetically impossible for Tier 1 to answer one at all.
 * Every fresh answer was coming from Tier 2, so the sweep was measuring nothing
 * about the parameter it was selecting.
 *
 * The reason is recurrence. A merchant learned from a correction only becomes a
 * Tier 1 answer if it appears again, often enough, before the walk ends — and six
 * months is not long enough for that to happen much. The population this
 * parameter governs is a store that is still learning, so the split is chosen to
 * produce one: six months of seed, twelve of walk. Still entirely inside history;
 * the golden holdout is not touched.
 */
const VALIDATION_SPLIT = '2024-08-01';
const history = load('history.json');
const fit = history.filter((t) => t.date < VALIDATION_SPLIT);
const validation = history.filter((t) => t.date >= VALIDATION_SPLIT);

/**
 * Which tier's gate is being selected. `--tier=2` fixes Tier 1 at its own
 * selected values and sweeps Tier 2 — sequential rather than joint, because the
 * cascade is sequential: Tier 2 only ever sees what Tier 1 declined, so its
 * population is defined by Tier 1's settings and not the reverse.
 */
const TIER = process.argv.includes('--tier=2') ? 2 : 1;

const CONFIDENCE_CANDIDATES = TIER === 1 ? [0.2, 0.3, 0.4, 0.5] : [0.3, 0.4, 0.5, 0.6];
const AGREEMENT_CANDIDATES = [0, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0];

/**
 * The floor applies to both numbers. Overall precision alone is what hid this
 * defect for two commits: 17 bad answers inside 1,400 good ones moves the
 * aggregate by tenths of a point. Freshly-learned merchants are a small
 * population with an outsized ability to corrupt a ledger, so they get their own
 * constraint rather than being averaged into everyone else's.
 */
const PRECISION_FLOOR = 0.97;

/**
 * A merchant seen exactly once must never answer, whatever the sweep says.
 *
 * This is a constraint the corpus cannot argue with, which is why it is stated
 * rather than measured. A single observation carries no information about a
 * merchant's variance — it cannot distinguish a merchant that is always
 * groceries from one that is groceries this time. The fixture's singleton
 * merchants happen to be consistent, so a sweep scores n=1 answers as accurate
 * and would happily select a gate that admits them; the next corpus with a
 * mixed-basket singleton would pay for that silently.
 *
 * Encoded here as part of the selection policy rather than applied afterwards by
 * hand, so the script still picks the settings and the rule stays reproducible.
 */
const REFUSES_SINGLE_SIGHTING = (minConfidence: number) =>
  wilsonLowerBound(1, 1, DEFAULT_MEMORY_CONFIG.z) < minConfidence;

interface Outcome {
  readonly answered: boolean;
  readonly correct: boolean;
  readonly fromMemory: boolean;
}

interface Result {
  minConfidence: number;
  minAgreement: number;
  transactions: number;
  answered: number;
  correct: number;
  /** Answers that exist only because corrections wrote back. The defect lives here. */
  marginal: number;
  marginalCorrect: number;
  marginalFromMemory: number;
  corrections: number;
}

const monthOf = (iso: string) => iso.slice(0, 7);

async function main(): Promise<void> {
  const results: Result[] = [];
  for (const minConfidence of CONFIDENCE_CANDIDATES) {
    for (const minAgreement of AGREEMENT_CANDIDATES) {
      const learning = await replay(minConfidence, minAgreement, true);
      const control = await replay(minConfidence, minAgreement, false);

      let marginal = 0;
      let marginalCorrect = 0;
      let marginalFromMemory = 0;
      for (const [id, l] of learning.outcomes) {
        if (!l.answered || (control.outcomes.get(id)?.answered ?? false)) continue;
        marginal++;
        if (l.correct) marginalCorrect++;
        if (l.fromMemory) marginalFromMemory++;
      }

      results.push({
        minConfidence,
        minAgreement,
        transactions: validation.length,
        answered: learning.answered,
        correct: learning.correct,
        marginal,
        marginalCorrect,
        marginalFromMemory,
        corrections: learning.corrections,
      });
    }
  }

  const pct = (n: number, d: number) => (d === 0 ? '   n/a' : `${((n / d) * 100).toFixed(1)}%`.padStart(6));

  const passes = (r: Result) =>
    r.answered > 0 &&
    (TIER === 2 || REFUSES_SINGLE_SIGHTING(r.minConfidence)) &&
    r.correct / r.answered >= PRECISION_FLOOR &&
    (r.marginal === 0 || r.marginalCorrect / r.marginal >= PRECISION_FLOOR);

  const lines: string[] = [
    `GATE SELECTION     selecting Tier ${TIER}'s gate against the population write-back creates`,
    `FIXED              ${TIER === 2 ? `tier 1 at ${DEFAULT_MEMORY_CONFIG.minConfidence}/${DEFAULT_MEMORY_CONFIG.minAgreement}` : 'nothing — this is the first tier'}`,
    `SPLITS             fit ${fit.length} → validation ${validation.length}, walked in date order`,
    `METHOD             two arms per setting (write-back on / off); the marginal set is every`,
    `                   answer that exists only because corrections reached the store`,
    `SCORED ON          tiers 1+2 (tier 3 excluded: a changed gate escalates transactions the`,
    `                   committed response cache never covered)`,
    ``,
    `  conf  agree   coverage  precision    marginal   marg prec   of which t1   corrections`,
  ];

  for (const r of results) {
    lines.push(
      `  ${r.minConfidence.toFixed(2)}  ${r.minAgreement.toFixed(2)}    ${pct(r.answered, r.transactions)}     ${pct(r.correct, r.answered)}      ` +
        `${String(r.marginal).padStart(6)}     ${pct(r.marginalCorrect, r.marginal)}        ${String(r.marginalFromMemory).padStart(6)}       ${String(r.corrections).padStart(4)}  ${passes(r) ? '✓' : ''}`,
    );
  }

  // Primary rule: widest coverage clearing both floors.
  const chosen = results.filter(passes).sort((a, b) => b.answered - a.answered || a.minAgreement - b.minAgreement)[0];

  // Fallback, used when the marginal floor is unreachable at any setting. That is
  // not a bug in the sweep — on a corpus with genuine merchant churn the answers
  // write-back creates are the hard residue by construction, and no threshold
  // makes them 97% precise. Falling back to the overall floor keeps the tier
  // shippable and reports the gap instead of hiding it behind "no setting".
  const fallbackPasses = (r: Result) =>
    r.answered > 0 &&
    (TIER === 2 || REFUSES_SINGLE_SIGHTING(r.minConfidence)) &&
    r.correct / r.answered >= PRECISION_FLOOR;
  const fallback = results.filter(fallbackPasses).sort((a, b) => b.answered - a.answered || b.minAgreement - a.minAgreement)[0];
  const bestMarginal = results
    .filter((r) => r.marginal >= 20)
    .sort((a, b) => b.marginalCorrect / b.marginal - a.marginalCorrect / a.marginal)[0];

  lines.push(
    ``,
    `CONSTRAINTS        ${(PRECISION_FLOOR * 100).toFixed(0)}% on overall precision AND on the marginal set,`,
    `                   and the gate must refuse a merchant seen exactly once`,
    `                   (Wilson(1,1) = ${wilsonLowerBound(1, 1, DEFAULT_MEMORY_CONFIG.z).toFixed(3)}, so minConfidence must exceed it)`,
    `                   the marginal floor is the point: a handful of bad answers inside a`,
    `                   thousand good ones moves an aggregate by tenths of a point`,
    ``,
  );

  if (chosen === undefined) {
    lines.push(
      `MARGINAL FLOOR UNREACHABLE`,
      `  No setting reaches ${(PRECISION_FLOOR * 100).toFixed(0)}% on the answers write-back creates.`,
      bestMarginal
        ? `  Best achievable: ${pct(bestMarginal.marginalCorrect, bestMarginal.marginal).trim()} at minConfidence ${bestMarginal.minConfidence.toFixed(2)} / minAgreement ${bestMarginal.minAgreement.toFixed(2)} over ${bestMarginal.marginal} answers.`
        : `  Too few marginal answers at any setting to say more.`,
      `  That population is the hard residue by construction — merchants learned from a`,
      `  handful of corrections, and contested near-miss neighbourhoods. No threshold`,
      `  makes it 97% precise; only a different mechanism would.`,
      ``,
      fallback
        ? `FALLING BACK       to the overall-precision floor alone`
        : `NO SETTING clears even the overall floor.`,
    );
  }
  const selected = chosen ?? fallback;
  if (selected !== undefined) {
    const chosenRow = selected;
    lines.push(
      `SELECTED           minConfidence ${chosenRow.minConfidence.toFixed(2)} · minAgreement ${chosenRow.minAgreement}`,
      `  coverage         ${pct(chosenRow.answered, chosenRow.transactions).trim()}`,
      `  precision        ${pct(chosenRow.correct, chosenRow.answered).trim()}`,
      `  marginal         ${pct(chosenRow.marginalCorrect, chosenRow.marginal).trim()} over ${chosenRow.marginal} answers write-back created`,
      ``,
      `CURRENT DEFAULT    minConfidence ${DEFAULT_MEMORY_CONFIG.minConfidence} · minAgreement ${DEFAULT_MEMORY_CONFIG.minAgreement}`,
    );
  }

  console.log(lines.join('\n'));
}

// ── The replay ──────────────────────────────────────────────────────────────
// Async only because NeighbourIndex is, for a hosted embedder that does not
// exist yet; the lexical one resolves immediately.

async function replay(
  minConfidence: number,
  minAgreement: number,
  learns: boolean,
): Promise<{ outcomes: Map<string, Outcome>; answered: number; correct: number; corrections: number }> {
  const memory =
    TIER === 1 ? new MerchantMemory({ minConfidence, minAgreement }) : new MerchantMemory();
  for (const t of fit) memory.remember(t.rawDescriptor, t.label.category);

  const tier2Config =
    TIER === 2 ? { minConfidence, minAgreement } : {};

  let index = await buildIndex(memory, tier2Config);
  let indexMonth = monthOf(validation[0]!.date);

  const outcomes = new Map<string, Outcome>();
  let answered = 0;
  let correct = 0;
  let corrections = 0;

  for (const txn of validation) {
    const month = monthOf(txn.date);
    if (month !== indexMonth) {
      index = await buildIndex(memory, tier2Config);
      indexMonth = month;
    }

    let predicted: string | undefined;
    let fromMemory = false;
    const tier1 = memory.lookup(txn.rawDescriptor);
    if (tier1.status === 'hit') {
      predicted = tier1.category;
      fromMemory = true;
    } else {
      const tier2 = await index.lookup(txn.rawDescriptor);
      if (tier2.status === 'hit') predicted = tier2.category;
    }

    const right = predicted === txn.label.category;
    outcomes.set(txn.id, { answered: predicted !== undefined, correct: right, fromMemory });

    if (predicted !== undefined) {
      answered++;
      if (right) correct++;
      continue;
    }

    corrections++;
    if (learns) memory.remember(txn.rawDescriptor, txn.label.category, 'confirmed');
  }

  return { outcomes, answered, correct, corrections };
}

async function buildIndex(
  memory: MerchantMemory,
  config: Partial<typeof DEFAULT_NEIGHBOUR_CONFIG>,
): Promise<NeighbourIndex> {
  const embedder = fitLexicalEmbedder([...memory.entries()].map((e) => e.key));
  return NeighbourIndex.build(memory, embedder, config);
}

main();
