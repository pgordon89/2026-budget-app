/**
 * Tier 1 — merchant memory.
 *
 * An exact normalized-key → category store, seeded from the user's own labeled
 * history. It answers the large, boring majority of transactions for zero
 * marginal cost: the same twelve merchants account for most of a household's
 * month, and once each has been seen labeled a few times there is nothing for a
 * model to add.
 *
 * Two decisions carry the weight here.
 *
 * **It abstains.** A key that has been seen is not the same as a key that has
 * been *answered*. `AMZN MKTP US` appears hundreds of times in history and is
 * genuinely split across household supplies, electronics, and general
 * merchandise; the plurality label is right barely half the time. Returning it
 * anyway would be a confident wrong answer, which is strictly worse than a cache
 * miss — a miss costs a fraction of a cent at the next tier, a wrong answer
 * corrupts a budget silently. So a hit has to clear a confidence gate, and the
 * threshold that gate uses is measured (`npm run analyze:memory`), not guessed.
 *
 * **Confidence is a lower bound, not a ratio.** The obvious score — winner count
 * over total count — rates a key seen once at 100%, which is exactly backwards:
 * one observation is the least evidence the store can have. This uses the Wilson
 * score lower bound instead, so confidence rises with agreement *and* with
 * sample size, and a single sighting scores ~0.06 rather than 1.0. That one
 * change is what lets the store warm up safely from an empty state.
 *
 * In-memory for Phase 2. The shape is deliberately a Postgres table waiting to
 * happen: one row per (key, category) with a weight and a count, upserted on
 * write.
 */

import { normalizeDescriptor } from './normalize.js';
import { isCategoryId, type CategoryId } from '../core/taxonomy.js';
import type { Prediction } from './types.js';

/**
 * Where an observation came from.
 *
 * `confirmed` — a human said so: imported history, or a user correction in the
 * review queue. Independent evidence.
 *
 * `inferred` — a downstream tier's own output, written back so the merchant is
 * cheap next month. This is the mechanism behind "the system gets cheaper as it
 * is used", but it is also a feedback loop: counted at full weight, the store
 * would grow confident by re-reading its own guesses, and confidence would climb
 * on zero new information. Hence the discount below.
 */
export type ObservationSource = 'confirmed' | 'inferred';

export interface MemoryConfig {
  /**
   * Minimum confidence to answer instead of escalating.
   *
   * The default is selected, not chosen: `npm run analyze:memory` sweeps the
   * gate on a validation slice carved out of history and takes the widest
   * coverage that still clears a 97% precision floor. Deliberately not tuned on
   * the golden holdout — a threshold fitted to the test set makes the accuracy
   * it produces an estimate of nothing.
   *
   * Raising it buys precision and spends money at the next tier.
   */
  readonly minConfidence: number;
  /**
   * Weight of one `inferred` observation relative to one `confirmed` one.
   *
   * A chosen prior, not a fitted constant: the fixture corpus is entirely
   * confirmed history, so nothing in the eval can justify a value and pretending
   * otherwise would be dishonest. 0.25 encodes "one user correction outweighs
   * three of the pipeline's own prior guesses", and combined with the Wilson
   * bound it means a merchant seen once by the LLM tier still cannot answer —
   * it takes repeated agreement before the cache is trusted.
   */
  readonly inferredWeight: number;
  /** Wilson z. 1.96 ≈ 95% one-sided-ish; larger = more conservative on small n. */
  readonly z: number;
}

export const DEFAULT_MEMORY_CONFIG: MemoryConfig = {
  minConfidence: 0.3,
  inferredWeight: 0.25,
  z: 1.96,
};

const SOURCE_WEIGHT = (config: MemoryConfig, source: ObservationSource): number =>
  source === 'confirmed' ? 1 : config.inferredWeight;

/** Aggregate for one (key, category) pair — one Postgres row, eventually. */
interface Tally {
  /** Sum of source weights. The effective sample size confidence is computed on. */
  weight: number;
  /** Raw sighting count. Reported for debugging; never used in the score. */
  count: number;
}

export type MemoryOutcome =
  /** Confident enough to answer. */
  | {
      readonly status: 'hit';
      readonly key: string;
      readonly category: CategoryId;
      readonly confidence: number;
      /** Effective (weighted) sample size behind the winner's key. */
      readonly support: number;
      /** Winner's share of that support, ungated. Diagnostic only. */
      readonly agreement: number;
    }
  /** Key known, but the evidence is thin or split. The interesting escalation. */
  | {
      readonly status: 'low_confidence';
      readonly key: string;
      readonly category: CategoryId;
      readonly confidence: number;
      readonly support: number;
      readonly agreement: number;
    }
  /** Never seen this merchant. The unavoidable escalation. */
  | { readonly status: 'unseen'; readonly key: string }
  /** Tier 0 could not produce a usable key. Refusing to key on garbage. */
  | { readonly status: 'degenerate'; readonly key: string };

/**
 * Wilson score interval, lower bound.
 *
 * Chosen over a raw proportion because the raw proportion cannot tell 1/1 from
 * 100/100, and over a fixed `minSupport` threshold because that would be a
 * second knob doing a worse version of the same job — Wilson already collapses
 * toward zero as n shrinks, continuously, with no cliff to tune.
 *
 * Accepts fractional counts: `inferred` observations contribute partial weight,
 * so n here is an effective sample size rather than an integer.
 */
export function wilsonLowerBound(successes: number, total: number, z: number): number {
  if (total <= 0) return 0;
  const p = successes / total;
  const z2 = z * z;
  const denominator = 1 + z2 / total;
  const centre = p + z2 / (2 * total);
  const margin = z * Math.sqrt((p * (1 - p) + z2 / (4 * total)) / total);
  return Math.max(0, Math.min(1, (centre - margin) / denominator));
}

export interface TallyScore {
  readonly category: CategoryId;
  readonly confidence: number;
  readonly support: number;
  readonly agreement: number;
}

/**
 * Turns a category-weight distribution into a scored winner.
 *
 * Extracted so the in-memory store and the Postgres-backed one cannot drift.
 * They hold their tallies differently — a Map in one, rows in the other — but a
 * merchant must score identically either way, and the only way to guarantee that
 * is for there to be one implementation rather than two that agree today.
 *
 * Returns undefined when there is nothing to score.
 */
export function scoreTallies(
  distribution: Iterable<readonly [CategoryId, number]>,
  z: number,
): TallyScore | undefined {
  let winner: CategoryId | undefined;
  let winnerWeight = 0;
  let totalWeight = 0;

  for (const [category, weight] of distribution) {
    totalWeight += weight;
    // Deterministic winner: weight desc, then category id asc. The tie-break is
    // not cosmetic — without it a 50/50 key resolves by iteration order, and
    // eval numbers move when row order changes.
    if (weight > winnerWeight || (weight === winnerWeight && (winner === undefined || category < winner))) {
      winner = category;
      winnerWeight = weight;
    }
  }

  if (winner === undefined || totalWeight <= 0) return undefined;

  return {
    category: winner,
    confidence: wilsonLowerBound(winnerWeight, totalWeight, z),
    support: totalWeight,
    agreement: winnerWeight / totalWeight,
  };
}

export class MerchantMemory {
  private readonly config: MemoryConfig;
  private readonly byKey = new Map<string, Map<CategoryId, Tally>>();

  constructor(config: Partial<MemoryConfig> = {}) {
    this.config = { ...DEFAULT_MEMORY_CONFIG, ...config };
  }

  /**
   * Rebuilds a store from already-weighted tallies — the rows a database holds.
   *
   * Deliberately not `remember()` in a loop: that would re-apply source weighting
   * to weights already discounted when they were first recorded, so a store
   * round-tripped through Postgres would score differently from the one that
   * wrote it.
   */
  static fromTallies(
    rows: Iterable<{ merchantKey: string; categoryId: CategoryId; weight: number; count: number }>,
    config: Partial<MemoryConfig> = {},
  ): MerchantMemory {
    const memory = new MerchantMemory(config);
    for (const row of rows) {
      const tallies = memory.byKey.get(row.merchantKey) ?? new Map<CategoryId, Tally>();
      tallies.set(row.categoryId, { weight: row.weight, count: row.count });
      memory.byKey.set(row.merchantKey, tallies);
    }
    return memory;
  }

  /** The weight one observation from this source contributes. */
  weightOf(source: ObservationSource): number {
    return SOURCE_WEIGHT(this.config, source);
  }

  /**
   * Record one labeled sighting.
   *
   * Takes the raw descriptor rather than a pre-normalized key on purpose: the
   * store owns its own key derivation, so a caller cannot half-normalize on the
   * write path and fully normalize on the read path and quietly miss forever.
   */
  remember(rawDescriptor: string, category: CategoryId, source: ObservationSource = 'confirmed'): void {
    if (!isCategoryId(category)) {
      // A bad id means a caller skipped validation upstream. Failing loudly here
      // is cheaper than discovering an unscoreable label in an eval report.
      throw new Error(`refusing to remember unknown category id: ${category}`);
    }

    const { key, degenerate } = normalizeDescriptor(rawDescriptor);
    // Degenerate keys are near-collisions by construction — whatever is left of
    // the descriptor identifies no merchant, so anything stored under it will be
    // served to unrelated transactions later.
    if (degenerate) return;

    const tallies = this.byKey.get(key) ?? new Map<CategoryId, Tally>();
    const tally = tallies.get(category) ?? { weight: 0, count: 0 };
    tally.weight += SOURCE_WEIGHT(this.config, source);
    tally.count += 1;
    tallies.set(category, tally);
    this.byKey.set(key, tallies);
  }

  /**
   * Drop everything known about a merchant.
   *
   * The recategorization path: when a user reassigns a merchant wholesale, the
   * accumulated inferred weight behind the old answer should not have to be
   * out-voted one correction at a time.
   */
  forget(rawDescriptor: string): boolean {
    return this.byKey.delete(normalizeDescriptor(rawDescriptor).key);
  }

  /** Full outcome, including why an escalation happened. What the eval scores. */
  lookup(rawDescriptor: string): MemoryOutcome {
    const { key, degenerate } = normalizeDescriptor(rawDescriptor);
    if (degenerate) return { status: 'degenerate', key };

    const tallies = this.byKey.get(key);
    if (!tallies || tallies.size === 0) return { status: 'unseen', key };

    const score = scoreTallies(
      [...tallies].map(([category, tally]) => [category, tally.weight] as const),
      this.config.z,
    );
    if (score === undefined) return { status: 'unseen', key };

    return score.confidence >= this.config.minConfidence
      ? { status: 'hit', key, ...score }
      : { status: 'low_confidence', key, ...score };
  }

  /** Router-facing form. `null` means escalate — for any of the three reasons. */
  predict(rawDescriptor: string): Prediction | null {
    const outcome = this.lookup(rawDescriptor);
    if (outcome.status !== 'hit') return null;
    return {
      category: outcome.category,
      confidence: outcome.confidence,
      tier: 'memory',
      costUsd: 0,
    };
  }

  /**
   * Every known key with its label distribution, as weights.
   *
   * Exists for Tier 2, which is this same store queried by approximate key
   * match instead of exact. Sharing the corpus rather than building a second
   * one keeps "what the system knows about a merchant" in a single place, so a
   * user correction cannot be visible to one tier and stale in the other.
   */
  *entries(): Generator<{ key: string; distribution: ReadonlyMap<CategoryId, number> }> {
    for (const [key, tallies] of this.byKey) {
      const distribution = new Map<CategoryId, number>();
      for (const [category, tally] of tallies) distribution.set(category, tally.weight);
      yield { key, distribution };
    }
  }

  /** Distinct merchant keys held. */
  get size(): number {
    return this.byKey.size;
  }

  /** Total sightings recorded, degenerate descriptors excluded. */
  get observations(): number {
    let total = 0;
    for (const tallies of this.byKey.values()) {
      for (const tally of tallies.values()) total += tally.count;
    }
    return total;
  }
}
