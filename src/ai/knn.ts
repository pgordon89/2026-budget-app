/**
 * Tier 2 — nearest-neighbour vote over merchant keys.
 *
 * Tier 1 answers merchants it has seen before. This one answers merchants it has
 * *not*, by finding the labeled merchants whose keys look most like the query
 * and letting them vote in proportion to how near they are. `PEETS COFFEE` has
 * never appeared in the user's history; `BLUE BOTTLE COFFEE` and `SIGHTGLASS
 * COFFEE` have, and they are close enough to be worth asking.
 *
 * Three decisions shape it.
 *
 * **The exact key is excluded from its own neighbourhood.** Without that, a
 * merchant Tier 1 refused to answer for — `AMZN MKTP US`, genuinely split
 * across three categories — reappears here as its own nearest neighbour at
 * similarity 1.0, and this tier confidently repeats the answer the previous tier
 * correctly declined to give. That is a pipeline routing around its own
 * abstention, which is worse than having no abstention at all. Each tier
 * answers a question the one before it could not.
 *
 * **It shares Tier 1's store rather than building a second one.** Same corpus,
 * different access path: exact lookup there, approximate here. A user
 * correction cannot then be live in one tier and stale in the other.
 *
 * **Confidence here is a heuristic, and is labelled as one.** Tier 1's score is
 * a Wilson lower bound — an actual statistical statement. This one is
 * `agreement × nearest-similarity`, which is a reasonable shape but not a bound
 * on anything, so it is checked the only way a heuristic can be: by measuring
 * observed accuracy per confidence bucket. See `npm run analyze:knn`.
 */

import { normalizeDescriptor } from './normalize.js';
import { type Embedder } from './embed.js';
import type { MerchantMemory } from './memory.js';
import type { CategoryId } from '../core/taxonomy.js';
import type { Prediction } from './types.js';

export interface NeighbourConfig {
  /** Neighbours allowed to vote. */
  readonly k: number;
  /** Minimum confidence to answer instead of escalating to the LLM tier. */
  readonly minConfidence: number;
  /**
   * Marginal cost attributed to one lookup, for the router's cost accounting.
   * Zero for the offline lexical embedder; non-zero once a hosted embedding
   * model is wired in, where it is the per-query embed cost after cache misses.
   */
  readonly costPerLookupUsd: number;
}

/**
 * Selected by sweep on a validation slice of history, never on the golden
 * holdout — `npm run analyze:knn`. k and the gate are chosen together because
 * they trade against each other: more voters smooth out a single bad neighbour
 * but dilute a good one, so the best gate moves with k.
 */
export const DEFAULT_NEIGHBOUR_CONFIG: NeighbourConfig = {
  k: 3,
  minConfidence: 0.5,
  costPerLookupUsd: 0,
};

export interface ScoredNeighbour {
  readonly key: string;
  readonly similarity: number;
}

export type NeighbourOutcome =
  | {
      readonly status: 'hit' | 'low_confidence';
      readonly key: string;
      readonly category: CategoryId;
      readonly confidence: number;
      /** Winner's share of the total vote weight. */
      readonly agreement: number;
      /** Similarity of the closest voting neighbour. */
      readonly nearest: number;
      readonly neighbours: readonly ScoredNeighbour[];
    }
  /** No labeled merchant shares a single feature with this key. */
  | { readonly status: 'no_neighbours'; readonly key: string }
  /** Tier 0 could not produce a usable key. */
  | { readonly status: 'degenerate'; readonly key: string };

/** Cosine similarity between a sparse indexed row and a dense query, both unit length. */
function sparseDot(row: { indices: Int32Array; values: Float32Array }, query: Float32Array): number {
  let total = 0;
  for (let j = 0; j < row.indices.length; j++) total += row.values[j]! * query[row.indices[j]!]!;
  return total;
}

export class NeighbourIndex {
  /** Per-key memo. Lookups repeat heavily — a corpus of 1,400 transactions
   *  resolves to a few hundred distinct keys — and the vote is pure. */
  private readonly cache = new Map<string, NeighbourOutcome>();

  private constructor(
    private readonly config: NeighbourConfig,
    private readonly embedder: Embedder,
    private readonly keys: readonly string[],
    private readonly distributions: readonly ReadonlyMap<CategoryId, number>[],
    /**
     * Indexed vectors in sparse form.
     *
     * The scan is brute force — a few hundred merchants does not justify an ANN
     * structure, and exactness is worth more here than the constant factor. But
     * a lexical vector carries ~40 non-zeros in a several-thousand-dimension
     * space, so a dense scan spends >99% of its multiplies on zeros. Iterating
     * the non-zeros instead makes the scan proportional to real features rather
     * than to vocabulary size. A hosted embedder's dense vectors cost the same
     * either way, so this stays a single code path.
     */
    private readonly rows: readonly { indices: Int32Array; values: Float32Array }[],
  ) {}

  static async build(
    memory: MerchantMemory,
    embedder: Embedder,
    config: Partial<NeighbourConfig> = {},
  ): Promise<NeighbourIndex> {
    const keys: string[] = [];
    const distributions: ReadonlyMap<CategoryId, number>[] = [];
    for (const entry of memory.entries()) {
      keys.push(entry.key);
      distributions.push(entry.distribution);
    }

    const rows = (await embedder.embed(keys)).map((vector) => {
      const indices: number[] = [];
      const values: number[] = [];
      for (let i = 0; i < vector.length; i++) {
        if (vector[i] !== 0) {
          indices.push(i);
          values.push(vector[i]!);
        }
      }
      return { indices: Int32Array.from(indices), values: Float32Array.from(values) };
    });

    return new NeighbourIndex(
      { ...DEFAULT_NEIGHBOUR_CONFIG, ...config },
      embedder,
      keys,
      distributions,
      rows,
    );
  }

  async lookup(rawDescriptor: string): Promise<NeighbourOutcome> {
    const { key, degenerate } = normalizeDescriptor(rawDescriptor);
    if (degenerate) return { status: 'degenerate', key };

    const cached = this.cache.get(key);
    if (cached) return cached;

    const outcome = await this.vote(key);
    this.cache.set(key, outcome);
    return outcome;
  }

  /** Router-facing form. `null` means escalate to the LLM tier. */
  async predict(rawDescriptor: string): Promise<Prediction | null> {
    const outcome = await this.lookup(rawDescriptor);
    if (outcome.status !== 'hit') return null;
    return {
      category: outcome.category,
      confidence: outcome.confidence,
      tier: 'embedding',
      costUsd: this.config.costPerLookupUsd,
    };
  }

  private async vote(key: string): Promise<NeighbourOutcome> {
    const [query] = await this.embedder.embed([key]);

    const scored: Array<{ index: number; similarity: number }> = [];
    for (let i = 0; i < this.keys.length; i++) {
      // The exclusion that keeps this tier from overruling Tier 1's abstention.
      if (this.keys[i] === key) continue;
      const value = sparseDot(this.rows[i]!, query!);
      // Vectors are non-negative, so zero means "shares no feature at all".
      if (value > 0) scored.push({ index: i, similarity: value });
    }

    if (scored.length === 0) return { status: 'no_neighbours', key };

    // Ties broken by key so the neighbourhood is reproducible across runs.
    scored.sort((a, b) => b.similarity - a.similarity || (this.keys[a.index]! < this.keys[b.index]! ? -1 : 1));
    const voters = scored.slice(0, this.config.k);

    // Each neighbour spends its similarity across its own label distribution, so
    // a merchant that is itself ambiguous casts a split vote rather than a whole
    // one for its plurality label.
    const votes = new Map<CategoryId, number>();
    let totalWeight = 0;
    for (const voter of voters) {
      const distribution = this.distributions[voter.index]!;
      let mass = 0;
      for (const weight of distribution.values()) mass += weight;
      if (mass <= 0) continue;
      for (const [category, weight] of distribution) {
        votes.set(category, (votes.get(category) ?? 0) + voter.similarity * (weight / mass));
      }
      totalWeight += voter.similarity;
    }

    if (totalWeight <= 0) return { status: 'no_neighbours', key };

    let winner: CategoryId | undefined;
    let winnerVote = 0;
    for (const [category, vote] of votes) {
      if (vote > winnerVote || (vote === winnerVote && (winner === undefined || category < winner))) {
        winner = category;
        winnerVote = vote;
      }
    }
    if (winner === undefined) return { status: 'no_neighbours', key };

    const agreement = winnerVote / totalWeight;
    const nearest = voters[0]!.similarity;
    // Both factors are necessary. Unanimous neighbours that are all far away are
    // agreeing about the wrong neighbourhood; one very near neighbour amid
    // disagreement is a coin flip with a confident-looking top match.
    const confidence = agreement * nearest;

    return {
      status: confidence >= this.config.minConfidence ? 'hit' : 'low_confidence',
      key,
      category: winner,
      confidence,
      agreement,
      nearest,
      neighbours: voters.map((v) => ({ key: this.keys[v.index]!, similarity: v.similarity })),
    };
  }

  /** Labeled merchant keys available to vote. */
  get size(): number {
    return this.keys.length;
  }
}
