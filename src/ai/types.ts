/**
 * The contract every categorization tier answers with.
 *
 * Kept in its own module so tiers do not import each other just to agree on a
 * return shape, and so the eval harness has one thing to score.
 *
 * The fields are chosen to make the routing claim falsifiable. "~90% of
 * transactions never reach a language model" is only checkable if every
 * prediction carries the tier that produced it and what that tier cost, so both
 * are required rather than optional.
 */

import type { CategoryId } from '../core/taxonomy.js';

/** Ordered cheapest → most expensive. Tier 0 (normalization) never predicts on
 *  its own; it produces the key the other tiers key off. */
export const TIERS = ['memory', 'embedding', 'llm'] as const;

export type Tier = (typeof TIERS)[number];

export interface Prediction {
  readonly category: CategoryId;
  /**
   * Calibrated-ish 0..1. Comparable *within* a tier; across tiers it is only
   * ever used against that tier's own accept threshold, never to pick a winner
   * between two tiers that both answered. Tiers are tried in cost order and the
   * first one to clear its threshold wins, so cross-tier comparison never arises.
   */
  readonly confidence: number;
  readonly tier: Tier;
  /** Marginal cost of producing this prediction. Zero for deterministic tiers. */
  readonly costUsd: number;
}
