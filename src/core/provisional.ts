/**
 * When is a label solid enough to appear in a budget total?
 *
 * The cascade already knows how to abstain: below its gates a transaction goes
 * to a human instead of into the ledger. That mechanism has a measured limit.
 * The answers the correction loop creates — merchants the store learned from a
 * handful of confirmations, and merchants answered by a contested neighbourhood
 * — top out at 92.3% (Tier 1) and 95.2% (Tier 2) precision, whatever threshold
 * is applied. `npm run analyze:gate` sweeps for a setting that reaches 97% on
 * that population and reports that none exists. It is the hard residue of a
 * corpus with real merchant churn, and a threshold is the wrong instrument for
 * it: no confidence cut separates those answers because the thing that makes
 * them unreliable is how little independent evidence stands behind them, not
 * how the pipeline scored them.
 *
 * So the answer changes shape instead. A `provisional` label is shown, and
 * pre-fills a one-tap confirmation, and is *excluded from every total* until it
 * earns promotion. The user sees a category on every row either way; what they
 * do not get is a wrong number silently summed into a monthly figure. A
 * category people confirm is a far better experience than a category they
 * discover was wrong after budgeting against it.
 *
 * Promotion is by accumulating evidence, not by asking a human — otherwise this
 * is just the review queue with extra steps. See `Ledger.correct`, which
 * upgrades a merchant's whole provisional backlog the moment its confirmed
 * support crosses the floor.
 */

export type CategoryStatus = 'provisional' | 'confirmed';

/** `categorySource` written by a human correction rather than by a tier. */
export const HUMAN_SOURCE = 'human';

export interface ProvisionalConfig {
  /**
   * Independent sightings of this merchant *and this category* before its
   * labels may be summed.
   *
   * Counts confirmed observations only — imported history and human
   * corrections. Deliberately not the weighted support the gates use: that
   * includes the pipeline's own write-backs, so a rule built on it would let a
   * merchant certify itself by being guessed at repeatedly. This counter cannot
   * move unless independent evidence arrives.
   */
  readonly minConfirmations: number;
  /**
   * Tiers whose answers stay provisional however well-evidenced the merchant is.
   *
   * The escape hatch for a tier measured below the floor on its own traffic.
   * A well-supported merchant can still be a genuinely mixed one — the exact
   * reason Tier 1 declined it and it reached a later tier at all — so
   * `minConfirmations` alone cannot express "this tier's answers do not count".
   */
  readonly unattestedSources: readonly string[];
}

/**
 * Selected by `npm run analyze:provisional` on the validation split and scored
 * once on the holdout, under a lexicographic objective: fewest wrong numbers in
 * the totals first, widest coverage as the tiebreak.
 *
 * In plain terms it comes out as: **a total is built from exact merchant matches
 * and human confirmations, and nothing else.** Similarity and the model produce
 * labels that are displayed, but not summed until something independent backs
 * them. On the holdout that counts 85.4% of transactions at 99.7% precision,
 * against 98.4% for summing everything the pipeline says.
 *
 * Two things the sweep settled that are easy to get backwards.
 *
 * **The tier does the work; the count barely matters.** Between
 * `minConfirmations` 1 and 2 the counted set differs by a single row out of 546.
 * It only starts costing at 3. Almost the entire effect comes from which tiers
 * are allowed to self-certify, which is the dimension a support floor alone
 * cannot express — and a support floor was the obvious first design.
 *
 * **The 97% floor decides nothing here.** All twelve candidates clear it, so a
 * coverage-maximising rule would take the loosest and book 7 wrong totals to
 * gain 3.8 points of coverage. Errors are ordered first instead, because the
 * fallback for not counting a row is one tap rather than a paid escalation, and
 * a wrong total costs more than a wrong row.
 */
export const DEFAULT_PROVISIONAL_CONFIG: ProvisionalConfig = {
  minConfirmations: 1,
  unattestedSources: ['llm', 'embedding'],
};

/**
 * The status one prediction earns.
 *
 * `confirmedSupport` is the count of independent sightings of the predicted
 * category for this merchant — zero for a merchant history has never seen, which
 * is why every answer about a new merchant starts provisional regardless of how
 * confident the tier that produced it was.
 */
export function statusFor(
  source: string,
  confirmedSupport: number,
  config: ProvisionalConfig = DEFAULT_PROVISIONAL_CONFIG,
): CategoryStatus {
  // A human looked at it. Nothing outranks that, and it is the one source that
  // is evidence rather than inference.
  if (source === HUMAN_SOURCE) return 'confirmed';
  if (config.unattestedSources.includes(source)) return 'provisional';
  return confirmedSupport >= config.minConfirmations ? 'confirmed' : 'provisional';
}
