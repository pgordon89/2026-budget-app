/**
 * What Tier 3 receives when Tier 2 gives up.
 *
 * One function, used by every script that measures the model tier, because the
 * A/B this exists to support is only valid if both arms build the same prompt
 * from the same evidence. Three call sites each assembling a `ClassificationInput`
 * by hand is three chances for one of them to attach the neighbours slightly
 * differently and turn a prompt difference into a measured "model improvement".
 *
 * The handover is deliberately more than the winning label. Tier 2 computes a
 * weighted distribution over categories and, when it cannot clear its agreement
 * floor, discards the whole thing and escalates — so the model has historically
 * been asked to re-derive, from three merchant keys, a distribution the previous
 * tier had already calculated exactly. Passing the vote costs nothing: it is
 * already computed, it sits in the volatile half of the prompt where it cannot
 * disturb the cache prefix, and it is a few dozen tokens.
 */

import type { NeighbourOutcome } from './knn.js';
import type { ClassificationInput, NeighbourExample, VotePrior } from './prompt.js';
import { normalizeDescriptor } from './normalize.js';

/**
 * Tier 2's vote as a prior, or `null` when there is no vote to hand over.
 *
 * `no_neighbours` and `degenerate` outcomes have no distribution at all — the
 * key shares no feature with anything labelled, or Tier 0 could not produce a
 * usable key. Those escalate with the descriptor alone, and saying so explicitly
 * is better than sending an empty table the model has to interpret.
 */
export function toVotePrior(outcome: NeighbourOutcome): VotePrior | null {
  if (outcome.status !== 'hit' && outcome.status !== 'low_confidence') return null;
  return {
    category: outcome.category,
    agreement: outcome.agreement,
    distribution: outcome.distribution,
  };
}

export function toNeighbourExamples(outcome: NeighbourOutcome): NeighbourExample[] {
  if (outcome.status !== 'hit' && outcome.status !== 'low_confidence') return [];
  return outcome.neighbours.map((n) => ({
    key: n.key,
    category: n.category,
    similarity: n.similarity,
  }));
}

/**
 * Builds the model tier's input for one escalated transaction.
 *
 * **`withVote` defaults to off, and that is the measured result rather than a
 * placeholder.** Handing Tier 2's distribution to the model was worth one answer
 * in the wrong direction across 109 matched-coverage holdout answers — see
 * `npm run analyze:prior` and the note on `VotePrior`. The parameter survives so
 * the experiment stays runnable and so the shipped default is an explicit
 * decision in code rather than a feature nobody got round to wiring up.
 */
export function escalationInput(
  rawDescriptor: string,
  amount: number,
  outcome: NeighbourOutcome,
  withVote = false,
): ClassificationInput {
  return {
    rawDescriptor,
    normalizedKey: normalizeDescriptor(rawDescriptor).key,
    amount,
    neighbours: toNeighbourExamples(outcome),
    vote: withVote ? toVotePrior(outcome) : null,
  };
}
