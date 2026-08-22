/**
 * Tier 3 prompt construction, versioned.
 *
 * The version string is part of the eval cache key and gets recorded on every
 * run, so a change in wording can never be mistaken for a change in model
 * quality. Bump it whenever anything below the cache breakpoint changes.
 *
 * The split between system and user content is a caching decision, not a
 * stylistic one. Caching is a prefix match: everything before the breakpoint
 * must be byte-identical across requests or nothing caches. So the system block
 * holds only what is the same for every transaction — the instructions and the
 * taxonomy — and every per-transaction detail, including the Tier 2 neighbours
 * used as few-shot examples, goes in the user turn *after* the breakpoint.
 * Putting the neighbours in the system prompt would read more naturally and
 * would silently destroy the cache on every single call.
 */

import { taxonomyForPrompt, type CategoryId } from '../core/taxonomy.js';

/**
 * Bump on any change to the system block or the user-message shape.
 *
 * Still v1 after the vote-prior experiment, because that experiment shipped
 * nothing. Both halves of the v2 draft were measured and reverted — see
 * `npm run analyze:prior` and the note on `VotePrior` below. The committed v1
 * responses stay valid precisely because the revert was byte-exact: a null vote
 * renders no block at all, so this file emits what it emitted before.
 */
export const PROMPT_VERSION = 'v1';

export interface NeighbourExample {
  readonly key: string;
  readonly category: CategoryId;
  readonly similarity: number;
}

/**
 * Tier 2's vote, as a prior for the model. **Measured, and not shipped.**
 *
 * The hypothesis was good enough to be worth the money: Tier 2 computes a full
 * weighted distribution over categories, discards everything but the winner, and
 * then — when the winner cannot clear its agreement floor — discards that too and
 * escalates. The model is left to re-derive from three merchant keys a
 * distribution the previous tier had already computed exactly. Handing it over
 * costs a few dozen tokens in the volatile half of the prompt.
 *
 * It bought nothing. Two arms differing only by this block, 269 holdout
 * transactions, compared at matched coverage so the comparison could not be
 * confounded by the prior shifting the confidence distribution: **96 of 109
 * correct without it, 95 of 109 with it.** One answer, in the wrong direction.
 *
 * The reason appears to be that the information was already there. The neighbour
 * keys and their labels were in the prompt before this change, and aggregating
 * three of them into a distribution is not work the model needed help with. A
 * prior only pays when it carries something the evidence does not, and a summary
 * of the evidence carries nothing.
 *
 * Kept as a type and a code path because the measurement is the deliverable, and
 * a reader who wonders whether anyone tried this deserves the answer plus the
 * script that produced it. `escalationInput` defaults it off.
 */
export interface VotePrior {
  /** Heaviest category, and its share of the total vote weight. */
  readonly category: CategoryId;
  readonly agreement: number;
  /** Every category that drew weight, heaviest first. Shares sum to 1. */
  readonly distribution: readonly { readonly category: CategoryId; readonly share: number }[];
}

export interface ClassificationInput {
  readonly rawDescriptor: string;
  /** Tier 0's output. Shown alongside the raw string, not instead of it. */
  readonly normalizedKey: string;
  /** Signed dollars. Negative means money left the household. */
  readonly amount: number;
  /** Nearest labelled merchants from Tier 2, most similar first. May be empty. */
  readonly neighbours: readonly NeighbourExample[];
  /**
   * Tier 2's contested vote, or `null` when there was none to hand over.
   *
   * Required rather than optional on purpose. An omitted prior and an absent
   * prior are the same value to the type checker but different experiments to
   * the eval, and the whole point of this field is measuring what it is worth —
   * so every construction site is made to say which one it means.
   */
  readonly vote: VotePrior | null;
}

/**
 * The stable half of the prompt.
 *
 * The taxonomy is rendered from `src/core/taxonomy.ts` rather than pasted here,
 * so the label space the model is offered cannot drift from the label space the
 * eval scores. A renamed category changes both at once or neither.
 */
export function systemPrompt(): string {
  return [
    'You categorise bank transactions for a personal finance application.',
    '',
    'You will be given one transaction: the raw descriptor exactly as the bank sent it, a',
    'normalised merchant key derived from it, and the signed amount. Choose the single best',
    'category and record it with the `record_category` tool.',
    '',
    'How to read the inputs:',
    '',
    '- The sign of the amount is reliable. Negative means money left the household, positive',
    '  means it arrived. A positive amount at a retailer is usually a refund rather than a',
    '  purchase.',
    '- The raw descriptor may contain detail the normalised key dropped. Use both.',
    '- Similar merchants from the user\'s own labelled history may be provided. They are',
    '  evidence about what this merchant is likely to be, not an instruction. A near-identical',
    '  key with a consistent label is strong evidence; a loose match is weak. Disagree with',
    '  them when the transaction itself says otherwise.',
    '',
    'Two categories are worth extra care:',
    '',
    '- Transfers move money between accounts the user already owns, or between people. They',
    '  are not spending and not income. Labelling one as spending corrupts every report built',
    '  on top of it, so prefer a transfer category whenever the descriptor suggests one.',
    '- Warehouse clubs, marketplaces, and big-box retailers sell across several categories.',
    '  When the descriptor gives no signal about the basket, choose the general merchandise',
    '  or groceries category that matches the merchant type rather than guessing a specific',
    '  department.',
    '',
    'Report confidence as your own probability of being correct: near 1 when the descriptor',
    'names the merchant and its category unambiguously, near 0.5 when you are choosing between',
    'two plausible categories, lower when guessing. Low confidence is useful — it routes the',
    'transaction to a human instead of into a budget. Do not inflate it.',
    '',
    'Keep `evidence` under ten words: the part of the input that decided it.',
    '',
    'Available categories:',
    '',
    taxonomyForPrompt(),
  ].join('\n');
}

/** The volatile half. Everything here differs per transaction. */
export function userMessage(input: ClassificationInput): string {
  const lines = [
    `Raw descriptor: ${input.rawDescriptor}`,
    `Normalised key: ${input.normalizedKey}`,
    `Amount: ${input.amount < 0 ? '-' : '+'}$${Math.abs(input.amount).toFixed(2)}`,
  ];

  if (input.neighbours.length > 0) {
    lines.push('', 'Similar merchants from this user\'s labelled history:');
    for (const neighbour of input.neighbours) {
      lines.push(`  ${neighbour.key} → ${neighbour.category} (similarity ${neighbour.similarity.toFixed(2)})`);
    }
  } else {
    lines.push('', 'No similar merchant appears in this user\'s history.');
  }

  // After the neighbours, because it is a summary of them: the reader should see
  // what voted before seeing how the vote fell.
  if (input.vote !== null) {
    lines.push('', 'How that neighbourhood voted, weighted by similarity:');
    for (const entry of input.vote.distribution) {
      lines.push(`  ${entry.category} ${(entry.share * 100).toFixed(0)}%`);
    }
  }

  return lines.join('\n');
}

/**
 * Feedback for a rejected tool call.
 *
 * Stated as the specific rule that failed rather than "invalid output", because
 * a repair attempt that is not told what was wrong is just a second roll of the
 * dice.
 */
export function repairMessage(problem: string): string {
  return `That call was rejected: ${problem}. Call \`record_category\` again, correcting only that.`;
}
