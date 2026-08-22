/**
 * The ledger, and the correction path that feeds the cascade.
 *
 * The correction path is the part that matters. The pipeline's cheap tiers learn
 * from confirmed labels, so a user fixing a category in the UI is not a UI event
 * — it is a training signal, and the only one this system gets. Everything here
 * is arranged so that signal is captured atomically and kept: the ledger row is
 * updated, an immutable correction is logged, and the merchant store is
 * reweighted, all inside one transaction. Any of those succeeding without the
 * others leaves the ledger and the learner disagreeing about what is true.
 */

import { and, asc, eq, isNull, or, sql } from 'drizzle-orm';

import { normalizeDescriptor } from '../ai/normalize.js';
import { DEFAULT_MEMORY_CONFIG, MerchantMemory, type MemoryConfig } from '../ai/memory.js';
import { isCategoryId, type CategoryId } from '../core/taxonomy.js';
import {
  DEFAULT_PROVISIONAL_CONFIG,
  HUMAN_SOURCE,
  statusFor,
  type CategoryStatus,
  type ProvisionalConfig,
} from '../core/provisional.js';
import { corrections, merchantMemory, transactions, type NewTransaction, type Transaction } from './schema.js';
import type { Database } from './client.js';

/** A transaction handle exposes the same query surface as the database. */
type Executor = Database | Parameters<Parameters<Database['transaction']>[0]>[0];

export interface ImportRow {
  readonly id: string;
  readonly accountId: string;
  readonly postedOn: string;
  readonly amountCents: number;
  readonly rawDescriptor: string;
}

export interface PredictionRecord {
  readonly categoryId: CategoryId;
  readonly source: string;
  readonly confidence: number;
  readonly costMicroUsd: number;
  /**
   * Independent sightings behind this merchant+category, from `MemoryOutcome`.
   *
   * Required rather than defaulted. A caller that does not know the support has
   * not consulted the store, and silently treating that as zero would mark every
   * such row provisional — which looks conservative but is really a missing
   * measurement wearing a safe-looking default.
   */
  readonly confirmedSupport: number;
}

export interface CorrectionResult {
  readonly transactionId: string;
  readonly merchantKey: string;
  readonly correctedTo: CategoryId;
  readonly previousCategoryId: string | null;
  readonly previousSource: string | null;
  /** True when the pipeline had committed to an answer and it was wrong. */
  readonly overturned: boolean;
  /**
   * Other rows this correction promoted out of `provisional`.
   *
   * The number that makes the mechanism worth having. If confirming one coffee
   * shop settles the nine earlier visits to it, the queue drains faster than a
   * human works through it — and if this is usually zero, the design has failed
   * and the measurement says so.
   */
  readonly upgraded: number;
}

export class Ledger {
  private readonly config: MemoryConfig;
  private readonly provisional: ProvisionalConfig;

  constructor(
    private readonly db: Database,
    config: Partial<MemoryConfig> = {},
    provisional: Partial<ProvisionalConfig> = {},
  ) {
    this.config = { ...DEFAULT_MEMORY_CONFIG, ...config };
    this.provisional = { ...DEFAULT_PROVISIONAL_CONFIG, ...provisional };
  }

  /**
   * Insert transactions, skipping ones already present.
   *
   * Idempotent by natural key, because re-importing an overlapping statement is
   * the normal case, not an error — banks hand you the last 90 days every time.
   * Returns how many were actually new.
   */
  async importTransactions(rows: readonly ImportRow[]): Promise<number> {
    if (rows.length === 0) return 0;

    const values: NewTransaction[] = rows.map((row) => ({
      ...row,
      // Derived once at import so every later lookup and join keys off the same
      // string rather than re-deriving it and risking a version skew.
      merchantKey: normalizeDescriptor(row.rawDescriptor).key,
    }));

    const inserted = await this.db
      .insert(transactions)
      .values(values)
      .onConflictDoNothing({
        target: [
          transactions.accountId,
          transactions.postedOn,
          transactions.amountCents,
          transactions.rawDescriptor,
        ],
      })
      .returning({ id: transactions.id });

    return inserted.length;
  }

  /**
   * Attach what the pipeline decided, and how far it may be trusted.
   *
   * Never writes `confirmed` on the strength of a tier's own confidence — that
   * status means independent evidence exists, and a tier agreeing with itself is
   * not evidence. See `src/core/provisional.ts`.
   */
  async recordPrediction(transactionId: string, prediction: PredictionRecord): Promise<CategoryStatus> {
    const status = statusFor(prediction.source, prediction.confirmedSupport, this.provisional);
    await this.db
      .update(transactions)
      .set({
        categoryId: prediction.categoryId,
        categorySource: prediction.source,
        categoryConfidence: prediction.confidence,
        categoryCostMicroUsd: prediction.costMicroUsd,
        categoryStatus: status,
      })
      .where(eq(transactions.id, transactionId));
    return status;
  }

  /**
   * What a human still needs to look at: uncategorised, or provisional. Oldest
   * first, because a review queue worked newest-first never reaches the bottom.
   *
   * The two are different jobs for the person doing them, and the UI should say
   * so. An uncategorised row is a question — what is this. A provisional row is
   * a proposal — we think it is X, one tap to agree. The second is the cheap one,
   * and it is most of the queue.
   */
  async reviewQueue(limit = 50): Promise<Transaction[]> {
    return this.db
      .select()
      .from(transactions)
      .where(or(isNull(transactions.categoryStatus), eq(transactions.categoryStatus, 'provisional')))
      .orderBy(asc(transactions.postedOn), asc(transactions.id))
      .limit(limit);
  }

  async pendingReviewCount(): Promise<number> {
    const [row] = await this.db
      .select({ count: sql<number>`count(*)::int` })
      .from(transactions)
      .where(or(isNull(transactions.categoryStatus), eq(transactions.categoryStatus, 'provisional')));
    return row?.count ?? 0;
  }

  /**
   * Total spend by category, over confirmed rows only.
   *
   * The whole point of the status column, in one query. A provisional label is
   * displayable and not summable, so every budget-facing figure filters here
   * rather than each caller remembering to — a total that quietly includes
   * provisional rows is exactly the silent wrong number this is built to prevent.
   */
  async totalsByCategory(from: string, to: string): Promise<{ categoryId: string; totalCents: number }[]> {
    return this.db
      .select({
        categoryId: sql<string>`${transactions.categoryId}`,
        totalCents: sql<number>`sum(${transactions.amountCents})::int`,
      })
      .from(transactions)
      .where(
        and(
          eq(transactions.categoryStatus, 'confirmed'),
          sql`${transactions.postedOn} >= ${from}`,
          sql`${transactions.postedOn} <= ${to}`,
        ),
      )
      .groupBy(transactions.categoryId);
  }

  /** What the totals above are leaving out, so the gap is reportable rather than invisible. */
  async provisionalExcludedCount(): Promise<number> {
    const [row] = await this.db
      .select({ count: sql<number>`count(*)::int` })
      .from(transactions)
      .where(eq(transactions.categoryStatus, 'provisional'));
    return row?.count ?? 0;
  }

  /**
   * Apply a human decision.
   *
   * One transaction covering three writes: the ledger row becomes confirmed, the
   * correction is logged immutably, and the merchant store gains a confirmed
   * observation. Splitting these would let the ledger say one thing while the
   * learner believes another, and the disagreement would be silent.
   *
   * Recording it even when the human agrees with the pipeline is deliberate:
   * a confirmation is evidence too, and it is what turns a tentative
   * model-inferred label into one the cheap tier can rely on.
   */
  async correct(transactionId: string, category: CategoryId): Promise<CorrectionResult> {
    if (!isCategoryId(category)) throw new Error(`refusing to record unknown category id: ${category}`);

    return this.db.transaction(async (tx) => {
      const [row] = await tx.select().from(transactions).where(eq(transactions.id, transactionId));
      if (!row) throw new Error(`no such transaction: ${transactionId}`);

      await tx.insert(corrections).values({
        id: `corr_${transactionId}`,
        transactionId,
        merchantKey: row.merchantKey,
        predictedCategoryId: row.categoryId,
        predictedSource: row.categorySource,
        predictedConfidence: row.categoryConfidence,
        correctedCategoryId: category,
      });

      await tx
        .update(transactions)
        .set({
          categoryId: category,
          categorySource: HUMAN_SOURCE,
          categoryConfidence: 1,
          categoryStatus: 'confirmed',
        })
        .where(eq(transactions.id, transactionId));

      await this.reinforce(tx, row.merchantKey, category);
      const upgraded = await this.promote(tx, row.merchantKey, category);

      return {
        transactionId,
        merchantKey: row.merchantKey,
        correctedTo: category,
        previousCategoryId: row.categoryId,
        previousSource: row.categorySource,
        overturned: row.categoryId !== null && row.categoryId !== category,
        upgraded,
      };
    });
  }

  /**
   * Add a confirmed observation for a merchant.
   *
   * Takes the already-normalised key rather than a descriptor: the key was
   * derived at import time and stored, so re-deriving it here would introduce a
   * second code path that could disagree with the first.
   */
  private async reinforce(tx: Executor, merchantKey: string, category: CategoryId): Promise<void> {
    const weight = new MerchantMemory(this.config).weightOf('confirmed');
    await tx
      .insert(merchantMemory)
      .values({ merchantKey, categoryId: category, weight, count: 1, confirmedCount: 1 })
      .onConflictDoUpdate({
        target: [merchantMemory.merchantKey, merchantMemory.categoryId],
        set: {
          weight: sql`${merchantMemory.weight} + ${weight}`,
          count: sql`${merchantMemory.count} + 1`,
          // A correction is by definition independent evidence, so this path
          // always increments. Missing it here once meant promotion could never
          // fire: `reinforce` is the only writer on the correction path, and the
          // repository's `remember` — which does maintain the counter — is not
          // on it.
          confirmedCount: sql`${merchantMemory.confirmedCount} + 1`,
          updatedAt: sql`now()`,
        },
      });
  }

  /**
   * Promote a merchant's provisional backlog, if this correction tipped it over.
   *
   * The mechanism that keeps `provisional` from being a second review queue. One
   * confirmation does not settle one row — it settles the *merchant*, so every
   * earlier transaction the pipeline labelled the same way stops being a question.
   * Confirm one visit to a coffee shop and the previous nine leave the queue.
   *
   * Scoped to the corrected category on purpose. A merchant genuinely split
   * across categories accumulates confirmations for each independently, so
   * confirming one `shopping.household` charge at a marketplace does not certify
   * its `shopping.electronics` rows — which is the case that made a mixed-basket
   * merchant dangerous in the first place.
   *
   * Runs inside the correction's transaction, so a promotion cannot survive a
   * rolled-back correction and leave rows certified by evidence that no longer
   * exists.
   */
  private async promote(tx: Executor, merchantKey: string, category: CategoryId): Promise<number> {
    const [row] = await tx
      .select({ confirmedCount: merchantMemory.confirmedCount })
      .from(merchantMemory)
      .where(and(eq(merchantMemory.merchantKey, merchantKey), eq(merchantMemory.categoryId, category)));

    if ((row?.confirmedCount ?? 0) < this.provisional.minConfirmations) return 0;

    const promoted = await tx
      .update(transactions)
      .set({ categoryStatus: 'confirmed' })
      .where(
        and(
          eq(transactions.merchantKey, merchantKey),
          eq(transactions.categoryId, category),
          eq(transactions.categoryStatus, 'provisional'),
        ),
      )
      .returning({ id: transactions.id });

    return promoted.length;
  }

  /** How often the pipeline was overruled, by the tier that answered. */
  async correctionStats(): Promise<{ total: number; overturned: number; bySource: Record<string, number> }> {
    const rows = await this.db
      .select({
        source: corrections.predictedSource,
        predicted: corrections.predictedCategoryId,
        corrected: corrections.correctedCategoryId,
      })
      .from(corrections);

    const bySource: Record<string, number> = {};
    let overturned = 0;
    for (const row of rows) {
      if (row.predicted !== null && row.predicted !== row.corrected) {
        overturned++;
        bySource[row.source ?? 'unknown'] = (bySource[row.source ?? 'unknown'] ?? 0) + 1;
      }
    }
    return { total: rows.length, overturned, bySource };
  }
}
