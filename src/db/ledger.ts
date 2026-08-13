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
}

export interface CorrectionResult {
  readonly transactionId: string;
  readonly merchantKey: string;
  readonly correctedTo: CategoryId;
  readonly previousCategoryId: string | null;
  readonly previousSource: string | null;
  /** True when the pipeline had committed to an answer and it was wrong. */
  readonly overturned: boolean;
}

export class Ledger {
  private readonly config: MemoryConfig;

  constructor(
    private readonly db: Database,
    config: Partial<MemoryConfig> = {},
  ) {
    this.config = { ...DEFAULT_MEMORY_CONFIG, ...config };
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

  /** Attach what the pipeline decided. Does not mark the row confirmed. */
  async recordPrediction(transactionId: string, prediction: PredictionRecord): Promise<void> {
    await this.db
      .update(transactions)
      .set({
        categoryId: prediction.categoryId,
        categorySource: prediction.source,
        categoryConfidence: prediction.confidence,
        categoryCostMicroUsd: prediction.costMicroUsd,
      })
      .where(eq(transactions.id, transactionId));
  }

  /**
   * What a human still needs to look at: uncategorised, or categorised by the
   * pipeline without confirmation. Oldest first, because a review queue worked
   * newest-first never reaches the bottom.
   */
  async reviewQueue(limit = 50): Promise<Transaction[]> {
    return this.db
      .select()
      .from(transactions)
      .where(and(eq(transactions.categoryConfirmed, false), or(isNull(transactions.categoryId), sql`true`)))
      .orderBy(asc(transactions.postedOn), asc(transactions.id))
      .limit(limit);
  }

  async pendingReviewCount(): Promise<number> {
    const [row] = await this.db
      .select({ count: sql<number>`count(*)::int` })
      .from(transactions)
      .where(eq(transactions.categoryConfirmed, false));
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
        .set({ categoryId: category, categorySource: 'human', categoryConfidence: 1, categoryConfirmed: true })
        .where(eq(transactions.id, transactionId));

      await this.reinforce(tx, row.merchantKey, category);

      return {
        transactionId,
        merchantKey: row.merchantKey,
        correctedTo: category,
        previousCategoryId: row.categoryId,
        previousSource: row.categorySource,
        overturned: row.categoryId !== null && row.categoryId !== category,
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
      .values({ merchantKey, categoryId: category, weight, count: 1 })
      .onConflictDoUpdate({
        target: [merchantMemory.merchantKey, merchantMemory.categoryId],
        set: {
          weight: sql`${merchantMemory.weight} + ${weight}`,
          count: sql`${merchantMemory.count} + 1`,
          updatedAt: sql`now()`,
        },
      });
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
