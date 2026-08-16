/**
 * Tier 1's store, persisted.
 *
 * The in-memory `MerchantMemory` remains the scoring engine — this holds the
 * same tallies in Postgres and hands them back. Two access paths, because batch
 * and request-time want different things:
 *
 *   `hydrate()`  — load every tally once and score in process. What an eval or a
 *                  nightly recategorisation run wants; one query, not 1,429.
 *   `lookup()`   — score a single merchant from a single indexed query. What a
 *                  web request wants; it must not read the whole store to
 *                  categorise one transaction.
 *
 * Both route through `scoreTallies`, so a merchant scores identically whichever
 * path reached it. There is a conformance test that asserts exactly that rather
 * than trusting it.
 */

import { and, eq, sql } from 'drizzle-orm';

import {
  acceptsScore,
  DEFAULT_MEMORY_CONFIG,
  MerchantMemory,
  scoreTallies,
  type MemoryConfig,
  type MemoryOutcome,
  type ObservationSource,
} from '../ai/memory.js';
import { normalizeDescriptor } from '../ai/normalize.js';
import { isCategoryId, type CategoryId } from '../core/taxonomy.js';
import { merchantMemory } from './schema.js';
import type { Database } from './client.js';

export class MerchantMemoryRepository {
  private readonly config: MemoryConfig;

  constructor(
    private readonly db: Database,
    config: Partial<MemoryConfig> = {},
  ) {
    this.config = { ...DEFAULT_MEMORY_CONFIG, ...config };
  }

  /**
   * Record one labelled sighting.
   *
   * An upsert that adds to the existing weight rather than replacing it, done in
   * SQL so two concurrent corrections for the same merchant cannot lose one
   * another — a read-modify-write in application code would.
   */
  async remember(
    rawDescriptor: string,
    category: CategoryId,
    source: ObservationSource = 'confirmed',
  ): Promise<void> {
    if (!isCategoryId(category)) throw new Error(`refusing to remember unknown category id: ${category}`);

    const { key, degenerate } = normalizeDescriptor(rawDescriptor);
    // Whatever survived normalisation identifies no merchant; anything stored
    // under it would later be served to unrelated transactions.
    if (degenerate) return;

    const weight = new MerchantMemory(this.config).weightOf(source);

    await this.db
      .insert(merchantMemory)
      .values({ merchantKey: key, categoryId: category, weight, count: 1 })
      .onConflictDoUpdate({
        target: [merchantMemory.merchantKey, merchantMemory.categoryId],
        set: {
          weight: sql`${merchantMemory.weight} + ${weight}`,
          count: sql`${merchantMemory.count} + 1`,
          updatedAt: sql`now()`,
        },
      });
  }

  /** Score one merchant without loading the store. */
  async lookup(rawDescriptor: string): Promise<MemoryOutcome> {
    const { key, degenerate } = normalizeDescriptor(rawDescriptor);
    if (degenerate) return { status: 'degenerate', key };

    const rows = await this.db
      .select({ categoryId: merchantMemory.categoryId, weight: merchantMemory.weight })
      .from(merchantMemory)
      .where(eq(merchantMemory.merchantKey, key));

    if (rows.length === 0) return { status: 'unseen', key };

    const score = scoreTallies(
      rows.map((row) => [row.categoryId as CategoryId, row.weight] as const),
      this.config.z,
    );
    if (score === undefined) return { status: 'unseen', key };

    return acceptsScore(score, this.config)
      ? { status: 'hit', key, ...score }
      : { status: 'low_confidence', key, ...score };
  }

  /** Load every tally into an in-process store, for batch scoring. */
  async hydrate(): Promise<MerchantMemory> {
    const rows = await this.db
      .select({
        merchantKey: merchantMemory.merchantKey,
        categoryId: merchantMemory.categoryId,
        weight: merchantMemory.weight,
        count: merchantMemory.count,
      })
      .from(merchantMemory);

    return MerchantMemory.fromTallies(
      rows.map((row) => ({ ...row, categoryId: row.categoryId as CategoryId })),
      this.config,
    );
  }

  /**
   * Drop everything known about a merchant.
   *
   * The wholesale-recategorisation path: when a user reassigns a merchant
   * outright, the accumulated weight behind the old answer should not have to be
   * out-voted one correction at a time.
   */
  async forget(rawDescriptor: string): Promise<number> {
    const { key } = normalizeDescriptor(rawDescriptor);
    const deleted = await this.db.delete(merchantMemory).where(eq(merchantMemory.merchantKey, key)).returning();
    return deleted.length;
  }

  async stats(): Promise<{ keys: number; observations: number }> {
    const [row] = await this.db
      .select({
        keys: sql<number>`count(distinct ${merchantMemory.merchantKey})::int`,
        observations: sql<number>`coalesce(sum(${merchantMemory.count}), 0)::int`,
      })
      .from(merchantMemory);
    return { keys: row?.keys ?? 0, observations: row?.observations ?? 0 };
  }

  /** Weight recorded for one merchant-category pair. Diagnostics and tests. */
  async weightFor(rawDescriptor: string, category: CategoryId): Promise<number> {
    const { key } = normalizeDescriptor(rawDescriptor);
    const [row] = await this.db
      .select({ weight: merchantMemory.weight })
      .from(merchantMemory)
      .where(and(eq(merchantMemory.merchantKey, key), eq(merchantMemory.categoryId, category)));
    return row?.weight ?? 0;
  }
}
