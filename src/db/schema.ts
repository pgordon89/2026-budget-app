/**
 * The ledger schema.
 *
 * Two rules shape it.
 *
 * **Money is integers.** Every amount is stored in cents as a bigint, never as a
 * float and never as `numeric` read back through a float. `0.1 + 0.2` is a
 * budgeting bug waiting to happen, and the one place this project must not have
 * probabilistic behaviour is the arithmetic.
 *
 * **The taxonomy lives in TypeScript, not here.** `src/core/taxonomy.ts` is the
 * single source of truth — it is the model's label space and the eval's label
 * space, and a second definition in SQL would eventually disagree with it. The
 * `categories` table exists purely so foreign keys can enforce that a stored
 * category is a real one, and it is *seeded from the TypeScript* rather than
 * maintained by hand. Renaming a category is still a breaking change; this makes
 * it a loud one.
 */

import {
  pgTable,
  text,
  bigint,
  integer,
  doublePrecision,
  timestamp,
  date,
  index,
  uniqueIndex,
  primaryKey,
} from 'drizzle-orm/pg-core';

/** Mirrors `CATEGORY_GROUPS` / `CATEGORIES`; seeded from the taxonomy module. */
export const categories = pgTable('categories', {
  id: text('id').primaryKey(),
  group: text('group').notNull(),
  label: text('label').notNull(),
  /** 'inflow' | 'outflow' | 'either' — the cheap sanity check on model output. */
  direction: text('direction').notNull(),
});

export const accounts = pgTable('accounts', {
  id: text('id').primaryKey(),
  name: text('name').notNull(),
  /** 'checking' | 'savings' | 'credit' */
  type: text('type').notNull(),
  mask: text('mask').notNull(),
});

export const transactions = pgTable(
  'transactions',
  {
    id: text('id').primaryKey(),
    accountId: text('account_id')
      .notNull()
      .references(() => accounts.id),
    /** Calendar date, not a timestamp — a posting date has no time zone. */
    postedOn: date('posted_on').notNull(),
    /** Signed cents. Negative leaves the household. Integer, deliberately. */
    amountCents: bigint('amount_cents', { mode: 'number' }).notNull(),
    /** Exactly as the bank sent it. Never overwritten — Tier 0 is lossy. */
    rawDescriptor: text('raw_descriptor').notNull(),
    /** Tier 0's output, stored so lookups and joins do not re-derive it. */
    merchantKey: text('merchant_key').notNull(),

    /** Null until something categorises it. */
    categoryId: text('category_id').references(() => categories.id),
    /** Which tier answered: 'memory' | 'embedding' | 'llm' | null. */
    categorySource: text('category_source'),
    categoryConfidence: doublePrecision('category_confidence'),
    /**
     * How much this label can be relied on: 'provisional' | 'confirmed'. Null
     * while uncategorised.
     *
     * Replaces a `category_confirmed` boolean, and the extra state earns its
     * keep. The boolean had to answer two questions at once — has a human seen
     * this, and may a budget total include it — and the measurement said those
     * come apart. No threshold gets the answers the correction loop creates above
     * 97% precision (92.3% for Tier 1, 95.2% for Tier 2 at their ceilings), so a
     * system that only knows "confirmed or not" must either book known-bad
     * numbers into totals or send every unconfirmed row to a human.
     *
     * `provisional` is the third option: the label is good enough to show and to
     * pre-fill a one-tap confirmation with, and not good enough to sum. Rows
     * leave it by an upgrade rule (see `Ledger.correct`), so the queue drains as
     * evidence accumulates rather than requiring a human per row.
     */
    categoryStatus: text('category_status'),
    /** Marginal cost of the prediction, in millionths of a dollar. Integer, so
     *  summing a million rows cannot drift. */
    categoryCostMicroUsd: bigint('category_cost_micro_usd', { mode: 'number' }).notNull().default(0),

    importedAt: timestamp('imported_at', { withTimezone: true }).notNull().defaultNow(),
  },
  (table) => [
    index('transactions_posted_on_idx').on(table.postedOn),
    index('transactions_merchant_key_idx').on(table.merchantKey),
    index('transactions_category_idx').on(table.categoryId),
    // The review queue and the budget-total filter both key off status.
    index('transactions_status_idx').on(table.categoryStatus, table.postedOn),
    // Import idempotency: the same descriptor, date, amount and account is the
    // same transaction. Re-importing an overlapping statement must not duplicate.
    uniqueIndex('transactions_natural_key_idx').on(
      table.accountId,
      table.postedOn,
      table.amountCents,
      table.rawDescriptor,
    ),
  ],
);

/**
 * Tier 1's store, as a table: one row per (merchant key, category).
 *
 * This is the shape the in-memory implementation always had — the aggregate was
 * a `Map<key, Map<category, tally>>` from the start specifically so moving it
 * here would be a transcription rather than a redesign.
 *
 * Weight and count are separate on purpose. `count` is how many times it was
 * seen; `weight` discounts observations the pipeline made itself, so the store
 * cannot grow confident by reading back its own guesses.
 */
export const merchantMemory = pgTable(
  'merchant_memory',
  {
    merchantKey: text('merchant_key').notNull(),
    categoryId: text('category_id')
      .notNull()
      .references(() => categories.id),
    weight: doublePrecision('weight').notNull().default(0),
    count: integer('count').notNull().default(0),
    /**
     * Sightings backed by independent evidence — imported history or a human
     * correction — as opposed to the pipeline's own write-backs.
     *
     * Separate from `weight` because `weight` includes inferred observations and
     * therefore moves when the system reads back its own guesses. A promotion
     * rule built on `weight` would let a merchant certify itself; built on this,
     * it cannot move without a human or history saying something new.
     */
    confirmedCount: integer('confirmed_count').notNull().default(0),
    updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
  },
  (table) => [primaryKey({ columns: [table.merchantKey, table.categoryId] })],
);

/**
 * Every human correction, kept forever.
 *
 * An append-only log rather than an in-place update, because this is the
 * training signal for the whole cascade. Overwriting `transactions.category_id`
 * alone would answer "what is it now" and destroy "what did the pipeline say,
 * and how often was it wrong" — which is the only way to measure whether the
 * system is actually learning.
 */
export const corrections = pgTable(
  'corrections',
  {
    id: text('id').primaryKey(),
    transactionId: text('transaction_id')
      .notNull()
      .references(() => transactions.id),
    merchantKey: text('merchant_key').notNull(),
    /** What the pipeline said, or null if it declined to answer. */
    predictedCategoryId: text('predicted_category_id').references(() => categories.id),
    predictedSource: text('predicted_source'),
    predictedConfidence: doublePrecision('predicted_confidence'),
    /** What the human said. */
    correctedCategoryId: text('corrected_category_id')
      .notNull()
      .references(() => categories.id),
    correctedAt: timestamp('corrected_at', { withTimezone: true }).notNull().defaultNow(),
  },
  (table) => [
    index('corrections_merchant_key_idx').on(table.merchantKey),
    index('corrections_corrected_at_idx').on(table.correctedAt),
  ],
);

export type Transaction = typeof transactions.$inferSelect;
export type NewTransaction = typeof transactions.$inferInsert;
export type Correction = typeof corrections.$inferSelect;
export type MerchantMemoryRow = typeof merchantMemory.$inferSelect;
