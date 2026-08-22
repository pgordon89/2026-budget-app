import { test, before, after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { eq } from 'drizzle-orm';

import { createMemoryDatabase, type Handle } from './client.js';
import { Ledger, type ImportRow } from './ledger.js';
import { MerchantMemoryRepository } from './merchant-memory-repository.js';
import { accounts, corrections, merchantMemory, transactions } from './schema.js';

let handle: Handle;

before(async () => {
  handle = await createMemoryDatabase();
  await handle.db.insert(accounts).values({ id: 'acct_cc', name: 'Card', type: 'credit', mask: '2288' });
});
after(async () => {
  await handle.close();
});
beforeEach(async () => {
  await handle.db.delete(corrections);
  await handle.db.delete(transactions);
  await handle.db.delete(merchantMemory);
});

const row = (id: string, descriptor: string, cents = -675, postedOn = '2026-01-05'): ImportRow => ({
  id,
  accountId: 'acct_cc',
  postedOn,
  amountCents: cents,
  rawDescriptor: descriptor,
});

test('derives and stores the merchant key at import', async () => {
  const ledger = new Ledger(handle.db);
  await ledger.importTransactions([row('t1', 'SQ *BLUE BOTTLE #4432 SANFRAN CA')]);

  const [stored] = await handle.db.select().from(transactions).where(eq(transactions.id, 't1'));
  assert.equal(stored?.merchantKey, 'BLUE BOTTLE');
  assert.equal(stored?.categoryStatus, null, 'nothing has a status until something categorises it');
});

test('re-importing an overlapping statement does not duplicate', async () => {
  // Banks hand you the last 90 days every time; this is the normal case.
  const ledger = new Ledger(handle.db);
  assert.equal(await ledger.importTransactions([row('t1', 'SQ *BLUE BOTTLE'), row('t2', 'WHOLEFDS #10234', -4210)]), 2);

  const again = await ledger.importTransactions([
    { ...row('t1_dup', 'SQ *BLUE BOTTLE') },
    row('t3', 'SHELL SERVICE STATION', -5500),
  ]);
  assert.equal(again, 1, 'only the genuinely new row lands');

  const all = await handle.db.select().from(transactions);
  assert.equal(all.length, 3);
});

test('stores money as integer cents', async () => {
  const ledger = new Ledger(handle.db);
  await ledger.importTransactions([row('t1', 'SQ *BLUE BOTTLE', -675)]);
  const [stored] = await handle.db.select().from(transactions).where(eq(transactions.id, 't1'));

  assert.equal(stored?.amountCents, -675);
  assert.equal(Number.isInteger(stored?.amountCents), true, 'never a float — this is money');
});

test('a correction confirms the row, logs it, and reinforces the merchant, atomically', async () => {
  const ledger = new Ledger(handle.db);
  const memory = new MerchantMemoryRepository(handle.db);
  await ledger.importTransactions([row('t1', 'SQ *PEETS COFFEE #221')]);
  await ledger.recordPrediction('t1', {
    categoryId: 'food.restaurants',
    source: 'llm',
    confidence: 0.71,
    costMicroUsd: 3300, confirmedSupport: 0
  });

  const result = await ledger.correct('t1', 'food.coffee');

  assert.equal(result.overturned, true, 'the pipeline had answered and was wrong');
  assert.equal(result.previousCategoryId, 'food.restaurants');
  assert.equal(result.previousSource, 'llm');

  const [stored] = await handle.db.select().from(transactions).where(eq(transactions.id, 't1'));
  assert.equal(stored?.categoryId, 'food.coffee');
  assert.equal(stored?.categoryStatus, 'confirmed');
  assert.equal(stored?.categorySource, 'human');

  const logged = await handle.db.select().from(corrections);
  assert.equal(logged.length, 1);
  assert.equal(logged[0]?.predictedCategoryId, 'food.restaurants', 'what the pipeline said is preserved');

  // The training signal actually reached the learner.
  assert.equal(await memory.weightFor('SQ *PEETS COFFEE #221', 'food.coffee'), 1);
});

test('confirming the pipeline is recorded too, and still reinforces', async () => {
  // Agreement is evidence. It is what promotes a tentative model label into one
  // the cheap tier can answer from next month.
  const ledger = new Ledger(handle.db);
  const memory = new MerchantMemoryRepository(handle.db);
  await ledger.importTransactions([row('t1', 'SQ *PEETS COFFEE #221')]);
  await ledger.recordPrediction('t1', { categoryId: 'food.coffee', source: 'llm', confidence: 0.95, costMicroUsd: 3300, confirmedSupport: 0 });

  const result = await ledger.correct('t1', 'food.coffee');

  assert.equal(result.overturned, false, 'agreeing is not an overturn');
  assert.equal(await memory.weightFor('SQ *PEETS COFFEE #221', 'food.coffee'), 1);
});

test('a correction on an unseen merchant teaches the cheap tier from nothing', async () => {
  const ledger = new Ledger(handle.db);
  const memory = new MerchantMemoryRepository(handle.db);
  await ledger.importTransactions([row('t1', 'SQ *NEW LOCAL CAFE')]);

  assert.equal((await memory.lookup('SQ *NEW LOCAL CAFE')).status, 'unseen');
  await ledger.correct('t1', 'food.coffee');

  // One confirmation is not yet enough to answer — Wilson still discounts a
  // sample of one — but the evidence is now there and accumulating.
  const after = await memory.lookup('SQ *NEW LOCAL CAFE');
  assert.notEqual(after.status, 'unseen');
  assert.equal(after.status === 'low_confidence' && after.category, 'food.coffee');
});

test('repeated corrections for one merchant compound into an answer', async () => {
  const ledger = new Ledger(handle.db);
  const memory = new MerchantMemoryRepository(handle.db);
  for (let i = 0; i < 5; i++) {
    await ledger.importTransactions([row(`t${i}`, 'SQ *NEW LOCAL CAFE', -600 - i, `2026-01-0${i + 1}`)]);
    await ledger.correct(`t${i}`, 'food.coffee');
  }

  const outcome = await memory.lookup('SQ *NEW LOCAL CAFE');
  assert.equal(outcome.status, 'hit', 'the cheap tier now answers a merchant it had never seen');
  assert.equal(outcome.status === 'hit' && outcome.category, 'food.coffee');
});

test('rolls back everything when the correction is invalid', async () => {
  const ledger = new Ledger(handle.db);
  await assert.rejects(() => ledger.correct('does-not-exist', 'food.coffee'), /no such transaction/);
  assert.equal((await handle.db.select().from(corrections)).length, 0);
  assert.equal((await handle.db.select().from(merchantMemory)).length, 0);
});

test('rejects a category outside the taxonomy', async () => {
  const ledger = new Ledger(handle.db);
  await ledger.importTransactions([row('t1', 'SQ *BLUE BOTTLE')]);
  await assert.rejects(() => ledger.correct('t1', 'food.coffee_shops' as 'food.coffee'), /unknown category/);
});

test('the review queue is unconfirmed rows, oldest first', async () => {
  const ledger = new Ledger(handle.db);
  await ledger.importTransactions([
    row('t_late', 'SHELL SERVICE STATION', -5500, '2026-03-01'),
    row('t_early', 'WHOLEFDS #10234', -4210, '2026-01-01'),
  ]);

  assert.equal(await ledger.pendingReviewCount(), 2);
  const queue = await ledger.reviewQueue();
  assert.deepEqual(queue.map((t) => t.id), ['t_early', 't_late'], 'a newest-first queue never reaches the bottom');

  await ledger.correct('t_early', 'food.groceries');
  assert.equal(await ledger.pendingReviewCount(), 1);
  assert.deepEqual((await ledger.reviewQueue()).map((t) => t.id), ['t_late']);
});

test('reports how often each tier was overruled', async () => {
  const ledger = new Ledger(handle.db);
  await ledger.importTransactions([
    row('t1', 'SQ *PEETS COFFEE', -675, '2026-01-01'),
    row('t2', 'WHOLEFDS #10234', -4210, '2026-01-02'),
    row('t3', 'SHELL SERVICE STATION', -5500, '2026-01-03'),
  ]);
  await ledger.recordPrediction('t1', { categoryId: 'food.restaurants', source: 'llm', confidence: 0.7, costMicroUsd: 0, confirmedSupport: 0 });
  await ledger.recordPrediction('t2', { categoryId: 'food.groceries', source: 'memory', confidence: 0.9, costMicroUsd: 0, confirmedSupport: 0 });
  await ledger.recordPrediction('t3', { categoryId: 'food.groceries', source: 'embedding', confidence: 0.6, costMicroUsd: 0, confirmedSupport: 0 });

  await ledger.correct('t1', 'food.coffee');
  await ledger.correct('t2', 'food.groceries');
  await ledger.correct('t3', 'transport.fuel');

  const stats = await ledger.correctionStats();
  assert.equal(stats.total, 3);
  assert.equal(stats.overturned, 2, 'the agreeing confirmation is not an overturn');
  assert.deepEqual(stats.bySource, { llm: 1, embedding: 1 });
});

test('an unbacked prediction is provisional, and the label is still written', async () => {
  // Isolates the support dimension: `memory` is an attesting source under the
  // shipped rule, so the only thing making this provisional is that nothing
  // independent has ever said anything about this merchant. A tier's confidence
  // is not evidence, however high the number is.
  const ledger = new Ledger(handle.db);
  await ledger.importTransactions([row('t1', 'RIDGELINE CLIMBING CO')]);

  const status = await ledger.recordPrediction('t1', {
    categoryId: 'health.fitness',
    source: 'memory',
    confidence: 0.99,
    costMicroUsd: 0,
    confirmedSupport: 0,
  });

  assert.equal(status, 'provisional');
  const [stored] = await handle.db.select().from(transactions).where(eq(transactions.id, 't1'));
  assert.equal(stored?.categoryId, 'health.fitness', 'the label is still written — it is shown, just not summed');
  assert.equal(stored?.categoryStatus, 'provisional');
});

test('an unattested tier stays provisional even on a well-evidenced merchant', async () => {
  // Isolates the source dimension, and it is the one doing nearly all the work:
  // varying the support floor between 1 and 2 moved a single row out of 546,
  // while which tiers may self-certify moved seven errors to zero.
  const ledger = new Ledger(handle.db);
  await ledger.importTransactions([row('t1', 'RIDGELINE CLIMBING CO')]);

  const status = await ledger.recordPrediction('t1', {
    categoryId: 'health.fitness',
    source: 'embedding',
    confidence: 0.99,
    costMicroUsd: 0,
    confirmedSupport: 25,
  });

  assert.equal(status, 'provisional', 'similarity does not certify a total, however familiar the merchant');
});

test('a well-evidenced merchant is confirmed without a human touching it', async () => {
  // Otherwise this is a review queue with extra steps.
  const ledger = new Ledger(handle.db);
  await ledger.importTransactions([row('t1', 'SQ *BLUE BOTTLE #4432')]);

  const status = await ledger.recordPrediction('t1', {
    categoryId: 'food.coffee',
    source: 'memory',
    confidence: 0.72,
    costMicroUsd: 0,
    confirmedSupport: 9,
  });

  assert.equal(status, 'confirmed');
});

test('the model tier stays provisional however well-evidenced the merchant is', async () => {
  // Same dimension as above, for the tier where it matters most: 90.4% precision
  // on its own traffic. A high confirmedSupport here means the merchant is known
  // *and* split — which is why Tier 1 declined it and the model saw it at all.
  const ledger = new Ledger(handle.db);
  await ledger.importTransactions([row('t1', 'AMZN MKTP US*2Z4X')]);

  const status = await ledger.recordPrediction('t1', {
    categoryId: 'shopping.household',
    source: 'llm',
    confidence: 0.95,
    costMicroUsd: 3300,
    confirmedSupport: 40,
  });

  assert.equal(status, 'provisional');
});

test('confirming one row promotes the merchant’s whole provisional backlog', async () => {
  // The mechanism that keeps the queue from growing linearly with transactions.
  // minConfirmations: 1 so a single correction tips it; the default of 2 is the
  // measured value, not a property this test is about.
  const ledger = new Ledger(handle.db, {}, { minConfirmations: 1 });
  // Distinct amounts: same descriptor, date and account is the *same* transaction
  // by natural key, and the importer is right to collapse it.
  await ledger.importTransactions([
    row('t1', 'RIDGELINE CLIMBING CO', -1200),
    row('t2', 'RIDGELINE CLIMBING CO', -1300),
    row('t3', 'RIDGELINE CLIMBING CO', -1400),
  ]);
  for (const id of ['t1', 't2', 't3']) {
    await ledger.recordPrediction(id, {
      categoryId: 'health.fitness',
      source: 'embedding',
      confidence: 0.8,
      costMicroUsd: 0,
      confirmedSupport: 0,
    });
  }

  const result = await ledger.correct('t3', 'health.fitness');

  assert.equal(result.overturned, false);
  assert.equal(result.upgraded, 2, 'the two earlier rows were settled by the same confirmation');
  assert.equal(await ledger.pendingReviewCount(), 0);
});

test('promotion is scoped to the confirmed category, not the merchant', async () => {
  // A mixed-basket merchant accumulates evidence per category. Confirming a
  // household charge at a marketplace must not certify its electronics rows —
  // that is exactly the merchant this whole mechanism is wary of.
  const ledger = new Ledger(handle.db, {}, { minConfirmations: 1 });
  await ledger.importTransactions([row('t1', 'AMZN MKTP US*A1'), row('t2', 'AMZN MKTP US*B2')]);
  await ledger.recordPrediction('t1', {
    categoryId: 'shopping.household',
    source: 'llm',
    confidence: 0.8,
    costMicroUsd: 0,
    confirmedSupport: 0,
  });
  await ledger.recordPrediction('t2', {
    categoryId: 'shopping.electronics',
    source: 'llm',
    confidence: 0.8,
    costMicroUsd: 0,
    confirmedSupport: 0,
  });

  const result = await ledger.correct('t1', 'shopping.household');

  assert.equal(result.upgraded, 0, 't1 was confirmed directly; t2 is a different category');
  const [other] = await handle.db.select().from(transactions).where(eq(transactions.id, 't2'));
  assert.equal(other?.categoryStatus, 'provisional', 'the electronics row is untouched');
});

test('budget totals exclude provisional rows', async () => {
  // The acceptance criterion for the whole mechanism, as a query. A provisional
  // label is displayable and not summable.
  const ledger = new Ledger(handle.db);
  await ledger.importTransactions([
    row('t1', 'SQ *BLUE BOTTLE #1', -500),
    row('t2', 'SQ *BLUE BOTTLE #2', -700),
  ]);
  await ledger.recordPrediction('t1', {
    categoryId: 'food.coffee',
    source: 'memory',
    confidence: 0.9,
    costMicroUsd: 0,
    confirmedSupport: 9,
  });
  await ledger.recordPrediction('t2', {
    categoryId: 'food.coffee',
    source: 'llm',
    confidence: 0.99,
    costMicroUsd: 0,
    confirmedSupport: 0,
  });

  const totals = await ledger.totalsByCategory('2000-01-01', '2100-01-01');
  assert.deepEqual(totals, [{ categoryId: 'food.coffee', totalCents: -500 }], 'only the confirmed row is summed');
  assert.equal(await ledger.provisionalExcludedCount(), 1, 'and the exclusion is reportable, not silent');
});
