import { test, before, after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';

import { createMemoryDatabase, type Handle } from './client.js';
import { MerchantMemoryRepository } from './merchant-memory-repository.js';
import { merchantMemory } from './schema.js';
import { MerchantMemory, type ObservationSource } from '../ai/memory.js';
import type { CategoryId } from '../core/taxonomy.js';

let handle: Handle;

before(async () => {
  handle = await createMemoryDatabase();
});
after(async () => {
  await handle.close();
});
beforeEach(async () => {
  await handle.db.delete(merchantMemory);
});

type Sighting = [descriptor: string, category: CategoryId, source?: ObservationSource];

/**
 * The behaviours Tier 1 is defined by. Each runs against both backends and the
 * outcomes must match — that is the actual contract, not "both look reasonable".
 */
const SCENARIOS: Array<{ name: string; seed: Sighting[]; query: string }> = [
  {
    name: 'a merchant seen consistently',
    seed: Array.from({ length: 20 }, () => ['SQ *BLUE BOTTLE #4432 SANFRAN', 'food.coffee'] as Sighting),
    query: 'SQ *BLUE BOTTLE #9981 OAKLAND CA',
  },
  {
    name: 'a single sighting',
    seed: [['NETFLIX.COM', 'entertainment.streaming']],
    query: 'NETFLIX.COM',
  },
  {
    name: 'a split merchant',
    seed: [
      ...Array.from({ length: 30 }, () => ['AMZN Mktp US*2K4LM9XY3', 'shopping.general'] as Sighting),
      ...Array.from({ length: 28 }, () => ['AMZN Mktp US*7QW3ZX1', 'shopping.household'] as Sighting),
    ],
    query: 'AMZN Mktp US*HM8FQ2XZW',
  },
  {
    name: 'an unseen merchant',
    seed: [['SQ *BLUE BOTTLE', 'food.coffee']],
    query: 'WHOLEFDS SANFRAN #10234',
  },
  {
    name: 'a degenerate key',
    seed: [['SQ *BLUE BOTTLE', 'food.coffee']],
    query: 'SQ *12345',
  },
  {
    name: 'a correction outweighing prior inferences',
    seed: [
      ['TARGET T-1049 BERKELEY', 'shopping.general', 'inferred'],
      ['TARGET T-1049 BERKELEY', 'shopping.general', 'inferred'],
      ['TARGET T-1049 BERKELEY', 'shopping.general', 'inferred'],
      ['TARGET T-1049 BERKELEY', 'shopping.household', 'confirmed'],
    ],
    query: 'TARGET T-1049 BERKELEY',
  },
  {
    name: 'a lexicographic tie',
    seed: [
      ...Array.from({ length: 10 }, () => ['SPLIT MERCHANT', 'food.groceries'] as Sighting),
      ...Array.from({ length: 10 }, () => ['SPLIT MERCHANT', 'food.coffee'] as Sighting),
    ],
    query: 'SPLIT MERCHANT',
  },
];

for (const scenario of SCENARIOS) {
  test(`postgres and in-memory agree: ${scenario.name}`, async () => {
    const inMemory = new MerchantMemory();
    const repository = new MerchantMemoryRepository(handle.db);

    for (const [descriptor, category, source] of scenario.seed) {
      inMemory.remember(descriptor, category, source);
      await repository.remember(descriptor, category, source);
    }

    const expected = inMemory.lookup(scenario.query);
    const actual = await repository.lookup(scenario.query);

    assert.equal(actual.status, expected.status, 'status');
    if (expected.status === 'hit' || expected.status === 'low_confidence') {
      assert.equal(actual.status === expected.status && actual.category, expected.category, 'category');
      const confidence = actual.status === 'hit' || actual.status === 'low_confidence' ? actual.confidence : -1;
      assert.ok(Math.abs(confidence - expected.confidence) < 1e-9, 'confidence');
    }
  });
}

test('hydrating reproduces the store that wrote it', async () => {
  // The round trip that matters: a batch eval loads tallies rather than
  // replaying sightings, so weights must survive rather than be recomputed.
  const inMemory = new MerchantMemory();
  const repository = new MerchantMemoryRepository(handle.db);

  const seed: Sighting[] = [
    ...Array.from({ length: 12 }, () => ['WHOLEFDS SANFRAN #10234', 'food.groceries'] as Sighting),
    ['AMZN Mktp US*2K4LM9XY3', 'shopping.general', 'inferred'],
    ['AMZN Mktp US*2K4LM9XY3', 'shopping.household', 'confirmed'],
    ['SHELL SERVICE STATION SANFRAN CA', 'transport.fuel'],
  ];
  for (const [descriptor, category, source] of seed) {
    inMemory.remember(descriptor, category, source);
    await repository.remember(descriptor, category, source);
  }

  const hydrated = await repository.hydrate();
  assert.equal(hydrated.size, inMemory.size);
  assert.equal(hydrated.observations, inMemory.observations);

  for (const [descriptor] of seed) {
    assert.deepEqual(hydrated.lookup(descriptor), inMemory.lookup(descriptor), descriptor);
  }
});

test('accumulates weight in SQL rather than read-modify-write', async () => {
  // Concurrent corrections for one merchant must not lose each other. An
  // application-side read-modify-write would drop all but the last.
  const repository = new MerchantMemoryRepository(handle.db);
  await Promise.all(
    Array.from({ length: 25 }, () => repository.remember('SQ *BLUE BOTTLE', 'food.coffee')),
  );

  assert.equal(await repository.weightFor('SQ *BLUE BOTTLE', 'food.coffee'), 25);
  assert.deepEqual(await repository.stats(), { keys: 1, observations: 25 });
});

test('discounts inferred observations exactly as the in-memory store does', async () => {
  const repository = new MerchantMemoryRepository(handle.db);
  await repository.remember('SQ *NEW CAFE', 'food.coffee', 'inferred');
  await repository.remember('SQ *NEW CAFE', 'food.coffee', 'confirmed');

  // 0.25 + 1.0 — the weighting that stops the store growing confident by
  // reading back its own guesses.
  assert.equal(await repository.weightFor('SQ *NEW CAFE', 'food.coffee'), 1.25);
});

test('refuses to store a degenerate key', async () => {
  const repository = new MerchantMemoryRepository(handle.db);
  await repository.remember('SQ *12345', 'food.coffee');

  assert.deepEqual(await repository.stats(), { keys: 0, observations: 0 });
  assert.equal((await repository.lookup('SQ *12345')).status, 'degenerate');
});

test('forget clears a merchant by normalised key, not raw string', async () => {
  const repository = new MerchantMemoryRepository(handle.db);
  for (let i = 0; i < 20; i++) await repository.remember('SQ *BLUE BOTTLE #4432 SANFRAN', 'food.coffee');

  assert.equal(await repository.forget('SQ *BLUE BOTTLE #77 OAKLAND'), 1);
  assert.equal((await repository.lookup('SQ *BLUE BOTTLE')).status, 'unseen');
});

test('rejects a category outside the taxonomy before it reaches the database', async () => {
  const repository = new MerchantMemoryRepository(handle.db);
  await assert.rejects(
    () => repository.remember('SQ *BLUE BOTTLE', 'food.coffee_shops' as CategoryId),
    /unknown category/,
  );
});

test('the categories table is seeded from the taxonomy module', async () => {
  // The foreign key is only worth having if the table it points at matches the
  // TypeScript that defines the label space.
  const { CATEGORIES } = await import('../core/taxonomy.js');
  const rows = await handle.db.query.categories.findMany();
  assert.equal(rows.length, CATEGORIES.length);
  assert.deepEqual(
    rows.map((r) => r.id).sort(),
    CATEGORIES.map((c) => c.id).sort(),
  );
});
