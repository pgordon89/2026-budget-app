import { test } from 'node:test';
import assert from 'node:assert/strict';

import { MerchantMemory } from './memory.js';
import { fitLexicalEmbedder } from './embed.js';
import { NeighbourIndex, type NeighbourConfig } from './knn.js';
import type { CategoryId } from '../core/taxonomy.js';

const COFFEE: Array<[string, CategoryId]> = [
  ['BLUE BOTTLE COFFEE', 'food.coffee'],
  ['SIGHTGLASS COFFEE', 'food.coffee'],
  ['RITUAL COFFEE ROASTERS', 'food.coffee'],
];

const FUEL: Array<[string, CategoryId]> = [
  ['SHELL SERVICE STATION', 'transport.fuel'],
  ['CHEVRON STATION', 'transport.fuel'],
  ['VALERO STATION', 'transport.fuel'],
];

async function build(
  seed: Array<[string, CategoryId]>,
  config: Partial<NeighbourConfig> = {},
): Promise<NeighbourIndex> {
  const memory = new MerchantMemory();
  for (const [descriptor, category] of seed) {
    // Enough sightings that the memory tier would be confident, so anything this
    // tier gets wrong is its own doing.
    for (let i = 0; i < 10; i++) memory.remember(descriptor, category);
  }
  const embedder = fitLexicalEmbedder([...memory.entries()].map((e) => e.key));
  return NeighbourIndex.build(memory, embedder, config);
}

test('categorises an unseen merchant from its labelled neighbours', async () => {
  // Explicit gate: the production default is selected by sweep against the real
  // corpus (`npm run analyze:knn`), and a unit test should not silently depend
  // on where that lands.
  const index = await build([...COFFEE, ...FUEL], { minConfidence: 0.2 });
  const outcome = await index.lookup('PEETS COFFEE');

  assert.equal(outcome.status, 'hit');
  assert.equal(outcome.status === 'hit' && outcome.category, 'food.coffee');
  assert.ok(outcome.status === 'hit' && outcome.agreement > 0.8, 'the coffee shops agree');
});

test('excludes the exact key from its own neighbourhood', async () => {
  // Otherwise this tier answers with the very key Tier 1 already ruled on, and
  // the pipeline routes around its own abstention.
  const index = await build([...COFFEE, ...FUEL]);
  const outcome = await index.lookup('BLUE BOTTLE COFFEE');

  assert.ok(outcome.status === 'hit' || outcome.status === 'low_confidence');
  const neighbours = outcome.status === 'hit' || outcome.status === 'low_confidence' ? outcome.neighbours : [];
  assert.ok(
    !neighbours.some((n) => n.key === 'BLUE BOTTLE COFFEE'),
    'a key must never be its own evidence',
  );
  assert.ok(neighbours.length > 0, 'it still gets to hear from the other coffee shops');
});

test('escalates when the nearest labelled merchant is not actually near', async () => {
  const index = await build([...COFFEE, ...FUEL]);
  const outcome = await index.lookup('ZURICH INSURANCE PREMIUM');

  assert.notEqual(outcome.status, 'hit');
  if (outcome.status === 'low_confidence') {
    assert.ok(outcome.nearest < 0.5, `weak neighbourhood, got nearest ${outcome.nearest.toFixed(3)}`);
  }
});

test('escalates when close neighbours disagree', async () => {
  // Two merchants with near-identical keys and opposite labels: the query sits
  // between them, so agreement collapses even though similarity is high.
  const index = await build([
    ['NORTHSIDE MARKET GROCERY', 'food.groceries'],
    ['NORTHSIDE MARKET DELI', 'food.restaurants'],
  ]);
  const outcome = await index.lookup('NORTHSIDE MARKET');

  assert.notEqual(outcome.status, 'hit');
  assert.ok(
    outcome.status === 'low_confidence' && outcome.agreement < 0.75,
    'a split neighbourhood must not read as confidence',
  );
});

test('a neighbour that is itself ambiguous casts a split vote', async () => {
  const memory = new MerchantMemory();
  for (let i = 0; i < 10; i++) memory.remember('NORTHSIDE MARKET DEPOT', 'food.groceries');
  for (let i = 0; i < 10; i++) memory.remember('NORTHSIDE MARKET DEPOT', 'shopping.household');
  const embedder = fitLexicalEmbedder([...memory.entries()].map((e) => e.key));
  const index = await NeighbourIndex.build(memory, embedder);

  const outcome = await index.lookup('NORTHSIDE MARKET');
  assert.notEqual(outcome.status, 'hit', 'inheriting a neighbour\'s ambiguity, not laundering it');
  assert.ok(outcome.status === 'low_confidence' && Math.abs(outcome.agreement - 0.5) < 1e-6);
});

test('reports no neighbours when nothing shares a single feature', async () => {
  const index = await build(COFFEE);
  assert.equal((await index.lookup('XQZ MPW')).status, 'no_neighbours');
});

test('refuses to answer on a degenerate key', async () => {
  const index = await build(COFFEE);
  assert.equal((await index.lookup('SQ *12345')).status, 'degenerate');
});

test('k bounds how many neighbours vote', async () => {
  const index = await build([...COFFEE, ...FUEL], { k: 2 });
  const outcome = await index.lookup('PEETS COFFEE');

  assert.ok(outcome.status === 'hit' || outcome.status === 'low_confidence');
  assert.equal(outcome.status === 'hit' || outcome.status === 'low_confidence' ? outcome.neighbours.length : -1, 2);
});

test('is deterministic across repeated lookups', async () => {
  const index = await build([...COFFEE, ...FUEL]);
  const first = await index.lookup('PEETS COFFEE ROASTERY');
  const second = await index.lookup('PEETS COFFEE ROASTERY');
  assert.deepEqual(first, second);
});

test('predict records the answering tier and its cost, and returns null to escalate', async () => {
  const index = await build([...COFFEE, ...FUEL], { costPerLookupUsd: 0.000002, minConfidence: 0.2 });

  const hit = await index.predict('PEETS COFFEE');
  assert.deepEqual(
    { category: hit?.category, tier: hit?.tier, costUsd: hit?.costUsd },
    { category: 'food.coffee', tier: 'embedding', costUsd: 0.000002 },
  );
  assert.equal(await index.predict('XQZ MPW'), null, 'no neighbourhood means escalate, not guess');
});
