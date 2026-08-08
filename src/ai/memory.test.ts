import { test } from 'node:test';
import assert from 'node:assert/strict';

import { MerchantMemory, wilsonLowerBound, DEFAULT_MEMORY_CONFIG } from './memory.js';

const Z = DEFAULT_MEMORY_CONFIG.z;

/** Seed one merchant `times` times so the gate tests read at a glance. */
function seeded(descriptor: string, category: Parameters<MerchantMemory['remember']>[1], times: number, config = {}) {
  const memory = new MerchantMemory(config);
  for (let i = 0; i < times; i++) memory.remember(descriptor, category);
  return memory;
}

test('answers a merchant it has seen labeled consistently', () => {
  const memory = seeded('SQ *BLUE BOTTLE #4432 SANFRAN', 'food.coffee', 20);
  const outcome = memory.lookup('SQ *BLUE BOTTLE #4432 SANFRAN');

  assert.equal(outcome.status, 'hit');
  assert.equal(outcome.status === 'hit' && outcome.category, 'food.coffee');
});

test('answers descriptor variants of a merchant it has only seen in one form', () => {
  // The whole point of pairing Tier 0 with Tier 1: store number and city differ,
  // the key does not.
  const memory = seeded('SQ *BLUE BOTTLE #4432 SANFRAN', 'food.coffee', 20);
  const outcome = memory.lookup('SQ *BLUE BOTTLE #9981 OAKLAND CA');

  assert.equal(outcome.status, 'hit');
  assert.equal(outcome.status === 'hit' && outcome.category, 'food.coffee');
});

test('one clean sighting is not enough evidence to answer', () => {
  // The failure mode a raw winner/total ratio walks straight into: 1/1 = 100%.
  const memory = seeded('NETFLIX.COM', 'entertainment.streaming', 1);
  const outcome = memory.lookup('NETFLIX.COM');

  assert.notEqual(outcome.status, 'hit');
  assert.equal(outcome.status === 'low_confidence' && outcome.agreement, 1, 'agreement is unanimous');
  assert.ok(
    outcome.status === 'low_confidence' && outcome.confidence < 0.25,
    'unanimous but near-zero confidence, because n = 1',
  );
});

test('confidence grows with sample size at constant agreement', () => {
  const conf = (times: number) => {
    const outcome = seeded('NETFLIX.COM', 'entertainment.streaming', times).lookup('NETFLIX.COM');
    assert.notEqual(outcome.status, 'unseen');
    return outcome.status === 'hit' || outcome.status === 'low_confidence' ? outcome.confidence : 0;
  };

  assert.ok(conf(1) < conf(5) && conf(5) < conf(25), 'monotone in n at 100% agreement');
  assert.ok(conf(25) > 0.8, '25 unanimous sightings should be trusted');
});

test('abstains on a genuinely split merchant however often it has been seen', () => {
  // A mixed-basket retailer. Plurality would answer and be right ~52% of the
  // time; that is a confident wrong answer on nearly half its volume.
  const memory = new MerchantMemory({ minConfidence: 0.6 });
  for (let i = 0; i < 30; i++) memory.remember('AMZN Mktp US*2K4LM9XY3', 'shopping.general');
  for (let i = 0; i < 28; i++) memory.remember('AMZN Mktp US*7QW3ZX1', 'shopping.household');

  const outcome = memory.lookup('AMZN Mktp US*HM8FQ2XZW');
  assert.equal(outcome.status, 'low_confidence');
  assert.ok(outcome.status === 'low_confidence' && outcome.support > 50, 'plenty of data, still not separable');
});

test('a user correction outweighs the pipeline\'s own prior guesses', () => {
  const memory = new MerchantMemory();
  for (let i = 0; i < 3; i++) memory.remember('TARGET T-1049 BERKELEY', 'shopping.general', 'inferred');
  memory.remember('TARGET T-1049 BERKELEY', 'shopping.household', 'confirmed');

  const outcome = memory.lookup('TARGET T-1049 BERKELEY');
  assert.equal(
    outcome.status === 'hit' || outcome.status === 'low_confidence' ? outcome.category : undefined,
    'shopping.household',
    'one confirmation beats three inferences',
  );
});

test('inferred observations warm the cache only after repeated agreement', () => {
  // Guards the self-reinforcement loop: the tier must not become confident by
  // reading back a single guess of its own.
  const memory = new MerchantMemory({ minConfidence: 0.5 });
  memory.remember('SQ *NEW CAFE', 'food.coffee', 'inferred');
  assert.notEqual(memory.lookup('SQ *NEW CAFE').status, 'hit', 'one inference proves nothing');

  for (let i = 0; i < 14; i++) memory.remember('SQ *NEW CAFE', 'food.coffee', 'inferred');
  assert.notEqual(memory.lookup('SQ *NEW CAFE').status, 'hit', 'still short of one confirmed-equivalent gate');

  memory.remember('SQ *NEW CAFE', 'food.coffee', 'inferred');
  assert.equal(memory.lookup('SQ *NEW CAFE').status, 'hit', '16 agreeing inferences = 4 confirmations');
});

test('refuses to store or answer on a degenerate key', () => {
  // Tier 0 stripped everything identifying; whatever is left would be served to
  // unrelated merchants.
  const memory = new MerchantMemory();
  memory.remember('SQ *12345', 'food.coffee');

  assert.equal(memory.size, 0, 'nothing was stored');
  assert.equal(memory.lookup('SQ *12345').status, 'degenerate');
});

test('reports an unseen merchant as unseen, not as a low-confidence guess', () => {
  const memory = seeded('SQ *BLUE BOTTLE', 'food.coffee', 20);
  assert.equal(memory.lookup('WHOLEFDS SANFRAN #10234').status, 'unseen');
});

test('resolves ties deterministically regardless of insertion order', () => {
  const build = (first: 'food.coffee' | 'food.groceries') => {
    const memory = new MerchantMemory();
    const second = first === 'food.coffee' ? 'food.groceries' : 'food.coffee';
    for (let i = 0; i < 10; i++) memory.remember('SPLIT MERCHANT', first);
    for (let i = 0; i < 10; i++) memory.remember('SPLIT MERCHANT', second);
    const outcome = memory.lookup('SPLIT MERCHANT');
    return outcome.status === 'hit' || outcome.status === 'low_confidence' ? outcome.category : undefined;
  };

  // Insertion-order-dependent winners would make eval numbers move whenever the
  // fixture's row order changed.
  assert.equal(build('food.coffee'), build('food.groceries'));
  assert.equal(build('food.coffee'), 'food.coffee', 'lexicographic tie-break');
});

test('forget clears accumulated weight for a merchant', () => {
  const memory = seeded('SQ *BLUE BOTTLE', 'food.coffee', 20);
  assert.equal(memory.forget('SQ *BLUE BOTTLE #77 OAKLAND'), true, 'forgets by normalized key, not raw string');
  assert.equal(memory.lookup('SQ *BLUE BOTTLE').status, 'unseen');
});

test('rejects a category id outside the taxonomy', () => {
  const memory = new MerchantMemory();
  assert.throws(() => memory.remember('SQ *BLUE BOTTLE', 'food.coffee_shops' as 'food.coffee'), /unknown category/);
});

test('predict records the answering tier and its cost, and returns null to escalate', () => {
  const memory = seeded('SQ *BLUE BOTTLE', 'food.coffee', 20);

  const hit = memory.predict('SQ *BLUE BOTTLE');
  assert.deepEqual(
    { category: hit?.category, tier: hit?.tier, costUsd: hit?.costUsd },
    { category: 'food.coffee', tier: 'memory', costUsd: 0 },
  );
  assert.equal(memory.predict('WHOLEFDS SANFRAN'), null, 'a miss escalates rather than guessing');
});

test('wilson lower bound penalises small samples and is bounded to [0,1]', () => {
  assert.equal(wilsonLowerBound(0, 0, Z), 0);
  assert.ok(wilsonLowerBound(1, 1, Z) < 0.25, '1/1 must not score as certainty');
  assert.ok(wilsonLowerBound(1, 1, Z) < wilsonLowerBound(10, 10, Z));
  assert.ok(wilsonLowerBound(10, 10, Z) < wilsonLowerBound(100, 100, Z));
  assert.ok(wilsonLowerBound(100, 100, Z) < 1);

  // Fractional counts are legal — inferred observations carry partial weight.
  assert.ok(wilsonLowerBound(0.25, 0.25, Z) < wilsonLowerBound(1, 1, Z));
});
