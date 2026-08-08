import { test } from 'node:test';
import assert from 'node:assert/strict';

import { MODELS, costOf, addUsage, readUsage, EMPTY_USAGE } from './client.js';

const HAIKU = MODELS['claude-haiku-4-5'];
const SONNET = MODELS['claude-sonnet-5'];

test('prices input and output at their separate rates', () => {
  const usage = { ...EMPTY_USAGE, inputTokens: 1_000_000, outputTokens: 1_000_000 };
  assert.equal(costOf(usage, HAIKU), HAIKU.inputPerMTok + HAIKU.outputPerMTok);
});

test('prices cache reads far below fresh input, and writes above it', () => {
  const fresh = costOf({ ...EMPTY_USAGE, inputTokens: 1_000_000 }, HAIKU);
  const read = costOf({ ...EMPTY_USAGE, cacheReadTokens: 1_000_000 }, HAIKU);
  const write = costOf({ ...EMPTY_USAGE, cacheCreationTokens: 1_000_000 }, HAIKU);

  assert.ok(read < fresh, 'a cache read is the cheap path');
  assert.ok(write > fresh, 'a cache write carries a premium — caching is not free on first use');
});

test('the routing claim is priced: the reasoning model costs multiples of the bulk one', () => {
  const usage = { ...EMPTY_USAGE, inputTokens: 500_000, outputTokens: 20_000 };
  assert.ok(costOf(usage, SONNET) > costOf(usage, HAIKU) * 2.5);
});

test('costs nothing when nothing was spent', () => {
  assert.equal(costOf(EMPTY_USAGE, HAIKU), 0);
});

test('reads the SDK usage block, tolerating absent cache fields', () => {
  assert.deepEqual(readUsage({ input_tokens: 10, output_tokens: 2 }), {
    inputTokens: 10,
    outputTokens: 2,
    cacheReadTokens: 0,
    cacheCreationTokens: 0,
  });

  assert.deepEqual(
    readUsage({ input_tokens: 10, output_tokens: 2, cache_read_input_tokens: 5, cache_creation_input_tokens: 7 }),
    { inputTokens: 10, outputTokens: 2, cacheReadTokens: 5, cacheCreationTokens: 7 },
  );
});

test('sums usage across the calls of one classification', () => {
  const a = { inputTokens: 1, outputTokens: 2, cacheReadTokens: 3, cacheCreationTokens: 4 };
  assert.deepEqual(addUsage(a, a), {
    inputTokens: 2,
    outputTokens: 4,
    cacheReadTokens: 6,
    cacheCreationTokens: 8,
  });
});

test('records the prefix length below which caching silently does nothing', () => {
  // Not trivia: a shared prompt under this size accepts cache_control and never
  // caches, with no error to notice. Haiku 4.5's floor is unusually high.
  assert.ok(HAIKU.minCacheablePrefixTokens >= 4096);
  assert.ok(SONNET.minCacheablePrefixTokens <= HAIKU.minCacheablePrefixTokens);
});
