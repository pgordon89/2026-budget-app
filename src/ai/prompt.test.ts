import { test } from 'node:test';
import assert from 'node:assert/strict';

import { systemPrompt, userMessage, PROMPT_VERSION, type ClassificationInput } from './prompt.js';
import { CATEGORY_IDS } from '../core/taxonomy.js';

const BASE: ClassificationInput = {
  rawDescriptor: 'SQ *PEETS COFFEE #221 SANFRAN CA',
  normalizedKey: 'PEETS COFFEE',
  amount: -6.75,
  neighbours: [
    { key: 'BLUE BOTTLE COFFEE', category: 'food.coffee', similarity: 0.42 },
    { key: 'SIGHTGLASS COFFEE', category: 'food.coffee', similarity: 0.38 },
  ],
};

test('offers exactly the taxonomy the eval scores', () => {
  // The model's label space and the eval's label space are the same array,
  // rendered — they cannot drift apart without a test failing here.
  const prompt = systemPrompt();
  for (const id of CATEGORY_IDS) {
    assert.ok(prompt.includes(id), `taxonomy id missing from prompt: ${id}`);
  }
});

test('the system block is identical for every transaction', () => {
  // This is the caching invariant, not a style preference: a prefix that differs
  // per transaction caches nothing, and the failure is silent.
  const other: ClassificationInput = {
    rawDescriptor: 'ACH DEBIT STERLING RIDGE APT',
    normalizedKey: 'STERLING RIDGE APT',
    amount: -2164.29,
    neighbours: [],
  };

  assert.equal(systemPrompt(), systemPrompt());
  assert.ok(!systemPrompt().includes(BASE.rawDescriptor));
  assert.ok(!systemPrompt().includes(other.rawDescriptor));
});

test('per-transaction detail lives in the user turn, after the cache breakpoint', () => {
  const message = userMessage(BASE);

  assert.ok(message.includes('SQ *PEETS COFFEE #221 SANFRAN CA'), 'raw descriptor');
  assert.ok(message.includes('PEETS COFFEE'), 'normalised key');
  assert.ok(message.includes('BLUE BOTTLE COFFEE'), 'neighbours are few-shot, and they vary per query');
  assert.ok(message.includes('food.coffee'));
});

test('keeps the raw descriptor as well as the normalised key', () => {
  // Tier 0 is deliberately lossy; the model should see what it threw away.
  const message = userMessage(BASE);
  assert.ok(message.indexOf('SANFRAN') > -1, 'city survived into the prompt via the raw string');
});

test('signs the amount explicitly rather than relying on a minus glyph', () => {
  assert.ok(userMessage({ ...BASE, amount: -6.75 }).includes('-$6.75'));
  assert.ok(userMessage({ ...BASE, amount: 42 }).includes('+$42.00'));
});

test('says so plainly when history offers no neighbours', () => {
  const message = userMessage({ ...BASE, neighbours: [] });
  assert.match(message, /No similar merchant/);
});

test('carries a prompt version, so wording changes cannot look like model changes', () => {
  assert.match(PROMPT_VERSION, /^v\d+$/);
});
