import { test } from 'node:test';
import assert from 'node:assert/strict';

import { escalationInput, toVotePrior } from './escalation.js';
import type { NeighbourOutcome } from './knn.js';

const CONTESTED: NeighbourOutcome = {
  status: 'low_confidence',
  key: 'PRESIDIO VETERINARY',
  category: 'health.dental_vision',
  confidence: 0.48,
  agreement: 0.55,
  nearest: 0.87,
  neighbours: [{ key: 'PRESIDIO DENTAL', similarity: 0.87, category: 'health.dental_vision' }],
  distribution: [
    { category: 'health.dental_vision', share: 0.55 },
    { category: 'personal.pets', share: 0.45 },
  ],
};

test('hands over the vote when the neighbourhood produced one', () => {
  const prior = toVotePrior(CONTESTED);
  assert.equal(prior?.category, 'health.dental_vision');
  assert.equal(prior?.distribution.length, 2);
});

test('hands over nothing when the key has no neighbours', () => {
  // Not an empty distribution — no distribution. The model is told the history
  // is silent rather than shown a table with no rows in it.
  assert.equal(toVotePrior({ status: 'no_neighbours', key: 'XYZZY' }), null);
  assert.equal(toVotePrior({ status: 'degenerate', key: '' }), null);
});

test('the control arm keeps the neighbours and drops only the vote', () => {
  // What makes the A/B measure the prior rather than "the model was told about
  // its neighbours", which it already was before any of this.
  const withVote = escalationInput('PRESIDIO VETERINARY 4432', -84.5, CONTESTED, true);
  const without = escalationInput('PRESIDIO VETERINARY 4432', -84.5, CONTESTED, false);

  assert.deepEqual(withVote.neighbours, without.neighbours);
  assert.notEqual(withVote.vote, null);
  assert.equal(without.vote, null);
});
