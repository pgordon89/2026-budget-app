import { test } from 'node:test';
import assert from 'node:assert/strict';

import { compareToBaseline, GATED_METRICS, type GatedMetric } from './gate.js';

const METRICS: GatedMetric[] = [
  { path: 'overall.precisionPct', direction: 'higher-is-better', tolerance: 0 },
  { path: 'cost.perThousandTransactionsUsd', direction: 'lower-is-better', tolerance: 0 },
];

const report = (precision: number, cost: number) => ({
  overall: { precisionPct: precision },
  cost: { perThousandTransactionsUsd: cost },
});

test('passes when nothing moved', () => {
  const gate = compareToBaseline(report(98.5, 0.42), report(98.5, 0.42), METRICS);
  assert.equal(gate.regressions.length, 0);
  assert.equal(gate.improvements.length, 0);
  assert.ok(gate.changes.every((c) => c.verdict === 'unchanged'));
});

test('fails when precision drops', () => {
  const gate = compareToBaseline(report(97.9, 0.42), report(98.5, 0.42), METRICS);
  assert.equal(gate.regressions.length, 1);
  assert.equal(gate.regressions[0]?.path, 'overall.precisionPct');
});

test('fails when cost rises — direction is per metric, not global', () => {
  // The bug this guards: treating every metric as higher-is-better would call a
  // cost increase an improvement and wave it through.
  const gate = compareToBaseline(report(98.5, 0.51), report(98.5, 0.42), METRICS);
  assert.equal(gate.regressions.length, 1);
  assert.equal(gate.regressions[0]?.path, 'cost.perThousandTransactionsUsd');
});

test('lets improvements through instead of blocking on any change', () => {
  const gate = compareToBaseline(report(99.1, 0.31), report(98.5, 0.42), METRICS);
  assert.equal(gate.regressions.length, 0);
  assert.equal(gate.improvements.length, 2, 'better precision and cheaper both count as wins');
});

test('respects a tolerance when one is set', () => {
  const loose: GatedMetric[] = [{ path: 'overall.precisionPct', direction: 'higher-is-better', tolerance: 0.5 }];
  assert.equal(compareToBaseline(report(98.2, 0), report(98.5, 0), loose).regressions.length, 0);
  assert.equal(compareToBaseline(report(97.9, 0), report(98.5, 0), loose).regressions.length, 1);
});

test('treats a metric missing from the baseline as new, not as a regression', () => {
  // Otherwise adding a measurement fails the build, and nobody adds measurements.
  const gate = compareToBaseline(report(98.5, 0.42), { overall: { precisionPct: 98.5 } }, METRICS);
  assert.equal(gate.regressions.length, 0);
  assert.deepEqual(gate.incomparable, ['cost.perThousandTransactionsUsd']);
});

test('does not silently pass when a metric vanishes from the current report', () => {
  const gate = compareToBaseline({ overall: { precisionPct: 98.5 } }, report(98.5, 0.42), METRICS);
  assert.deepEqual(gate.incomparable, ['cost.perThousandTransactionsUsd']);
});

test('ignores non-numeric values at a gated path rather than coercing them', () => {
  const gate = compareToBaseline({ overall: { precisionPct: 'n/a' } }, report(98.5, 0.42), METRICS);
  assert.ok(gate.incomparable.includes('overall.precisionPct'));
  assert.equal(gate.regressions.length, 0);
});

test('the shipped gate covers precision, cost, transfers, and collisions', () => {
  // The four things that would each be a silent disaster if they slipped.
  const paths = GATED_METRICS.map((m) => m.path);
  assert.ok(paths.includes('overall.precisionPct'));
  assert.ok(paths.includes('cost.perThousandTransactionsUsd'));
  assert.ok(paths.includes('transfers.misclassifiedAsSpend'));
  assert.ok(paths.includes('normalizer.collisions'));
});

test('every shipped cost-like metric is gated in the direction that costs money', () => {
  for (const metric of GATED_METRICS) {
    const isCostLike = /cost|collision|misclassified/i.test(metric.path);
    assert.equal(
      metric.direction,
      isCostLike ? 'lower-is-better' : 'higher-is-better',
      `${metric.path} is gated the wrong way round`,
    );
  }
});
