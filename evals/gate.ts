/**
 * Regression gate: compares a fresh eval report against the committed baseline.
 *
 * Separated from the runner so the comparison logic is unit-testable without
 * running an eval. The gate is the thing CI actually enforces, so a bug in it is
 * worse than a bug in a metric — a gate that silently passes everything looks
 * exactly like a gate that is working.
 *
 * Direction matters and is declared per metric. Precision going up is a win;
 * cost going up is a regression; both are "the number changed". A gate that only
 * checked for change would block every improvement.
 */

export type Direction = 'higher-is-better' | 'lower-is-better';

export interface GatedMetric {
  /** Dotted path into the report, e.g. `overall.precisionPct`. */
  readonly path: string;
  readonly direction: Direction;
  /**
   * How much worse a value may get before it fails.
   *
   * Mostly zero. Every tier is deterministic and the model tier is replayed from
   * a committed cache, so the same commit produces byte-identical numbers — this
   * gate can be exact rather than fuzzy, which is a stronger guarantee than the
   * usual "within a few percent" band.
   */
  readonly tolerance: number;
}

export const GATED_METRICS: readonly GatedMetric[] = [
  { path: 'normalizer.collisions', direction: 'lower-is-better', tolerance: 0 },
  { path: 'overall.precisionPct', direction: 'higher-is-better', tolerance: 0 },
  { path: 'overall.resolvedPct', direction: 'higher-is-better', tolerance: 0 },
  { path: 'overall.coveragePct', direction: 'higher-is-better', tolerance: 0 },
  { path: 'slices.hard.precisionPct', direction: 'higher-is-better', tolerance: 0 },
  { path: 'transfers.misclassifiedAsSpend', direction: 'lower-is-better', tolerance: 0 },
  { path: 'cost.perThousandTransactionsUsd', direction: 'lower-is-better', tolerance: 0 },
  { path: 'tiers.memory.precisionPct', direction: 'higher-is-better', tolerance: 0 },
  { path: 'tiers.embedding.precisionPct', direction: 'higher-is-better', tolerance: 0 },
  { path: 'tiers.llm.precisionPct', direction: 'higher-is-better', tolerance: 0 },
];

export interface Change {
  readonly path: string;
  readonly baseline: number;
  readonly current: number;
  readonly delta: number;
  readonly verdict: 'regressed' | 'improved' | 'unchanged';
}

export interface GateResult {
  readonly changes: readonly Change[];
  readonly regressions: readonly Change[];
  readonly improvements: readonly Change[];
  /** Metrics present in one report but not the other — usually a new tier. */
  readonly incomparable: readonly string[];
}

function readPath(report: unknown, path: string): number | undefined {
  let node: unknown = report;
  for (const segment of path.split('.')) {
    if (typeof node !== 'object' || node === null) return undefined;
    node = (node as Record<string, unknown>)[segment];
  }
  return typeof node === 'number' ? node : undefined;
}

export function compareToBaseline(
  current: unknown,
  baseline: unknown,
  metrics: readonly GatedMetric[] = GATED_METRICS,
): GateResult {
  const changes: Change[] = [];
  const incomparable: string[] = [];

  for (const metric of metrics) {
    const before = readPath(baseline, metric.path);
    const after = readPath(current, metric.path);

    // A metric only one side has cannot be a regression — it is a new
    // measurement, and failing the build for adding one would discourage adding
    // any. Reported so it is visible rather than silently dropped.
    if (before === undefined || after === undefined) {
      incomparable.push(metric.path);
      continue;
    }

    const delta = after - before;
    const worse = metric.direction === 'higher-is-better' ? -delta : delta;

    changes.push({
      path: metric.path,
      baseline: before,
      current: after,
      delta,
      verdict: worse > metric.tolerance ? 'regressed' : delta === 0 ? 'unchanged' : 'improved',
    });
  }

  return {
    changes,
    regressions: changes.filter((c) => c.verdict === 'regressed'),
    improvements: changes.filter((c) => c.verdict === 'improved'),
    incomparable,
  };
}
