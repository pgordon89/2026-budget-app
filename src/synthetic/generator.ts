/**
 * Seeded synthetic transaction generator.
 *
 * Purpose is twofold:
 *   1. A public demo account with realistic finances and zero real-world exposure.
 *   2. A *reproducible* labeled corpus for the categorization eval. Same seed in,
 *      byte-identical corpus out — otherwise accuracy numbers across commits are
 *      not comparable and the CI gate is meaningless.
 *
 * Determinism rules, enforced by review: no Date.now(), no Math.random(), no
 * Intl-dependent formatting. Every source of variation flows from `seed`.
 */

import { MERCHANTS, type Cadence, type MerchantDef } from './merchants.js';
import { getCategory, type CategoryId } from '../core/taxonomy.js';

// ── Deterministic RNG ───────────────────────────────────────────────────────

/** mulberry32 — small, fast, good enough distribution for fixture generation. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

class Rng {
  private readonly next: () => number;
  constructor(seed: number) {
    this.next = mulberry32(seed);
  }
  float(min = 0, max = 1): number {
    return min + this.next() * (max - min);
  }
  int(min: number, max: number): number {
    return Math.floor(this.float(min, max + 1));
  }
  pick<T>(items: readonly T[]): T {
    if (items.length === 0) throw new Error('pick() on empty array');
    return items[Math.floor(this.next() * items.length)]!;
  }
  bool(probability: number): boolean {
    return this.next() < probability;
  }
  /** Box–Muller, clamped to +/- 3 sigma so fixtures never contain absurd outliers. */
  normal(mean: number, stdDev: number): number {
    const u = Math.max(this.next(), 1e-9);
    const v = this.next();
    const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    return mean + stdDev * Math.max(-3, Math.min(3, z));
  }
  /**
   * Right-skewed draw across [min, max]: mode near min, long thin tail toward max.
   *
   * Real spend at a given merchant is skewed this way — most Amazon orders are $30
   * and a few are $600. Drawing from the midpoint instead (the obvious first
   * implementation) makes every wide-range merchant behave like its rare expensive
   * case, which inflated total outflow to ~2x income in the first fixture run.
   */
  skewed(min: number, max: number, exponent = 2.2): number {
    return min + (max - min) * Math.pow(this.next(), exponent);
  }
}

// ── Date helpers (UTC-only, no locale dependence) ───────────────────────────

const DAY_MS = 86_400_000;

function toDayNumber(iso: string): number {
  const [y, m, d] = iso.split('-').map(Number) as [number, number, number];
  return Math.floor(Date.UTC(y, m - 1, d) / DAY_MS);
}

function toIso(dayNumber: number): string {
  const dt = new Date(dayNumber * DAY_MS);
  const y = dt.getUTCFullYear();
  const m = String(dt.getUTCMonth() + 1).padStart(2, '0');
  const d = String(dt.getUTCDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

function monthStartsBetween(startDay: number, endDay: number): number[] {
  const out: number[] = [];
  const start = new Date(startDay * DAY_MS);
  let y = start.getUTCFullYear();
  let m = start.getUTCMonth();
  for (;;) {
    const day = Math.floor(Date.UTC(y, m, 1) / DAY_MS);
    if (day > endDay) break;
    if (day >= startDay - 31) out.push(day);
    m += 1;
    if (m > 11) {
      m = 0;
      y += 1;
    }
  }
  return out;
}

// ── Descriptor expansion ────────────────────────────────────────────────────

const CITY_FRAGMENTS = ['SANFRAN', 'OAKLAND', 'BERKELEY', 'SF', 'DALY CITY', 'SAN MATEO', 'EMERYVLE', 'ALAMEDA'] as const;
const STATES = ['CA', 'CA', 'CA', 'NV', 'OR'] as const;
const REF_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ0123456789';

function expandDescriptor(template: string, rng: Rng): string {
  return template
    .replace(/\{n(\d+)\}/g, (_m, digits: string) => {
      const len = Number(digits);
      let s = '';
      for (let i = 0; i < len; i++) s += String(rng.int(0, 9));
      return s;
    })
    .replace(/\{city\}/g, () => rng.pick(CITY_FRAGMENTS))
    .replace(/\{st\}/g, () => rng.pick(STATES))
    .replace(/\{ref\}/g, () => {
      const len = rng.int(6, 11);
      let s = '';
      for (let i = 0; i < len; i++) s += REF_ALPHABET[rng.int(0, REF_ALPHABET.length - 1)];
      return s;
    });
}

/**
 * Post-processing that mimics real bank feeds: some institutions upper-case and
 * truncate, some collapse whitespace, some leave a trailing city/state blob.
 * Applied probabilistically so the corpus contains both clean and mangled forms.
 */
function applyFeedNoise(descriptor: string, rng: Rng): string {
  let out = descriptor;
  if (rng.bool(0.18)) out = out.replace(/\s+/g, ' ');
  if (rng.bool(0.12) && out.length > 22) out = out.slice(0, rng.int(20, Math.min(32, out.length)));
  if (rng.bool(0.08)) out = `POS DEBIT ${out}`;
  if (rng.bool(0.05)) out = `${out}  ${rng.pick(CITY_FRAGMENTS)} ${rng.pick(STATES)}`;
  return out.trim();
}

// ── Accounts ────────────────────────────────────────────────────────────────

export interface SyntheticAccount {
  readonly id: string;
  readonly name: string;
  readonly type: 'checking' | 'savings' | 'credit';
  readonly mask: string;
}

export const ACCOUNTS: readonly SyntheticAccount[] = [
  { id: 'acct_chk', name: 'Everyday Checking', type: 'checking', mask: '4417' },
  { id: 'acct_sav', name: 'High-Yield Savings', type: 'savings', mask: '9032' },
  { id: 'acct_cc', name: 'Sapphire Credit Card', type: 'credit', mask: '2288' },
];

/** Bills, payroll, and ACH run through checking; day-to-day spend hits the card. */
function routeAccount(merchant: MerchantDef, rng: Rng): string {
  const group = getCategory(merchant.category)?.group;
  switch (group) {
    case 'income':
      return 'acct_chk';
    case 'transfer':
      return merchant.category === 'transfer.internal' ? 'acct_sav' : 'acct_chk';
    case 'housing':
    case 'financial':
      return rng.bool(0.85) ? 'acct_chk' : 'acct_cc';
    default:
      return rng.bool(0.82) ? 'acct_cc' : 'acct_chk';
  }
}

// ── Occurrence scheduling ───────────────────────────────────────────────────

function occurrenceDays(cadence: Cadence, startDay: number, endDay: number, rng: Rng): number[] {
  const days: number[] = [];
  const span = endDay - startDay + 1;

  switch (cadence.kind) {
    case 'monthly': {
      for (const monthStart of monthStartsBetween(startDay, endDay)) {
        const jitter = cadence.jitterDays ? rng.int(-cadence.jitterDays, cadence.jitterDays) : 0;
        days.push(monthStart + (cadence.dayOfMonth - 1) + jitter);
      }
      break;
    }
    case 'biweekly': {
      // Anchor to the first matching weekday so paychecks land consistently.
      let d = startDay;
      while (new Date(d * DAY_MS).getUTCDay() !== cadence.anchorDay) d += 1;
      for (; d <= endDay; d += 14) days.push(d);
      break;
    }
    case 'weekly': {
      const weeks = Math.ceil(span / 7);
      for (let w = 0; w < weeks; w++) {
        // Fractional rates become a probability; integral parts become guaranteed hits.
        const guaranteed = Math.floor(cadence.timesPerWeek);
        const extra = rng.bool(cadence.timesPerWeek - guaranteed) ? 1 : 0;
        for (let i = 0; i < guaranteed + extra; i++) {
          days.push(startDay + w * 7 + rng.int(0, 6));
        }
      }
      break;
    }
    case 'sporadic': {
      const expected = (cadence.timesPerYear * span) / 365;
      const count = Math.floor(expected) + (rng.bool(expected % 1) ? 1 : 0);
      for (let i = 0; i < count; i++) days.push(startDay + rng.int(0, span - 1));
      break;
    }
  }

  return days.filter((d) => d >= startDay && d <= endDay).sort((a, b) => a - b);
}

/** Modest seasonal lift so forecasting and anomaly features have signal to find. */
function seasonalMultiplier(dayNumber: number, category: CategoryId): number {
  const month = new Date(dayNumber * DAY_MS).getUTCMonth(); // 0 = Jan
  if (category === 'shopping.gifts' && month === 11) return 3.2;
  if (category === 'housing.utilities' && (month === 0 || month === 1 || month === 7)) return 1.35;
  if (category === 'travel.flights' && (month === 5 || month === 6 || month === 11)) return 1.6;
  if (category === 'food.restaurants' && month === 11) return 1.25;
  return 1;
}

// ── Output shape ────────────────────────────────────────────────────────────

export interface SyntheticTransaction {
  readonly id: string;
  readonly date: string;
  /** Negative = money leaving the household. Stored in dollars, 2dp. */
  readonly amount: number;
  readonly rawDescriptor: string;
  readonly accountId: string;
  /** Ground truth. Never exposed to the categorization pipeline at inference time. */
  readonly label: {
    readonly category: CategoryId;
    readonly merchant: string;
    /** True when the descriptor alone is genuinely ambiguous to a human annotator. */
    readonly hard: boolean;
  };
}

export interface GenerateOptions {
  readonly seed: number;
  readonly startDate: string;
  readonly endDate: string;
  /** Scales every amount — lets one catalog serve several income levels. */
  readonly incomeScale?: number;
}

function weightedPick(mix: readonly (readonly [CategoryId, number])[], rng: Rng): CategoryId {
  const total = mix.reduce((s, [, w]) => s + w, 0);
  let r = rng.float(0, total);
  for (const [id, w] of mix) {
    r -= w;
    if (r <= 0) return id;
  }
  return mix[mix.length - 1]![0];
}

export function generateTransactions(opts: GenerateOptions): SyntheticTransaction[] {
  const rng = new Rng(opts.seed);
  const scale = opts.incomeScale ?? 1;
  const startDay = toDayNumber(opts.startDate);
  const endDay = toDayNumber(opts.endDate);
  if (endDay < startDay) throw new Error('endDate must not precede startDate');

  const out: SyntheticTransaction[] = [];

  for (const merchant of MERCHANTS) {
    const direction = getCategory(merchant.category)?.direction ?? 'outflow';
    const [min, max] = merchant.amount;

    // Merchant lifecycle. A merchant that starts partway through is genuinely
    // new to the user at that point, which is the only traffic Tier 2 exists to
    // serve — and which a catalog where every merchant spans the whole range
    // produces none of.
    const activeFrom = merchant.activeFrom ? Math.max(startDay, toDayNumber(merchant.activeFrom)) : startDay;
    const activeUntil = merchant.activeUntil ? Math.min(endDay, toDayNumber(merchant.activeUntil)) : endDay;
    if (activeUntil < activeFrom) continue;

    for (const day of occurrenceDays(merchant.cadence, activeFrom, activeUntil, rng)) {
      let amount: number;
      if (merchant.fixedAmount) {
        // Fixed-price merchants drift only at contract boundaries, not per charge.
        amount = min + (max - min) * ((merchant.name.length % 7) / 7);
      } else {
        // Income is roughly symmetric; discretionary spend is not.
        amount =
          direction === 'inflow'
            ? rng.normal((min + max) / 2, (max - min) / 5)
            : rng.skewed(min, max);
      }
      // Multi-department merchants draw a category per basket. The descriptor
      // carries no signal about which one, so this is the irreducible ambiguity
      // the pipeline's accuracy ceiling is made of.
      const category = merchant.categoryMix ? weightedPick(merchant.categoryMix, rng) : merchant.category;

      amount *= seasonalMultiplier(day, category) * scale;
      amount = Math.max(min * 0.6, Math.min(max * 1.8, amount));

      const signed = direction === 'inflow' ? amount : -amount;
      const descriptor = applyFeedNoise(expandDescriptor(rng.pick(merchant.descriptors), rng), rng);

      out.push({
        id: `stx_${out.length.toString().padStart(6, '0')}`,
        date: toIso(day),
        amount: Math.round(signed * 100) / 100,
        rawDescriptor: descriptor,
        accountId: routeAccount(merchant, rng),
        label: {
          category,
          merchant: merchant.name,
          hard: merchant.hard ?? (merchant.categoryMix?.length ?? 0) > 1,
        },
      });
    }
  }

  // Stable sort: date, then original emission order. Keeps output byte-identical.
  return out
    .map((t, i) => ({ t, i }))
    .sort((a, b) => (a.t.date === b.t.date ? a.i - b.i : a.t.date < b.t.date ? -1 : 1))
    .map(({ t }, idx) => ({ ...t, id: `stx_${idx.toString().padStart(6, '0')}` }));
}
