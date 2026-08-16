/**
 * Merchant catalog for the synthetic data generator.
 *
 * Every entry pairs a canonical merchant identity with the *raw descriptor templates*
 * a bank would actually emit for it. The mess is the point: processor prefixes
 * (`SQ *`, `TST*`, `PAYPAL *`), truncation, store numbers, embedded city codes, and
 * inconsistent spacing are exactly what the categorization pipeline has to survive.
 *
 * Placeholders expanded by the generator:
 *   {n3} {n4} {n5}  — zero-padded random digits
 *   {city}          — 3–8 char city fragment
 *   {st}            — 2-letter state
 *   {ref}           — alphanumeric reference blob
 */

import type { CategoryId } from '../core/taxonomy.js';

/** How often the merchant shows up in a persona's ledger. */
export type Cadence =
  | { kind: 'monthly'; dayOfMonth: number; jitterDays?: number }
  | { kind: 'biweekly'; anchorDay: number }
  | { kind: 'weekly'; timesPerWeek: number }
  | { kind: 'sporadic'; timesPerYear: number };

export interface MerchantDef {
  /** Canonical, human-facing name. Also the target for merchant normalization evals. */
  readonly name: string;
  /** Modal category. Used directly unless `categoryMix` is present. */
  readonly category: CategoryId;
  /**
   * Weighted category distribution for merchants that sell across departments.
   *
   * This is what makes the task genuinely hard. An Amazon order is household
   * supplies, electronics, or pet food depending on the basket, and the descriptor
   * says none of that. Without a mix, every merchant maps to exactly one category,
   * a memory lookup is correct by construction, and the eval reports 100% precision
   * for a system that has learned nothing — which is exactly what the first version
   * of this fixture did.
   *
   * The consequence is a real accuracy ceiling below 100%: the share of spend that
   * cannot be resolved from a descriptor alone. That ceiling is the argument for
   * line-item receipt ingestion in Phase 5.
   */
  readonly categoryMix?: readonly (readonly [CategoryId, number])[];
  readonly descriptors: readonly string[];
  /** Amount range in dollars. Sign is applied by the generator from category direction. */
  readonly amount: readonly [min: number, max: number];
  readonly cadence: Cadence;
  /** Fixed-amount merchants (rent, subscriptions) don't vary run to run. */
  readonly fixedAmount?: boolean;
  /**
   * Marks descriptors that are genuinely ambiguous from text alone — a human
   * annotator would need account context. These form the "hard" eval slice.
   */
  readonly hard?: boolean;
  /**
   * Merchant lifecycle. Absent means present for the whole range.
   *
   * Added because the first corpus had every merchant spanning all 30 months, so
   * *no merchant was ever new*. Tier 2 exists precisely to answer merchants the
   * user's history has never seen, and the fixture gave it none — its escalation
   * population was only descriptor variants of merchants Tier 1 already knew.
   * That made Tier 2's gate impossible to select against: sweeps reported 100%
   * precision on a population of a handful of answers.
   *
   * Real ledgers churn. People change gyms, move, cancel a subscription, find a
   * new dentist. Modelling that is both more realistic and the only way to
   * generate the traffic the tier is built for.
   */
  readonly activeFrom?: string;
  readonly activeUntil?: string;
}

// Annotated rather than `as const satisfies` on purpose: literal narrowing would
// drop the optional fields from entries that omit them, and the generator reads
// `fixedAmount`/`hard` uniformly across the catalog.
export const MERCHANTS: readonly MerchantDef[] = [
  // ── Recurring bills ───────────────────────────────────────────────────────
  { name: 'Sterling Ridge Apartments', category: 'housing.rent', descriptors: ['STERLING RIDGE APTS RENT', 'ACH DEBIT STERLING RIDGE APT', 'STERLINGRIDGE PROPERTY MGMT'], amount: [1850, 2400], cadence: { kind: 'monthly', dayOfMonth: 1 }, fixedAmount: true },
  { name: 'Pacific Gas & Electric', category: 'housing.utilities', descriptors: ['PG&E ONLINE PMT {ref}', 'ACH DEBIT PACIFIC GAS ELECTRI', 'PGANDE WEB ONLINE PMT'], amount: [62, 210], cadence: { kind: 'monthly', dayOfMonth: 12, jitterDays: 2 } },
  { name: 'City Water Utility', category: 'housing.utilities', descriptors: ['CITY OF {city} WATER DEPT', 'MUNICIPAL WATER SVCS EPAY'], amount: [38, 92], cadence: { kind: 'monthly', dayOfMonth: 18, jitterDays: 3 } },
  { name: 'Comcast Xfinity', category: 'housing.internet_phone', descriptors: ['COMCAST CALIFORNIA', 'XFINITY MOBILE {n4}', 'COMCAST {n5} AUTOPAY'], amount: [79, 129], cadence: { kind: 'monthly', dayOfMonth: 8 }, fixedAmount: true },
  { name: 'Verizon Wireless', category: 'housing.internet_phone', descriptors: ['VERIZON WIRELESS PAYMENTS', 'VZWRLSS*APOCC VISE', 'VERIZON WRLS {n5}-{n3}'], amount: [85, 145], cadence: { kind: 'monthly', dayOfMonth: 22 }, fixedAmount: true },
  { name: 'State Farm Insurance', category: 'transport.insurance', descriptors: ['STATE FARM INSURANCE', 'STATEFARM RO {n2} AUTOPAY', 'SF INS PREM {ref}'], amount: [118, 186], cadence: { kind: 'monthly', dayOfMonth: 5 }, fixedAmount: true },
  { name: 'Lemonade Renters Insurance', category: 'housing.insurance', descriptors: ['LEMONADE INSURANCE', 'LMNDE INS PREMIUM'], amount: [14, 28], cadence: { kind: 'monthly', dayOfMonth: 15 }, fixedAmount: true },

  // ── Subscriptions (small, recurring, easily confused with each other) ──────
  { name: 'Netflix', category: 'entertainment.streaming', descriptors: ['NETFLIX.COM', 'NETFLIX {n4}', 'Netflix 866-579-7172'], amount: [15.49, 22.99], cadence: { kind: 'monthly', dayOfMonth: 14 }, fixedAmount: true },
  { name: 'Spotify', category: 'entertainment.streaming', descriptors: ['SPOTIFY USA', 'PAYPAL *SPOTIFYUSA', 'Spotify P{n7}'], amount: [11.99, 17.99], cadence: { kind: 'monthly', dayOfMonth: 3 }, fixedAmount: true },
  { name: 'HBO Max', category: 'entertainment.streaming', descriptors: ['HBO MAX', 'HBOMAX.COM {n4}', 'WARNERMEDIA DIRECT'], amount: [9.99, 20.99], cadence: { kind: 'monthly', dayOfMonth: 19 }, fixedAmount: true },
  { name: 'The New York Times', category: 'entertainment.streaming', descriptors: ['NYTIMES*NYTIMES', 'NYT DIGITAL SUBSCRIPTION'], amount: [4, 25], cadence: { kind: 'monthly', dayOfMonth: 27 }, fixedAmount: true, hard: true },
  { name: 'iCloud+', category: 'personal.software', descriptors: ['APPLE.COM/BILL', 'APL*ITUNES.COM/BILL', 'APPLE SERVICES {n4}'], amount: [0.99, 9.99], cadence: { kind: 'monthly', dayOfMonth: 11 }, hard: true },
  { name: 'GitHub', category: 'personal.software', descriptors: ['GITHUB.COM {ref}', 'MSFT*GITHUB'], amount: [4, 21], cadence: { kind: 'monthly', dayOfMonth: 6 }, fixedAmount: true },
  { name: 'Anthropic', category: 'personal.software', descriptors: ['ANTHROPIC', 'ANTHROPIC PBC CLAUDE.AI'], amount: [20, 200], cadence: { kind: 'monthly', dayOfMonth: 9 } },
  { name: 'Adobe Creative Cloud', category: 'personal.software', descriptors: ['ADOBE *CREATIVE CLOUD', 'ADOBE INC {n4}'], amount: [22.99, 59.99], cadence: { kind: 'monthly', dayOfMonth: 24 }, fixedAmount: true },
  { name: 'Dropbox', category: 'personal.software', descriptors: ['DROPBOX*{n4}', 'DBX*DROPBOX PLUS'], amount: [11.99, 19.99], cadence: { kind: 'monthly', dayOfMonth: 2 }, fixedAmount: true },
  { name: 'PlayStation Plus', category: 'entertainment.games', descriptors: ['PLAYSTATION NETWORK', 'SONY *PLAYSTATION {n4}'], amount: [9.99, 17.99], cadence: { kind: 'monthly', dayOfMonth: 17 }, fixedAmount: true },
  { name: '24 Hour Fitness', category: 'health.fitness', descriptors: ['24 HOUR FITNESS {n5}', '24HRFIT*{city} {st}'], amount: [44, 69], cadence: { kind: 'monthly', dayOfMonth: 1 }, fixedAmount: true, activeUntil: '2025-12-31' },
  { name: 'ClassPass', category: 'health.fitness', descriptors: ['CLASSPASS INC', 'CLASSPASS*{n5}'], amount: [49, 99], cadence: { kind: 'monthly', dayOfMonth: 20 }, fixedAmount: true, activeUntil: '2025-09-30' },

  // ── Groceries ─────────────────────────────────────────────────────────────
  { name: 'Whole Foods Market', category: 'food.groceries', descriptors: ['WHOLEFDS {city} #{n5}', 'WHOLE FOODS MARKET {n4}', 'AMZN WHOLE FOODS {city}'], amount: [24, 210], cadence: { kind: 'weekly', timesPerWeek: 0.75 } },
  { name: 'Trader Joe\'s', category: 'food.groceries', descriptors: ['TRADER JOE\'S #{n3} QPS', 'TRADER JOES {n3}', 'TJ\'S {city} {st}'], amount: [18, 145], cadence: { kind: 'weekly', timesPerWeek: 0.6 } },
  { name: 'Safeway', category: 'food.groceries', descriptors: ['SAFEWAY #{n4}', 'SAFEWAY STORE {n4} {city}', 'POS DEBIT SAFEWAY {n4}'], amount: [15, 175], cadence: { kind: 'weekly', timesPerWeek: 0.5 } },
  { name: 'Costco Wholesale', category: 'food.groceries', categoryMix: [['food.groceries', 0.62], ['shopping.household', 0.22], ['transport.fuel', 0.10], ['shopping.electronics', 0.06]], descriptors: ['COSTCO WHSE #{n4}', 'COSTCO WHOLESALE {city}'], amount: [85, 420], cadence: { kind: 'sporadic', timesPerYear: 12 }, hard: true },
  { name: 'Instacart', category: 'food.delivery', descriptors: ['INSTACART*{ref}', 'INSTACART SAN FRANCISCO', 'IC* INSTACART'], amount: [45, 190], cadence: { kind: 'sporadic', timesPerYear: 10 }, hard: true },

  // ── Restaurants / coffee / bars ───────────────────────────────────────────
  { name: 'Chipotle', category: 'food.restaurants', descriptors: ['CHIPOTLE {n4}', 'TST* CHIPOTLE - {city}', 'CHIPOTLE ONLINE {n5}'], amount: [11, 34], cadence: { kind: 'weekly', timesPerWeek: 0.8 } },
  { name: 'Blue Bottle Coffee', category: 'food.coffee', descriptors: ['SQ *BLUE BOTTLE COFFEE', 'SQ *BLUE BOTTLE #{n4} {city}', 'BLUE BOTTLE COFFEE {n3}'], amount: [4.5, 19], cadence: { kind: 'weekly', timesPerWeek: 2.0 } },
  { name: 'Starbucks', category: 'food.coffee', descriptors: ['STARBUCKS STORE {n5}', 'SBUX {city} {st}', 'STARBUCKS {n5} {city}'], amount: [3.75, 22], cadence: { kind: 'weekly', timesPerWeek: 1.6 } },
  { name: 'Sightglass Coffee', category: 'food.coffee', descriptors: ['SQ *SIGHTGLASS COFFEE', 'TST* SIGHTGLASS - {city}'], amount: [4, 16], cadence: { kind: 'weekly', timesPerWeek: 0.8 } },
  { name: 'Zuni Cafe', category: 'food.restaurants', descriptors: ['TST* ZUNI CAFE', 'ZUNI CAFE {city} {st}'], amount: [55, 210], cadence: { kind: 'sporadic', timesPerYear: 9 } },
  { name: 'Tartine Bakery', category: 'food.restaurants', descriptors: ['SQ *TARTINE BAKERY', 'TARTINE MANUFACTORY {n3}'], amount: [8, 48], cadence: { kind: 'weekly', timesPerWeek: 0.5 }, hard: true },
  { name: 'Shake Shack', category: 'food.restaurants', descriptors: ['SHAKE SHACK {n4}', 'TST* SHAKE SHACK {city}'], amount: [14, 46], cadence: { kind: 'sporadic', timesPerYear: 20 } },
  { name: 'DoorDash', category: 'food.delivery', descriptors: ['DD *DOORDASH {ref}', 'DOORDASH*{ref}', 'DD DOORDASH THAIFOOD'], amount: [18, 78], cadence: { kind: 'weekly', timesPerWeek: 0.6 } },
  { name: 'Uber Eats', category: 'food.delivery', descriptors: ['UBER *EATS {ref}', 'UBER EATS HELP.UBER.COM'], amount: [16, 72], cadence: { kind: 'weekly', timesPerWeek: 0.5 }, hard: true },
  { name: 'Zeitgeist', category: 'food.bars', descriptors: ['SQ *ZEITGEIST', 'ZEITGEIST BAR {city}'], amount: [22, 95], cadence: { kind: 'sporadic', timesPerYear: 16 } },
  { name: 'Total Wine & More', category: 'food.bars', descriptors: ['TOTAL WINE AND MORE {n4}', 'TOTALWINE #{n4} {city}'], amount: [28, 165], cadence: { kind: 'sporadic', timesPerYear: 14 } },

  // ── Transportation ────────────────────────────────────────────────────────
  { name: 'Uber', category: 'transport.rideshare', descriptors: ['UBER *TRIP {ref}', 'UBER TRIP HELP.UBER.COM', 'UBER *TRIP'], amount: [9, 68], cadence: { kind: 'weekly', timesPerWeek: 0.8 }, hard: true },
  { name: 'Toyota Financial Services', category: 'transport.auto_payment', descriptors: ['TOYOTA FINANCIAL SVC PMT', 'TFS AUTO PAY {n10}', 'ACH DEBIT TOYOTA FIN SVCS'], amount: [389, 462], cadence: { kind: 'monthly', dayOfMonth: 10 }, fixedAmount: true },
  { name: 'Lyft', category: 'transport.rideshare', descriptors: ['LYFT *RIDE {ref}', 'LYFT   *RIDE THU 3PM'], amount: [8, 55], cadence: { kind: 'weekly', timesPerWeek: 0.6 } },
  { name: 'Shell', category: 'transport.fuel', descriptors: ['SHELL OIL {n10}', 'SHELL SERVICE STATION {n4}', 'SHELL {city} {st}'], amount: [32, 88], cadence: { kind: 'weekly', timesPerWeek: 0.4 } },
  { name: 'Chevron', category: 'transport.fuel', descriptors: ['CHEVRON {n5}', 'CHEVRON/CSI {n4}'], amount: [30, 92], cadence: { kind: 'sporadic', timesPerYear: 22 } },
  { name: 'Tesla Supercharger', category: 'transport.fuel', descriptors: ['TESLA SUPERCHARGER US', 'TESLA INC SUPERCHARGE {n4}'], amount: [8, 34], cadence: { kind: 'weekly', timesPerWeek: 0.5 }, hard: true },
  { name: 'BART', category: 'transport.transit', descriptors: ['BART-{city}', 'BAY AREA RAPID TRANSIT', 'CLIPPER CARD RELOAD'], amount: [10, 60], cadence: { kind: 'sporadic', timesPerYear: 30 } },
  { name: 'SP Plus Parking', category: 'transport.parking_tolls', descriptors: ['SP PLUS PARKING {n4}', 'PARKING METER {city} {st}', 'SPPLUS*{n5}'], amount: [4, 45], cadence: { kind: 'sporadic', timesPerYear: 28 } },
  { name: 'FasTrak', category: 'transport.parking_tolls', descriptors: ['BATA FASTRAK CSC', 'FASTRAK REPLENISH {n5}'], amount: [25, 50], cadence: { kind: 'sporadic', timesPerYear: 10 }, fixedAmount: true },
  { name: 'Jiffy Lube', category: 'transport.maintenance', descriptors: ['JIFFY LUBE #{n4}', 'JIFFYLUBE {city} {st}'], amount: [65, 180], cadence: { kind: 'sporadic', timesPerYear: 3 } },
  { name: 'CA DMV', category: 'transport.maintenance', descriptors: ['DMV RENEWAL FEE {st}', 'CA DEPT MOTOR VEHICLES'], amount: [95, 340], cadence: { kind: 'sporadic', timesPerYear: 1 }, hard: true },

  // ── Shopping ──────────────────────────────────────────────────────────────
  { name: 'Amazon', category: 'shopping.general', categoryMix: [['shopping.general', 0.42], ['shopping.household', 0.22], ['shopping.electronics', 0.14], ['shopping.hobbies', 0.10], ['personal.pets', 0.06], ['health.pharmacy', 0.06]], descriptors: ['AMZN Mktp US*{ref}', 'AMAZON.COM*{ref} AMZN.COM/BILL', 'Amazon.com*{ref}', 'AMZN Mktp US {ref}'], amount: [8, 240], cadence: { kind: 'weekly', timesPerWeek: 1.0 }, hard: true },
  { name: 'Grove Collaborative', category: 'shopping.household', descriptors: ['GROVE COLLABORATIVE', 'GROVE.CO {n5}'], amount: [28, 95], cadence: { kind: 'monthly', dayOfMonth: 13, jitterDays: 5 } },
  { name: 'Target', category: 'shopping.general', categoryMix: [['shopping.general', 0.40], ['shopping.household', 0.30], ['food.groceries', 0.20], ['shopping.clothing', 0.10]], descriptors: ['TARGET {n8}', 'TARGET.COM * {ref}', 'TARGET T-{n4} {city}'], amount: [18, 195], cadence: { kind: 'sporadic', timesPerYear: 16 }, hard: true },
  { name: 'Uniqlo', category: 'shopping.clothing', descriptors: ['UNIQLO USA {n4}', 'UNIQLO {city} {st}'], amount: [35, 220], cadence: { kind: 'sporadic', timesPerYear: 7 } },
  { name: 'Nordstrom', category: 'shopping.clothing', descriptors: ['NORDSTROM #{n4}', 'NORDSTROM.COM {ref}'], amount: [60, 480], cadence: { kind: 'sporadic', timesPerYear: 5 } },
  { name: 'Best Buy', category: 'shopping.electronics', descriptors: ['BEST BUY {n5}', 'BESTBUYCOM{n12}'], amount: [40, 1200], cadence: { kind: 'sporadic', timesPerYear: 2 } },
  { name: 'Apple Store', category: 'shopping.electronics', categoryMix: [['shopping.electronics', 0.75], ['personal.software', 0.25]], descriptors: ['APPLE STORE #R{n3}', 'APPLE.COM/US {n4}'], amount: [99, 2400], cadence: { kind: 'sporadic', timesPerYear: 1 }, hard: true },
  { name: 'IKEA', category: 'housing.furnishing', descriptors: ['IKEA {city}', 'IKEA.COM {n5}'], amount: [45, 890], cadence: { kind: 'sporadic', timesPerYear: 2 } },
  { name: 'Home Depot', category: 'housing.maintenance', categoryMix: [['housing.maintenance', 0.68], ['shopping.hobbies', 0.18], ['housing.furnishing', 0.14]], descriptors: ['THE HOME DEPOT {n4}', 'HOMEDEPOT.COM {ref}'], amount: [22, 380], cadence: { kind: 'sporadic', timesPerYear: 11 }, hard: true },
  { name: 'REI', category: 'shopping.hobbies', descriptors: ['REI #{n3} {city}', 'REI.COM {ref}'], amount: [45, 620], cadence: { kind: 'sporadic', timesPerYear: 5 } },
  { name: 'Etsy', category: 'shopping.gifts', descriptors: ['ETSY.COM - {ref}', 'ETSY INC {n5}'], amount: [18, 145], cadence: { kind: 'sporadic', timesPerYear: 9 }, hard: true },

  // ── Health ────────────────────────────────────────────────────────────────
  { name: 'One Medical', category: 'health.medical', descriptors: ['ONE MEDICAL GROUP', '1LIFE HEALTHCARE {n4}'], amount: [35, 320], cadence: { kind: 'sporadic', timesPerYear: 5 } },
  { name: 'CVS Pharmacy', category: 'health.pharmacy', categoryMix: [['health.pharmacy', 0.55], ['shopping.household', 0.28], ['personal.care', 0.17]], descriptors: ['CVS/PHARMACY #{n5}', 'CVS {n5} {city} {st}'], amount: [9, 145], cadence: { kind: 'sporadic', timesPerYear: 20 }, hard: true },
  { name: 'Walgreens', category: 'health.pharmacy', categoryMix: [['health.pharmacy', 0.58], ['shopping.household', 0.26], ['personal.care', 0.16]], descriptors: ['WALGREENS #{n5}', 'WALGREENS STORE {n5}'], amount: [7, 98], cadence: { kind: 'sporadic', timesPerYear: 12 }, hard: true },
  { name: 'Bright Smile Dental', category: 'health.dental_vision', descriptors: ['BRIGHT SMILE DENTAL {city}', 'SQ *BRIGHT SMILE DENTAL'], amount: [85, 1400], cadence: { kind: 'sporadic', timesPerYear: 2 } },
  { name: 'Warby Parker', category: 'health.dental_vision', descriptors: ['WARBY PARKER {n4}', 'WARBYPARKER.COM'], amount: [95, 395], cadence: { kind: 'sporadic', timesPerYear: 1 }, hard: true },
  { name: 'Blue Shield of California', category: 'health.insurance', descriptors: ['BLUE SHIELD CA PREMIUM', 'BSCA HEALTH PLAN {n8}', 'ACH DEBIT BLUE SHIELD OF CA'], amount: [242, 318], cadence: { kind: 'monthly', dayOfMonth: 4 }, fixedAmount: true },

  // ── Entertainment / travel ────────────────────────────────────────────────
  { name: 'AMC Theatres', category: 'entertainment.events', descriptors: ['AMC {city} {n2}', 'AMC ONLINE {n7}'], amount: [16, 78], cadence: { kind: 'sporadic', timesPerYear: 8 } },
  { name: 'Ticketmaster', category: 'entertainment.events', descriptors: ['TICKETMASTER {n10}', 'TICKETMASTER.COM {ref}'], amount: [55, 640], cadence: { kind: 'sporadic', timesPerYear: 4 } },
  { name: 'Steam', category: 'entertainment.games', descriptors: ['STEAMGAMES.COM {n10}', 'VALVE *STEAM PURCHASE'], amount: [5, 90], cadence: { kind: 'sporadic', timesPerYear: 11 } },
  { name: 'United Airlines', category: 'travel.flights', descriptors: ['UNITED {n13}', 'UNITED AIRLINES {city}'], amount: [180, 1250], cadence: { kind: 'sporadic', timesPerYear: 3 } },
  { name: 'Airbnb', category: 'travel.lodging', descriptors: ['AIRBNB * HM{ref}', 'AIRBNB HM{ref}'], amount: [180, 1600], cadence: { kind: 'sporadic', timesPerYear: 2 } },
  { name: 'Marriott', category: 'travel.lodging', descriptors: ['MARRIOTT {city} {n5}', 'COURTYARD BY MARRIOTT {n4}'], amount: [165, 980], cadence: { kind: 'sporadic', timesPerYear: 2 } },
  { name: 'Hertz', category: 'travel.rental_car', descriptors: ['HERTZ RENT-A-CAR {n5}', 'HERTZ {city} APO'], amount: [95, 540], cadence: { kind: 'sporadic', timesPerYear: 4 } },
  { name: 'Allianz Travel Insurance', category: 'travel.other', descriptors: ['ALLIANZ TRAVEL INS {n6}', 'AGA SERVICE COMPANY', 'GLOBAL ENTRY TTP FEE'], amount: [28, 120], cadence: { kind: 'sporadic', timesPerYear: 3 }, hard: true },

  // ── Personal ──────────────────────────────────────────────────────────────
  { name: 'Barber & Blade', category: 'personal.care', descriptors: ['SQ *BARBER AND BLADE', 'BARBER & BLADE {city}'], amount: [35, 85], cadence: { kind: 'sporadic', timesPerYear: 10 } },
  { name: 'Coursera', category: 'personal.education', descriptors: ['COURSERA.ORG {ref}', 'COURSERA INC {n5}'], amount: [39, 399], cadence: { kind: 'sporadic', timesPerYear: 3 } },
  { name: 'Bayside Veterinary', category: 'personal.pets', descriptors: ['BAYSIDE VETERINARY CLINIC', 'SQ *BAYSIDE VET {city}'], amount: [75, 890], cadence: { kind: 'sporadic', timesPerYear: 3 } },
  { name: 'Chewy', category: 'personal.pets', descriptors: ['CHEWY.COM {n6}', 'CHEWY INC {ref}'], amount: [38, 165], cadence: { kind: 'monthly', dayOfMonth: 21, jitterDays: 4 } },
  { name: 'Care.com Sitter', category: 'personal.childcare', descriptors: ['CARE.COM *SITTER {n6}', 'SQ *SITTER SERVICES', 'CARE COM INC {n5}'], amount: [60, 240], cadence: { kind: 'sporadic', timesPerYear: 8 }, hard: true },

  // ── Financial ─────────────────────────────────────────────────────────────
  { name: 'Bank Monthly Fee', category: 'financial.fees', descriptors: ['MONTHLY MAINTENANCE FEE', 'MONTHLY SERVICE CHARGE'], amount: [12, 15], cadence: { kind: 'sporadic', timesPerYear: 4 }, fixedAmount: true },
  { name: 'ATM Withdrawal Fee', category: 'financial.fees', descriptors: ['NON-CHASE ATM FEE', 'ATM SURCHARGE FEE {city}'], amount: [3, 5.5], cadence: { kind: 'sporadic', timesPerYear: 6 } },
  { name: 'Credit Card Interest', category: 'financial.interest', descriptors: ['INTEREST CHARGED ON PURCHASES', 'PURCHASE INTEREST CHARGE'], amount: [8, 140], cadence: { kind: 'sporadic', timesPerYear: 5 } },
  { name: 'Nelnet Student Loan', category: 'financial.loan_payment', descriptors: ['NELNET STUDENT LOAN PMT', 'NELNET {n10} WEB PMT'], amount: [285, 420], cadence: { kind: 'monthly', dayOfMonth: 16 }, fixedAmount: true },
  { name: 'Vanguard', category: 'financial.investment', descriptors: ['VANGUARD BUY INVESTMENT', 'VANGUARD GROUP {n5} ACH'], amount: [400, 700], cadence: { kind: 'monthly', dayOfMonth: 2 }, fixedAmount: true, hard: true },
  { name: 'TurboTax', category: 'financial.taxes', descriptors: ['INTUIT *TURBOTAX', 'TURBOTAX {n8}'], amount: [59, 189], cadence: { kind: 'sporadic', timesPerYear: 1 } },
  { name: 'IRS Estimated Tax', category: 'financial.taxes', descriptors: ['IRS USATAXPYMT {n10}', 'US TREASURY TAX PYMT', 'FRANCHISE TAX BD CASTTAXRFD'], amount: [420, 1800], cadence: { kind: 'sporadic', timesPerYear: 4 }, hard: true },

  // ── Income ────────────────────────────────────────────────────────────────
  { name: 'Employer Payroll', category: 'income.salary', descriptors: ['NORTHSTAR SYSTEMS DIRECT DEP', 'NORTHSTAR SYS PAYROLL {ref}', 'DIRECT DEPOSIT NORTHSTAR'], amount: [3100, 3600], cadence: { kind: 'biweekly', anchorDay: 5 } },
  { name: 'Freelance Client', category: 'income.freelance', descriptors: ['STRIPE TRANSFER {ref}', 'DEPOSIT MERIDIAN LABS LLC', 'ACH CREDIT MERIDIAN LABS'], amount: [400, 3200], cadence: { kind: 'sporadic', timesPerYear: 9 }, hard: true },
  { name: 'Savings Interest', category: 'income.investment', descriptors: ['INTEREST PAYMENT', 'INTEREST EARNED APY'], amount: [2, 48], cadence: { kind: 'monthly', dayOfMonth: 28 } },
  { name: 'Amazon Refund', category: 'income.refund', descriptors: ['AMZN Mktp US*{ref} REFUND', 'AMAZON.COM REFUND {ref}'], amount: [12, 180], cadence: { kind: 'sporadic', timesPerYear: 7 }, hard: true },
  { name: 'Card Rewards Redemption', category: 'income.other', descriptors: ['CASH BACK REWARD REDEMPTION', 'STATEMENT CREDIT REWARDS', 'REWARDS REDEMPTION {n6}'], amount: [15, 250], cadence: { kind: 'sporadic', timesPerYear: 5 }, hard: true },

  // ── Transfers (the highest-cost error class) ──────────────────────────────
  { name: 'Credit Card Payment', category: 'transfer.credit_payment', descriptors: ['CHASE CREDIT CRD AUTOPAY {ref}', 'PAYMENT THANK YOU - WEB', 'AMEX EPAYMENT ACH PMT'], amount: [400, 2800], cadence: { kind: 'monthly', dayOfMonth: 25 }, hard: true },
  { name: 'Transfer to Savings', category: 'transfer.internal', descriptors: ['ONLINE TRANSFER TO SAV {n4}', 'TRANSFER TO SAVINGS {n4}'], amount: [200, 1500], cadence: { kind: 'monthly', dayOfMonth: 6 }, hard: true },
  { name: 'Venmo', category: 'transfer.person', descriptors: ['VENMO PAYMENT {n10}', 'VENMO CASHOUT {n10}', 'VENMO *{ref}'], amount: [10, 320], cadence: { kind: 'weekly', timesPerWeek: 0.7 }, hard: true },
  { name: 'Zelle', category: 'transfer.person', descriptors: ['ZELLE PAYMENT TO {ref}', 'ZELLE FROM {ref} {n8}'], amount: [25, 900], cadence: { kind: 'sporadic', timesPerYear: 14 }, hard: true },
  // ── Merchant churn and near-miss families ─────────────────────────────────
  //
  // Two jobs. The `activeFrom` merchants are genuinely new to the user partway
  // through, which is the only traffic Tier 2 is built for and which the first
  // corpus contained none of. And they come in *families* that share a leading
  // token while spanning different categories — PRESIDIO DENTAL against PRESIDIO
  // VETERINARY — so a lexical neighbourhood is contested rather than trivially
  // unanimous. A near-miss neighbourhood is the case a nearest-neighbour vote is
  // supposed to be careful about, and a fixture with none of them cannot show
  // whether it is.
  //
  // Regional-name families are not a contrivance: real high streets are full of
  // them, and they are exactly where a descriptor-only classifier should hesitate.
  { name: 'Presidio Dental Group', category: 'health.dental_vision', descriptors: ['PRESIDIO DENTAL GRP', 'PRESIDIO DENTAL {n4}'], amount: [95, 420], cadence: { kind: 'sporadic', timesPerYear: 2 }, activeFrom: '2025-09-01' },
  { name: 'Presidio Veterinary', category: 'personal.pets', descriptors: ['PRESIDIO VETERINARY', 'PRESIDIO VET CLINIC {n3}'], amount: [65, 260], cadence: { kind: 'sporadic', timesPerYear: 3 }, activeFrom: '2025-09-01' },
  { name: 'Presidio Cleaners', category: 'housing.maintenance', descriptors: ['PRESIDIO CLEANERS', 'PRESIDIO DRY CLEAN {n3}'], amount: [18, 74], cadence: { kind: 'sporadic', timesPerYear: 8 }, activeFrom: '2025-10-01' },

  { name: 'Marina Market', category: 'food.groceries', descriptors: ['MARINA MARKET {n3}', 'MARINA MKT {city}'], amount: [12, 60], cadence: { kind: 'weekly', timesPerWeek: 0.3 }, activeFrom: '2025-08-15' },
  { name: 'Marina Deli & Grill', category: 'food.restaurants', descriptors: ['MARINA DELI GRILL', 'TST* MARINA DELI'], amount: [14, 52], cadence: { kind: 'weekly', timesPerWeek: 0.3 }, activeFrom: '2025-08-15' },
  { name: 'Marina Pet Supply', category: 'personal.pets', descriptors: ['MARINA PET SUPPLY', 'MARINA PET {n4}'], amount: [22, 90], cadence: { kind: 'sporadic', timesPerYear: 6 }, activeFrom: '2025-11-01' },

  { name: 'Bayside Pharmacy', category: 'health.pharmacy', descriptors: ['BAYSIDE PHARMACY {n4}', 'BAYSIDE RX {city}'], amount: [8, 165], cadence: { kind: 'sporadic', timesPerYear: 11 }, activeFrom: '2025-08-01' },
  { name: 'Bayside Wine & Spirits', category: 'food.bars', descriptors: ['BAYSIDE WINE SPIRITS', 'BAYSIDE WINE {n3}'], amount: [19, 72], cadence: { kind: 'sporadic', timesPerYear: 9 }, activeFrom: '2025-08-01' },
  { name: 'Bayside Hardware', category: 'housing.maintenance', descriptors: ['BAYSIDE HARDWARE {n3}', 'BAYSIDE HDW {city} {st}'], amount: [11, 120], cadence: { kind: 'sporadic', timesPerYear: 6 }, activeFrom: '2025-12-01' },

  { name: 'Golden Gate Auto Care', category: 'transport.maintenance', descriptors: ['GOLDEN GATE AUTO CARE', 'GG AUTO CARE {n4}'], amount: [58, 420], cadence: { kind: 'sporadic', timesPerYear: 2 }, activeFrom: '2026-01-01' },
  { name: 'Golden Gate Optometry', category: 'health.dental_vision', descriptors: ['GOLDEN GATE OPTOMETRY', 'GG OPTOMETRY {n3}'], amount: [85, 445], cadence: { kind: 'sporadic', timesPerYear: 2 }, activeFrom: '2026-01-01' },
  { name: 'Golden Gate Athletic Club', category: 'health.fitness', descriptors: ['GOLDEN GATE ATHLETIC', 'GG ATHLETIC CLUB {n4}'], amount: [79, 129], cadence: { kind: 'monthly', dayOfMonth: 4 }, fixedAmount: true, activeFrom: '2026-01-01' },

  // Churn, modelled as genuine switches rather than additive spend: ClassPass
  // ends as Ridgeline begins, and 24 Hour Fitness ends as Golden Gate Athletic
  // begins. The old keys go quiet and the new ones must be learned from scratch
  // mid-ledger, which is the realistic shape and keeps the persona solvent —
  // a fixture whose spend quietly climbs past its income stops being a
  // believable demo, which is how the amount distributions got fixed the first
  // time round.
  { name: 'Ridgeline Climbing Gym', category: 'health.fitness', descriptors: ['RIDGELINE CLIMBING', 'RIDGELINE CLMB {n4}'], amount: [95, 155], cadence: { kind: 'monthly', dayOfMonth: 7 }, fixedAmount: true, activeFrom: '2025-10-01' },
  { name: 'Ridgeline Pro Shop', category: 'shopping.hobbies', descriptors: ['RIDGELINE PRO SHOP', 'RIDGELINE SHOP {n3}'], amount: [24, 260], cadence: { kind: 'sporadic', timesPerYear: 6 }, activeFrom: '2025-10-01' },

];

export const HARD_MERCHANTS: readonly string[] = MERCHANTS.filter((m) => m.hard).map((m) => m.name);
