/**
 * The category taxonomy.
 *
 * Two levels: `group` (what a human thinks in) and `category` (what we budget and
 * report on). Leaf ids are stable, machine-readable slugs — they are the label space
 * for the categorization model, so renaming one is a breaking change that invalidates
 * the golden dataset. Add freely; rename never.
 */

export const CATEGORY_GROUPS = [
  'income',
  'housing',
  'transportation',
  'food',
  'shopping',
  'health',
  'entertainment',
  'travel',
  'personal',
  'financial',
  'transfer',
] as const;

export type CategoryGroup = (typeof CATEGORY_GROUPS)[number];

export interface CategoryDef {
  readonly id: string;
  readonly group: CategoryGroup;
  readonly label: string;
  /** Guidance shown to the model and to the user. Doubles as prompt context. */
  readonly hint: string;
  /** Money normally flows this way. Used as a cheap sanity check on model output. */
  readonly direction: 'inflow' | 'outflow' | 'either';
}

export const CATEGORIES = [
  // ── Income ────────────────────────────────────────────────────────────────
  { id: 'income.salary', group: 'income', label: 'Salary', hint: 'Regular paycheck from an employer, including direct deposit payroll.', direction: 'inflow' },
  { id: 'income.freelance', group: 'income', label: 'Freelance & Contract', hint: 'Self-employment, 1099, consulting, or gig platform payouts.', direction: 'inflow' },
  { id: 'income.investment', group: 'income', label: 'Investment Income', hint: 'Dividends, interest earned, capital gains distributions.', direction: 'inflow' },
  { id: 'income.refund', group: 'income', label: 'Refunds & Reimbursements', hint: 'Returned purchases, expense reimbursements, tax refunds.', direction: 'inflow' },
  { id: 'income.other', group: 'income', label: 'Other Income', hint: 'Gifts received, rebates, cash back rewards, anything inbound that fits nowhere else.', direction: 'inflow' },

  // ── Housing ───────────────────────────────────────────────────────────────
  { id: 'housing.rent', group: 'housing', label: 'Rent & Mortgage', hint: 'Monthly rent payments or mortgage principal and interest.', direction: 'outflow' },
  { id: 'housing.utilities', group: 'housing', label: 'Utilities', hint: 'Electric, gas, water, sewer, trash. Not internet or phone.', direction: 'outflow' },
  { id: 'housing.internet_phone', group: 'housing', label: 'Internet & Phone', hint: 'Home internet, mobile carrier bills, landline.', direction: 'outflow' },
  { id: 'housing.maintenance', group: 'housing', label: 'Home Maintenance', hint: 'Repairs, cleaning services, lawn care, hardware store runs, HOA dues.', direction: 'outflow' },
  { id: 'housing.insurance', group: 'housing', label: 'Home & Renters Insurance', hint: 'Homeowners, renters, or umbrella property insurance premiums.', direction: 'outflow' },
  { id: 'housing.furnishing', group: 'housing', label: 'Furniture & Appliances', hint: 'Durable goods for the home: furniture, mattresses, major appliances.', direction: 'outflow' },

  // ── Transportation ────────────────────────────────────────────────────────
  { id: 'transport.auto_payment', group: 'transportation', label: 'Auto Loan & Lease', hint: 'Car loan or lease payments.', direction: 'outflow' },
  { id: 'transport.fuel', group: 'transportation', label: 'Gas & Fuel', hint: 'Gas stations and EV charging networks.', direction: 'outflow' },
  { id: 'transport.maintenance', group: 'transportation', label: 'Auto Maintenance', hint: 'Oil changes, repairs, tires, car washes, registration and DMV fees.', direction: 'outflow' },
  { id: 'transport.insurance', group: 'transportation', label: 'Auto Insurance', hint: 'Vehicle insurance premiums.', direction: 'outflow' },
  { id: 'transport.rideshare', group: 'transportation', label: 'Rideshare & Taxi', hint: 'Uber, Lyft, taxis. Rideshare food delivery belongs in Food Delivery.', direction: 'outflow' },
  { id: 'transport.transit', group: 'transportation', label: 'Public Transit', hint: 'Subway, bus, commuter rail, transit cards.', direction: 'outflow' },
  { id: 'transport.parking_tolls', group: 'transportation', label: 'Parking & Tolls', hint: 'Parking garages, meters, bridge and highway tolls.', direction: 'outflow' },

  // ── Food & Drink ──────────────────────────────────────────────────────────
  { id: 'food.groceries', group: 'food', label: 'Groceries', hint: 'Supermarkets and grocery delivery. Warehouse clubs default here.', direction: 'outflow' },
  { id: 'food.restaurants', group: 'food', label: 'Restaurants', hint: 'Sit-down and fast-casual dining, takeout ordered directly from the restaurant.', direction: 'outflow' },
  { id: 'food.coffee', group: 'food', label: 'Coffee Shops', hint: 'Cafes and coffee chains.', direction: 'outflow' },
  { id: 'food.bars', group: 'food', label: 'Bars & Alcohol', hint: 'Bars, breweries, liquor stores, wine shops.', direction: 'outflow' },
  { id: 'food.delivery', group: 'food', label: 'Food Delivery', hint: 'Third-party delivery platforms: DoorDash, Uber Eats, Grubhub, Instacart fees.', direction: 'outflow' },

  // ── Shopping ──────────────────────────────────────────────────────────────
  { id: 'shopping.general', group: 'shopping', label: 'General Merchandise', hint: 'Marketplaces and big-box retail where the basket is mixed or unknown.', direction: 'outflow' },
  { id: 'shopping.clothing', group: 'shopping', label: 'Clothing & Accessories', hint: 'Apparel, shoes, jewelry, bags.', direction: 'outflow' },
  { id: 'shopping.electronics', group: 'shopping', label: 'Electronics', hint: 'Consumer electronics, computers, phones, components.', direction: 'outflow' },
  { id: 'shopping.household', group: 'shopping', label: 'Household Supplies', hint: 'Cleaning products, paper goods, toiletries bought as supplies.', direction: 'outflow' },
  { id: 'shopping.hobbies', group: 'shopping', label: 'Hobbies & Sporting Goods', hint: 'Craft supplies, musical instruments, outdoor and sports gear.', direction: 'outflow' },
  { id: 'shopping.gifts', group: 'shopping', label: 'Gifts & Donations', hint: 'Presents for others, charitable giving, fundraisers.', direction: 'outflow' },

  // ── Health ────────────────────────────────────────────────────────────────
  { id: 'health.medical', group: 'health', label: 'Doctor & Medical', hint: 'Office visits, labs, hospitals, specialists, copays.', direction: 'outflow' },
  { id: 'health.pharmacy', group: 'health', label: 'Pharmacy', hint: 'Prescriptions and drugstore purchases.', direction: 'outflow' },
  { id: 'health.dental_vision', group: 'health', label: 'Dental & Vision', hint: 'Dentists, orthodontists, optometrists, eyewear.', direction: 'outflow' },
  { id: 'health.fitness', group: 'health', label: 'Fitness', hint: 'Gym memberships, studio classes, fitness apps, personal training.', direction: 'outflow' },
  { id: 'health.insurance', group: 'health', label: 'Health Insurance', hint: 'Medical, dental, or vision insurance premiums paid directly.', direction: 'outflow' },

  // ── Entertainment ─────────────────────────────────────────────────────────
  { id: 'entertainment.streaming', group: 'entertainment', label: 'Streaming & Media', hint: 'Video, music, and news subscriptions.', direction: 'outflow' },
  { id: 'entertainment.events', group: 'entertainment', label: 'Events & Attractions', hint: 'Concerts, movies, sports, museums, theme parks.', direction: 'outflow' },
  { id: 'entertainment.games', group: 'entertainment', label: 'Games', hint: 'Video games, in-game purchases, gaming subscriptions, board games.', direction: 'outflow' },

  // ── Travel ────────────────────────────────────────────────────────────────
  { id: 'travel.flights', group: 'travel', label: 'Flights', hint: 'Airfare, seat upgrades, baggage fees.', direction: 'outflow' },
  { id: 'travel.lodging', group: 'travel', label: 'Lodging', hint: 'Hotels, vacation rentals, hostels.', direction: 'outflow' },
  { id: 'travel.rental_car', group: 'travel', label: 'Rental Car', hint: 'Car rental agencies and associated fees.', direction: 'outflow' },
  { id: 'travel.other', group: 'travel', label: 'Other Travel', hint: 'Travel insurance, luggage fees, tours, currency exchange.', direction: 'outflow' },

  // ── Personal ──────────────────────────────────────────────────────────────
  { id: 'personal.care', group: 'personal', label: 'Personal Care', hint: 'Salons, barbers, spas, cosmetics services.', direction: 'outflow' },
  { id: 'personal.education', group: 'personal', label: 'Education', hint: 'Tuition, courses, textbooks, professional certification.', direction: 'outflow' },
  { id: 'personal.childcare', group: 'personal', label: 'Childcare & Kids', hint: 'Daycare, babysitting, school fees, kids activities.', direction: 'outflow' },
  { id: 'personal.pets', group: 'personal', label: 'Pets', hint: 'Vet, pet food, grooming, boarding.', direction: 'outflow' },
  { id: 'personal.software', group: 'personal', label: 'Software & Cloud', hint: 'SaaS tools, app subscriptions, cloud storage, developer services.', direction: 'outflow' },

  // ── Financial ─────────────────────────────────────────────────────────────
  { id: 'financial.fees', group: 'financial', label: 'Bank Fees', hint: 'Maintenance fees, overdraft, ATM, wire, and foreign transaction fees.', direction: 'outflow' },
  { id: 'financial.interest', group: 'financial', label: 'Interest Charges', hint: 'Credit card and loan interest assessed.', direction: 'outflow' },
  { id: 'financial.taxes', group: 'financial', label: 'Taxes', hint: 'Estimated tax payments, tax prep, property tax paid directly.', direction: 'outflow' },
  { id: 'financial.investment', group: 'financial', label: 'Investment Contribution', hint: 'Money moved into brokerage, IRA, or 401k outside of payroll.', direction: 'outflow' },
  { id: 'financial.loan_payment', group: 'financial', label: 'Loan Payment', hint: 'Student loan, personal loan, or other non-auto debt payments.', direction: 'outflow' },

  // ── Transfers ─────────────────────────────────────────────────────────────
  // Excluded from spend reporting. Misclassifying a transfer as spend is the single
  // most damaging error the model can make, so these get their own eval slice.
  { id: 'transfer.internal', group: 'transfer', label: 'Internal Transfer', hint: 'Movement between the user’s own accounts. Not income and not spending.', direction: 'either' },
  { id: 'transfer.credit_payment', group: 'transfer', label: 'Credit Card Payment', hint: 'Paying a credit card balance from a bank account.', direction: 'either' },
  { id: 'transfer.person', group: 'transfer', label: 'Person to Person', hint: 'Venmo, Zelle, Cash App transfers between people.', direction: 'either' },
] as const satisfies readonly CategoryDef[];

export type CategoryId = (typeof CATEGORIES)[number]['id'];

/** Sentinel for "the pipeline declined to guess". Never a valid model output. */
export const UNCATEGORIZED = 'uncategorized' as const;

const BY_ID = new Map<string, CategoryDef>(CATEGORIES.map((c) => [c.id, c]));

export function getCategory(id: string): CategoryDef | undefined {
  return BY_ID.get(id);
}

export function isCategoryId(id: string): id is CategoryId {
  return BY_ID.has(id);
}

export const CATEGORY_IDS: readonly CategoryId[] = CATEGORIES.map((c) => c.id);

/**
 * Compact taxonomy rendering for prompts. Kept here rather than in the prompt file
 * so the label space the model sees can never drift from the label space we score.
 */
export function taxonomyForPrompt(): string {
  const lines: string[] = [];
  for (const group of CATEGORY_GROUPS) {
    lines.push(`\n[${group.toUpperCase()}]`);
    for (const c of CATEGORIES.filter((x) => x.group === group)) {
      lines.push(`  ${c.id} — ${c.label}: ${c.hint}`);
    }
  }
  return lines.join('\n').trim();
}
