/**
 * Tier 0 — descriptor normalization.
 *
 * Turns a raw bank descriptor into a stable key that identifies the *merchant* and
 * nothing else. This is the cheapest and highest-leverage stage in the pipeline:
 * every store number, reference blob, and city suffix left in the string is a cache
 * miss downstream, and a cache miss is either an embedding call or an LLM call.
 *
 *   "SQ *BLUE BOTTLE #4432 SANFRAN CA"  →  "BLUE BOTTLE"
 *   "AMZN Mktp US*2K4LM9XY3"            →  "AMZN MKTP US"
 *   "POS DEBIT SAFEWAY 0412"            →  "SAFEWAY"
 *
 * Deliberately conservative. Over-stripping collapses distinct merchants into one
 * key and produces confident wrong answers, which are far more expensive than the
 * cache misses caused by under-stripping.
 */

/**
 * Payment processors and network markers that carry no merchant identity.
 *
 * Only genuinely merchant-agnostic prefixes belong here. `UBER *` and `DD *` are
 * deliberately absent: Uber and DoorDash are the merchant, so stripping them would
 * reduce "UBER *EATS" to "EATS" and destroy the only signal in the string.
 */
const PROCESSOR_PREFIXES = [
  'POS DEBIT',
  'POS PURCHASE',
  'DEBIT CARD PURCHASE',
  'DEBIT PURCHASE',
  'CHECKCARD',
  'PURCHASE AUTHORIZED ON',
  'RECURRING PAYMENT',
  'ACH DEBIT',
  'ACH CREDIT',
  'ACH PMT',
  'VISA DDA PUR',
  'ELECTRONIC PAYMENT',
  'PREAUTHORIZED DEBIT',
  'SQ *',
  'SQC*',
  'TST*',
  'TST *',
  'PAYPAL *',
  'PAYPAL*',
  'PP*',
  'PY *',
] as const;

/** Trailing location blobs. Matched only at the end of the string. */
const CITY_FRAGMENTS = [
  'SANFRAN', 'SAN FRANCISCO', 'SF', 'OAKLAND', 'BERKELEY', 'DALY CITY',
  'SAN MATEO', 'EMERYVLE', 'EMERYVILLE', 'ALAMEDA',
] as const;

const STATE_CODES = new Set([
  'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS',
  'KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY',
  'NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV',
  'WI','WY','DC',
]);

/**
 * Trailing identifier digits fused to a brand token: `BESTBUYCOM123456789012`.
 *
 * Stripping the digits rather than dropping the whole token matters — dropping it
 * would erase `BESTBUYCOM` along with the id and leave nothing to key on.
 */
function stripFusedDigits(token: string): string {
  if (!/[A-Z]/.test(token)) return token;
  return token.replace(/\d{3,}$/, '');
}

const VOWELS = /[AEIOU]/;

/**
 * Tokens that are references, not names.
 *
 * The discriminator is vowels, not digit density. Randomly generated ids are drawn
 * from an alphabet with no vowel bias and reliably come out unpronounceable —
 * `2K4LM9XY3`, `HM8FQ2XZW`, `KX82LM`. Brand tokens keep their vowels even when they
 * carry digits: `24HRFIT`, `1LIFE`, `7-ELEVEN`.
 *
 * An earlier digit-ratio rule missed `HM8FQ2XZW` (only 22% digits) and would have
 * leaked a unique reference into every Airbnb key.
 */
function isReferenceToken(token: string): boolean {
  if (token.length < 5) return false;
  if (!/[A-Z]/.test(token) || !/\d/.test(token)) return false;
  return !VOWELS.test(token);
}

export interface NormalizedDescriptor {
  /** Stable merchant key. Uppercase, punctuation-light, no identifiers. */
  readonly key: string;
  /** The input, unchanged. Kept so the LLM tier can see what we threw away. */
  readonly raw: string;
  /** Set when normalization stripped so much that the key is unusable. */
  readonly degenerate: boolean;
}

export function normalizeDescriptor(raw: string): NormalizedDescriptor {
  let s = raw.toUpperCase().trim();

  // 1. Peel processor prefixes, repeatedly — feeds stack them ("POS DEBIT SQ *X").
  for (let changed = true; changed; ) {
    changed = false;
    for (const prefix of PROCESSOR_PREFIXES) {
      if (s.startsWith(prefix)) {
        s = s.slice(prefix.length).trim();
        changed = true;
      }
    }
  }

  // 2. `*` is a field separator in most feeds, never part of a name.
  s = s.replace(/\*/g, ' ');

  // 3. Store numbers: "#4432", "#R203", "STORE 0412", "T-1234".
  s = s.replace(/#\s*[A-Z]?\d+/g, ' ');
  s = s.replace(/\b(?:STORE|STR|LOC|UNIT)\s+\d+\b/g, ' ');
  s = s.replace(/\b[A-Z]-\d{3,}\b/g, ' ');

  // 4. Fused digit suffixes, reference blobs, and bare identifier runs.
  s = s
    .split(/\s+/)
    .map(stripFusedDigits)
    .filter((tok) => tok.length > 1 && !isReferenceToken(tok) && !/^\d{2,}$/.test(tok))
    .join(' ');

  // 5. Trailing state code, then trailing city fragment. Order matters: the state
  //    sits outside the city ("... SANFRAN CA"), so it has to come off first.
  const tokens = s.split(/\s+/).filter(Boolean);
  if (tokens.length > 1 && STATE_CODES.has(tokens[tokens.length - 1]!)) tokens.pop();
  s = tokens.join(' ');
  for (const city of [...CITY_FRAGMENTS].sort((a, b) => b.length - a.length)) {
    if (s.length > city.length + 1 && s.endsWith(` ${city}`)) {
      s = s.slice(0, -(city.length + 1));
      break;
    }
  }

  // 6. Trailing noise punctuation and collapsed whitespace.
  s = s.replace(/[\s\-_.,;:/\\]+$/g, '').replace(/\s+/g, ' ').trim();

  // A key of one or two characters carries no merchant signal — better to admit
  // that and let a later tier see the raw string than to cache on garbage.
  const degenerate = s.length < 3;

  return { key: degenerate ? raw.toUpperCase().trim() : s, raw, degenerate };
}
