import { test } from 'node:test';
import assert from 'node:assert/strict';

import { normalizeDescriptor } from './normalize.js';

const key = (s: string) => normalizeDescriptor(s).key;

test('strips square and toast processor prefixes', () => {
  assert.equal(key('SQ *BLUE BOTTLE COFFEE'), 'BLUE BOTTLE COFFEE');
  assert.equal(key('TST* CHIPOTLE - SANFRAN'), 'CHIPOTLE');
  assert.equal(key('PAYPAL *SPOTIFYUSA'), 'SPOTIFYUSA');
});

test('peels stacked prefixes', () => {
  assert.equal(key('POS DEBIT SQ *SIGHTGLASS COFFEE'), 'SIGHTGLASS COFFEE');
});

test('removes store numbers in every common form', () => {
  assert.equal(key('WHOLEFDS SANFRAN #10234'), 'WHOLEFDS');
  assert.equal(key('SAFEWAY STORE 0412 OAKLAND'), 'SAFEWAY');
  assert.equal(key('TARGET T-1049 BERKELEY'), 'TARGET');
  assert.equal(key('CVS/PHARMACY #08812'), 'CVS/PHARMACY');
});

test('removes reference blobs but keeps brand tokens', () => {
  assert.equal(key('AMZN Mktp US*2K4LM9XY3'), 'AMZN MKTP US');
  assert.equal(key('AIRBNB * HM8FQ2XZW'), 'AIRBNB');
  // Short mixed-alphanumeric brands must survive.
  assert.equal(key('7-ELEVEN 34122'), '7-ELEVEN');
});

test('strips trailing state then city, in that order', () => {
  assert.equal(key('SHELL SERVICE STATION SANFRAN CA'), 'SHELL SERVICE STATION');
  assert.equal(key('EQUINOX OAKLAND'), 'EQUINOX');
});

test('does not strip merchant-identifying prefixes', () => {
  // Uber and DoorDash *are* the merchant; losing them loses the category signal.
  assert.match(key('UBER *EATS 8FJ2K1'), /^UBER EATS/);
  assert.match(key('UBER *TRIP 9DK2LX'), /^UBER TRIP/);
  assert.match(key('DD *DOORDASH KX82LM'), /^DD DOORDASH/);
});

test('collapses descriptor variants of one merchant onto one key', () => {
  const variants = ['SQ *BLUE BOTTLE #4432 SANFRAN', 'SQ *BLUE BOTTLE #9981 OAKLAND', 'SQ *BLUE BOTTLE'];
  const keys = new Set(variants.map(key));
  assert.equal(keys.size, 1, `expected one key, got ${[...keys].join(' | ')}`);
});

test('flags degenerate results instead of caching on garbage', () => {
  const r = normalizeDescriptor('SQ *12345');
  assert.equal(r.degenerate, true);
  assert.equal(r.key, 'SQ *12345', 'degenerate keys fall back to the raw string');
});

test('is idempotent', () => {
  for (const s of ['SQ *BLUE BOTTLE #4432 SANFRAN CA', 'AMZN Mktp US*2K4LM9XY3', 'NETFLIX.COM']) {
    assert.equal(key(key(s)), key(s), s);
  }
});
