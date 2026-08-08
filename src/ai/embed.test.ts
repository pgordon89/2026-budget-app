import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { fitLexicalEmbedder, similarity, withDiskCache, type Embedder } from './embed.js';

const CORPUS = [
  'BLUE BOTTLE COFFEE',
  'SIGHTGLASS COFFEE',
  'SAFEWAY',
  'WHOLEFDS',
  'SHELL SERVICE STATION',
  'CHEVRON',
  'CVS PHARMACY',
];

async function embedOne(embedder: Embedder, text: string): Promise<Float32Array> {
  return (await embedder.embed([text]))[0]!;
}

test('embeds deterministically and to unit length', async () => {
  const embedder = fitLexicalEmbedder(CORPUS);
  const a = await embedOne(embedder, 'BLUE BOTTLE COFFEE');
  const b = await embedOne(embedder, 'BLUE BOTTLE COFFEE');

  assert.deepEqual([...a], [...b], 'same input, same vector — eval baselines depend on it');
  assert.ok(Math.abs(similarity(a, a) - 1) < 1e-5, 'unit length, so cosine is a dot product');
});

test('places an unseen merchant nearer its category peers than an unrelated one', async () => {
  // The whole premise of the tier: PEETS was never in the corpus, but the word
  // COFFEE puts it in the right neighbourhood.
  const embedder = fitLexicalEmbedder(CORPUS);
  const query = await embedOne(embedder, 'PEETS COFFEE');

  const toCoffee = similarity(query, await embedOne(embedder, 'BLUE BOTTLE COFFEE'));
  const toGas = similarity(query, await embedOne(embedder, 'SHELL SERVICE STATION'));

  assert.ok(toCoffee > toGas, `expected coffee to win, got ${toCoffee.toFixed(3)} vs ${toGas.toFixed(3)}`);
});

test('survives the truncation banks apply to descriptors', async () => {
  const embedder = fitLexicalEmbedder(CORPUS);
  const full = await embedOne(embedder, 'SIGHTGLASS COFFEE');
  const cut = await embedOne(embedder, 'SIGHTGLASS COFF');

  assert.ok(similarity(full, cut) > 0.5, 'character trigrams carry the match across a truncated token');
});

test('idf discounts a token the corpus uses everywhere', async () => {
  // ZED is ubiquitous in one corpus and near-unique in the other. Two strings
  // sharing only ZED should be judged less alike when ZED is uninformative.
  const ubiquitous = fitLexicalEmbedder(['ALPHA ZED', 'BETA ZED', 'GAMMA ZED', 'DELTA ZED', 'EPSILON ZED']);
  const rare = fitLexicalEmbedder(['ALPHA ONE', 'BETA TWO', 'GAMMA THREE', 'DELTA FOUR', 'EPSILON ZED']);

  const pairwise = async (embedder: Embedder) =>
    similarity(await embedOne(embedder, 'ALPHA ZED'), await embedOne(embedder, 'BETA ZED'));

  assert.ok(
    (await pairwise(rare)) > (await pairwise(ubiquitous)),
    'a shared common token is weaker evidence than a shared rare one',
  );
});

test('scores no similarity between keys sharing nothing', async () => {
  const embedder = fitLexicalEmbedder(CORPUS);
  const value = similarity(await embedOne(embedder, 'XQZ'), await embedOne(embedder, 'MPW'));
  assert.equal(value, 0, 'no shared feature means no vote, not a weak vote');
});

// ── Disk cache ──────────────────────────────────────────────────────────────

function countingEmbedder(id = 'fake-v1', dimensions = 4) {
  const state = { calls: 0, embedded: [] as string[] };
  const embedder: Embedder = {
    id,
    dimensions,
    async embed(texts) {
      state.calls++;
      state.embedded.push(...texts);
      return texts.map((t) => {
        const v = new Float32Array(dimensions);
        v[t.length % dimensions] = 1;
        return v;
      });
    },
  };
  return { embedder, state };
}

test('disk cache calls the underlying embedder only for texts it has not seen', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'fiscus-embed-'));
  try {
    const file = join(dir, 'cache.json');
    const { embedder, state } = countingEmbedder();
    const cached = withDiskCache(embedder, file);

    await cached.embed(['ALPHA', 'BETA']);
    assert.deepEqual(state.embedded, ['ALPHA', 'BETA']);

    await cached.embed(['ALPHA', 'BETA']);
    assert.equal(state.calls, 1, 'a repeat run must not re-embed, or every eval is a bill');

    await cached.embed(['ALPHA', 'GAMMA']);
    assert.deepEqual(state.embedded, ['ALPHA', 'BETA', 'GAMMA'], 'only the miss is sent');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('disk cache reloads across process boundaries and preserves vectors', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'fiscus-embed-'));
  try {
    const file = join(dir, 'cache.json');
    const first = countingEmbedder();
    const before = await withDiskCache(first.embedder, file).embed(['ALPHA', 'BETA']);

    const second = countingEmbedder();
    const after = await withDiskCache(second.embedder, file).embed(['ALPHA', 'BETA']);

    assert.equal(second.state.calls, 0, 'served entirely from the file');
    assert.deepEqual(after.map((v) => [...v]), before.map((v) => [...v]));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('disk cache discards vectors built by a different embedder', async () => {
  // Silently mixing two models' vectors in one index would be undetectable in
  // any accuracy number, so identity is checked rather than assumed.
  const dir = mkdtempSync(join(tmpdir(), 'fiscus-embed-'));
  try {
    const file = join(dir, 'cache.json');
    await withDiskCache(countingEmbedder('model-a').embedder, file).embed(['ALPHA']);

    const { embedder, state } = countingEmbedder('model-b');
    await withDiskCache(embedder, file).embed(['ALPHA']);

    assert.equal(state.calls, 1, 'cold start under a new embedder id');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('disk cache treats an unreadable file as a cold start, not an error', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'fiscus-embed-'));
  try {
    const file = join(dir, 'cache.json');
    writeFileSync(file, 'not json at all', 'utf8');

    const { embedder, state } = countingEmbedder();
    await withDiskCache(embedder, file).embed(['ALPHA']);
    assert.equal(state.calls, 1);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
