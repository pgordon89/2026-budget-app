/**
 * Vector representations of merchant keys, and the plumbing around them.
 *
 * Tier 2 needs to find "merchants that look like this one" for descriptors the
 * exact-match tier has never seen. What produces those vectors is a swappable
 * decision, so it sits behind an interface with two things on the other side of
 * it: a lexical baseline that runs offline, and (later) a hosted embedding
 * model that costs money per call.
 *
 * The baseline goes first on purpose. A hosted embedding model is a recurring
 * bill and an availability dependency, and "we should use embeddings here" is an
 * assumption until something measures it. The lexical embedder costs nothing,
 * runs in CI with no key, and produces the number a paid model has to beat
 * before it earns its place in the pipeline. If it does beat it, the gap is the
 * justification; if it doesn't, the cheap thing ships and that is a result too.
 *
 * Merchant keys are unusually friendly to lexical matching — Tier 0 has already
 * stripped the noise, and what survives often contains the category outright
 * (`... COFFEE`, `... PHARMACY`, `... DENTAL`, `... AUTO PARTS`). That is
 * exactly why the baseline is worth measuring rather than assuming away.
 */

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

export interface Embedder {
  /**
   * Stable identity of *what produced these vectors* — provider, model, and a
   * version bump for any change to featurization. It namespaces the disk cache,
   * so changing the embedder can never silently serve vectors built by the old
   * one against an index built by the new one.
   */
  readonly id: string;
  readonly dimensions: number;
  /** Returns L2-normalised vectors, so cosine similarity is a plain dot product. */
  embed(texts: readonly string[]): Promise<Float32Array[]>;
}

/** Cosine similarity for vectors this module guarantees are unit length. */
export function similarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i]! * b[i]!;
  return dot;
}

// ── Lexical baseline ────────────────────────────────────────────────────────

const GRAM = 3;

/**
 * Word tokens plus character trigrams.
 *
 * Words carry the category signal that makes this baseline competitive at all
 * (`COFFEE`, `PHARMACY`). Trigrams carry robustness: bank feeds truncate
 * mid-word and glue tokens together, so `SIGHTGLASS COFFEE` and `SIGHTGLASS
 * COFF` have to land near each other.
 */
function features(text: string): string[] {
  const tokens = text.toUpperCase().split(/[^A-Z0-9]+/).filter(Boolean);
  const out: string[] = [];
  for (const token of tokens) {
    out.push(`W:${token}`);
    const padded = `^${token}$`;
    for (let i = 0; i + GRAM <= padded.length; i++) out.push(`G:${padded.slice(i, i + GRAM)}`);
  }
  return out;
}

/**
 * Fits an exact feature vocabulary and its inverse document frequencies, then
 * embeds against it.
 *
 * IDF is the difference between a useful baseline and a useless one. Merchant
 * keys share a lot of uninformative surface — `COM`, `US`, `INC`, `THE` — and
 * without down-weighting it, everything is a little alike and the nearest
 * neighbour is noise.
 *
 * An exact vocabulary rather than feature hashing, which this originally used.
 * Hashing ~40 features per key into a tractable number of buckets collides often
 * enough that unrelated merchants score a small non-zero similarity — a noise
 * floor under every score, and the loss of an honest "these share nothing"
 * signal. Widening the table until collisions vanished made the brute-force scan
 * too slow to be worth it. An exact vocabulary has neither problem.
 *
 * Features absent from the vocabulary are still counted **in the norm** even
 * though they get no dimension. This is deliberate: `PEETS COFFEE` should not be
 * treated as if it were merely `COFFEE` because `PEETS` is unrecognised. Keeping
 * the unknown mass in the denominator means an unfamiliar merchant scores lower
 * against its category peers than a familiar one does, which is exactly the
 * caution the tier should have. The dot product is unaffected — unknown features
 * can never match anything anyway — so `similarity` still returns a true cosine
 * in the full implicit feature space.
 *
 * Fit on history only. The corpus is a training input like any other, and
 * fitting on the holdout would leak the test set into the features.
 */
export function fitLexicalEmbedder(corpus: readonly string[]): Embedder {
  const vocabulary = new Map<string, number>();
  const documentFrequency: number[] = [];

  for (const text of corpus) {
    for (const feature of new Set(features(text))) {
      let index = vocabulary.get(feature);
      if (index === undefined) {
        index = vocabulary.size;
        vocabulary.set(feature, index);
        documentFrequency.push(0);
      }
      documentFrequency[index]! += 1;
    }
  }

  const n = Math.max(corpus.length, 1);
  // Smoothed, and never zero: a feature seen in every document is weak evidence,
  // not zero evidence.
  const idf = documentFrequency.map((df) => Math.log(1 + n / (1 + df)));
  const unknownIdf = Math.log(1 + n);

  function embedOne(text: string): Float32Array {
    const counts = new Map<string, number>();
    for (const feature of features(text)) counts.set(feature, (counts.get(feature) ?? 0) + 1);

    const vector = new Float32Array(vocabulary.size);
    let normSquared = 0;
    for (const [feature, count] of counts) {
      // Sublinear tf: a token repeated five times is not five times the evidence.
      const index = vocabulary.get(feature);
      const weight = (1 + Math.log(count)) * (index === undefined ? unknownIdf : idf[index]!);
      normSquared += weight * weight;
      if (index !== undefined) vector[index] = weight;
    }

    const norm = Math.sqrt(normSquared);
    if (norm > 0) for (let i = 0; i < vector.length; i++) vector[i]! /= norm;
    return vector;
  }

  return {
    id: `lexical-tfidf-g${GRAM}-v2`,
    dimensions: vocabulary.size,
    embed: async (texts) => texts.map(embedOne),
  };
}

// ── Disk cache ──────────────────────────────────────────────────────────────

interface CacheFile {
  readonly embedderId: string;
  readonly dimensions: number;
  readonly vectors: Record<string, number[]>;
}

/**
 * Memoises embeddings to a JSON file keyed by the embedder's id.
 *
 * Exists so an eval run is not a bill. A hosted embedding model charges per
 * call and the eval re-embeds the same ~600 merchant keys on every invocation;
 * without this, iterating on the *voting* logic would cost money each time even
 * though not one vector changed. The id in the file is checked on load, so a
 * cache built by a different model is discarded rather than silently mixed.
 */
export function withDiskCache(base: Embedder, file: string): Embedder {
  const vectors = new Map<string, Float32Array>();

  try {
    const parsed = JSON.parse(readFileSync(file, 'utf8')) as CacheFile;
    if (parsed.embedderId === base.id && parsed.dimensions === base.dimensions) {
      for (const [text, values] of Object.entries(parsed.vectors)) {
        vectors.set(text, Float32Array.from(values));
      }
    }
  } catch {
    // Absent or unreadable cache is not an error — it is a cold start.
  }

  function persist(): void {
    const out: CacheFile = {
      embedderId: base.id,
      dimensions: base.dimensions,
      // Sorted so the committed file has a stable diff instead of churning on
      // insertion order every run.
      vectors: Object.fromEntries([...vectors.entries()].sort(([a], [b]) => (a < b ? -1 : 1)).map(([text, v]) => [text, [...v]])),
    };
    mkdirSync(dirname(file), { recursive: true });
    writeFileSync(file, JSON.stringify(out) + '\n', 'utf8');
  }

  return {
    id: base.id,
    dimensions: base.dimensions,
    async embed(texts) {
      const missing = [...new Set(texts.filter((t) => !vectors.has(t)))];
      if (missing.length > 0) {
        const fresh = await base.embed(missing);
        missing.forEach((text, i) => vectors.set(text, fresh[i]!));
        persist();
      }
      return texts.map((text) => vectors.get(text)!);
    },
  };
}
