/**
 * Anthropic client construction, model routing, and cost accounting.
 *
 * Isolated from the classifier so that "what did this cost" is answerable from
 * one table rather than recomputed at each call site. The routing rule the
 * project claims — cheap model for bulk, expensive model for reasoning — is only
 * a claim if nothing prices the difference, so the price table lives next to the
 * client that uses it.
 */

import Anthropic from '@anthropic-ai/sdk';

export interface ModelSpec {
  readonly id: string;
  /** USD per million input tokens, at list price. */
  readonly inputPerMTok: number;
  /** USD per million output tokens, at list price. */
  readonly outputPerMTok: number;
  /**
   * Shortest prefix this model will cache. Below it, `cache_control` is accepted
   * and silently does nothing — no error, just a permanent cache miss. Recorded
   * here because that failure is invisible unless something checks it.
   */
  readonly minCacheablePrefixTokens: number;
}

/**
 * List prices, deliberately not promotional ones. Sonnet 5 carries an
 * introductory rate through 2026-08-31; a cost-per-1k figure computed against a
 * discount that expires is a number that quietly becomes wrong, so the table
 * uses standard pricing and any headline stays honest after the promo ends.
 */
export const MODELS = {
  'claude-haiku-4-5': {
    id: 'claude-haiku-4-5',
    inputPerMTok: 1,
    outputPerMTok: 5,
    minCacheablePrefixTokens: 4096,
  },
  'claude-sonnet-5': {
    id: 'claude-sonnet-5',
    inputPerMTok: 3,
    outputPerMTok: 15,
    minCacheablePrefixTokens: 1024,
  },
} as const satisfies Record<string, ModelSpec>;

export type ModelId = keyof typeof MODELS;

/** Cache writes cost more than fresh input; cache reads cost far less. */
const CACHE_WRITE_MULTIPLIER = 1.25;
const CACHE_READ_MULTIPLIER = 0.1;

export interface TokenUsage {
  readonly inputTokens: number;
  readonly outputTokens: number;
  readonly cacheReadTokens: number;
  readonly cacheCreationTokens: number;
}

export const EMPTY_USAGE: TokenUsage = {
  inputTokens: 0,
  outputTokens: 0,
  cacheReadTokens: 0,
  cacheCreationTokens: 0,
};

export function addUsage(a: TokenUsage, b: TokenUsage): TokenUsage {
  return {
    inputTokens: a.inputTokens + b.inputTokens,
    outputTokens: a.outputTokens + b.outputTokens,
    cacheReadTokens: a.cacheReadTokens + b.cacheReadTokens,
    cacheCreationTokens: a.cacheCreationTokens + b.cacheCreationTokens,
  };
}

/** Reads the SDK's usage block into our shape, tolerating absent cache fields. */
export function readUsage(usage: {
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens?: number | null;
  cache_creation_input_tokens?: number | null;
}): TokenUsage {
  return {
    inputTokens: usage.input_tokens,
    outputTokens: usage.output_tokens,
    cacheReadTokens: usage.cache_read_input_tokens ?? 0,
    cacheCreationTokens: usage.cache_creation_input_tokens ?? 0,
  };
}

/**
 * Cost in USD, with cached tokens priced at their own rates.
 *
 * Counting cache reads at full input price would overstate spend and make
 * caching look like it did nothing; ignoring cache writes would understate it.
 */
export function costOf(usage: TokenUsage, model: ModelSpec): number {
  const input =
    usage.inputTokens +
    usage.cacheCreationTokens * CACHE_WRITE_MULTIPLIER +
    usage.cacheReadTokens * CACHE_READ_MULTIPLIER;
  return (input * model.inputPerMTok + usage.outputTokens * model.outputPerMTok) / 1_000_000;
}

/**
 * Builds a client that works both with a real key and behind a credential-
 * injecting proxy.
 *
 * The proxy path exists because this project's development sandbox authenticates
 * at the gateway named by `ANTHROPIC_BASE_URL` and injects the credential itself
 * — but only when the request carries no `x-api-key` header at all. The SDK
 * refuses to construct without a key and then sends whatever placeholder it was
 * given, which the proxy rejects; deleting the header is what lets it through.
 *
 * CI and production set a real `ANTHROPIC_API_KEY` and take the first branch, so
 * the same code runs in all three places.
 */
export function createClient(): Anthropic {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (apiKey) return new Anthropic({ apiKey });

  return new Anthropic({
    apiKey: 'unused-proxy-injects-credential',
    defaultHeaders: { 'x-api-key': null },
  });
}

/** True when a real key is present, i.e. calls are billed to this project. */
export function hasDirectCredentials(): boolean {
  return Boolean(process.env.ANTHROPIC_API_KEY);
}
