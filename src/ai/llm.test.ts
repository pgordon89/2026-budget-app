import { test } from 'node:test';
import assert from 'node:assert/strict';
import type Anthropic from '@anthropic-ai/sdk';

import { LlmClassifier, validate, type MessageCreator } from './llm.js';
import { costOf, MODELS } from './client.js';
import type { ClassificationInput } from './prompt.js';

const INPUT: ClassificationInput = {
  rawDescriptor: 'SQ *PEETS COFFEE #221 SANFRAN CA',
  normalizedKey: 'PEETS COFFEE',
  amount: -6.75,
  neighbours: [{ key: 'BLUE BOTTLE COFFEE', category: 'food.coffee', similarity: 0.42 }],
  vote: null,
};

const USAGE = { input_tokens: 1200, output_tokens: 25, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 };

/** A response carrying one tool call, as forced `tool_choice` guarantees. */
function toolCall(input: unknown, overrides: Partial<Anthropic.Messages.Message> = {}) {
  return {
    id: 'msg_test',
    type: 'message',
    role: 'assistant',
    model: 'claude-haiku-4-5',
    stop_reason: 'tool_use',
    stop_sequence: null,
    usage: USAGE,
    content: [{ type: 'tool_use', id: 'toolu_test', name: 'record_category', input }],
    ...overrides,
  } as unknown as Anthropic.Messages.Message;
}

/** Records what the classifier sent, and replays scripted responses in order. */
function fakeTransport(...responses: Anthropic.Messages.Message[]) {
  const sent: Anthropic.Messages.MessageCreateParamsNonStreaming[] = [];
  let i = 0;
  const create: MessageCreator = async (params) => {
    sent.push(params);
    const response = responses[Math.min(i, responses.length - 1)];
    i++;
    return response!;
  };
  return { create, sent, calls: () => i };
}

const GOOD = { category: 'food.coffee', confidence: 0.92, evidence: 'coffee in descriptor' };

test('records a valid categorisation on the first attempt', async () => {
  const t = fakeTransport(toolCall(GOOD));
  const outcome = await new LlmClassifier(t.create).classify(INPUT);

  assert.equal(outcome.status, 'ok');
  assert.equal(outcome.status === 'ok' && outcome.category, 'food.coffee');
  assert.equal(outcome.telemetry.attempts, 1);
  assert.deepEqual(outcome.telemetry.repairs, [], 'nothing needed repairing');
});

test('forces the one tool and pins the schema', async () => {
  const t = fakeTransport(toolCall(GOOD));
  await new LlmClassifier(t.create).classify(INPUT);

  const [request] = t.sent;
  assert.deepEqual(request?.tool_choice, { type: 'tool', name: 'record_category' });

  const tool = request?.tools?.[0] as unknown as Record<string, unknown>;
  assert.equal(tool.strict, true, 'strict is what makes the enum binding rather than advisory');
  const schema = tool.input_schema as Record<string, unknown>;
  assert.equal(schema.additionalProperties, false, 'required by strict schemas');
});

test('marks a cache breakpoint on the stable system block', async () => {
  const t = fakeTransport(toolCall(GOOD));
  await new LlmClassifier(t.create).classify(INPUT);

  const system = t.sent[0]?.system as unknown as Array<Record<string, unknown>>;
  assert.deepEqual(system[0]?.cache_control, { type: 'ephemeral' });
  assert.ok(!JSON.stringify(system).includes('PEETS'), 'nothing per-transaction above the breakpoint');
});

test('records tokens, cost, and latency on every call', async () => {
  const t = fakeTransport(toolCall(GOOD));
  const outcome = await new LlmClassifier(t.create).classify(INPUT);

  assert.equal(outcome.telemetry.usage.inputTokens, 1200);
  assert.equal(outcome.telemetry.usage.outputTokens, 25);
  assert.ok(outcome.telemetry.costUsd > 0, 'a tier that costs money has to say so');
  assert.ok(outcome.telemetry.latencyMs >= 0);
  assert.equal(outcome.telemetry.model, 'claude-haiku-4-5');
});

test('repairs a confidence the strict schema cannot reject', async () => {
  // Strict schemas carry no numeric bounds, so 4.2 is schema-valid and absurd.
  // This is the gap Zod exists to close.
  const t = fakeTransport(
    toolCall({ ...GOOD, confidence: 4.2 }),
    toolCall(GOOD),
  );
  const outcome = await new LlmClassifier(t.create).classify(INPUT);

  assert.equal(outcome.status, 'ok');
  assert.equal(outcome.telemetry.attempts, 2);
  assert.match(outcome.telemetry.repairs[0] ?? '', /confidence/);
});

test('repairs a category whose direction contradicts the amount', async () => {
  // Well-formed, in-taxonomy, and wrong: an inflow label on money going out.
  const t = fakeTransport(
    toolCall({ category: 'income.refund', confidence: 0.8, evidence: 'refund' }),
    toolCall(GOOD),
  );
  const outcome = await new LlmClassifier(t.create).classify(INPUT);

  assert.equal(outcome.status, 'ok');
  assert.match(outcome.telemetry.repairs[0] ?? '', /inflow category but the amount is negative/);
});

test('tells the model exactly which rule failed', async () => {
  const t = fakeTransport(toolCall({ ...GOOD, confidence: 4.2 }), toolCall(GOOD));
  await new LlmClassifier(t.create).classify(INPUT);

  const followUp = JSON.stringify(t.sent[1]?.messages);
  assert.match(followUp, /rejected/);
  assert.match(followUp, /confidence/, 'a retry not told what was wrong is just another sample');
  assert.match(followUp, /"is_error":true/);
});

test('bounds the repair loop instead of paying for an unbounded one', async () => {
  const t = fakeTransport(toolCall({ ...GOOD, confidence: 9 }));
  const outcome = await new LlmClassifier(t.create, { maxRepairs: 2 }).classify(INPUT);

  assert.equal(outcome.status, 'unresolved');
  assert.equal(t.calls(), 3, 'the first attempt plus exactly maxRepairs retries');
  assert.equal(outcome.telemetry.attempts, 3);
});

test('accumulates cost across repair attempts rather than reporting the last one', async () => {
  const t = fakeTransport(toolCall({ ...GOOD, confidence: 9 }));
  const outcome = await new LlmClassifier(t.create, { maxRepairs: 1 }).classify(INPUT);

  assert.equal(outcome.telemetry.usage.inputTokens, 2400, 'two calls of 1200');
  assert.ok(
    Math.abs(outcome.telemetry.costUsd - costOf(outcome.telemetry.usage, MODELS['claude-haiku-4-5'])) < 1e-12,
  );
});

test('surfaces a refusal rather than treating it as an answer', async () => {
  const t = fakeTransport(toolCall(GOOD, { stop_reason: 'refusal', content: [] }));
  const outcome = await new LlmClassifier(t.create).classify(INPUT);

  assert.equal(outcome.status, 'unresolved');
  assert.match(outcome.status === 'unresolved' ? outcome.reason : '', /declined/);
});

test('handles a response with no tool call at all', async () => {
  const t = fakeTransport(
    toolCall(GOOD, { content: [{ type: 'text', text: 'food.coffee' }] as never, stop_reason: 'end_turn' }),
  );
  const outcome = await new LlmClassifier(t.create).classify(INPUT);

  assert.equal(outcome.status, 'unresolved');
});

test('predict routes a low-confidence answer to review instead of the ledger', async () => {
  const t = fakeTransport(toolCall({ ...GOOD, confidence: 0.4 }));
  const classifier = new LlmClassifier(t.create, { minConfidence: 0.7 });

  assert.equal(await classifier.predict(INPUT), null);
});

test('predict records the answering tier and what it cost', async () => {
  const t = fakeTransport(toolCall(GOOD));
  const prediction = await new LlmClassifier(t.create).predict(INPUT);

  assert.equal(prediction?.category, 'food.coffee');
  assert.equal(prediction?.tier, 'llm');
  assert.ok((prediction?.costUsd ?? 0) > 0);
});

test('validate accepts either-direction categories against any sign', () => {
  const transfer = { category: 'transfer.internal', confidence: 0.9, evidence: 'transfer' };
  assert.equal(validate(transfer, -500), undefined);
  assert.equal(validate(transfer, 500), undefined);
});

test('validate rejects a category outside the taxonomy', () => {
  assert.match(
    validate({ category: 'food.coffee_shops', confidence: 0.9, evidence: 'x' }, -5) ?? '',
    /taxonomy/,
  );
});
