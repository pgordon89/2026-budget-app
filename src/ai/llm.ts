/**
 * Tier 3 — model categorisation, for what the deterministic tiers could not settle.
 *
 * This is the only tier that costs money per transaction, so it is also the only
 * one that has to justify itself: every call records its tokens, its cost, its
 * latency, and how many attempts it took. Those numbers are the point. A
 * categoriser that is 3% more accurate and 40× more expensive is a different
 * product decision than one that is 3% more accurate and free, and neither can
 * be made without the figures.
 *
 * Output is structured three times over, deliberately, because the three layers
 * catch different things:
 *
 *   1. **Forced tool use with a strict schema.** `tool_choice` pins the model to
 *      one tool and `strict: true` makes the API enforce the schema, so a
 *      category outside the taxonomy is not something the model can emit. This
 *      removes the entire class of "the model returned prose instead of JSON"
 *      failures rather than handling them.
 *   2. **Zod validation.** Strict schemas cannot express numeric bounds, so a
 *      confidence of 4.2 satisfies the API and is still nonsense. Zod covers
 *      what the schema cannot, and keeps the guarantee if `strict` is ever
 *      unavailable on a model we route to.
 *   3. **A domain check.** The taxonomy records which direction money normally
 *      flows for each category; an inflow label on a negative amount is
 *      well-formed, in-taxonomy, and wrong. Nothing upstream can catch that.
 *
 * The repair loop exists for layers 2 and 3 — the semantic failures. It is
 * bounded because an unbounded repair loop is an unbounded bill, and it feeds
 * back the specific rule that failed, since a retry that isn't told what was
 * wrong is just another sample from the same distribution.
 */

import type Anthropic from '@anthropic-ai/sdk';
import { z } from 'zod';

import {
  CATEGORY_IDS,
  getCategory,
  isCategoryId,
  type CategoryId,
} from '../core/taxonomy.js';
import {
  MODELS,
  addUsage,
  costOf,
  readUsage,
  EMPTY_USAGE,
  type ModelId,
  type TokenUsage,
} from './client.js';
import { PROMPT_VERSION, repairMessage, systemPrompt, userMessage, type ClassificationInput } from './prompt.js';
import type { Prediction } from './types.js';

/** The one tool the model is allowed to call. */
const TOOL_NAME = 'record_category';

/**
 * `strict: true` is what makes the enum binding rather than advisory. It also
 * requires `additionalProperties: false` and a complete `required` list, so the
 * shape is fully pinned. Numeric bounds are deliberately absent — strict schemas
 * do not support them, which is exactly why Zod runs afterwards.
 */
function toolDefinition(): Anthropic.Messages.Tool {
  return {
    name: TOOL_NAME,
    description:
      'Record the category for the transaction, with your confidence and the evidence that decided it.',
    strict: true,
    input_schema: {
      type: 'object',
      properties: {
        category: {
          type: 'string',
          enum: [...CATEGORY_IDS],
          description: 'The single best category id from the list provided.',
        },
        confidence: {
          type: 'number',
          description: 'Your probability of being correct, from 0 to 1.',
        },
        evidence: {
          type: 'string',
          description: 'Under ten words: the part of the input that decided it.',
        },
      },
      required: ['category', 'confidence', 'evidence'],
      additionalProperties: false,
    },
  } as Anthropic.Messages.Tool;
}

const RecordCategorySchema = z.object({
  category: z.string().refine(isCategoryId, { message: 'category is not in the taxonomy' }),
  confidence: z.number().min(0).max(1),
  evidence: z.string().min(1).max(300),
});

export interface LlmConfig {
  readonly model: ModelId;
  /** Repair attempts after the first call. Zero means one shot, no repairs. */
  readonly maxRepairs: number;
  readonly maxTokens: number;
  /** Below this, the prediction goes to the review queue instead of the ledger. */
  readonly minConfidence: number;
}

export const DEFAULT_LLM_CONFIG: LlmConfig = {
  // Bulk categorisation is the cheap model's job; the routing claim is priced in
  // `analyze:llm`, which scores both tiers of model on the same transactions.
  model: 'claude-haiku-4-5',
  maxRepairs: 1,
  // Generous for one tool call with a short evidence string; the tool schema,
  // not this limit, is what bounds the response.
  maxTokens: 256,
  minConfidence: 0.7,
};

export interface LlmTelemetry {
  readonly model: string;
  readonly promptVersion: string;
  /** Calls made, including repairs. 1 means it validated first time. */
  readonly attempts: number;
  /** Validation failures, in order. Empty when nothing needed repairing. */
  readonly repairs: readonly string[];
  readonly usage: TokenUsage;
  readonly costUsd: number;
  readonly latencyMs: number;
}

export type LlmOutcome =
  | {
      readonly status: 'ok';
      readonly category: CategoryId;
      readonly confidence: number;
      readonly evidence: string;
      readonly telemetry: LlmTelemetry;
    }
  | {
      /** Repairs exhausted, refused, or malformed beyond recovery. */
      readonly status: 'unresolved';
      readonly reason: string;
      readonly telemetry: LlmTelemetry;
    };

/** The slice of the SDK this tier uses, so tests can supply a fake. */
export type MessageCreator = (
  params: Anthropic.Messages.MessageCreateParamsNonStreaming,
) => Promise<Anthropic.Messages.Message>;

export class LlmClassifier {
  private readonly config: LlmConfig;

  constructor(
    private readonly create: MessageCreator,
    config: Partial<LlmConfig> = {},
  ) {
    this.config = { ...DEFAULT_LLM_CONFIG, ...config };
  }

  async classify(input: ClassificationInput): Promise<LlmOutcome> {
    const startedAt = Date.now();
    const model = MODELS[this.config.model];
    const repairs: string[] = [];
    let usage = EMPTY_USAGE;
    let attempts = 0;

    const messages: Anthropic.Messages.MessageParam[] = [
      { role: 'user', content: userMessage(input) },
    ];

    const telemetry = (): LlmTelemetry => ({
      model: model.id,
      promptVersion: PROMPT_VERSION,
      attempts,
      repairs: [...repairs],
      usage,
      costUsd: costOf(usage, model),
      latencyMs: Date.now() - startedAt,
    });

    for (let attempt = 0; attempt <= this.config.maxRepairs; attempt++) {
      attempts++;

      const response = await this.create({
        model: model.id,
        max_tokens: this.config.maxTokens,
        // The breakpoint sits at the end of the only content that repeats across
        // transactions. Everything volatile is in `messages`, after it.
        system: [
          {
            type: 'text',
            text: systemPrompt(),
            cache_control: { type: 'ephemeral' },
          },
        ],
        tools: [toolDefinition()],
        tool_choice: { type: 'tool', name: TOOL_NAME },
        messages,
      });

      usage = addUsage(usage, readUsage(response.usage));

      if (response.stop_reason === 'refusal') {
        return { status: 'unresolved', reason: 'model declined the request', telemetry: telemetry() };
      }

      const toolUse = response.content.find(
        (block): block is Anthropic.Messages.ToolUseBlock => block.type === 'tool_use',
      );
      if (!toolUse) {
        return { status: 'unresolved', reason: 'no tool call in response', telemetry: telemetry() };
      }

      const problem = validate(toolUse.input, input.amount);
      if (problem === undefined) {
        const parsed = RecordCategorySchema.parse(toolUse.input);
        return {
          status: 'ok',
          category: parsed.category as CategoryId,
          confidence: parsed.confidence,
          evidence: parsed.evidence,
          telemetry: telemetry(),
        };
      }

      repairs.push(problem);
      messages.push(
        { role: 'assistant', content: response.content },
        {
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: toolUse.id,
              is_error: true,
              content: repairMessage(problem),
            },
          ],
        },
      );
    }

    return {
      status: 'unresolved',
      reason: `failed validation after ${attempts} attempts: ${repairs[repairs.length - 1]}`,
      telemetry: telemetry(),
    };
  }

  /** Router-facing form. `null` routes the transaction to human review. */
  async predict(input: ClassificationInput): Promise<Prediction | null> {
    const outcome = await this.classify(input);
    if (outcome.status !== 'ok' || outcome.confidence < this.config.minConfidence) return null;
    return {
      category: outcome.category,
      confidence: outcome.confidence,
      tier: 'llm',
      costUsd: outcome.telemetry.costUsd,
    };
  }
}

/**
 * Returns a human-readable problem, or undefined when the call is acceptable.
 *
 * The direction check is the one that earns its keep. `income.refund` on a
 * -$40 charge parses cleanly, sits inside the taxonomy, and is still wrong —
 * neither the schema nor Zod can see it, because it is only wrong relative to
 * the transaction.
 */
export function validate(raw: unknown, amount: number): string | undefined {
  const parsed = RecordCategorySchema.safeParse(raw);
  if (!parsed.success) {
    return parsed.error.issues.map((i) => `${i.path.join('.') || 'input'}: ${i.message}`).join('; ');
  }

  const direction = getCategory(parsed.data.category)?.direction;
  if (direction === 'inflow' && amount < 0) {
    return `${parsed.data.category} is an inflow category but the amount is negative`;
  }
  if (direction === 'outflow' && amount > 0) {
    return `${parsed.data.category} is an outflow category but the amount is positive`;
  }

  return undefined;
}

/** Adapts a real SDK client to the injectable transport. */
export function messageCreator(client: Anthropic): MessageCreator {
  return (params) => client.messages.create(params);
}
