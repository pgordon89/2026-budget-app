-- Replace the confirmed/unconfirmed boolean with a three-state label trust level,
-- and give merchant_memory a counter that only independent evidence can move.
--
-- Written by hand rather than generated. drizzle-kit cannot tell a rename from a
-- drop-and-add without being asked, and this is neither: the old boolean carries
-- real information that has to be translated, not discarded. A generated
-- migration would have dropped `category_confirmed` and left every previously
-- confirmed row with a NULL status — silently returning the whole ledger to the
-- review queue on deploy.

ALTER TABLE "transactions" ADD COLUMN "category_status" text;--> statement-breakpoint

-- Translate the old boolean. A row a human had confirmed stays confirmed. A row
-- the pipeline had answered becomes provisional: it was never independently
-- evidenced, and under the old schema it was being summed into totals anyway,
-- which is the defect this column exists to fix. Rows with no category keep a
-- NULL status and stay in the queue as questions rather than proposals.
UPDATE "transactions" SET "category_status" =
  CASE
    WHEN "category_confirmed" THEN 'confirmed'
    WHEN "category_id" IS NOT NULL THEN 'provisional'
    ELSE NULL
  END;--> statement-breakpoint

DROP INDEX IF EXISTS "transactions_review_idx";--> statement-breakpoint
ALTER TABLE "transactions" DROP COLUMN "category_confirmed";--> statement-breakpoint
CREATE INDEX "transactions_status_idx" ON "transactions" USING btree ("category_status","posted_on");--> statement-breakpoint

ALTER TABLE "merchant_memory" ADD COLUMN "confirmed_count" integer DEFAULT 0 NOT NULL;--> statement-breakpoint

-- Existing tallies predate the split, so their provenance is unrecoverable from
-- the weight alone: 3.0 could be three confirmed sightings or twelve inferred
-- ones. Backfilled to zero, which is the conservative reading — every merchant
-- starts unattested and re-earns promotion from the next real confirmation.
-- The alternative, inferring counts from weight, would fabricate evidence.
UPDATE "merchant_memory" SET "confirmed_count" = 0;
