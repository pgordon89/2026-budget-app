CREATE TABLE "accounts" (
	"id" text PRIMARY KEY NOT NULL,
	"name" text NOT NULL,
	"type" text NOT NULL,
	"mask" text NOT NULL
);
--> statement-breakpoint
CREATE TABLE "categories" (
	"id" text PRIMARY KEY NOT NULL,
	"group" text NOT NULL,
	"label" text NOT NULL,
	"direction" text NOT NULL
);
--> statement-breakpoint
CREATE TABLE "corrections" (
	"id" text PRIMARY KEY NOT NULL,
	"transaction_id" text NOT NULL,
	"merchant_key" text NOT NULL,
	"predicted_category_id" text,
	"predicted_source" text,
	"predicted_confidence" double precision,
	"corrected_category_id" text NOT NULL,
	"corrected_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "merchant_memory" (
	"merchant_key" text NOT NULL,
	"category_id" text NOT NULL,
	"weight" double precision DEFAULT 0 NOT NULL,
	"count" integer DEFAULT 0 NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "merchant_memory_merchant_key_category_id_pk" PRIMARY KEY("merchant_key","category_id")
);
--> statement-breakpoint
CREATE TABLE "transactions" (
	"id" text PRIMARY KEY NOT NULL,
	"account_id" text NOT NULL,
	"posted_on" date NOT NULL,
	"amount_cents" bigint NOT NULL,
	"raw_descriptor" text NOT NULL,
	"merchant_key" text NOT NULL,
	"category_id" text,
	"category_source" text,
	"category_confidence" double precision,
	"category_confirmed" boolean DEFAULT false NOT NULL,
	"category_cost_micro_usd" bigint DEFAULT 0 NOT NULL,
	"imported_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
ALTER TABLE "corrections" ADD CONSTRAINT "corrections_transaction_id_transactions_id_fk" FOREIGN KEY ("transaction_id") REFERENCES "public"."transactions"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "corrections" ADD CONSTRAINT "corrections_predicted_category_id_categories_id_fk" FOREIGN KEY ("predicted_category_id") REFERENCES "public"."categories"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "corrections" ADD CONSTRAINT "corrections_corrected_category_id_categories_id_fk" FOREIGN KEY ("corrected_category_id") REFERENCES "public"."categories"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "merchant_memory" ADD CONSTRAINT "merchant_memory_category_id_categories_id_fk" FOREIGN KEY ("category_id") REFERENCES "public"."categories"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "transactions" ADD CONSTRAINT "transactions_account_id_accounts_id_fk" FOREIGN KEY ("account_id") REFERENCES "public"."accounts"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "transactions" ADD CONSTRAINT "transactions_category_id_categories_id_fk" FOREIGN KEY ("category_id") REFERENCES "public"."categories"("id") ON DELETE no action ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "corrections_merchant_key_idx" ON "corrections" USING btree ("merchant_key");--> statement-breakpoint
CREATE INDEX "corrections_corrected_at_idx" ON "corrections" USING btree ("corrected_at");--> statement-breakpoint
CREATE INDEX "transactions_posted_on_idx" ON "transactions" USING btree ("posted_on");--> statement-breakpoint
CREATE INDEX "transactions_merchant_key_idx" ON "transactions" USING btree ("merchant_key");--> statement-breakpoint
CREATE INDEX "transactions_category_idx" ON "transactions" USING btree ("category_id");--> statement-breakpoint
CREATE INDEX "transactions_review_idx" ON "transactions" USING btree ("category_confirmed","posted_on");--> statement-breakpoint
CREATE UNIQUE INDEX "transactions_natural_key_idx" ON "transactions" USING btree ("account_id","posted_on","amount_cents","raw_descriptor");