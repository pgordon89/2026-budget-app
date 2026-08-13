import { defineConfig } from 'drizzle-kit';

/**
 * Migrations are generated files, checked in, and applied identically in tests
 * (against PGlite) and in production (against Neon). A schema that is only ever
 * pushed rather than migrated cannot be tested before it is deployed.
 */
export default defineConfig({
  dialect: 'postgresql',
  schema: './src/db/schema.ts',
  out: './src/db/migrations',
  dbCredentials: { url: process.env.DATABASE_URL ?? 'postgres://localhost:5432/fiscus' },
});
