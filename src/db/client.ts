/**
 * Database connections.
 *
 * Local development, tests, and CI all run against PGlite — Postgres compiled to
 * WebAssembly, in-process, no server and no credentials. That is not a stand-in
 * for Postgres the way SQLite would be: it is Postgres, so the migrations that
 * run here are the migrations that run against Neon, and a schema mistake
 * surfaces in CI rather than on deploy.
 *
 * The alternative — Docker Postgres locally, nothing in CI — leaves the schema
 * untested on every push, which for a project whose whole argument is "if it
 * isn't measured, it isn't done" would be an odd place to stop measuring.
 */

import { PGlite } from '@electric-sql/pglite';
import { drizzle, type PgliteDatabase } from 'drizzle-orm/pglite';
import { migrate } from 'drizzle-orm/pglite/migrator';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import * as schema from './schema.js';
import { CATEGORIES } from '../core/taxonomy.js';

const MIGRATIONS = join(dirname(fileURLToPath(import.meta.url)), 'migrations');

export type Database = PgliteDatabase<typeof schema>;

export interface Handle {
  readonly db: Database;
  close(): Promise<void>;
}

/**
 * Seeds `categories` from the taxonomy module.
 *
 * This is what keeps the table honest. The categories are defined once, in
 * TypeScript, and projected into SQL so foreign keys can do their job — rather
 * than being typed into a migration where the two definitions would drift apart
 * silently and the eval would start scoring against a label space the database
 * disagreed with.
 */
export async function seedCategories(db: Database): Promise<void> {
  await db
    .insert(schema.categories)
    .values(
      CATEGORIES.map((category) => ({
        id: category.id,
        group: category.group,
        label: category.label,
        direction: category.direction,
      })),
    )
    .onConflictDoUpdate({
      target: schema.categories.id,
      set: {
        group: schema.categories.group,
        label: schema.categories.label,
        direction: schema.categories.direction,
      },
    });
}

/** A fresh, empty, migrated, seeded database held entirely in memory. */
export async function createMemoryDatabase(): Promise<Handle> {
  const client = new PGlite();
  const db = drizzle(client, { schema });
  await migrate(db, { migrationsFolder: MIGRATIONS });
  await seedCategories(db);
  return { db, close: () => client.close() };
}

/** A database persisted to disk, for a local dev server that outlives a process. */
export async function createFileDatabase(path: string): Promise<Handle> {
  const client = new PGlite(path);
  const db = drizzle(client, { schema });
  await migrate(db, { migrationsFolder: MIGRATIONS });
  await seedCategories(db);
  return { db, close: () => client.close() };
}

export { schema };
