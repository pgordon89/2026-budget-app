/**
 * Test entry point.
 *
 * Node 20's built-in runner only auto-discovers `.js` test files, so pointing
 * `node --test` at a directory of `.ts` tests silently matches nothing and exits 0.
 * A suite that passes because it ran no tests is worse than no suite at all —
 * especially once CI gates on it. This walks for `*.test.ts` and passes an explicit
 * file list, then fails loudly if the list is empty.
 */

import { readdirSync, statSync } from 'node:fs';
import { join, dirname, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const SEARCH_DIRS = ['src', 'evals'];
const SKIP = new Set(['node_modules', '.git', '.next', 'dist', 'datasets']);

function walk(dir: string, found: string[] = []): string[] {
  let entries: string[];
  try {
    entries = readdirSync(dir);
  } catch {
    return found;
  }
  for (const entry of entries) {
    if (SKIP.has(entry)) continue;
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) walk(full, found);
    else if (entry.endsWith('.test.ts')) found.push(full);
  }
  return found;
}

const files = SEARCH_DIRS.flatMap((d) => walk(join(ROOT, d))).sort();

if (files.length === 0) {
  console.error('No *.test.ts files found. Refusing to report a passing run over an empty suite.');
  process.exit(1);
}

console.error(`running ${files.length} test file(s):`);
for (const f of files) console.error(`  ${relative(ROOT, f)}`);

const result = spawnSync(
  process.execPath,
  ['--import', 'tsx', '--test', ...files],
  { stdio: 'inherit', cwd: ROOT },
);

process.exit(result.status ?? 1);
