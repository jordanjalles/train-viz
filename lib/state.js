// Per-run curation state shared by every viewer: which runs are kept (starred) and which are
// archived (hidden from the default list). Lives in train-viz's own state/ directory; run
// directories are never written. Nothing here deletes data.

import fs from 'node:fs';
import path from 'node:path';

const OPS = new Set(['keep', 'unkeep', 'archive', 'unarchive']);

export class RunState {
  constructor(file) {
    this.file = file;
    this.kept = new Set();
    this.archived = new Map(); // id -> archived-at (ms)
    try {
      const s = JSON.parse(fs.readFileSync(file, 'utf8'));
      for (const id of s.kept || []) this.kept.add(id);
      for (const [id, t] of Object.entries(s.archived || {})) this.archived.set(id, t);
    } catch { /* first run, or unreadable: start empty */ }
  }

  /** Apply one op to many ids. A kept run is never archived. Returns {changed, skipped}. */
  apply(op, ids) {
    if (!OPS.has(op)) throw new Error(`unknown op ${op}`);
    if (!Array.isArray(ids) || ids.length > 2000 || ids.some((x) => typeof x !== 'string' || x.length > 300)) {
      throw new Error('ids must be an array of run ids');
    }
    let changed = 0;
    const skipped = [];
    for (const id of ids) {
      if (op === 'keep' && !this.kept.has(id)) { this.kept.add(id); this.archived.delete(id); changed++; }
      else if (op === 'unkeep' && this.kept.delete(id)) changed++;
      else if (op === 'archive') {
        if (this.kept.has(id)) skipped.push(id);
        else if (!this.archived.has(id)) { this.archived.set(id, Date.now()); changed++; }
      } else if (op === 'unarchive' && this.archived.delete(id)) changed++;
    }
    if (changed) this.save();
    return { changed, skipped };
  }

  save() {
    fs.mkdirSync(path.dirname(this.file), { recursive: true });
    const tmp = `${this.file}.tmp`;
    fs.writeFileSync(tmp, JSON.stringify({ kept: [...this.kept], archived: Object.fromEntries(this.archived) }, null, 1));
    fs.renameSync(tmp, this.file);
  }

  toJSON() { return { kept: [...this.kept], archived: [...this.archived.keys()] }; }
}
