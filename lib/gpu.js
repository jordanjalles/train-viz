// nvidia-smi poller. One child process per sample, never concurrent.
// The GB10 reports [N/A] for memory.used/total in the query API on this
// driver, so memory comes from the per-process list instead.

import { execFile } from 'node:child_process';
import { EventEmitter } from 'node:events';

const GPU_FIELDS = 'name,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,memory.total';
const PROC_FIELDS = 'pid,process_name,used_gpu_memory';

function num(s) {
  const v = parseFloat(String(s).trim());
  return Number.isFinite(v) ? v : null;
}

function run(args) {
  return new Promise((resolve) => {
    execFile('nvidia-smi', args, { timeout: 4000 }, (err, stdout) => resolve(err ? null : stdout));
  });
}

export class GpuPoller extends EventEmitter {
  constructor({ intervalMs = 2000, historySize = 300 } = {}) {
    super();
    this.intervalMs = intervalMs;
    this.history = [];
    this.historySize = historySize;
    this.latest = null;
    this.available = null;
    this._busy = false;
  }

  start() {
    this.tick();
    this._timer = setInterval(() => this.tick(), this.intervalMs);
    this._timer.unref();
  }

  async tick() {
    if (this._busy) return;
    this._busy = true;
    try {
      const [g, p] = await Promise.all([
        run(['--query-gpu=' + GPU_FIELDS, '--format=csv,noheader,nounits']),
        run(['--query-compute-apps=' + PROC_FIELDS, '--format=csv,noheader,nounits']),
      ]);
      if (g == null) { this.available = false; return; }
      this.available = true;
      const f = g.trim().split(',').map((s) => s.trim());
      const procs = (p || '').trim().split('\n').filter(Boolean).map((line) => {
        const [pid, name, mem] = line.split(',').map((s) => s.trim());
        return { pid: parseInt(pid, 10), name, memMiB: num(mem) };
      });
      const sample = {
        t: Date.now(),
        name: f[0],
        util: num(f[1]),
        memUtil: num(f[2]),
        temp: num(f[3]),
        power: num(f[4]),
        powerLimit: num(f[5]),
        clockSm: num(f[6]),
        clockMem: num(f[7]),
        memUsed: num(f[8]) ?? procs.reduce((a, q) => a + (q.memMiB || 0), 0),
        memTotal: num(f[9]),
        procs,
      };
      this.latest = sample;
      this.history.push(sample);
      if (this.history.length > this.historySize) this.history.shift();
      this.emit('sample', sample);
    } finally {
      this._busy = false;
    }
  }
}
