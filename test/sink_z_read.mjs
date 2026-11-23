// Parse a z snapshot written by the sink cell with train-viz's own reader.
// Usage: node test/sink_z_read.mjs <path to z_NNNNNN.pt>
import { loadTensor } from '../lib/torchpt.js';

const t = await loadTensor(process.argv[2]);
const ok = t.shape.join(',') === '16,48,48' && t.data.length === 16 * 48 * 48 && Number.isFinite(t.data[0]);
console.log(`${ok ? 'ok  ' : 'FAIL'} train-viz reads the snapshot: shape ${t.shape.join('x')}`);
process.exit(ok ? 0 : 1);
