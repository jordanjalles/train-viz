// Minimal reader for `torch.save(tensor)` zip archives (the modern
// "PyTorch file format"): one stored (uncompressed) `data/0` entry with the
// raw storage bytes, and a small pickle `data.pkl` whose only interesting
// content is the tensor's dtype, shape and stride.
//
// No torch, no numpy, no dependencies. Only what train.py writes for the
// z_*.pt snapshots is supported: a single contiguous tensor, fp16/fp32/bf16,
// plus uint8 for the z_images.pt thumbnails (returned as a Uint8Array).

import { readFile } from 'node:fs/promises';

const STORAGE_DTYPES = {
  HalfStorage: 'f16',
  FloatStorage: 'f32',
  BFloat16Storage: 'bf16',
  DoubleStorage: 'f64',
  ByteStorage: 'u8',
};

/** Parse the zip central directory; return { name -> {offset, size} } for stored entries. */
function zipEntries(buf) {
  // End of central directory record: signature 0x06054b50, search back from the end.
  let eocd = -1;
  for (let i = buf.length - 22; i >= Math.max(0, buf.length - 65557); i--) {
    if (buf.readUInt32LE(i) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error('not a zip file (no EOCD)');
  let count = buf.readUInt16LE(eocd + 10);
  let cdOffset = buf.readUInt32LE(eocd + 16);
  // zip64 (torch writes zip64 for large files; z snapshots are ~2 MB but be safe)
  if (cdOffset === 0xffffffff || count === 0xffff) {
    const loc = eocd - 20;
    if (buf.readUInt32LE(loc) !== 0x07064b50) throw new Error('zip64 locator missing');
    const z64 = Number(buf.readBigUInt64LE(loc + 8));
    count = Number(buf.readBigUInt64LE(z64 + 32));
    cdOffset = Number(buf.readBigUInt64LE(z64 + 48));
  }
  const entries = {};
  let p = cdOffset;
  for (let i = 0; i < count; i++) {
    if (buf.readUInt32LE(p) !== 0x02014b50) throw new Error('bad central directory');
    const method = buf.readUInt16LE(p + 10);
    let size = buf.readUInt32LE(p + 24);
    const nameLen = buf.readUInt16LE(p + 28);
    const extraLen = buf.readUInt16LE(p + 30);
    const commentLen = buf.readUInt16LE(p + 32);
    let localOffset = buf.readUInt32LE(p + 42);
    const name = buf.toString('utf8', p + 46, p + 46 + nameLen);
    if (size === 0xffffffff || localOffset === 0xffffffff) {
      // zip64 extra field
      let e = p + 46 + nameLen;
      const end = e + extraLen;
      while (e + 4 <= end) {
        const id = buf.readUInt16LE(e), len = buf.readUInt16LE(e + 2);
        if (id === 1) {
          let q = e + 4;
          if (size === 0xffffffff) { size = Number(buf.readBigUInt64LE(q)); q += 8; }
          if (buf.readUInt32LE(p + 20) === 0xffffffff) q += 8; // compressed size
          if (localOffset === 0xffffffff) localOffset = Number(buf.readBigUInt64LE(q));
        }
        e += 4 + len;
      }
    }
    entries[name] = { method, size, localOffset };
    p += 46 + nameLen + extraLen + commentLen;
  }
  // Resolve data offsets through the local headers.
  for (const e of Object.values(entries)) {
    const h = e.localOffset;
    if (buf.readUInt32LE(h) !== 0x04034b50) throw new Error('bad local header');
    const nameLen = buf.readUInt16LE(h + 26), extraLen = buf.readUInt16LE(h + 28);
    e.offset = h + 30 + nameLen + extraLen;
  }
  return entries;
}

/**
 * Extract (dtype, shape) from torch's tensor pickle. We walk the opcodes we
 * care about and take the first GLOBAL `torch.XStorage` and the first two
 * tuples of ints after it (size, then stride).
 */
function parsePickle(pk) {
  let i = 0;
  let dtype = null;
  const ints = [];
  const tuples = [];
  const marks = [];   // ints.length at each MARK, so MARK…TUPLE (ndim > 3) becomes one tuple
  const flush = (n) => { tuples.push(ints.splice(ints.length - n, n)); };
  while (i < pk.length) {
    const op = pk[i++];
    switch (op) {
      case 0x80: i += 1; break;                    // PROTO
      case 0x63: {                                 // GLOBAL module\nname\n
        const end1 = pk.indexOf(0x0a, i); const mod = pk.toString('latin1', i, end1);
        const end2 = pk.indexOf(0x0a, end1 + 1); const name = pk.toString('latin1', end1 + 1, end2);
        i = end2 + 1;
        if (mod === 'torch' && STORAGE_DTYPES[name] && !dtype) dtype = STORAGE_DTYPES[name];
        break;
      }
      case 0x28: marks.push(ints.length); break;   // MARK
      case 0x58: { const n = pk.readUInt32LE(i); i += 4 + n; break; } // BINUNICODE
      case 0x8c: { const n = pk[i]; i += 1 + n; break; }              // SHORT_BINUNICODE
      case 0x71: i += 1; break;                    // BINPUT
      case 0x72: i += 4; break;                    // LONG_BINPUT
      case 0x68: i += 1; break;                    // BINGET
      case 0x6a: i += 4; break;                    // LONG_BINGET
      case 0x4b: ints.push(pk[i]); i += 1; break;                 // BININT1
      case 0x4d: ints.push(pk.readUInt16LE(i)); i += 2; break;    // BININT2
      case 0x4a: ints.push(pk.readInt32LE(i)); i += 4; break;     // BININT
      case 0x85: flush(1); break;                  // TUPLE1
      case 0x86: flush(2); break;                  // TUPLE2
      case 0x87: flush(3); break;                  // TUPLE3
      case 0x74: {                                 // TUPLE: everything since the last MARK
        const start = Math.min(marks.pop() ?? ints.length, ints.length);
        tuples.push(ints.splice(start));
        break;
      }
      case 0x29: tuples.push([]); break;           // EMPTY_TUPLE
      case 0x88: case 0x89: break;                 // NEWTRUE/NEWFALSE
      case 0x52: break;                            // REDUCE
      case 0x51: tuples.pop(); break;              // BINPERSID: drop the storage id tuple (holds numel)
      case 0x2e: i = pk.length; break;             // STOP
      case 0x7d: break;                            // EMPTY_DICT
      case 0x75: break;                            // SETITEMS
      case 0x73: break;                            // SETITEM
      case 0x5d: break;                            // EMPTY_LIST
      case 0x4e: break;                            // NONE
      case 0x81: break;                            // NEWOBJ
      case 0x62: break;                            // BUILD
      default:
        throw new Error(`unsupported pickle opcode 0x${op.toString(16)} at ${i - 1}`);
    }
  }
  // The first two int tuples left are (size), (stride): TUPLEn for ndim <= 3,
  // MARK…TUPLE above that. The storage id tuple was dropped at BINPERSID.
  // Fallback for an unexpected layout: split the remaining ints in half.
  let shape, stride;
  const tup = tuples.filter((t) => t.length > 0);
  if (tup.length >= 2) [shape, stride] = tup;
  else if (ints.length >= 2 && ints.length % 2 === 0) {
    shape = ints.slice(0, ints.length / 2); stride = ints.slice(ints.length / 2);
  } else throw new Error('could not find tensor shape in pickle');
  return { dtype, shape, stride };
}

const BYTES = { f16: 2, bf16: 2, f32: 4, f64: 8, u8: 1 };

/** Load a single-tensor .pt file. Returns { dtype, shape, data: Float32Array } (Uint8Array for u8) */
export async function loadTensor(path) {
  const buf = await readFile(path);
  const entries = zipEntries(buf);
  const pklName = Object.keys(entries).find((n) => n.endsWith('/data.pkl') || n === 'data.pkl');
  const dataName = Object.keys(entries).find((n) => n.endsWith('/data/0') || n === 'data/0');
  if (!pklName || !dataName) throw new Error('not a torch tensor archive');
  if (entries[pklName].method !== 0 || entries[dataName].method !== 0) throw new Error('compressed torch archives are unsupported');
  const meta = parsePickle(buf.subarray(entries[pklName].offset, entries[pklName].offset + entries[pklName].size));
  const numel = meta.shape.reduce((a, b) => a * b, 1);
  const bytes = BYTES[meta.dtype];
  if (!bytes) throw new Error(`unsupported dtype ${meta.dtype}`);
  const e = entries[dataName];
  if (numel * bytes > e.size) throw new Error(`storage too small: ${e.size} < ${numel * bytes}`);
  // Contiguity check: stride must be the row-major stride of shape.
  let expect = 1;
  for (let d = meta.shape.length - 1; d >= 0; d--) {
    if (meta.stride[d] !== expect) throw new Error('non-contiguous tensor unsupported');
    expect *= meta.shape[d];
  }
  // Copy into an aligned buffer (zip offsets are arbitrary).
  const raw = Buffer.alloc(numel * bytes);
  buf.copy(raw, 0, e.offset, e.offset + numel * bytes);
  if (meta.dtype === 'u8') return { dtype: 'u8', shape: meta.shape, data: new Uint8Array(raw.buffer, raw.byteOffset, numel) };
  const data = new Float32Array(numel);
  if (meta.dtype === 'f32') data.set(new Float32Array(raw.buffer, raw.byteOffset, numel));
  else if (meta.dtype === 'f64') data.set(new Float64Array(raw.buffer, raw.byteOffset, numel));
  else if (meta.dtype === 'f16') data.set(new Float16Array(raw.buffer, raw.byteOffset, numel));
  else { // bf16: upper 16 bits of an f32
    const u16 = new Uint16Array(raw.buffer, raw.byteOffset, numel);
    const u32 = new Uint32Array(data.buffer);
    for (let k = 0; k < numel; k++) u32[k] = u16[k] << 16;
  }
  return { dtype: meta.dtype, shape: meta.shape, data };
}
