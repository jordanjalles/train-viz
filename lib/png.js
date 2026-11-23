// Minimal PNG encoder for 8-bit RGB images (no dependencies; zlib from node).
// Used for z-image thumbnails and the thumbnail atlas of the Latent view.

import zlib from 'node:zlib';

function chunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
  const td = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4); crc.writeUInt32BE(zlib.crc32(td) >>> 0);
  return Buffer.concat([len, td, crc]);
}

/** rgb: Buffer/Uint8Array of w*h*3 bytes, row-major, interleaved. */
export function encodePng(w, h, rgb) {
  const raw = Buffer.alloc(h * (1 + 3 * w));
  for (let y = 0; y < h; y++) {
    raw[y * (1 + 3 * w)] = 0;   // filter: none
    Buffer.from(rgb.buffer, rgb.byteOffset + y * 3 * w, 3 * w).copy(raw, y * (1 + 3 * w) + 1);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;   // 8-bit RGB
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw, { level: 6 })), chunk('IEND', Buffer.alloc(0)),
  ]);
}

/** Area-average resample of an interleaved RGB image to s×s. */
export function resizeRgb(src, w, h, s) {
  const out = Buffer.alloc(s * s * 3);
  for (let y = 0; y < s; y++) {
    const y0 = (y * h) / s, y1 = ((y + 1) * h) / s;
    for (let x = 0; x < s; x++) {
      const x0 = (x * w) / s, x1 = ((x + 1) * w) / s;
      const acc = [0, 0, 0]; let wsum = 0;
      for (let sy = Math.floor(y0); sy < Math.ceil(y1); sy++) {
        const wy = Math.min(y1, sy + 1) - Math.max(y0, sy);
        for (let sx = Math.floor(x0); sx < Math.ceil(x1); sx++) {
          const wgt = wy * (Math.min(x1, sx + 1) - Math.max(x0, sx));
          const p = 3 * (sy * w + sx);
          acc[0] += src[p] * wgt; acc[1] += src[p + 1] * wgt; acc[2] += src[p + 2] * wgt; wsum += wgt;
        }
      }
      const q = 3 * (y * s + x);
      for (let k = 0; k < 3; k++) out[q + k] = Math.round(acc[k] / wsum);
    }
  }
  return out;
}
