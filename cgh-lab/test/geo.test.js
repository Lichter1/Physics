// Run with:  node --test test/
const test = require('node:test');
const assert = require('node:assert/strict');
const G = require('../src/js/geo.js');

const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg || ''} got ${a}, want ${b} (±${tol})`);

test('street coordinates round-trip and the local frame is orthonormal', () => {
  for (const [a, l, h] of [[0, 0, 0], [-40, 3, 1.7], [700, -20, 30], [-1150, 9, 5]]) {
    const p = G.fromWorld(G.toWorld(a, l, h));
    close(p.a, a, 1e-9); close(p.l, l, 1e-9); close(p.h, h, 1e-9);
    const f = G.frame(a, l), e = 1e-4;
    for (const v of [f.up, f.along, f.across]) close(G.len(v), 1, 1e-12);
    close(G.dot(f.up, f.along), 0, 1e-12); close(G.dot(f.up, f.across), 0, 1e-12); close(G.dot(f.along, f.across), 0, 1e-12);
    // the frame points the way the coordinates grow (along is tilted by l slightly; compare directions)
    const p0 = G.toWorld(a, l, 0), pa = G.toWorld(a + e, l, 0), pl = G.toWorld(a, l + e, 0);
    const da = pa.map((x, i) => x - p0[i]), dl = pl.map((x, i) => x - p0[i]);
    close(G.dot(da, f.along) / G.len(da), 1, 1e-6); close(G.dot(dl, f.across) / G.len(dl), 1, 1e-6);
  }
  // at the start, +l is to the right of someone facing +a with +y up
  const f = G.frame(0, 0), right = G.cross(f.along, f.up);
  close(G.dot(right, f.across), 1, 1e-12);
});

test('a distant ray bends by 2rs/b plus the second-order term', () => {
  const rs = 1, b = 1000, D = 1e5;
  const res = G.trace([-D, b, 0], [1, 0, 0], rs, { escape: D, maxStep: 0.002 });
  const bend = Math.atan2(-res.v[1], res.v[0]);
  const want = 2 * rs / b + (15 * Math.PI / 16) * (rs / b) ** 2;
  close(bend, want, 2e-6, 'deflection');
});

test('a tangential ray at 1.5 rs circles (the photon sphere)', () => {
  const rs = 1, x = [1.5, 0, 0], v = G.launch(x, [0, 1, 0], rs);
  close(G.len(v), Math.sqrt(3), 1e-12, 'tangential speed in coordinates');
  const res = G.trace(x, v, rs, { maxStep: 0.001, steps: 1571 });
  close(G.len(res.x), 1.5, 2e-3, 'radius after a quarter turn');
  close(Math.atan2(res.x[1], res.x[0]), Math.PI / 2, 0.01, 'angle swept');
});

test('a radial ray takes the Shapiro time', () => {
  const rs = 10, r1 = 20, res = G.trace([r1, 0, 0], G.launch([r1, 0, 0], [1, 0, 0], rs), rs, { escape: 1000, maxStep: 0.001 });
  const r2 = G.len(res.x);
  close(res.ct, r2 - r1 + rs * Math.log((r2 - rs) / (r1 - rs)), 1e-4 * r2);
});

test('the sky edge from the formula matches traced rays', () => {
  const Rp = G.R, re = Rp + G.EYE;
  for (const comp of [0, 0.3, 0.6, 0.8]) {
    const rs = comp * Rp, x = [0, re, 0], up = [0, 1, 0], fwd = [0, 0, 1];
    const hitsSky = (e) => {
      const n = [0, Math.sin(e), Math.cos(e)];
      return G.trace(x, G.launch(x, n, rs), rs, { ground: Rp, escape: 30 * Rp, maxStep: 0.0005 }).end === 'sky';
    };
    let lo = -Math.PI / 2 + 1e-6, hi = Math.PI / 2 - 1e-6;                 // lo: ground, hi: sky
    for (let i = 0; i < 40; i++) { const m = (lo + hi) / 2; if (hitsSky(m)) hi = m; else lo = m; }
    close((lo + hi) / 2, G.horizon(rs, Rp, re), 2e-3, `compactness ${comp}`);
    void up; void fwd;
  }
});

test('footprints: open road, the wall between the doors, the doors, a house, and the wrap', () => {
  assert.ok(G.footprint(G.START.a, G.START.l) > 1, 'start is clear');
  assert.ok(G.footprint(G.WALL.a, 0) < 0, 'wall between the doors');
  assert.ok(G.footprint(G.WALL.a, G.WALL.door) > 0.4, 'right door open');
  assert.ok(G.footprint(G.WALL.a, -G.WALL.door) > 0.4, 'left door open');
  assert.ok(G.footprint(10, 16) < 0, 'inside a house');
  assert.ok(G.footprint(G.TOWER.a, 0) < 0, 'inside the tower');
  close(G.footprint(10 + 2 * Math.PI * G.R, 7), G.footprint(10, 7), 1e-9, 'wraps round the planet');
  assert.ok(G.footprint(G.WALL.a + 120, 0) > 0, 'the wall is only in the first block');
});

test('clocks higher up run faster, by the Schwarzschild factor', () => {
  close(G.clockRate(0, 50), 1, 1e-15);
  const rs = 0.6 * G.R, want = Math.sqrt((1 - rs / (G.R + 50)) / (1 - rs / (G.R + G.EYE)));
  close(G.clockRate(rs, 50), want, 1e-15);
  assert.ok(want > 1.075 && want < 1.085, `tower top runs ${want}× faster`);
});
