// Run with:  node --test test/
const test = require('node:test');
const assert = require('node:assert/strict');
const L = require('../src/js/labs.js');

test('Einstein’s orbit precesses by 3π rs / p each turn in a weak field; Newton’s does not', () => {
  const GM = 1, c = 100, rs = 2 * GM / (c * c), r0 = 1;
  const E = L.orbit(GM, c, r0, 0.9 * L.circularSpeed(GM, c, r0, true), true, { turns: 4, dt: 2 * Math.PI / 20000 });
  const N = L.orbit(GM, c, r0, 0.9 * L.circularSpeed(GM, c, r0, false), false, { turns: 4, dt: 2 * Math.PI / 20000 });
  let rp = Infinity, ra = 0;
  E.path.forEach(p => { const r = Math.hypot(p[0], p[1]); rp = Math.min(rp, r); ra = Math.max(ra, r); });
  const p = 2 * rp * ra / (rp + ra), want = 3 * Math.PI * rs / p;
  assert.ok(E.peri.length >= 3, 'several closest approaches');
  assert.ok(Math.abs(E.precess - want) < 0.03 * want, `precession ${E.precess} vs ${want}`);
  assert.ok(Math.abs(N.precess) < 0.02 * want, `Newton ${N.precess}`);
});

test('two staircases: the record and the stripes trade off exactly, and Δτ matches the weak-field estimate', () => {
  const R = 400, rs = 1e-3, c = 10, h = 20, T = 30;
  const f = r => 1 - rs / r;
  const s = L.staircases(f(R + h), f(R), T, 70, 30, c, 0.7);
  // weak field: √(1 − rs/(R+h)) − √(1 − rs/R) ≈ rs h / (2R(R+h)), i.e. g h / c² with g taken between the two heights
  assert.ok(Math.abs(s.dtau - T * rs * h / (2 * R * (R + h))) < 1e-3 * s.dtau, `Δτ ${s.dtau}`);
  for (const tick of [0, 0.3, 1.7, 5]) {
    const t = L.staircases(f(R + h), f(R), T, 70, 30, c, tick);
    assert.ok(Math.abs(t.D * t.D + t.V * t.V - 1) < 1e-12);
  }
  // a whole tick of difference brings the stripes back
  const full = L.staircases(f(R + h), f(R), T, 70, 30, c, 1 / s.dtau);
  assert.ok(full.V > 0.999, 'revival after one whole tick');
});
