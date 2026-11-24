// Run with:  node --test test/
const test = require('node:test');
const assert = require('node:assert/strict');
require('../src/js/geo.js'); require('../src/js/qcore.js'); require('../src/js/wave.js');
const WV = globalThis.CGH.wave, G = globalThis.CGH.geo;
const HBAR = 2 * 70 * 1.4 / (2 * Math.PI);                     // a 2 m wavelength at walking pace

function walk(bell, seconds) {
  WV.start(G.DOORS.a, 0, -1.4, 0, HBAR, 0, 0.6);
  WV.W.bell = bell;
  for (let t = 0; t < seconds; t += 1 / 30) WV.step(1 / 30, -1.4, 0);
}
// Fringe contrast on a line 4.5 m behind the wall, across the middle of the street.
function contrast() {
  const W = WV.W, vals = [];
  for (let l = -4; l <= 4.01; l += 0.25) {
    const g = WV.toGrid(G.WALL.a - 4.5, l), i = Math.floor(g.x / WV.dx), j = Math.floor((WV.L - g.y) / WV.dx), q = j * WV.N + i;
    vals.push(W.sim.re[q] ** 2 + W.sim.im[q] ** 2 + (W.sim1 ? W.sim1.re[q] ** 2 + W.sim1.im[q] ** 2 : 0));
  }
  const mx = Math.max(...vals), mn = Math.min(...vals);
  return (mx - mn) / (mx + mn);
}
let seed = 7;
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);

test('a bell at one door washes out the stripes, and erasing its record brings them back', () => {
  walk(0, 12); const v0 = contrast();
  walk(1, 12); const v1 = contrast(), rang = WV.W.rang;
  WV.erase(rand); const v2 = contrast();
  assert.ok(rang > 0.05, `some of the wave rang the bell (${rang})`);
  assert.ok(v0 > 0.85, `stripes without the bell: ${v0}`);
  assert.ok(v1 < v0 - 0.15, `the bell washes them out: ${v1} vs ${v0}`);
  assert.ok(v2 > v1 + 0.1, `erasing restores them: ${v2} vs ${v1}`);
});

test('a sharp look narrows your position and widens your momentum; Δx·Δp never beats ħ/2', () => {
  WV.start(0, 0, 0, 0, HBAR, 0, 1.0);
  const before = WV.kstats();
  for (const sigma of [1.0, 0.25]) {
    WV.look(sigma, rand);
    const m = WV.W.pos, k = WV.kstats();
    const product = m.sx * HBAR * k.dk[0] / (HBAR / 2);
    assert.ok(product >= 0.99, `Δx·Δp = ${product} × ħ/2`);
    if (sigma === 0.25) assert.ok(k.dk[0] > 2 * before.dk[0], 'a sharp look kicks: momentum spread grows');
  }
  // looking again at once gives (almost) the same place: the state was updated, not reset
  const a = WV.look(0.25, rand), b = WV.look(0.25, rand);
  assert.ok(Math.hypot(a.a - b.a, a.l - b.l) < 1.2, 'repeat look lands nearby');
});
