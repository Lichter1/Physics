// Run with:  node --test test/
const test = require('node:test');
const assert = require('node:assert/strict');
const P = require('../src/js/physics.js');
const Q = require('../src/js/qcore.js');

const close = (a, b, rel, msg) => assert.ok(Math.abs(a - b) <= rel * Math.max(Math.abs(a), Math.abs(b)), `${msg || ''} got ${a}, want ${b}`);
const norm = v => Math.hypot(v[0], v[1], v[2]);
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

test('effort saturates below c and is Newtonian when c is huge', () => {
  close(P.speedFromEffort(8, P.REAL.c), 8, 1e-12);
  assert.ok(P.speedFromEffort(1000, 10) < 10);
  assert.ok(Number.isFinite(P.gamma(P.speedFromEffort(1e9, 10) / 10)));
  close(P.speedFromEffort(7, 10) / 10, Math.tanh(0.7), 1e-12);
});

test('seen direction of a static point follows the aberration formula', () => {
  const beta = 0.8, g = P.gamma(beta);
  for (const deg of [10, 60, 90, 135, 170]) {
    const th = deg * Math.PI / 180;
    const d0 = [30 * Math.sin(th), 0, 30 * Math.cos(th)];       // observer moves along +z
    const a = P.apparent(d0, [0, 0, 0], [0, 0, beta], 'seen');
    const cosSeen = a.rp[2] / norm(a.rp);
    close(cosSeen, (Math.cos(th) + beta) / (1 + beta * Math.cos(th)), 1e-12, 'aberration');
    close(a.D, g * (1 + beta * Math.cos(th)), 1e-12, 'doppler (world angle)');
    close(a.D, 1 / (g * (1 - beta * cosSeen)), 1e-12, 'doppler (seen angle)');
    // the photon still travels at c in the observer's frame
    close(norm(a.rp), g * (30 + beta * d0[2]), 1e-12, 'light travel distance');
  }
});

test('measured mode is plain Lorentz contraction', () => {
  const a = P.apparent([3, 2, 40], [0, 0, 0], [0, 0, 0.6], 'measured');
  close(a.rp[2], 40 / 1.25, 1e-12); close(a.rp[0], 3, 1e-12); close(a.rp[1], 2, 1e-12);
});

test('a moving point is seen on its own worldline and on the past light cone', () => {
  const d0 = [4, 1, 25], bu = [0.1, 0, -0.55], bv = [0.3, 0, 0.5];
  const a = P.apparent(d0, bu, bv, 'seen');
  close(norm(a.r), -a.cdt, 1e-12, 'light cone');
  for (let i = 0; i < 3; i++) close(a.r[i], d0[i] + bu[i] * a.cdt, 1e-12, 'worldline');
  // the interval between "now" and the event is null in the observer's frame as well
  const gv = P.gamma(norm(bv));
  const cdtPrime = gv * (a.cdt - dot(bv, a.r));
  close(norm(a.rp), -cdtPrime, 1e-12, 'null in the moving frame');
  // nothing changes when both move together
  const same = P.apparent(d0, bv, bv, 'seen');
  close(same.D, 1, 1e-12);
});

test('head-on Doppler factor is the textbook one', () => {
  const a = P.apparent([0, 0, 100], [0, 0, 0], [0, 0, 0.6], 'seen');
  close(a.D, Math.sqrt(1.6 / 0.4), 1e-12);
});

test('Schwarzschild limits of the Kerr formulas', () => {
  const k = P.kerr(1);
  close(k.rIsco, 6, 1e-12); close(k.rPhoton, 3, 1e-12); close(k.rHorizon, 2, 1e-12);
  close(P.kerrDilation(6, 0), Math.SQRT2, 1e-12);
  close(P.kerrDilation(1e9, 0), 1, 1e-8);
});

test("Interstellar: Miller's planet needs a = 1 - 1.3e-14 and gives ~61,000", () => {
  const k = P.kerr(1.3e-14);
  close(k.rIsco - 1, Math.cbrt(4 * 1.3e-14), 2e-3, 'near-extremal ISCO');
  const f = P.kerrDilation(k.rIsco, k.a);
  assert.ok(f > 5.5e4 && f < 6.8e4, `dilation ${f}`);
  // "one hour is seven years"
  const yearsPerHour = f * 3600 / P.K.year;
  assert.ok(yearsPerHour > 6.3 && yearsPerHour < 7.8, `years per hour ${yearsPerHour}`);
  // and it orbits Gargantua (1e8 suns) every ~1.7 hours of far-away time
  const tg = P.REAL.G * 1e8 * P.K.Msun / P.REAL.c ** 3;
  const hours = 2 * Math.PI / P.kerrOmega(k.rIsco, k.a) * tg / 3600;
  close(hours, 1.72, 0.03, 'orbital period');
});

test('Hawking temperature of a solar-mass black hole', () => {
  close(P.hawking(P.K.Msun, 1, P.REAL).temperature, 6.17e-8, 5e-3);
});

test('uniform gravity: one storey on Earth, and the slow-light town', () => {
  const g = P.earthGravity(P.REAL.G);
  close(g, 9.82, 2e-3);
  close(P.rindlerExcess(g, 1, P.REAL.c), 1.09e-16, 1e-2);            // the famous 1e-16 per metre
  close(P.rindlerRate(g, 30, 10), 1 + g * 0.3, 1e-12);
});

test('quantum scales', () => {
  close(P.bouncer(P.K.mn, 9.80665, P.REAL.hbar).z0, 5.87e-6, 2e-3);  // qBounce neutrons
  close(P.bouncer(P.K.mn, 9.80665, P.REAL.hbar).energies[0] / P.K.e, 1.407e-12, 5e-3);  // 1.41 peV
  close(P.deBroglie(0.43, 2, 0.1), 0.7306, 1e-3);
  close(P.fineStructure(P.REAL), 1 / 137.036, 1e-5);
  close(P.bohrRadius(P.REAL), 5.2918e-11, 1e-4);
  close(P.planck(P.REAL).length, 1.616e-35, 1e-3);
  close(P.planck(P.REAL).mass, 2.176e-8, 1e-3);
  // thick barrier: T ≈ 16 E(V-E)/V² · exp(-2κw)
  const E = 1, V = 2, w = 3, m = 1, h = 1, kap = Math.sqrt(2 * (V - E));
  close(P.barrierT(E, V, w, m, h), 16 * E * (V - E) / (V * V) * Math.exp(-2 * kap * w), 1e-3);
});

test('the real world sits at the Newtonian-gravity corner of the cube', () => {
  const r = P.regime(P.REAL);
  assert.equal(r.name, 'Newtonian gravity');
  assert.equal(P.regime({ c: 10, G: P.REAL.G, hbar: P.REAL.hbar }).name, 'General relativity');
  assert.equal(P.regime({ c: P.REAL.c, G: P.REAL.G, hbar: 0.1 }).name, 'Quantum mechanics in gravity');
  assert.equal(P.regime({ c: 10, G: P.REAL.G, hbar: 0.1 }).name, 'Quantum gravity');
  assert.equal(P.regime({ c: 10, G: P.REAL.G * 1e-3, hbar: P.REAL.hbar }).name, 'Special relativity');
});

test('FFT round trip and a known transform', () => {
  const f = new Q.FFT(64);
  const re = new Float64Array(64), im = new Float64Array(64);
  for (let i = 0; i < 64; i++) { re[i] = Math.cos(2 * Math.PI * 5 * i / 64) + 0.3 * Math.sin(i); im[i] = 0.1 * i; }
  const r0 = re.slice(), i0 = im.slice();
  f.forward(re, im); f.inverse(re, im);
  for (let i = 0; i < 64; i++) { assert.ok(Math.abs(re[i] - r0[i]) < 1e-12); assert.ok(Math.abs(im[i] - i0[i]) < 1e-12); }
  const a = new Float64Array(64), b = new Float64Array(64);
  for (let i = 0; i < 64; i++) a[i] = Math.cos(2 * Math.PI * 5 * i / 64);
  f.forward(a, b);
  close(a[5], 32, 1e-12); close(a[59], 32, 1e-12); assert.ok(Math.abs(a[4]) < 1e-9);
});

test('a free packet drifts at v, spreads as the textbook says, and keeps its norm', () => {
  const m = 0.43, hbar = 0.1, s0 = 0.3;
  const sim = new Q.Sim({ N: 128, L: 16, m, hbar });
  sim.gaussian(5, 8, s0, s0, 1.5, 0);
  close(sim.norm(), 1, 1e-12);
  const dt = 0.02; for (let i = 0; i < 100; i++) sim.step(dt);
  const mo = sim.moments();
  close(mo.norm, 1, 1e-10, 'norm');
  close(mo.x, 5 + 1.5 * 2, 2e-3, 'drift');
  close(mo.sx, P.spread(s0, 2, m, hbar), 5e-3, 'spread x');
  close(mo.sy, P.spread(s0, 2, m, hbar), 5e-3, 'spread y');
});

test('walls reflect and conserve probability; absorbers collect it on the screen', () => {
  const m = 0.43, hbar = 0.1, L = 8;
  const sim = new Q.Sim({ N: 128, L, m, hbar });
  sim.build((x, y) => ({ wall: x > 5 && x < 5.4 }));
  sim.gaussian(2.5, 4, 0.4, 0.8, 2, 0);
  const E = 0.5 * m * 4, dt = 0.1 * hbar / E;
  for (let i = 0; i < Math.round(1.6 / dt); i++) sim.step(dt);
  close(sim.norm(), 1, 1e-9, 'unitary with walls');
  assert.ok(sim.probRightOf(5.4) < 1e-3, 'nothing leaks through a thick wall: ' + sim.probRightOf(5.4));
  assert.ok(sim.moments().x < 4.6, 'packet came back');

  const s2 = new Q.Sim({ N: 128, L, m, hbar });
  s2.screenFrom = Math.round(7 / s2.dx);
  s2.build((x) => ({ damp: x > 7 ? 12 * (x - 7) * (x - 7) / 1 * 4 : 0 }));
  s2.gaussian(3, 4, 0.4, 0.8, 2, 0);
  for (let i = 0; i < Math.round(3.5 / dt); i++) s2.step(dt);
  const onScreen = s2.screen.reduce((a, b) => a + b, 0);
  assert.ok(onScreen > 0.95, 'absorbed on the screen: ' + onScreen);
  close(onScreen + s2.norm(), 1, 2e-2, 'absorbed + remaining');
});

test('tunnelling through a fence is close to the plane-wave formula', () => {
  const m = 0.43, hbar = 0.1, L = 8, v = 2, E = 0.5 * m * v * v, V0 = 1.2 * E, w = 0.25;
  const sim = new Q.Sim({ N: 128, L, m, hbar });
  sim.build((x) => ({ V: (x > 4 && x < 4 + w) ? V0 : 0 }));
  sim.gaussian(2, 4, 0.7, 1.2, v, 0);
  const dt = 0.1 * hbar / E;
  for (let i = 0; i < Math.round(1.9 / dt); i++) sim.step(dt);
  const T = sim.probRightOf(4 + w), want = P.barrierT(E, V0, w, m, hbar);
  assert.ok(Math.abs(T - want) < 0.05, `T=${T} formula=${want}`);
});

test('Airy shape has its first zeros in the right place', () => {
  const xs = []; for (let x = -6; x <= 3; x += 0.001) xs.push(x);
  const ai = Q.airyShape(xs), zeros = [];
  for (let i = xs.length - 2; i >= 0; i--) if (ai[i] * ai[i + 1] < 0) zeros.push(xs[i]);
  close(-zeros[0], P.AIRY_ZEROS[0], 1e-3); close(-zeros[1], P.AIRY_ZEROS[1], 1e-3); close(-zeros[2], P.AIRY_ZEROS[2], 1e-3);
});

test('a ball dropped on a floor: the 1-D solver follows the classical bounce and keeps its norm', () => {
  const m = 0.43, hbar = 0.1, g = 9.82, L = 4, zf = 0.5, h0 = 2.0;
  const sim = new Q.Sim1D({ N: 1024, L, m, hbar });
  sim.build(z => ({ V: m * g * (z - zf), wall: z < zf }));
  sim.set(z => [Math.exp(-((z - zf - h0) ** 2) / (4 * 0.2 * 0.2)), 0]);
  const Emax = m * g * h0, dt = 0.02 * hbar / Emax;
  const tFall = Math.sqrt(2 * h0 / g);
  let n = Math.round(0.5 * tFall / dt); for (let i = 0; i < n; i++) sim.step(dt);
  close(sim.mean() - zf, h0 - 0.5 * g * sim.time ** 2, 2e-2, 'free fall (Ehrenfest)');
  n = Math.round(1.5 * tFall / dt); for (let i = 0; i < n; i++) sim.step(dt);    // through the bounce and back up
  close(sim.norm(), 1, 1e-9, 'norm after a bounce');
  assert.ok(sim.mean() - zf > 1.5, 'came back up to ' + (sim.mean() - zf));
});

test('the Airy ground state of the bouncer is stationary', () => {
  const m = 0.43, hbar = 0.3, g = 9.82, L = 4, zf = 0.5;
  const b = P.bouncer(m, g, hbar);
  const sim = new Q.Sim1D({ N: 1024, L, m, hbar });
  sim.build(z => ({ V: m * g * (z - zf), wall: z < zf }));
  const zs = []; for (let i = 0; i < 1024; i++) zs.push((sim.z(i) - zf) / b.z0 - P.AIRY_ZEROS[0]);
  const ai = Q.airyShape(zs);
  sim.set(z => { const i = Math.round(z / sim.dz - 0.5); return [z > zf ? ai[i] : 0, 0]; });
  const mean0 = sim.mean() - zf;
  close(mean0, (2 / 3) * b.heights[0], 2e-2, 'mean height = 2/3 of the turning point');
  const dt = 0.005 * hbar / b.energies[0];       // wall = 200 E1: hard enough to hold the Airy state
  let maxDev = 0; for (let i = 0; i < 4000; i++) { sim.step(dt); maxDev = Math.max(maxDev, Math.abs(sim.mean() - zf - mean0)); }
  assert.ok(maxDev < 0.05 * mean0, 'hovers in place, drift ' + maxDev / mean0);
});
