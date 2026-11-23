/* cGħ Lab — physics core.
 * Pure functions, no DOM. Loaded as a plain script in the page (window.CGH.physics)
 * and with require() in the Node tests.
 * All quantities are SI unless a comment says otherwise.
 */
(function (root) {
  'use strict';

  var REAL = { c: 299792458, G: 6.6743e-11, hbar: 1.054571817e-34 };

  var K = {
    kB: 1.380649e-23, e: 1.602176634e-19, eps0: 8.8541878128e-12,
    me: 9.1093837015e-31, mp: 1.67262192369e-27, mn: 1.67492749804e-27,
    Mearth: 5.9722e24, Rearth: 6.371e6, Msun: 1.98847e30,
    AU: 1.495978707e11, year: 3.15576e7, day: 86400
  };

  // The everyday objects every scene refers to.
  var REF = {
    walk: 1.4,      // m/s, brisk walk
    bike: 8,        // m/s, easy cycling
    person: 70,     // kg
    ball: 0.43,     // kg, a football
    roll: 2,        // m/s, a gently kicked ball
    floor: 3,       // m, one storey
    door: 0.9       // m
  };

  /* ---------- special relativity ---------- */

  function gamma(beta) { return 1 / Math.sqrt(1 - beta * beta); }

  // "Effort" is the speed the same push would give you in Newton's world (rapidity × c).
  // The speed you actually reach saturates below c (capped at 0.9999 c so γ stays finite: γ ≤ 71).
  var BETA_MAX = 0.9999;
  function speedFromEffort(effort, c) { return c * Math.min(Math.tanh(effort / c), BETA_MAX); }

  // 1 - 1/γ without cancellation: the fraction of time a moving clock loses.
  function clockLoss(beta) { var s = Math.sqrt(1 - beta * beta); return beta * beta / (1 + s); }

  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

  /* Where and when a point is seen. This is the same arithmetic the vertex shader does;
   * it lives here too so it can be tested and used for the head-up display.
   *
   *   d0   position of the point relative to the observer at the observer's "now" (world frame)
   *   bu   the point's velocity / c in the world frame
   *   bv   the observer's velocity / c in the world frame
   *   mode 'seen'      the event on the observer's past light cone (what a camera records)
   *        'measured'  the event simultaneous with "now" in the observer's own frame
   *        'newton'    no relativity at all
   *
   * Returns r (world-frame offset of the event), cdt (c × event time relative to now),
   * rp (the same offset in the observer's frame, which is what gets drawn) and
   * D (received / emitted frequency; only meaningful for 'seen').
   */
  function apparent(d0, bu, bv, mode) {
    var bu2 = dot(bu, bu), bv2 = dot(bv, bv);
    var gu = 1 / Math.sqrt(1 - bu2), gv = 1 / Math.sqrt(1 - bv2);
    var r, cdt;
    if (mode === 'seen') {
      var a = 1 - bu2, b = dot(d0, bu);
      var cs = (-b + Math.sqrt(b * b + a * dot(d0, d0))) / a;   // c × light travel time
      r = [d0[0] - bu[0] * cs, d0[1] - bu[1] * cs, d0[2] - bu[2] * cs];
      cdt = -cs;
    } else if (mode === 'measured') {
      cdt = dot(bv, d0) / (1 - dot(bv, bu));
      r = [d0[0] + bu[0] * cdt, d0[1] + bu[1] * cdt, d0[2] + bu[2] * cdt];
    } else {
      r = d0.slice(); cdt = 0;
    }
    var rp = r.slice();
    if (mode !== 'newton' && bv2 > 0) {
      var inv = 1 / Math.sqrt(bv2);
      var vh = [bv[0] * inv, bv[1] * inv, bv[2] * inv];
      var par = dot(r, vh);
      for (var i = 0; i < 3; i++) rp[i] = r[i] + (gv - 1) * par * vh[i] - gv * bv[i] * cdt;
    }
    var D = 1;
    if (mode === 'seen') {
      var rl = Math.sqrt(dot(r, r)) || 1e-30;
      var rh = [r[0] / rl, r[1] / rl, r[2] / rl];
      D = gv * (1 + dot(rh, bv)) / (gu * (1 + dot(rh, bu)));
    }
    return { r: r, cdt: cdt, rp: rp, D: D };
  }

  /* ---------- gravity ---------- */

  // Surface gravity of an Earth with its real mass and radius under a different G.
  function earthGravity(G) { return G * K.Mearth / (K.Rearth * K.Rearth); }

  // Uniform gravity (the frame of a rocket accelerating at g): a clock at height h
  // runs faster than the floor clock by exactly this factor.
  function rindlerRate(g, h, c) { return 1 + g * h / (c * c); }
  // The same thing minus one, for when the difference is 1e-16 and would vanish in the sum.
  function rindlerExcess(g, h, c) { return g * h / (c * c); }

  // Depth below the floor at which clocks stop, as judged from the floor.
  function rindlerHorizon(g, c) { return c * c / g; }

  /* Kerr black hole, prograde circular orbits in the equatorial plane.
   * Lengths are in units of GM/c², times in GM/c³. Spin is a = 1 - delta, 0 <= a < 1;
   * delta is the argument because the film's Gargantua needs delta ~ 1e-14.
   */
  function kerr(delta) {
    var a = 1 - delta;
    var oma2 = delta * (2 - delta);                 // 1 - a²
    var s = Math.sqrt(oma2);
    var z1 = 1 + Math.cbrt(oma2) * (Math.cbrt(2 - delta) + Math.cbrt(delta));
    var z2 = Math.sqrt(3 * a * a + z1 * z1);
    return {
      a: a, delta: delta,
      rHorizon: 1 + s,
      rIsco: 3 + z2 - Math.sqrt((3 - z1) * (3 + z1 + 2 * z2)),
      rPhoton: 2 * (1 + Math.cos((2 / 3) * Math.acos(-a))),
      kappa: s / (2 * (1 + s))                       // horizon surface gravity in units of c⁴/(GM)
    };
  }

  // dt/dτ: far-away seconds per second on the orbiting clock.
  function kerrDilation(r, a) {
    var x = Math.pow(r, -1.5);
    return (1 + a * x) / Math.sqrt(1 - 3 / r + 2 * a * x);
  }

  // Orbital angular velocity seen from far away, in units of c³/(GM).
  function kerrOmega(r, a) { return 1 / (Math.pow(r, 1.5) + a); }

  function hawking(M, delta, k) {
    var kk = kerr(delta);
    var T = k.hbar * Math.pow(k.c, 3) * kk.kappa / (2 * Math.PI * K.kB * k.G * M);
    return {
      temperature: T,
      // Schwarzschild, photons only: the standard order-of-magnitude lifetime.
      lifetime: 5120 * Math.PI * k.G * k.G * M * M * M / (k.hbar * Math.pow(k.c, 4)),
      power: k.hbar * Math.pow(k.c, 6) / (15360 * Math.PI * k.G * k.G * M * M)
    };
  }

  /* ---------- quantum ---------- */

  function deBroglie(m, v, hbar) { return 2 * Math.PI * hbar / (m * v); }

  // Width of a free Gaussian packet after time t.
  function spread(sigma0, t, m, hbar) {
    var x = hbar * t / (2 * m * sigma0 * sigma0);
    return sigma0 * Math.sqrt(1 + x * x);
  }

  // A ball on a floor under gravity: natural height scale and the allowed energies.
  var AIRY_ZEROS = [2.338107410, 4.087949444, 5.520559828, 6.786708090, 7.944133587, 9.022650853];
  function bouncer(m, g, hbar) {
    var z0 = Math.cbrt(hbar * hbar / (2 * m * m * g));
    return {
      z0: z0,
      heights: AIRY_ZEROS.map(function (a) { return a * z0; }),       // classical turning points
      energies: AIRY_ZEROS.map(function (a) { return a * m * g * z0; })
    };
  }

  // Plane-wave transmission through a rectangular barrier of height V0 and thickness w.
  function barrierT(E, V0, w, m, hbar) {
    if (V0 <= 0) return 1;
    if (Math.abs(E - V0) < 1e-12 * V0) {
      var k0w = Math.sqrt(2 * m * E) * w / hbar;
      return 1 / (1 + k0w * k0w / 4);
    }
    if (E < V0) {
      var kap = Math.sqrt(2 * m * (V0 - E)) / hbar;
      var sh = Math.sinh(kap * w);
      return 1 / (1 + V0 * V0 * sh * sh / (4 * E * (V0 - E)));
    }
    var kk = Math.sqrt(2 * m * (E - V0)) / hbar;
    var sn = Math.sin(kk * w);
    return 1 / (1 + V0 * V0 * sn * sn / (4 * E * (E - V0)));
  }

  /* ---------- all three ---------- */

  function planck(k) {
    return {
      length: Math.sqrt(k.hbar * k.G / Math.pow(k.c, 3)),
      mass: Math.sqrt(k.hbar * k.c / k.G),
      time: Math.sqrt(k.hbar * k.G / Math.pow(k.c, 5))
    };
  }
  function fineStructure(k) { return K.e * K.e / (4 * Math.PI * K.eps0 * k.hbar * k.c); }
  function bohrRadius(k) { return 4 * Math.PI * K.eps0 * k.hbar * k.hbar / (K.me * K.e * K.e); }

  function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
  function logLevel(x, lo, hi) { return clamp01((Math.log10(x) - lo) / (hi - lo)); }

  /* Where today's dial settings sit in Bronstein's cube of theories.
   * Each axis is 0 when that constant leaves no trace at human scale and 1 when it dominates.
   *   rel   a 5 m/s run compared with c
   *   grav  surface gravity compared with Earth's (so the real world sits at 1)
   *   quant a rolling football's wavelength compared with a metre
   */
  var CORNERS = [
    'Newtonian mechanics', 'Special relativity', 'Newtonian gravity', 'General relativity',
    'Quantum mechanics', 'Quantum field theory', 'Quantum mechanics in gravity', 'Quantum gravity'
  ];
  function regime(k) {
    var rel = logLevel(5 / k.c, -3, -0.3);
    var grav = logLevel(k.G / REAL.G, -3, 0);
    var quant = logLevel(deBroglie(REF.ball, REF.roll, k.hbar), -3, -0.3);
    var bits = (rel > 0.5 ? 1 : 0) | (grav > 0.5 ? 2 : 0) | (quant > 0.5 ? 4 : 0);
    return { rel: rel, grav: grav, quant: quant, corner: bits, name: CORNERS[bits] };
  }

  var api = {
    REAL: REAL, K: K, REF: REF, CORNERS: CORNERS, AIRY_ZEROS: AIRY_ZEROS,
    gamma: gamma, speedFromEffort: speedFromEffort, clockLoss: clockLoss, apparent: apparent,
    earthGravity: earthGravity, rindlerRate: rindlerRate, rindlerExcess: rindlerExcess, BETA_MAX: BETA_MAX, rindlerHorizon: rindlerHorizon,
    kerr: kerr, kerrDilation: kerrDilation, kerrOmega: kerrOmega, hawking: hawking,
    deBroglie: deBroglie, spread: spread, bouncer: bouncer, barrierT: barrierT,
    planck: planck, fineStructure: fineStructure, bohrRadius: bohrRadius,
    regime: regime, logLevel: logLevel, clamp01: clamp01
  };

  root.CGH = root.CGH || {};
  root.CGH.physics = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
