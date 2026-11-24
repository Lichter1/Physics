/* cGħ World — you, as a wave.
 * Your position on the ground is a 2-D wavefunction, solved on a 32 m × 32 m window that follows you.
 * Everything standing on the ground is a wall. Your legs act as a uniform force: they shift the whole
 * momentum distribution toward the speed you are trying to walk at. Each frame the camera is placed at
 * one position drawn from |ψ|², and the screen keeps a running average: every place you might be.
 */
(function () {
  'use strict';
  var G = CGH.geo, QC = CGH.qcore;
  var N = 128, L = 32, M = 70, dx = L / N, EDGE = 2.5, BODY = 0.2, AMAX = 4;
  var W = { sim: null, ca: 0, cl: 0, hbar: 0, c: 0, on: false, cum: new Float64Array(N * N), spread: 0, mean: null };

  function toStreet(x, y) { return { a: G.wrapA(W.ca + y - L / 2), l: W.cl + x - L / 2 }; }
  function walls() {
    var s = W.sim;
    s.build(function (x, y) {
      var e = Math.min(x, L - x, y, L - y), p = toStreet(x, y);
      return { wall: G.footprint(p.a, p.l) < BODY, damp: e < EDGE ? 6 * Math.pow((EDGE - e) / EDGE, 2) : 0 };
    });
  }

  // Start (or restart) as a packet of width sigma at street point (a, l), moving at (va, vl).
  function start(a, l, va, vl, hbar, c, sigma) {
    W.hbar = hbar; W.c = c; W.ca = a; W.cl = l; W.on = true; W.vel = [va, vl];
    W.sim = new QC.Sim({ N: N, L: L, m: M, hbar: hbar, c: c });
    walls();
    W.sim.gaussian(L / 2, L / 2, sigma, sigma, 0, 0);
    var k = momentumK(va, vl);
    W.sim.kick(k[0], k[1]);
    measure();
  }
  function stop() { W.on = false; W.sim = null; }

  // Wavenumber of someone walking at (va, vl): p = γ m v, capped well inside what the grid can hold.
  function momentumK(va, vl) {
    var v2 = va * va + vl * vl, g = W.c ? 1 / Math.sqrt(Math.max(1e-6, 1 - v2 / (W.c * W.c))) : 1;
    var kx = g * M * vl / W.hbar, ky = g * M * va / W.hbar, kk = Math.hypot(kx, ky), cap = 0.45 * Math.PI / dx;
    return kk > cap ? [kx * cap / kk, ky * cap / kk] : [kx, ky];
  }
  function topSpeed() {                                   // the fastest walk the window can represent
    var p = 0.45 * Math.PI / dx * W.hbar;
    return W.c ? p * W.c / Math.sqrt(p * p + M * M * W.c * W.c) : p / M;
  }

  function measure() {
    var m = W.sim.moments(), p = toStreet(m.x, m.y);
    W.mean = p; W.spread = Math.hypot(m.sx, m.sy);
    return m;
  }

  // Advance by dt seconds of your own time while trying to walk at (va, vl).
  function step(dt, va, vl) {
    var s = W.sim, k = s.meanK(), kt = momentumK(va, vl);
    var lim = M * AMAX * dt / W.hbar, dkx = kt[0] - k[0], dky = kt[1] - k[1], dk = Math.hypot(dkx, dky);
    if (dk > lim) { dkx *= lim / dk; dky *= lim / dk; }
    if (dk > 1e-9) s.kick(dkx, dky);
    // group velocity of the mean momentum: c²p/E, or p/m without relativity
    var kx = k[0] + dkx, ky = k[1] + dky, px = W.hbar * kx, py = W.hbar * ky;
    var inv = W.c ? W.c * W.c / Math.sqrt((px * px + py * py) * W.c * W.c + Math.pow(M * W.c * W.c, 2)) : 1 / M;
    W.vel = [py * inv, px * inv];
    // walls must stand far above your energy: their height is wallPhase·ħ/dt, so take short steps
    var v2 = va * va + vl * vl, E = 0.5 * M * Math.max(v2, 1) + W.hbar * W.hbar / (2 * M * 0.25);
    var n = Math.max(1, Math.min(6, Math.ceil(dt * 10 * E / W.hbar)));
    for (var i = 0; i < n; i++) s.step(dt / n);
    s.normalise();
    var m = measure();
    // keep the wave in the middle of the window
    var di = Math.round((m.x - L / 2) / dx), dj = Math.round((m.y - L / 2) / dx);
    if (Math.abs(di) > 12 || Math.abs(dj) > 12) {
      s.shift(di, dj); W.cl += di * dx; W.ca = G.wrapA(W.ca + dj * dx); walls(); measure();
    }
  }

  // One place you might be, drawn from |ψ|².
  function sample(rand) {
    var re = W.sim.re, im = W.sim.im, cum = W.cum, tot = 0, q;
    for (q = 0; q < re.length; q++) { tot += re[q] * re[q] + im[q] * im[q]; cum[q] = tot; }
    var t = rand() * tot, lo = 0, hi = re.length - 1;
    while (lo < hi) { var mid = (lo + hi) >> 1; if (cum[mid] < t) lo = mid + 1; else hi = mid; }
    var i = lo % N, j = (lo - i) / N;
    return toStreet((i + rand()) * dx, L - (j + rand()) * dx);
  }

  // Find out where you are: keep only the part of the wave near one outcome.
  function look(sigma, rand) {
    var spot = W.sim.measure(sigma, rand);
    measure();
    return toStreet(spot.x, spot.y);
  }

  CGH.wave = { W: W, N: N, L: L, M: M, dx: dx, start: start, stop: stop, step: step, sample: sample, look: look, topSpeed: topSpeed, toStreet: toStreet };
})();
