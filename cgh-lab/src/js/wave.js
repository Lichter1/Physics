/* cGħ World — you, as a wave, and what it means to measure you.
 *
 * Your position on the ground is a 2-D wavefunction on a 32 m × 32 m window that follows you.
 * Everything standing on the ground is a wall. Your legs act as a uniform force (they shift the
 * whole momentum distribution). Each frame the camera stands at one place drawn from |ψ|².
 *
 * Measurement, three ways:
 *  - look(σ): a position measurement of width σ. The outcome is random with probability |ψ|²; the wave
 *    is then multiplied by a Gaussian of width σ around it. A narrow look gives a wide momentum spread.
 *  - the bell: a detector at the left door. Passing near it rotates your state from "bell silent" (ψ0)
 *    into "bell rang" (ψ1). Nobody has to read the bell: the two parts no longer interfere.
 *  - erase(): measure the bell in the basis (silent ± rang)/√2. That removes the record and brings the
 *    stripes back (shifted, for the minus outcome).
 */
(function () {
  'use strict';
  var G = CGH.geo, QC = CGH.qcore;
  var N = 128, L = 32, M = 70, dx = L / N, EDGE = 2.5, BODY = 0.2, AMAX = 4;
  var BELL = { a: G.WALL.a, l: G.WALL.door, ha: 0.7, hl: 0.75 };
  var W = { sim: null, sim1: null, ca: 0, cl: 0, hbar: 0, c: 0, on: false, bell: 0, rang: 0,
            cum: new Float64Array(N * N), spread: 0, mean: null, vel: [0, 0], marks: [], saved: null, stats: null };

  function toStreet(x, y) { return { a: G.wrapA(W.ca + y - L / 2), l: W.cl + x - L / 2 }; }
  function toGrid(a, l) { return { x: l - W.cl + L / 2, y: G.wrapA(a - W.ca) + L / 2 }; }
  function both(fn) { fn(W.sim); if (W.sim1) fn(W.sim1); }
  function walls() {
    var build = function (x, y) {
      var e = Math.min(x, L - x, y, L - y), p = toStreet(x, y);
      return { wall: G.footprint(p.a, p.l) < BODY, damp: e < EDGE ? 6 * Math.pow((EDGE - e) / EDGE, 2) : 0 };
    };
    both(function (s) { s.build(build); });
  }

  function start(a, l, va, vl, hbar, c, sigma) {
    W.hbar = hbar; W.c = c; W.ca = a; W.cl = l; W.on = true; W.vel = [va, vl]; W.sim1 = null; W.rang = 0; W.saved = null;
    W.sim = new QC.Sim({ N: N, L: L, m: M, hbar: hbar, c: c });
    walls();
    W.sim.gaussian(L / 2, L / 2, sigma, sigma, 0, 0);
    var k = momentumK(va, vl);
    W.sim.kick(k[0], k[1]);
    measure();
  }
  function stop() { W.on = false; W.sim = null; W.sim1 = null; }

  function momentumK(va, vl) {
    var v2 = va * va + vl * vl, g = W.c ? 1 / Math.sqrt(Math.max(1e-6, 1 - v2 / (W.c * W.c))) : 1;
    var kx = g * M * vl / W.hbar, ky = g * M * va / W.hbar, kk = Math.hypot(kx, ky), cap = 0.45 * Math.PI / dx;
    return kk > cap ? [kx * cap / kk, ky * cap / kk] : [kx, ky];
  }
  function topSpeed() {
    var p = 0.45 * Math.PI / dx * W.hbar;
    return W.c ? p * W.c / Math.sqrt(p * p + M * M * W.c * W.c) : p / M;
  }

  // Position statistics over both parts of the wave.
  function measure() {
    var m0 = W.sim.moments(), n0 = m0.norm, m = m0;
    if (W.sim1) {
      var m1 = W.sim1.moments(), n1 = m1.norm, t = n0 + n1 || 1;
      var x = (m0.x * n0 + m1.x * n1) / t, y = (m0.y * n0 + m1.y * n1) / t;
      var vx = ((m0.sx * m0.sx + m0.x * m0.x) * n0 + (m1.sx * m1.sx + m1.x * m1.x) * n1) / t - x * x;
      var vy = ((m0.sy * m0.sy + m0.y * m0.y) * n0 + (m1.sy * m1.sy + m1.y * m1.y) * n1) / t - y * y;
      m = { x: x, y: y, sx: Math.sqrt(Math.max(vx, 0)), sy: Math.sqrt(Math.max(vy, 0)), norm: t };
      W.rang = n1 / t;
    }
    W.mean = toStreet(m.x, m.y); W.spread = Math.hypot(m.sx, m.sy); W.pos = m;
    return m;
  }
  // Momentum statistics over both parts: mean wavenumber and spreads.
  function kstats() {
    var a = W.sim.kStats(), out = a;
    if (W.sim1) {
      var b = W.sim1.kStats(), t = a[4] + b[4] || 1;
      out = [0, 1, 2, 3].map(function (i) { return (a[i] * a[4] + b[i] * b[4]) / t; });
    }
    return { k: [out[0], out[1]], dk: [Math.sqrt(Math.max(0, out[2] - out[0] * out[0])), Math.sqrt(Math.max(0, out[3] - out[1] * out[1]))] };
  }
  function normalise() {
    var t = W.sim.norm() + (W.sim1 ? W.sim1.norm() : 0), f = 1 / Math.sqrt(t || 1);
    both(function (s) { for (var q = 0; q < s.re.length; q++) { s.re[q] *= f; s.im[q] *= f; } });
  }

  // The bell: inside the zone by the left door, rotate (silent, rang) at a rate set by the walking speed,
  // so that walking through it at a steady pace turns `strength` of a full record.
  function ring(dt, speed) {
    if (!W.bell) return;
    if (!W.sim1) { W.sim1 = new QC.Sim({ N: N, L: L, m: M, hbar: W.hbar, c: W.c }); walls(); W.sim1.re.fill(0); W.sim1.im.fill(0); }
    var g = toGrid(BELL.a, BELL.l);
    var i0 = Math.max(0, Math.floor((g.x - BELL.hl) / dx)), i1 = Math.min(N - 1, Math.ceil((g.x + BELL.hl) / dx));
    var y0 = g.y - BELL.ha, y1 = g.y + BELL.ha;
    if (i1 < 0 || i0 > N - 1 || y1 < 0 || y0 > L) return;
    var th = W.bell * (Math.PI / 2) * Math.max(speed, 0.8) * dt / (2 * BELL.ha), c = Math.cos(th), s = Math.sin(th);
    var a = W.sim, b = W.sim1;
    for (var j = 0; j < N; j++) {
      var y = a.y(j); if (y < y0 || y > y1) continue;
      for (var i = i0; i <= i1; i++) {
        var q = j * N + i, ar = a.re[q], ai = a.im[q], br = b.re[q], bi = b.im[q];
        a.re[q] = c * ar - s * br; a.im[q] = c * ai - s * bi;
        b.re[q] = s * ar + c * br; b.im[q] = s * ai + c * bi;
      }
    }
  }

  function step(dt, va, vl) {
    var st = kstats(), k = st.k, kt = momentumK(va, vl);
    var lim = M * AMAX * dt / W.hbar, dkx = kt[0] - k[0], dky = kt[1] - k[1], dk = Math.hypot(dkx, dky);
    if (dk > lim) { dkx *= lim / dk; dky *= lim / dk; }
    if (dk > 1e-9) both(function (s) { s.kick(dkx, dky); });
    var kx = k[0] + dkx, ky = k[1] + dky, px = W.hbar * kx, py = W.hbar * ky;
    var inv = W.c ? W.c * W.c / Math.sqrt((px * px + py * py) * W.c * W.c + Math.pow(M * W.c * W.c, 2)) : 1 / M;
    W.vel = [py * inv, px * inv];
    W.stats = { dk: st.dk };
    var v2 = va * va + vl * vl, E = 0.5 * M * Math.max(v2, 1) + W.hbar * W.hbar / (2 * M * 0.25);
    var n = Math.max(1, Math.min(6, Math.ceil(dt * 10 * E / W.hbar)));
    for (var i = 0; i < n; i++) { both(function (s) { s.step(dt / n); }); ring(dt / n, Math.hypot(va, vl)); }
    normalise();
    var m = measure();
    var di = Math.round((m.x - L / 2) / dx), dj = Math.round((m.y - L / 2) / dx);
    if (Math.abs(di) > 12 || Math.abs(dj) > 12) {
      both(function (s) { s.shift(di, dj); });
      W.cl += di * dx; W.ca = G.wrapA(W.ca + dj * dx); walls(); measure();
    }
  }

  // The yard behind the two doors: where a detector screen would catch you.
  function behind(x, y) { var p = toStreet(x, y); return p.a < G.WALL.a - G.WALL.half - 0.3 && p.a > -38 && Math.abs(p.l) < 10; }
  function chanceBehind() {
    var a = W.sim, b = W.sim1, s = 0, t = 0;
    for (var j = 0; j < N; j++) {
      var y = a.y(j), rowIn = behind(L / 2, y), rs = 0;
      for (var i = 0; i < N; i++) {
        var q = j * N + i, p = a.re[q] * a.re[q] + a.im[q] * a.im[q] + (b ? b.re[q] * b.re[q] + b.im[q] * b.im[q] : 0);
        t += p; if (rowIn && behind(a.x(i), y)) rs += p;
      }
      s += rs;
    }
    return t ? s / t : 0;
  }
  // Draw one place from the combined probability |ψ0|² + |ψ1|², optionally only from the yard behind the doors.
  function pick(rand, screen) {
    var a = W.sim, b = W.sim1, cum = W.cum, tot = 0, q;
    for (q = 0; q < cum.length; q++) {
      var i0 = q % N, j0 = (q - i0) / N;
      if (!screen || behind(a.x(i0), a.y(j0))) {
        tot += a.re[q] * a.re[q] + a.im[q] * a.im[q];
        if (b) tot += b.re[q] * b.re[q] + b.im[q] * b.im[q];
      }
      cum[q] = tot;
    }
    if (tot <= 1e-12) return null;
    var t = rand() * tot, lo = 0, hi = cum.length - 1;
    while (lo < hi) { var mid = (lo + hi) >> 1; if (cum[mid] < t) lo = mid + 1; else hi = mid; }
    var i = lo % N, j = (lo - i) / N;
    return { x: (i + rand()) * dx, y: L - (j + rand()) * dx };
  }
  function sample(rand) { var p = pick(rand); return toStreet(p.x, p.y); }

  function snapshot() {
    var copy = function (s) { return s ? { re: Float64Array.from(s.re), im: Float64Array.from(s.im) } : null; };
    return { s0: copy(W.sim), s1: copy(W.sim1), ca: W.ca, cl: W.cl };
  }
  function restore(snap) {
    W.ca = snap.ca; W.cl = snap.cl; walls();
    W.sim.re = Float64Array.from(snap.s0.re); W.sim.im = Float64Array.from(snap.s0.im);
    if (snap.s1) {
      if (!W.sim1) { W.sim1 = new QC.Sim({ N: N, L: L, m: M, hbar: W.hbar, c: W.c }); walls(); }
      W.sim1.re = Float64Array.from(snap.s1.re); W.sim1.im = Float64Array.from(snap.s1.im);
    } else W.sim1 = null;
  }

  // Look where you are, with a microscope of resolution sigma (m). Returns the outcome.
  function look(sigma, rand, screen) {
    var snap = snapshot(), p = pick(rand, screen);
    if (!p) return null;
    W.saved = snap;
    both(function (s) {
      for (var j = 0; j < N; j++) for (var i = 0; i < N; i++) {
        var ex = s.x(i) - p.x, ey = s.y(j) - p.y, w = Math.exp(-(ex * ex + ey * ey) / (4 * sigma * sigma)), q = j * N + i;
        s.re[q] *= w; s.im[q] *= w;
      }
    });
    normalise(); measure();
    var out = toStreet(p.x, p.y);
    W.marks.push(out); if (W.marks.length > 600) W.marks.shift();
    return out;
  }
  // The same experiment again: go back to the wave just before the last look, and look again.
  function relook(sigma, rand, screen) {
    if (!W.saved) return null;
    var snap = W.saved; restore(snap);
    var out = look(sigma, rand, screen); W.saved = snap;
    return out;
  }

  // Erase the bell's record: read the bell in the basis (silent ± rang)/√2. Returns +1 or -1.
  function erase(rand) {
    if (!W.sim1) return 0;
    var a = W.sim, b = W.sim1, n = a.re.length, pp = 0, pm = 0, q;
    for (q = 0; q < n; q++) {
      var sr = a.re[q] + b.re[q], si = a.im[q] + b.im[q], dr = a.re[q] - b.re[q], di = a.im[q] - b.im[q];
      pp += sr * sr + si * si; pm += dr * dr + di * di;
    }
    var sign = rand() * (pp + pm) < pp ? 1 : -1;
    for (q = 0; q < n; q++) { a.re[q] += sign * b.re[q]; a.im[q] += sign * b.im[q]; }
    W.sim1 = null; normalise(); measure();
    return sign;
  }

  CGH.wave = {
    W: W, N: N, L: L, M: M, dx: dx, BELL: BELL, start: start, stop: stop, step: step, sample: sample,
    look: look, relook: relook, erase: erase, chanceBehind: chanceBehind, topSpeed: topSpeed, toStreet: toStreet, toGrid: toGrid, kstats: kstats
  };
})();
