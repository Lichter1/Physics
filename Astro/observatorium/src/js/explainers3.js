/* Observatorium — explainers added after the November 2026 audit, each aimed at a misconception the reviewers met:
 * every star moved to 10 pc (1.3), a microlensing builder (3.3), a redshifting blackbody (6.2), a cost function with a
 * false valley (9.3) and a pulsar's life on the P–Ṗ chart (2.4). Same contract as explainers.js: X[id](host) -> {stop}. */
(function (G) {
  'use strict';
  var OBS = G.OBS, A = OBS.astro, UI = OBS.ui, X = OBS.EXPLAINERS;

  function stage(host, ratio) {
    var wrap = document.createElement('div'); wrap.className = 'ex-stage';
    var cv = document.createElement('canvas'); wrap.appendChild(cv); host.appendChild(wrap);
    var ctl = document.createElement('div'); ctl.className = 'ex-ctl'; host.appendChild(ctl);
    var out = document.createElement('p'); out.className = 'ex-out'; host.appendChild(out);
    var s = { cv: cv, ctx: cv.getContext('2d'), ctl: ctl, out: out, ratio: ratio || 0.5 };
    s.fit = function () {
      var w = Math.max(240, wrap.clientWidth), h = Math.round(w * s.ratio), dpr = Math.min(2, G.devicePixelRatio || 1);
      if (cv.width !== w * dpr) { cv.width = w * dpr; cv.height = h * dpr; cv.style.height = h + 'px'; }
      s.W = w; s.H = h; s.ctx.setTransform(dpr, 0, 0, dpr, 0, 0); s.C = OBS.colours();
      s.ctx.clearRect(0, 0, w, h); s.ctx.fillStyle = s.C.panel; s.ctx.fillRect(0, 0, w, h);
      s.ctx.font = '12px ' + getComputedStyle(document.documentElement).getPropertyValue('--sans');
      return s;
    };
    return s;
  }
  function line(c, pts, col, w, dash) { c.beginPath(); c.strokeStyle = col; c.lineWidth = w || 1.5; c.setLineDash(dash || []); pts.forEach(function (p, i) { if (i) c.lineTo(p[0], p[1]); else c.moveTo(p[0], p[1]); }); c.stroke(); c.setLineDash([]); }
  function text(c, s, x, y, col, align) { c.fillStyle = col; c.textAlign = align || 'left'; c.fillText(s, x, y); c.textAlign = 'left'; }
  function disc(c, x, y, r, fill) { c.beginPath(); c.arc(x, y, Math.max(0.5, r), 0, 7); c.fillStyle = fill; c.fill(); }
  function box(s, x0, y0, w, h, xr, yr, o) {
    o = o || {};
    function tx(v) { v = o.logx ? Math.log10(v) : v; return x0 + (v - xr[0]) / (xr[1] - xr[0]) * w; }
    function ty(v) { v = o.logy ? Math.log10(v) : v; return y0 + h - (v - yr[0]) / (yr[1] - yr[0]) * h; }
    s.ctx.strokeStyle = s.C.rule; s.ctx.lineWidth = 1; s.ctx.strokeRect(x0 + 0.5, y0 + 0.5, w, h);
    if (o.xl) text(s.ctx, o.xl, x0 + w / 2, y0 + h + 28, s.C['ink-2'], 'center');
    if (o.yl) { s.ctx.save(); s.ctx.translate(x0 - (o.ylx || 30), y0 + h / 2); s.ctx.rotate(-Math.PI / 2); text(s.ctx, o.yl, 0, 0, s.C['ink-2'], 'center'); s.ctx.restore(); }
    return { x: tx, y: ty, x0: x0, y0: y0, w: w, h: h };
  }
  function fx(v, d) { return OBS.fix(v, d); }
  function fm(v, n) { return OBS.fmt(v, n || 3); }
  function tick(s, b, vals, axis, fmt) {
    var c = s.ctx; c.fillStyle = s.C['ink-3'];
    vals.forEach(function (v) {
      if (axis === 'x') { var x = b.x(v); if (x < b.x0 - 1 || x > b.x0 + b.w + 1) return; text(c, fmt ? fmt(v) : String(v), x, b.y0 + b.h + 13, s.C['ink-3'], 'center'); }
      else { var y = b.y(v); if (y < b.y0 - 1 || y > b.y0 + b.h + 1) return; text(c, fmt ? fmt(v) : String(v), b.x0 - 5, y + 4, s.C['ink-3'], 'right'); }
    });
  }

  // ------------------------------------------------------------------------------------------ 1.3 move every star to 10 pc
  X.tenpc = function (host) {
    var s = stage(host, 0.62), st = { t: 0, play: false }, data = null, raf = 0;
    function draw() {
      s.fit(); var c = s.ctx, C = s.C;
      var b = box(s, 46, 12, s.W - 62, s.H - 52, [-0.4, 2.2], [9, -8], { xl: 'colour B − V (blue → red)', yl: st.t < 0.02 ? 'apparent brightness V' : st.t > 0.98 ? 'absolute magnitude M_V' : 'brightness, part way to 10 pc' });
      tick(s, b, [0, 0.5, 1, 1.5, 2], 'x'); tick(s, b, [-6, -3, 0, 3, 6, 9], 'y');
      if (!data) { text(c, 'Loading the 8,900 naked-eye stars…', b.x0 + 12, b.y0 + 22, C['ink-3']); return; }
      var n = data.v.length;
      for (var i = 0; i < n; i++) {
        var y = data.v[i] - st.t * data.dm[i], bv = data.bv[i];
        var r = Math.round(150 + 105 * Math.min(1, Math.max(0, (bv + 0.2) / 1.8))), g = Math.round(190 + 40 * (1 - Math.abs(bv - 0.6) / 1.4)), bl = Math.round(255 - 140 * Math.min(1, Math.max(0, (bv + 0.2) / 1.8)));
        c.fillStyle = 'rgba(' + r + ',' + g + ',' + bl + ',0.6)'; c.fillRect(b.x(bv) - 1, b.y(y) - 1, 2, 2);
      }
      s.out.innerHTML = st.t < 0.02 ? 'Apparent brightness mixes power with distance: the naked-eye stars form a <b>smear</b>. Slide them all to 10 parsecs.'
        : st.t > 0.98 ? 'Every star at 10 pc: the smear has become <b>bands</b>. The main sequence runs from bright blue stars to faint red ones; the giants sit above it on the red side. The pattern was always there, hidden by distance.'
        : 'Each star moves by its own distance modulus, 5 log₁₀(d / 10 pc): far stars rise a lot, near ones barely move.';
    }
    var sl = UI.slider(s.ctl, { label: 'Move every star to 10 pc', min: 0, max: 1, value: 0, fmt: function (x) { return Math.round(100 * x) + '%'; }, oninput: function (x) { st.t = x; draw(); } });
    UI.button(s.ctl, 'Play', function () {
      if (st.play) return; st.play = true; var t0 = performance.now(), from = st.t >= 0.99 ? 0 : st.t;
      (function step(now) { if (!st.play) return; var u = Math.min(1, from + (now - t0) / 2500); st.t = u; sl.set(u, true); draw(); if (u < 1) raf = G.requestAnimationFrame(step); else st.play = false; })(t0);
    }, 'small');
    OBS.open('sky').then(function (ds) {
      var c = ds.table.cols, v = [], bv = [], dm = [];
      for (var i = 0; i < ds.table.n; i++) if (c.dist[i] > 0 && isFinite(c.ci[i]) && isFinite(c.mag[i])) { v.push(c.mag[i]); bv.push(c.ci[i]); dm.push(5 * Math.log10(c.dist[i] / 10)); }
      data = { v: v, bv: bv, dm: dm }; draw();
    });
    draw();
    return { stop: function () { st.play = false; G.cancelAnimationFrame(raf); } };
  };

  // ------------------------------------------------------------------------------------------ 3.3 build a microlensing event
  X.lensbuild = function (host) {
    var s = stage(host, 0.5), st = { M: 0.3, DL: 4, v: 150, u0: 0.3 }, DS = 8;
    function calc() {
      var kpc = 3.0857e19, DLm = st.DL * kpc, DSm = DS * kpc, rs = 2953 * st.M;          // 2GM/c² = 2.95 km per solar mass
      var thE = Math.sqrt(2 * rs * (DSm - DLm) / (DLm * DSm)), RE = thE * DLm, tE = RE / (st.v * 1000) / 86400;
      return { thE: thE * 206265e3, RE: RE / 1.496e11, tE: tE };
    }
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, q = calc(), half = s.W / 2;
      // left: the Einstein ring and the source's path, in units of the ring
      var cx = half * 0.5, cy = s.H * 0.46, R = Math.min(half * 0.32, s.H * 0.3);
      c.strokeStyle = C.accent; c.lineWidth = 2; c.setLineDash([4, 3]); c.beginPath(); c.arc(cx, cy, R, 0, 7); c.stroke(); c.setLineDash([]);
      disc(c, cx, cy, 5, C.ink); text(c, 'lens', cx + 8, cy - 6, C['ink-2']);
      line(c, [[Math.max(6, cx - 2.2 * R), cy - st.u0 * R], [cx + 2.2 * R, cy - st.u0 * R]], C.c1, 1.6);
      text(c, 'source path', Math.max(8, cx - 2.1 * R), cy - st.u0 * R - 6, C.c1);
      text(c, 'Einstein ring: θ_E = ' + fx(q.thE, 2) + ' mas = ' + fx(q.RE, 2) + ' AU at the lens', 10, s.H - 10, C['ink-2']);
      // right: the light curve, on a fixed ±100-day axis so that t_E visibly changes
      var b = box(s, half + 40, 14, half - 56, s.H - 56, [-100, 100], [1, 11], { xl: 'days from the peak', yl: 'magnification' });
      tick(s, b, [-100, -50, 0, 50, 100], 'x'); tick(s, b, [1, 3, 5, 7, 9, 11], 'y');
      var pts = []; for (var t = -100; t <= 100; t += 0.5) { var u = Math.sqrt(st.u0 * st.u0 + (t / q.tE) * (t / q.tE)); pts.push([b.x(t), b.y(Math.min(11, A.magnification(u)))]); }
      line(c, pts, C.accent, 2.2);
      var Amax = A.magnification(st.u0);
      s.out.innerHTML = 'Ring crossing time <b>t_E = ' + fx(q.tE, 1) + ' days</b>, peak magnification <b>' + fx(Amax, 2) + '</b>. The light curve fixes only t_E and u₀: a heavier lens, a slower one, or one nearer the middle can all give the same t_E. That degeneracy is why one event rarely gives a lens mass; t_E ∝ √M, so the average duration of many events weighs the population.';
    }
    UI.slider(s.ctl, { label: 'Lens mass', min: 0.001, max: 10, value: st.M, log: true, fmt: function (x) { return x < 0.01 ? fx(x / 0.000954, 1) + ' M_Jup' : fx(x, 2) + ' M☉'; }, oninput: function (x) { st.M = x; draw(); } });
    UI.slider(s.ctl, { label: 'Lens distance (source at 8 kpc)', min: 0.5, max: 7.5, value: st.DL, unit: 'kpc', fmt: function (x) { return x.toFixed(1); }, oninput: function (x) { st.DL = x; draw(); } });
    UI.slider(s.ctl, { label: 'Relative speed', min: 30, max: 300, value: st.v, unit: 'km/s', fmt: function (x) { return Math.round(x); }, oninput: function (x) { st.v = x; draw(); } });
    UI.slider(s.ctl, { label: 'Closest approach u₀ (ring radii)', min: 0.02, max: 1.5, value: st.u0, fmt: function (x) { return x.toFixed(2); }, oninput: function (x) { st.u0 = x; draw(); } });
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 6.2 a blackbody, redshifted
  X.cmbz = function (host) {
    var s = stage(host, 0.5), st = { lz: Math.log10(1101) };
    function B(nu, T) { var x = 6.62607e-34 * nu / (1.380649e-23 * T); return x > 700 ? 0 : 2 * 6.62607e-34 * nu * nu * nu / (2.998e8 * 2.998e8) / Math.expm1(x); }
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, z = Math.pow(10, st.lz) - 1, T = 2.725 * (1 + z);
      var b = box(s, 74, 12, s.W - 90, s.H - 54, [8, 16], [-22, -6], { logx: true, logy: true, xl: 'frequency [Hz]', yl: 'B_ν [W m⁻² Hz⁻¹ sr⁻¹]', ylx: 60 });
      tick(s, b, [1e8, 1e10, 1e12, 1e14, 1e16], 'x', function (v) { return '10^' + Math.round(Math.log10(v)); });
      tick(s, b, [1e-21, 1e-17, 1e-13, 1e-9], 'y', function (v) { return '10^' + Math.round(Math.log10(v)); });
      c.fillStyle = 'rgba(255,200,120,0.16)'; c.fillRect(b.x(4e14), b.y0, b.x(7.5e14) - b.x(4e14), b.h); text(c, 'visible', b.x(5.5e14), b.y0 + 14, C['ink-3'], 'center');
      function curve(Tk, col, wd, dash) { var pts = []; for (var lx = 8; lx <= 16; lx += 0.02) { var y = B(Math.pow(10, lx), Tk); if (y > 1e-22) pts.push([b.x(Math.pow(10, lx)), b.y(y)]); } line(c, pts, col, wd, dash); }
      curve(2.725, C['ink-3'], 1.4, [4, 3]); curve(T, C.accent, 2.4);
      var nuPk = 5.879e10 * T;
      s.out.innerHTML = 'At <b>z = ' + fm(z, 3) + '</b> the radiation was a blackbody at <b>' + fm(T, 3) + ' K</b>, peaking at ' + (nuPk > 1e13 ? fx(2.998e8 / nuPk * 1e9, 0) + ' nm' : fx(nuPk / 1e9, 0) + ' GHz') + ', with ' + fm(410 * Math.pow(1 + z, 3), 2) + ' photons per cm³. ' +
        'Expansion slides the curve down and to the left without changing its shape: each photon’s frequency falls as 1/(1+z), and their number per volume as (1+z)³. The dashed curve is today’s 2.725 K.' + (z > 900 ? ' At recombination the sky glowed orange, like a dim star’s surface, in every direction.' : '');
    }
    UI.slider(s.ctl, { label: 'Redshift z', min: 0, max: Math.log10(1101), value: st.lz, fmt: function (x) { return fm(Math.pow(10, x) - 1, 3); }, oninput: function (x) { st.lz = x; draw(); } });
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 9.3 a cost function with two valleys
  X.costfn = function (host) {
    var s = stage(host, 0.5), st = { a0: 0.1, mode: null, path: [], grid: [], chain: [] };
    function f(a) { return 20 + 10 * (200 * (a - 0.2) * (a - 0.2) * (a - 0.78) * (a - 0.78) - 2 * a); }
    function df(a) { var h = 1e-4; return (f(a + h) - f(a - h)) / (2 * h); }
    function run(mode) {
      st.mode = mode; st.path = []; st.grid = []; st.chain = [];
      if (mode === 'grid') for (var k = 0; k <= 10; k++) st.grid.push(k / 10);
      if (mode === 'down') { var a = st.a0; for (var n = 0; n < 2000; n++) { st.path.push(a); var g = df(a); if (Math.abs(g) < 1e-3) break; var step = Math.max(-0.004, Math.min(0.004, -0.0008 * g)); a = Math.max(0, Math.min(1, a + step)); } st.path.push(a); }
      if (mode === 'chain') { var p = st.a0, fp = f(p); for (var m = 0; m < 3000; m++) { var q = p + 0.06 * (Math.random() + Math.random() + Math.random() - 1.5) * 1.4; if (q >= 0 && q <= 1) { var fq = f(q); if (fq <= fp || Math.random() < Math.exp(-(fq - fp) / 2)) { p = q; fp = fq; } } st.chain.push(p); } }
      draw();
    }
    function draw() {
      s.fit(); var c = s.ctx, C = s.C;
      var b = box(s, 48, 12, s.W - 64, s.H - 56, [0, 1], [0, 32], { xl: 'parameter a', yl: 'misfit χ²(a)' });
      tick(s, b, [0, 0.2, 0.4, 0.6, 0.8, 1], 'x'); tick(s, b, [0, 10, 20, 30], 'y');
      var pts = []; for (var a = 0; a <= 1.0001; a += 0.005) pts.push([b.x(a), b.y(Math.min(32, f(a)))]); line(c, pts, C.c1, 2);
      text(c, 'a false valley', b.x(0.2), b.y(f(0.2)) + 18, C['ink-3'], 'center'); text(c, 'the best fit', b.x(0.78), b.y(f(0.78)) + 18, C['ink-3'], 'center');
      disc(c, b.x(st.a0), b.y(f(st.a0)), 5, C.sel);
      var msg = 'Pick a starting value, then try a method.';
      if (st.mode === 'grid') { var best = 0; st.grid.forEach(function (g) { disc(c, b.x(g), b.y(f(g)), 4, C.accent); if (f(g) < f(best)) best = g; }); msg = 'A grid of 11 values finds the right valley (best at a = ' + best.toFixed(1) + ') whatever the start, but it costs n^d evaluations: 11 here, 11¹⁰ ≈ 2.6×10¹⁰ for ten parameters.'; }
      if (st.mode === 'down') { line(c, st.path.map(function (a) { return [b.x(a), b.y(f(a))]; }), C.accent, 3); var end = st.path[st.path.length - 1]; disc(c, b.x(end), b.y(f(end)), 6, C.accent); msg = 'Walking downhill from a = ' + st.a0.toFixed(2) + ' stopped at a = ' + end.toFixed(2) + ' (χ² = ' + f(end).toFixed(1) + ')' + (end < 0.5 ? ': a local minimum. The best fit, χ² = ' + f(0.78).toFixed(1) + ' at 0.78, is never seen from here.' : ': the best fit, because it started on that side of the hill.'); }
      if (st.mode === 'chain') {
        var nb = 40, h = new Array(nb).fill(0); st.chain.forEach(function (a) { h[Math.min(nb - 1, Math.floor(a * nb))]++; }); var mx = Math.max.apply(null, h);
        c.fillStyle = 'rgba(89,227,201,0.45)'; h.forEach(function (v, k) { var bh = 0.35 * b.h * v / mx; c.fillRect(b.x(k / nb) + 1, b.y0 + b.h - bh, b.w / nb - 2, bh); });
        var right = st.chain.filter(function (a) { return a > 0.5; }).length / st.chain.length;
        msg = 'A Markov chain from a = ' + st.a0.toFixed(2) + ': it sometimes steps uphill, so it climbs out of the false valley and spends ' + Math.round(100 * right) + '% of its 3,000 steps near the best fit (the histogram at the bottom). Where it spends its time is the probability of each value.';
      }
      s.out.innerHTML = msg;
    }
    UI.slider(s.ctl, { label: 'Start at a =', min: 0, max: 1, value: st.a0, fmt: function (x) { return x.toFixed(2); }, oninput: function (x) { st.a0 = x; if (st.mode && st.mode !== 'grid') run(st.mode); else draw(); } });
    var row = UI.row(s.ctl);
    UI.button(row, 'Grid', function () { run('grid'); }, 'small'); UI.button(row, 'Walk downhill', function () { run('down'); }, 'small'); UI.button(row, 'Markov chain', function () { run('chain'); }, 'small');
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 2.4 a pulsar's life on the P–Ṗ chart
  X.pulsarlife = function (host) {
    var s = stage(host, 0.62), P0 = 0.0334, PD0 = 4.21e-13, k = P0 * PD0, st = { lt: Math.log10(1000), recycled: false };
    var Pdeath = Math.pow(k / 2.822e-17, 0.25);                     // where the spin-down track meets the death line
    function state() {
      if (st.recycled) return { P: 0.003, Pd: 1e-20 };
      var t = Math.pow(10, st.lt) * 3.156e7, P = Math.min(Pdeath, Math.sqrt(P0 * P0 + 2 * k * t));
      return { P: P, Pd: k / P, dead: P >= Pdeath * 0.999 };
    }
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, q = state();
      var b = box(s, 74, 12, s.W - 90, s.H - 54, [-3, 1.3], [-22, -10], { logx: true, logy: true, xl: 'spin period P [s]', yl: 'slow-down Ṗ [s/s]', ylx: 60 });
      tick(s, b, [0.001, 0.01, 0.1, 1, 10], 'x', function (v) { return v >= 1 ? String(v) : v.toString(); });
      tick(s, b, [1e-21, 1e-18, 1e-15, 1e-12], 'y', function (v) { return '10^' + Math.round(Math.log10(v)); });
      [8, 10, 12, 14].forEach(function (lB) { var kk = Math.pow(Math.pow(10, lB) / 3.2e19, 2); line(c, [[b.x(1e-3), b.y(kk / 1e-3)], [b.x(20), b.y(kk / 20)]], C.rule, 1, [3, 3]); var px = Math.max(1.5e-3, Math.min(8, kk / 2e-11)); text(c, 'B = 10^' + lB + ' G', b.x(px) + 4, b.y(kk / px) - 4, C['ink-3']); });
      [3, 6, 9].forEach(function (lt) { var tau = Math.pow(10, lt) * 3.156e7; line(c, [[b.x(1e-3), b.y(1e-3 / (2 * tau))], [b.x(20), b.y(20 / (2 * tau))]], C.c4, 1, [1, 3]); });
      var dl = []; for (var lp = -3; lp <= 1.3; lp += 0.05) { var P = Math.pow(10, lp); dl.push([b.x(P), b.y(2.822e-17 * P * P * P)]); } line(c, dl, C.c3, 2);
      text(c, 'death line', b.x(2), b.y(2.822e-17 * 8) + 14, C.c3); text(c, 'graveyard', b.x(6), b.y(1e-17), C['ink-3'], 'center');
      text(c, 'magnetars', b.x(6), b.y(1e-11), C['ink-3'], 'center'); text(c, 'recycled', b.x(0.003), b.y(1e-21) - 4, C['ink-3'], 'center');
      // the track so far
      var tr = []; for (var u = 2; u <= st.lt + 1e-9; u += 0.05) { var t = Math.pow(10, u) * 3.156e7, Pt = Math.min(Pdeath, Math.sqrt(P0 * P0 + 2 * k * t)); tr.push([b.x(Pt), b.y(k / Pt)]); }
      if (!st.recycled) line(c, tr, C.accent, 2);
      disc(c, b.x(q.P), b.y(q.Pd), 6, st.recycled ? C.sel : q.dead ? C['ink-3'] : C.accent);
      var B = 3.2e19 * Math.sqrt(q.P * q.Pd), tc = q.P / (2 * q.Pd) / 3.156e7;
      s.out.innerHTML = st.recycled ? 'Recycled: a companion star spilled matter onto the dead pulsar and spun it up to 3 ms. With a weak field (' + fm(B, 2) + ' G) it slows so little that its characteristic age, ' + fm(tc, 2) + ' yr, is longer than the Universe’s: it will shine for billions of years.'
        : 'Age <b>' + fm(Math.pow(10, st.lt), 2) + ' yr</b>: P = ' + fm(q.P, 3) + ' s, Ṗ = ' + fm(q.Pd, 2) + '. With a constant field of ' + fm(B, 2) + ' G it slides down and to the right along its line, its characteristic age P/2Ṗ = ' + fm(tc, 2) + ' yr' + (q.dead ? ' (its spin-down age; it no longer shines to be measured)' : ', tracking its true age') + '.' + (q.dead ? ' It has crossed the <b>death line</b> at P ≈ ' + fx(Pdeath, 1) + ' s: too slow to make the pairs that power its beams. It falls silent.' : '');
    }
    UI.slider(s.ctl, { label: 'Age of the pulsar', min: 2, max: 9, value: st.lt, fmt: function (x) { return fm(Math.pow(10, x), 2) + ' yr'; }, oninput: function (x) { st.lt = x; st.recycled = false; draw(); } });
    UI.button(s.ctl, 'Recycle it with a companion', function () { st.recycled = !st.recycled; draw(); }, 'small');
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 8.2 de-disperse a burst
  X.dedisperse = function (host) {
    var s = stage(host, 0.62), NCH = 48, NT = 240, T = 4.0, DMtrue = 180, st = { dm: 0 }, F0 = 0.4, F1 = 0.8;   // CHIME's 400-800 MHz band
    var rnd = (function (seed) { return function () { seed = (seed * 16807) % 2147483647; return seed / 2147483647; }; })(12345);
    function gauss() { return Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd()); }
    function delay(dm, f) { return 4.149e-3 * dm * (1 / (f * f) - 1 / (F1 * F1)); }               // seconds, f in GHz
    var data = [];
    for (var c = 0; c < NCH; c++) { var f = F1 - (F1 - F0) * (c + 0.5) / NCH, t0 = 0.4 + delay(DMtrue, f), row = [];
      for (var k = 0; k < NT; k++) { var t = (k + 0.5) * T / NT; row.push(gauss() + 2.2 * Math.exp(-0.5 * Math.pow((t - t0) / 0.012, 2)) * (T / NT / 0.012 > 1 ? 1 : 1) + (Math.abs(t - t0) < T / NT ? 2.2 : 0)); }
      data.push(row); }
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, L = 54, R = s.W - 12, top = 10, H1 = s.H * 0.58, H2 = s.H * 0.22, y2 = top + H1 + 26;
      var prof = new Array(NT).fill(0);
      for (var ch = 0; ch < NCH; ch++) {
        var f = F1 - (F1 - F0) * (ch + 0.5) / NCH, sh = Math.round(delay(st.dm, f) / T * NT);
        for (var k = 0; k < NT; k++) {
          var v = data[ch][k + sh]; if (v === undefined) continue; prof[k] += v / NCH;
          var g = Math.max(0, Math.min(255, Math.round(60 + 45 * v))); c.fillStyle = 'rgb(' + g + ',' + g + ',' + Math.min(255, g + 20) + ')';
          c.fillRect(L + (R - L) * k / NT, top + H1 * ch / NCH, (R - L) / NT + 0.6, H1 / NCH + 0.6);
        }
      }
      c.strokeStyle = C.rule; c.strokeRect(L + 0.5, top + 0.5, R - L, H1);
      text(c, '800 MHz', L - 4, top + 10, C['ink-3'], 'right'); text(c, '400 MHz', L - 4, top + H1, C['ink-3'], 'right');
      var mx = Math.max.apply(null, prof), sd = Math.sqrt(1 / NCH), pts = prof.map(function (v, k) { return [L + (R - L) * (k + 0.5) / NT, y2 + H2 - H2 * Math.max(0, v) / Math.max(0.5, 2.3)]; });
      c.strokeStyle = C.rule; c.strokeRect(L + 0.5, y2 + 0.5, R - L, H2); line(c, pts, C.accent, 1.6);
      text(c, 'all channels added', L + 6, y2 + 12, C['ink-3']); text(c, 'time (4 s)', (L + R) / 2, s.H - 4, C['ink-3'], 'center');
      var snr = mx / sd;
      s.out.innerHTML = 'Trial DM = <b>' + Math.round(st.dm) + ' pc/cm³</b>: the summed pulse peaks at S/N ≈ <b>' + fx(snr, 1) + '</b>. Low frequencies arrive later, by Δt = 4.15 ms × DM × (ν⁻² − ν_top⁻²) with ν in GHz. At the right DM the sweep stands upright and the burst adds up across all 48 channels; at a wrong one it smears and sinks into the noise. Searches try thousands of trial DMs on every beam, every second.';
    }
    UI.slider(s.ctl, { label: 'Trial dispersion measure', min: 0, max: 400, value: 0, step: 1, unit: 'pc/cm³', fmt: function (x) { return Math.round(x); }, oninput: function (x) { st.dm = x; draw(); } });
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 4.1 the solar system from above
  var PLANETS = [   // J2000 elements (Standish): a, e, i, L, ϖ, Ω, and L's rate in degrees per century
    ['Mercury', 0.38710, 0.20563, 7.005, 252.2503, 77.4578, 48.3308, 149472.674], ['Venus', 0.72333, 0.00677, 3.3947, 181.9791, 131.6025, 76.68, 58517.8156],
    ['Earth', 1.00000, 0.01671, 0, 100.4644, 102.9377, 0, 35999.3724], ['Mars', 1.52371, 0.09339, 1.8497, -4.5534, -23.9436, 49.5596, 19140.3027],
    ['Jupiter', 5.20289, 0.04839, 1.3044, 34.3965, 14.7284, 100.474, 3034.7461], ['Saturn', 9.53668, 0.05386, 2.4859, 49.9548, 92.5989, 113.6624, 1222.4936],
    ['Uranus', 19.18916, 0.04726, 0.7726, 313.2381, 170.9543, 74.0168, 428.482], ['Neptune', 30.06992, 0.00859, 1.77, -55.1203, 44.9648, 131.7841, 218.4562]];
  function kep(a, e, i, Om, w, M) {      // heliocentric ecliptic x, y (AU); angles in degrees
    var d = Math.PI / 180, Mr = M * d, E = Mr; for (var k = 0; k < 12; k++) E -= (E - e * Math.sin(E) - Mr) / (1 - e * Math.cos(E));
    var xv = a * (Math.cos(E) - e), yv = a * Math.sqrt(1 - e * e) * Math.sin(E), ww = w * d, OO = Om * d, ii = i * d;
    var xo = xv * Math.cos(ww) - yv * Math.sin(ww), yo = xv * Math.sin(ww) + yv * Math.cos(ww);
    return [xo * Math.cos(OO) - yo * Math.sin(OO) * Math.cos(ii), xo * Math.sin(OO) + yo * Math.cos(OO) * Math.cos(ii)];
  }
  function planetAt(p, years) { var T = 0.26 + years / 100, L = p[4] + p[7] * T; return kep(p[1], p[2], p[3], p[6], p[5] - p[6], L - p[5]); }   // 2026 Jan 1 is T = 0.26
  X.solartop = function (host) {
    var s = stage(host, 0.85), st = { yr: 0, zoom: 6, play: false }, data = null, raf = 0;
    var COL = { MBA: '#9fb3d9', OMB: '#c9a3e6', IMB: '#9fb3d9', TJN: '#ffb35c', TNO: '#7fd6a8', CEN: '#e87f7f', MCA: '#e8e87f' };
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, cx = s.W / 2, cy = s.H / 2, sc = Math.min(s.W, s.H) / 2 / st.zoom;
      function P(x, y) { return [cx + x * sc, cy - y * sc]; }
      PLANETS.forEach(function (p) { if (p[1] > st.zoom * 1.5) return; c.strokeStyle = C.rule; c.beginPath(); for (var m = 0; m <= 360; m += 4) { var q = kep(p[1], p[2], p[3], p[6], p[5] - p[6], m), r = P(q[0], q[1]); if (m) c.lineTo(r[0], r[1]); else c.moveTo(r[0], r[1]); } c.stroke(); });
      if (data) for (var i = 0; i < data.a.length; i++) {
        var n = 0.9856076686 / Math.pow(data.a[i], 1.5), q = kep(data.a[i], data.e[i], data.i[i], data.om[i], data.w[i], data.M[i] + n * st.yr * 365.25), r = P(q[0], q[1]);
        if (r[0] < 0 || r[0] > s.W || r[1] < 0 || r[1] > s.H) continue; c.fillStyle = COL[data.cls[i]] || C['ink-3']; c.fillRect(r[0] - 0.7, r[1] - 0.7, 1.5, 1.5);
      }
      disc(c, cx, cy, 4, '#ffd34d');
      PLANETS.forEach(function (p) { if (p[1] > st.zoom * 1.2) return; var q = planetAt(p, st.yr), r = P(q[0], q[1]); disc(c, r[0], r[1], p[0] === 'Jupiter' ? 5 : 3.5, C.ink); text(c, p[0], r[0] + 6, r[1] - 4, C.ink); });
      text(c, 'orange: Jupiter’s Trojans · violet: outer belt and Hildas · blue: main belt · green: beyond Neptune · red: centaurs', 8, s.H - 8, C['ink-3']);
      s.out.innerHTML = '<b>' + (2026 + st.yr).toFixed(1) + '</b>, seen from above the ecliptic, ' + Math.round(2 * st.zoom) + ' AU across. The Kirkwood gaps, so clear in a histogram of orbit sizes, vanish here: eccentric orbits smear every asteroid over a range of distances. What does stand out is resonance in angle: the Trojans travel 60° ahead of and behind Jupiter, and as time runs the Hildas keep a triangle with Jupiter at one corner (3:2).';
    }
    UI.slider(s.ctl, { label: 'Years from 2026', min: 0, max: 24, value: 0, fmt: function (x) { return x.toFixed(1); }, oninput: function (x) { st.yr = x; draw(); } });
    UI.slider(s.ctl, { label: 'Field (half-width)', min: 2, max: 60, value: 6, log: true, unit: 'AU', fmt: function (x) { return x.toFixed(1); }, oninput: function (x) { st.zoom = x; draw(); } });
    OBS.open('orbits').then(function (ds) { var c = ds.table.cols; data = { a: c.a, e: c.e, i: c.i, om: c.om, w: c.w, M: c.M, cls: c['class'] }; draw(); });
    draw(); return { stop: function () { st.play = false; G.cancelAnimationFrame(raf); } };
  };

  // ------------------------------------------------------------------------------------------ the scale ladder: powers of ten
  X.powers = function (host) {
    var s = stage(host, 0.75), st = { lw: 1.3 }, D = {}, AU = 1, PC = 206265, KPC = 2.06265e8, MPC = 2.06265e11;
    function load(id, f) { OBS.open(id).then(function (ds) { D[id] = f(ds.table.cols, ds.table.n); draw(); }); }
    function xyz(ra, dec, d) { var r = Math.PI / 180; return [d * Math.cos(dec * r) * Math.cos(ra * r), d * Math.cos(dec * r) * Math.sin(ra * r)]; }
    load('orbits', function (c, n) { var o = [], b = []; for (var i = 0; i < n; i++) if (c['class'][i] === 'TNO' || i % 4 === 0) { var q = kep(c.a[i], c.e[i], c.i[i], c.om[i], c.w[i], c.M[i]); (c['class'][i] === 'TNO' ? o : b).push(q); } D.belt = b; return o; });
    load('nearby', function (c, n) { var o = []; for (var i = 0; i < n; i++) { var d = 1000 / (c.parallax[i] + 0.017); if (d > 0) o.push(xyz(c.ra[i], c.dec[i], d * PC)); } return o; });
    load('globulars', function (c, n) { var o = []; for (var i = 0; i < n; i++) if (isFinite(c.X[i])) o.push([-c.X[i] * KPC, c.Y[i] * KPC]); return o; });     // the catalogue's X points from the Sun to the centre, drawn to the left
    load('galaxies', function (c, n) { var o = []; for (var i = 0; i < n; i++) o.push(xyz(c.ra[i], c.dec[i], c.z[i] * 299792.458 / 70 * MPC)); return o; });
    load('supernovae', function (c, n) { var o = []; for (var i = 0; i < n; i++) o.push(xyz(c.ra[i], c.dec[i], OBS.astro.dM(c.z[i], 0.31, 0.69) * 299792.458 / 70 * MPC)); return o; });
    function unit(au) { return au < 0.5 * PC ? fm(au, 2) + ' AU' : au < 0.5 * KPC ? fm(au / PC, 2) + ' pc (' + fm(au / PC * 3.26, 2) + ' light-years)' : au < 0.5 * MPC ? fm(au / KPC, 2) + ' kpc' : au < 2000 * MPC ? fm(au / MPC, 2) + ' Mpc' : fm(au / MPC / 1000, 2) + ' Gpc'; }
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, W = Math.pow(10, st.lw), cx = s.W / 2, cy = s.H / 2, sc = Math.min(s.W, s.H) / W;
      function pt(set, col, sz, alpha) { if (!set) return; c.fillStyle = col; c.globalAlpha = alpha || 1; set.forEach(function (q) { var x = cx + q[0] * sc, y = cy - q[1] * sc; if (x > -2 && x < s.W + 2 && y > -2 && y < s.H + 2) c.fillRect(x - sz / 2, y - sz / 2, sz, sz); }); c.globalAlpha = 1; }
      function ring(r, col, lab, dash) { if (r * sc < 3 || r * sc > s.W * 3) return; c.strokeStyle = col; c.setLineDash(dash || []); c.beginPath(); c.arc(cx, cy, r * sc, 0, 7); c.stroke(); c.setLineDash([]); if (lab) text(c, lab, cx + r * sc * 0.72 + 4, cy - r * sc * 0.72, col); }
      PLANETS.forEach(function (p) { ring(p[1], C.rule, W < 80 ? p[0] : ''); });
      if (W > 5 && W < 2e3) { pt(D.belt, '#9fb3d9', 1.3, 0.7); pt(D.orbits, '#7fd6a8', 1.5, 0.85); }
      ring(2000, C.c4, W > 3e3 && W < 4e5 ? 'inner Oort cloud (inferred)' : '', [3, 3]); ring(1e5, C.c4, W > 1e5 && W < 4e6 ? 'outer Oort cloud' : '', [3, 3]);
      if (W > 2e5 && W < 3e8) pt(D.nearby, '#cfd8ff', 1.4, 0.7);
      if (W > 3e8 && W < 3e10) { ring(8.2 * KPC, C.rule, 'the Sun’s orbit', [2, 4]); pt(D.globulars, C.accent, 3, 0.9); c.fillStyle = 'rgba(200,190,160,0.10)'; c.beginPath(); c.arc(cx - 8.2 * KPC * sc, cy, 15 * KPC * sc, 0, 7); c.fill(); text(c, 'Galactic centre', cx - 8.2 * KPC * sc + 6, cy - 6, C['ink-2']); }
      if (W > 3e10 && W < 1e13) { [['M31', 780 * 1000 * PC, 2.3], ['LMC', 50 * KPC, 1.3]].forEach(function (g) { var x = cx + g[1] * Math.cos(g[2]) * sc, y = cy - g[1] * Math.sin(g[2]) * sc; disc(c, x, y, 3, C.ink); text(c, g[0], x + 5, y - 4, C.ink); }); }
      if (W > 1e12 && W < 5e15) pt(D.galaxies, '#ffb35c', 1.4, 0.75);
      if (W > 3e13) { pt(D.supernovae, '#ff7f9f', 2, 0.9); ring(14000 * MPC, C.c1, 'the CMB: the farthest we can see', [5, 3]); }
      disc(c, cx, cy, 3, '#ffd34d'); text(c, W < 1e4 ? 'Sun' : 'you are here', cx + 5, cy + 13, '#ffd34d');
      var bar = W / 5; c.strokeStyle = C.ink; c.beginPath(); c.moveTo(14, s.H - 18); c.lineTo(14 + bar * sc, s.H - 18); c.stroke(); text(c, unit(bar), 14, s.H - 24, C.ink);
      var what = W < 80 ? 'the planets' : W < 2e5 ? 'the asteroid belt (blue), the Kuiper belt (green) and, far beyond, the Oort cloud of comets' : W < 3e8 ? 'the nearest stars, from Gaia: within 100 pc they look evenly spread' : W < 3e10 ? 'the Milky Way: globular clusters (gold) around the centre that Shapley found' : W < 1e13 ? 'the Local Group: the Magellanic Clouds and Andromeda' : W < 5e15 ? 'galaxies from SDSS (orange) in walls and voids' : 'type Ia supernovae (pink) out to z ≈ 2, and the CMB beyond them';
      s.out.innerHTML = 'Field <b>' + unit(W) + '</b> across: ' + what + '. Each step of the slider is a factor of ten; the whole range is 26 of them.';
    }
    UI.slider(s.ctl, { label: 'Field of view (powers of ten)', min: 0.6, max: 15.5, value: st.lw, step: 0.05, fmt: function (x) { return unit(Math.pow(10, x)); }, oninput: function (x) { st.lw = x; draw(); } });
    draw(); return { stop: function () { } };
  };
})(typeof window !== 'undefined' ? window : globalThis);
