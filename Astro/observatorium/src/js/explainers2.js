/* Observatorium — explainers for chapters 6-9 (lectures 7-13): the expanding Universe, sound in the early plasma,
 * the first three minutes, telescopes and the atmosphere, detectors, radio interferometry, dispersed radio pulses,
 * gravitational-wave chirps and photometric redshifts. Same contract as explainers.js: X[id](host, opts) -> {stop}. */
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
  function loop(draw) {
    var on = true, t0 = performance.now(), still = OBS.reducedMotion();
    (function f(now) { if (!on) return; draw(still ? 2.5 : (now - t0) / 1000); if (!still) G.requestAnimationFrame(f); })(t0);
    return function () { on = false; };
  }
  function line(c, pts, col, w, dash) { c.beginPath(); c.strokeStyle = col; c.lineWidth = w || 1.5; c.setLineDash(dash || []); pts.forEach(function (p, i) { if (i) c.lineTo(p[0], p[1]); else c.moveTo(p[0], p[1]); }); c.stroke(); c.setLineDash([]); c.lineWidth = 1; }
  function text(c, s, x, y, col, align) { c.fillStyle = col; c.textAlign = align || 'left'; c.fillText(s, x, y); c.textAlign = 'left'; }
  function disc(c, x, y, r, fill) { c.beginPath(); c.arc(x, y, Math.max(0.5, r), 0, 7); c.fillStyle = fill; c.fill(); }
  function box(s, x0, y0, w, h, xr, yr, o) {
    o = o || {};
    function tx(v) { v = o.logx ? Math.log10(v) : v; return x0 + (v - xr[0]) / (xr[1] - xr[0]) * w; }
    function ty(v) { v = o.logy ? Math.log10(v) : v; return y0 + h - (v - yr[0]) / (yr[1] - yr[0]) * h; }
    s.ctx.strokeStyle = s.C.rule; s.ctx.strokeRect(x0 + 0.5, y0 + 0.5, w, h);
    if (o.xl) text(s.ctx, o.xl, x0 + w / 2, y0 + h + 15, s.C['ink-2'], 'center');
    if (o.yl) { s.ctx.save(); s.ctx.translate(x0 - 8, y0 + h / 2); s.ctx.rotate(-Math.PI / 2); text(s.ctx, o.yl, 0, 0, s.C['ink-2'], 'center'); s.ctx.restore(); }
    return { x: tx, y: ty };
  }
  function fx(v, d) { return OBS.fix(v, d); }
  function fm(v, s) { return OBS.fmt(v, s || 3); }

  // ------------------------------------------------------------------------------------------ 6.1 how the Universe expands
  /** Scale factor a(t) for matter + Lambda universes, from the Friedmann equation. */
  function scaleHistory(Om, OL) {
    // integrate dt/da = 1/(a H) backwards and forwards from a = 1 (t in units of 1/H0)
    var out = [], a, t = 0;
    function H(a) { var e2 = Om / (a * a * a) + (1 - Om - OL) / (a * a) + OL; return e2 > 0 ? Math.sqrt(e2) : NaN; }
    var pts = [];
    for (a = 1e-3; a <= 2.5; a += 0.005) { var h = H(a); if (!isFinite(h)) break; t += 0.005 / (a * h); pts.push([t, a]); }
    var t1 = null; for (var k = 0; k < pts.length; k++) if (pts[k][1] >= 1) { t1 = pts[k][0]; break; }
    pts.forEach(function (p) { out.push([p[0] - t1, p[1]]); });
    return { pts: out, age: t1 };
  }
  X.expansion = function (host) {
    var s = stage(host, 0.55), st = { Om: 0.31, OL: 0.69 };
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, b = box(s, 50, 12, s.W - 70, s.H - 48, [-1.6, 1.2], [0, 2.2], { xl: 'time from today, in units of 1/H₀ (14 Gyr)', yl: 'size of the Universe a(t)' });
      line(c, [[b.x(0), b.y(0)], [b.x(0), b.y(2.2)]], C.rule, 1, [3, 3]); line(c, [[b.x(-1.6), b.y(1)], [b.x(1.2), b.y(1)]], C.rule, 1, [3, 3]);
      [[0, 0, 'empty (coasting)', C['ink-3']], [1, 0, 'matter only', C.c4], [st.Om, st.OL, 'your universe', C.c1]].forEach(function (m) {
        var h = scaleHistory(m[0], m[1]);
        line(c, h.pts.filter(function (p) { return p[0] > -1.6 && p[0] < 1.2; }).map(function (p) { return [b.x(p[0]), b.y(Math.min(2.2, p[1]))]; }), m[3], m[2] === 'your universe' ? 2.6 : 1.6);
        var q = h.pts.filter(function (p) { return p[0] < 1.15; }).pop(); if (q) text(c, m[2], b.x(q[0]) - 4, b.y(Math.min(2.15, q[1])) - 6, m[3], 'right');
      });
      var age = A.age(0, 70, st.Om, st.OL), q0 = A.q0(st.Om, st.OL), z = 1, aEm = 1 / (1 + z);
      text(c, 'a = 1 today', b.x(-1.55), b.y(1) - 5, C['ink-3']);
      s.out.innerHTML = 'Ωm = ' + fx(st.Om, 2) + ', ΩΛ = ' + fx(st.OL, 2) + ': age ' + fx(age, 1) + ' Gyr (for H₀ = 70), and the expansion is ' + (q0 < 0 ? '<b>speeding up</b>' : '<b>slowing down</b>') + ' today (q₀ = ' + fx(q0, 2) + '). Light from z = 1 left when the Universe was half its present size: every wavelength since stretched by 1 + z = 1/a.';
    }
    UI.slider(s.ctl, { label: 'Ωm (matter)', min: 0.05, max: 1.5, value: st.Om, fmt: function (x) { return x.toFixed(2); }, oninput: function (x) { st.Om = x; draw(); } });
    UI.slider(s.ctl, { label: 'ΩΛ (dark energy)', min: 0, max: 1.2, value: st.OL, fmt: function (x) { return x.toFixed(2); }, oninput: function (x) { st.OL = x; draw(); } });
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 6.3 sound waves frozen at recombination
  X.acoustic = function (host) {
    var s = stage(host, 0.5), st = { t: 1 };
    function draw(tt) {
      s.fit(); var c = s.ctx, C = s.C, t = st.t, W = s.W, H = s.H;
      var b = box(s, 50, 14, W - 70, H - 52, [0, 1], [-1.15, 1.15], { xl: 'position across a region of the early Universe', yl: 'density' });
      // modes k = 1, 2, 3 that started together at the end of inflation; at recombination (t = 1) mode 1 has done half an oscillation
      [[1, C.c1], [2, C.c2], [3, C.c3]].forEach(function (m) {
        var k = m[0], amp = Math.cos(Math.PI * k * t), pts = [];
        for (var x = 0; x <= 1.0001; x += 0.01) pts.push([b.x(x), b.y(amp * Math.cos(Math.PI * k * x) * (1 - 0.12 * k))]);
        line(c, pts, m[1], 2);
        text(c, 'mode ' + k + (Math.abs(amp) > 0.98 ? ' (at its extreme)' : ''), b.x(0.02), b.y(1.1 - 0.17 * k), m[1]);
      });
      line(c, [[b.x(0), b.y(0)], [b.x(1), b.y(0)]], C.rule, 1, [3, 3]);
      s.out.innerHTML = 'Time since the end of inflation: <b>' + fx(t, 2) + '</b> × the time of recombination (380,000 years). ' + (Math.abs(t - 1) < 0.02 ? 'At recombination the fundamental (mode 1) has done exactly half an oscillation, and modes 2 and 3 one and one and a half: all at maximum compression or rarefaction. That pattern is frozen into the CMB as the acoustic peaks.' : 'Slide to t = 1, the moment the plasma became transparent.');
    }
    UI.slider(s.ctl, { label: 'Time (recombination = 1)', min: 0, max: 1.5, value: st.t, fmt: function (x) { return x.toFixed(2); }, oninput: function (x) { st.t = x; draw(); } });
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 6.x the first three minutes
  X.bbn = function (host) {
    var s = stage(host, 0.5), st = { tf: 0.92 };      // 0.74 MeV: the freeze-out that gives the observed Y ≈ 0.245
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, Q = 1.293, tau = 879.4;
      // freeze-out: n/p = exp(-Q/kT) at T_f (MeV); T ~ 1 MeV / sqrt(t / 1 s)
      var Tf = 0.8 * st.tf, x0 = Math.exp(-Q / Tf), tnuc = 180, x = x0 * Math.exp(-tnuc / tau), Y = 2 * x / (1 + x);
      var b = box(s, 56, 14, s.W - 76, s.H - 52, [-1, 3], [0, 1.05], { xl: 'time after the Big Bang (log s)', yl: 'neutrons per proton' });
      var pts = [];
      for (var lt = -1; lt <= 3; lt += 0.02) {
        var t = Math.pow(10, lt), T = 1 / Math.sqrt(t), r = T > Tf ? Math.exp(-Q / T) : x0 * Math.exp(-Math.max(0, t - 1 / (Tf * Tf)) / tau);
        if (t > tnuc) r = NaN;
        pts.push([b.x(lt), b.y(r)]);
      }
      line(c, pts.filter(function (p) { return isFinite(p[1]); }), C.c1, 2.4);
      line(c, [[b.x(Math.log10(tnuc)), b.y(0)], [b.x(Math.log10(tnuc)), b.y(1.05)]], C.c4, 1, [4, 3]);
      text(c, 'deuterium survives: neutrons locked into helium', b.x(Math.log10(tnuc)) - 4, b.y(0.95), C.c4, 'right');
      s.out.innerHTML = 'Reactions freeze out at kT = <b>' + fx(Tf, 2) + ' MeV</b> with n/p = e^(−1.29/' + fx(Tf, 2) + ') = ' + fx(x0, 3) + '; free neutrons then decay (lifetime 15 min) until nucleosynthesis at about 3 minutes leaves n/p = <b>' + fx(x, 3) + '</b>. Nearly all of them end in helium: Y = 2(n/p)/(1 + n/p) = <b>' + fx(Y, 3) + '</b> of the mass. The observed 0.245 needs freeze-out near 0.74 MeV (n/p ≈ 1/6 at freeze-out, about 1/7 once some neutrons have decayed), which the weak interaction sets.';
    }
    UI.slider(s.ctl, { label: 'Freeze-out temperature (× 0.8 MeV)', min: 0.6, max: 1.6, value: st.tf, fmt: function (x) { return (0.8 * x).toFixed(2) + ' MeV'; }, oninput: function (x) { st.tf = x; draw(); } });
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 7 resolution: two Airy patterns
  function airy(x) { if (Math.abs(x) < 1e-6) return 1; var j = besselJ1(x); return Math.pow(2 * j / x, 2); }
  function besselJ1(x) {   // Abramowitz & Stegun polynomial approximations
    var ax = Math.abs(x), y, ans;
    if (ax < 8) {
      y = x * x;
      var a1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
      var a2 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y))));
      return a1 / a2;
    }
    var z = 8 / ax; y = z * z; var xx = ax - 2.356194491;
    var p1 = 1 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
    var p2 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
    ans = Math.sqrt(0.636619772 / ax) * (Math.cos(xx) * p1 - z * Math.sin(xx) * p2);
    return x < 0 ? -ans : ans;
  }
  X.airy = function (host) {
    var s = stage(host, 0.48), st = { D: 0.1, sep: 1.5, lam: 550 };
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, th = 1.22 * st.lam * 1e-9 / st.D * 206265;   // Rayleigh limit, arcsec
      var b = box(s, 50, 14, s.W - 70, s.H - 52, [-4, 4], [0, 2.1], { xl: 'angle on the sky (arcsec)', yl: 'brightness' });
      var k = 3.8317 / (th / 1.0), sum = [], p1 = [], p2 = [];
      for (var x = -4; x <= 4.0001; x += 0.02) {
        var a = airy(k * (x + st.sep / 2) / 1.0), bb = airy(k * (x - st.sep / 2) / 1.0);
        p1.push([b.x(x), b.y(a)]); p2.push([b.x(x), b.y(bb)]); sum.push([b.x(x), b.y(a + bb)]);
      }
      line(c, p1, C.c1, 1.2, [4, 3]); line(c, p2, C.c2, 1.2, [4, 3]); line(c, sum, C.ink, 2.2);
      var ok = st.sep >= th;
      s.out.innerHTML = 'A ' + fx(st.D * 100, 0) + ' cm aperture at ' + fx(st.lam, 0) + ' nm: Rayleigh limit 1.22λ/D = <b>' + fx(th, 2) + '″</b>. Two stars ' + fx(st.sep, 2) + '″ apart are <b>' + (ok ? 'resolved' : 'not resolved') + '</b>' + (ok ? '' : ': the dip between them is gone') + '. (On the ground, seeing blurs everything to about 1″ whatever the size of the mirror.)';
    }
    UI.slider(s.ctl, { label: 'Aperture D', min: 0.007, max: 10, value: st.D, log: true, fmt: function (x) { return x < 1 ? (x * 100).toFixed(1) + ' cm' : x.toFixed(1) + ' m'; }, oninput: function (x) { st.D = x; draw(); } });
    UI.slider(s.ctl, { label: 'Separation of the two stars', min: 0.02, max: 4, value: st.sep, log: true, fmt: function (x) { return x.toFixed(2) + '″'; }, oninput: function (x) { st.sep = x; draw(); } });
    UI.slider(s.ctl, { label: 'Wavelength', min: 300, max: 2200, value: st.lam, step: 10, fmt: function (x) { return Math.round(x) + ' nm'; }, oninput: function (x) { st.lam = x; draw(); } });
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 7 airmass and extinction
  X.airmass = function (host) {
    var s = stage(host, 0.48), st = { alt: 40, k: 0.21 };
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, W = s.W, H = s.H, R = W * 1.6, cx = W * 0.32, cy = H + R - 40, atm = 46;
      c.fillStyle = C.dark ? 'rgba(80,120,190,0.25)' : 'rgba(80,120,190,0.18)'; c.beginPath(); c.arc(cx, cy, R + atm, Math.PI * 1.15, Math.PI * 1.85); c.lineTo(cx, cy); c.fill();
      c.fillStyle = C.dark ? '#1b1814' : '#c9bfae'; c.beginPath(); c.arc(cx, cy, R, Math.PI * 1.15, Math.PI * 1.85); c.lineTo(cx, cy); c.fill();
      var ox = cx, oy = cy - R, a = st.alt * Math.PI / 180, L = 600;
      line(c, [[ox, oy], [ox + L * Math.cos(a), oy - L * Math.sin(a)]], C.c2, 2);
      line(c, [[ox, oy], [ox, oy - atm - 20]], C['ink-3'], 1, [3, 3]); text(c, 'zenith', ox + 4, oy - atm - 10, C['ink-3']);
      disc(c, ox, oy, 4, C.ink);
      var X1 = A.airmass(st.alt), dm = st.k * X1;
      text(c, 'starlight', ox + 140 * Math.cos(a) + 6, oy - 140 * Math.sin(a), C.c2);
      s.out.innerHTML = 'Altitude ' + fx(st.alt, 0) + '° (zenith angle ' + fx(90 - st.alt, 0) + '°): the light crosses <b>' + fx(X1, 2) + '</b> times as much air as straight up (≈ sec z). With ' + fx(st.k, 2) + ' mag of extinction per airmass the star is dimmed by <b>' + fx(dm, 2) + ' mag</b>, and reddened, because blue light scatters more. Below about 30° it gets worse fast.';
    }
    UI.slider(s.ctl, { label: 'Altitude of the star', min: 3, max: 90, value: st.alt, step: 1, fmt: function (x) { return Math.round(x) + '°'; }, oninput: function (x) { st.alt = x; draw(); } });
    UI.slider(s.ctl, { label: 'Extinction per airmass', min: 0.1, max: 0.4, value: st.k, step: 0.01, fmt: function (x) { return x.toFixed(2) + ' mag (' + (x > 0.26 ? 'sea level' : x > 0.18 ? '1 km' : 'high mountain') + ')'; }, oninput: function (x) { st.k = x; draw(); } });
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 7 a CCD frame and its calibration (simulated)
  X.ccd = function (host) {
    var s = stage(host, 0.5), st = { bias: false, dark: false, flat: false, crs: false };
    var N = 96, rng = 7; function rnd() { rng = (rng * 16807) % 2147483647; return rng / 2147483647; }
    function gauss() { var u = 1 - rnd(), v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
    var stars = []; for (var k = 0; k < 18; k++) stars.push([4 + rnd() * (N - 8), 4 + rnd() * (N - 8), 30 + 900 * Math.pow(rnd(), 3)]);
    var cr = []; for (k = 0; k < 9; k++) cr.push([Math.floor(rnd() * N), Math.floor(rnd() * N)]);
    var sky = new Float32Array(N * N), flat = new Float32Array(N * N), biasRow = new Float32Array(N), noise = new Float32Array(N * N);
    for (var j = 0; j < N; j++) { biasRow[j] = 300 + 25 * Math.sin(j / 9); for (var i = 0; i < N; i++) {
      var r2 = ((i - N / 2) * (i - N / 2) + (j - N / 2) * (j - N / 2)) / (N * N / 4), v = 1 - 0.35 * r2, d = Math.hypot(i - 70, j - 28);
      if (d < 7 && d > 4) v *= 0.82;   // a dust donut
      flat[j * N + i] = v * (1 + 0.03 * gauss()); noise[j * N + i] = gauss();
      var f = 40; stars.forEach(function (p) { f += p[2] * Math.exp(-((i - p[0]) * (i - p[0]) + (j - p[1]) * (j - p[1])) / 4.5); });
      sky[j * N + i] = f;
    } }
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, size = Math.min(s.H - 10, s.W * 0.55), x0 = 10, y0 = 5, im = c.createImageData(N, N);
      for (var j = 0; j < N; j++) for (var i = 0; i < N; i++) {
        var k = j * N + i, raw = sky[k] * flat[k] + biasRow[j] + 12 + 6 * noise[k];   // light x flat + bias + dark current + noise
        if (!st.crs && cr.some(function (p) { return p[0] === i && (p[1] === j || p[1] + 1 === j); })) raw += 2500;
        var v = raw; if (st.bias) v -= biasRow[j]; if (st.dark) v -= 12; if (st.flat) v /= flat[k];
        var g = Math.max(0, Math.min(255, Math.round(255 * Math.asinh(Math.max(0, v - (st.bias ? 20 : 300)) / 30) / Math.asinh(900 / 30))));
        im.data[4 * k] = g; im.data[4 * k + 1] = g; im.data[4 * k + 2] = g; im.data[4 * k + 3] = 255;
      }
      var tmp = document.createElement('canvas'); tmp.width = N; tmp.height = N; tmp.getContext('2d').putImageData(im, 0, 0);
      c.imageSmoothingEnabled = false; c.drawImage(tmp, x0, y0, size, size);
      var tx = x0 + size + 16;
      text(c, 'reduced = (raw − bias − dark) / flat', tx, 26, C.ink);
      ['raw: light × flat + bias + dark current', 'stripes: the bias level of each row', 'darker corners and a ring: vignetting and dust (flat)', 'bright streaks: cosmic rays'].forEach(function (t, k) { text(c, t, tx, 52 + 18 * k, C['ink-2']); });
      s.out.innerHTML = 'A simulated raw frame. ' + (st.bias && st.dark && st.flat && st.crs ? 'Fully reduced: what is left is light from the sky, and noise.' : 'Switch on each calibration and watch what it removes.');
    }
    var row = document.createElement('div'); row.className = 'ctl-row';
    [['bias', 'Subtract the bias'], ['dark', 'Subtract the dark'], ['flat', 'Divide by the flat'], ['crs', 'Combine frames (median: no cosmic rays)']].forEach(function (b) {
      var l = document.createElement('label'); l.className = 'check'; l.innerHTML = '<input type="checkbox"> ' + b[1];
      l.querySelector('input').addEventListener('change', function (e) { st[b[0]] = e.target.checked; draw(); });
      row.appendChild(l);
    });
    s.ctl.appendChild(row);
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 8 two-element interferometer
  X.interferometer = function (host) {
    var s = stage(host, 0.5), st = { B: 1, lam: 0.21 }, D = 25;      // dishes 25 m across
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, th = st.lam / (st.B * 1000), thArc = th * 206265, beam = 1.03 * st.lam / D * 206265;   // B in km
      // on a fixed angular range set by one dish's beam: a longer baseline packs more, finer fringes into it
      var half = 1.5 * beam, b = box(s, 50, 14, s.W - 70, s.H - 56, [-half, half], [0, 1.05], { xl: 'position on the sky [arcsec]', yl: 'signal' });
      [-1, -0.5, 0, 0.5, 1].forEach(function (u) { var v = u * half; text(c, fm(v, 2), b.x(v), b.y(0) + 13, C['ink-3'], 'center'); });
      var env = []; for (var x = -half; x <= half * 1.0001; x += half / 200) env.push([b.x(x), b.y(Math.exp(-2.77 * (x / beam) * (x / beam)))]);
      line(c, env, C['ink-3'], 1.4, [4, 3]);
      var nfr = 2 * half / thArc;
      if (nfr < 400) {
        var pts = [], dx = Math.min(half / 400, thArc / 24); for (x = -half; x <= half * 1.0001; x += dx) pts.push([b.x(x), b.y(Math.exp(-2.77 * (x / beam) * (x / beam)) * (0.5 + 0.5 * Math.cos(2 * Math.PI * x / thArc)))]);
        line(c, pts, C.c1, 1.6);
      } else { c.fillStyle = C.dark ? 'rgba(120,170,255,0.25)' : 'rgba(40,90,200,0.2)'; c.beginPath(); c.moveTo(env[0][0], b.y(0)); env.forEach(function (p) { c.lineTo(p[0], p[1]); }); c.lineTo(env[env.length - 1][0], b.y(0)); c.fill(); text(c, fm(nfr, 2) + ' fringes across the beam: too fine to draw', b.x(-half) + 8, b.y(0.95), C['ink-2']); }
      var res = thArc > 60 ? fx(thArc / 60, 1) + '′' : thArc > 1 ? fx(thArc, 2) + '″' : thArc > 1e-3 ? fx(thArc * 1e3, 2) + ' mas' : fx(thArc * 1e6, 1) + ' μas';
      s.out.innerHTML = 'One 25 m dish sees a patch ' + fm(beam, 2) + '″ across (dashed). Two of them ' + fm(st.B, 3) + ' km apart, at λ = ' + (st.lam < 0.01 ? fx(st.lam * 1000, 1) + ' mm' : fx(st.lam * 100, 0) + ' cm') + ', lay fringes λ/B = <b>' + res + '</b> apart across it: the resolution of a single dish ' + fm(st.B, 3) + ' km wide. The dish size sets the field of view; the baseline sets the detail.';
    }
    UI.slider(s.ctl, { label: 'Baseline B', min: 0.03, max: 12000, value: st.B, log: true, fmt: function (x) { return fm(x, 2) + ' km'; }, oninput: function (x) { st.B = x; draw(); } });
    UI.slider(s.ctl, { label: 'Wavelength', min: 0.0013, max: 2, value: st.lam, log: true, fmt: function (x) { return x < 0.01 ? (x * 1000).toFixed(1) + ' mm' : (x * 100).toFixed(0) + ' cm'; }, oninput: function (x) { st.lam = x; draw(); } });
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 8 a dispersed radio pulse
  X.dispersion = function (host) {
    var s = stage(host, 0.5), st = { dm: 500 };
    var stop = loop(function (t) {
      s.fit(); var c = s.ctx, C = s.C, f0 = 0.4, f1 = 0.8;  // GHz (CHIME's band)
      var b = box(s, 56, 14, s.W - 76, s.H - 52, [0, 12], [f0, f1], { xl: 'arrival time (s)', yl: 'frequency (GHz)' });
      for (var f = f0; f <= f1 + 1e-9; f += 0.01) {
        var delay = 4.149e-3 * st.dm * (1 / (f * f) - 1 / (f1 * f1)), tt = (t % 12);
        var x = b.x(Math.min(12, delay)), y = b.y(f);
        c.fillStyle = tt > delay ? C.c1 : C.rule; c.fillRect(x - 2, y - 2, 4, 4);
      }
      var spread = 4.149e-3 * st.dm * (1 / (f0 * f0) - 1 / (f1 * f1));
      s.out.innerHTML = 'DM = <b>' + Math.round(st.dm) + '</b> pc/cm³: the 400 MHz end arrives <b>' + fx(spread, 2) + ' s</b> after the 800 MHz end (Δt = 4.15 ms × DM × [(ν/GHz)⁻² − ...]). Free electrons slow low frequencies most, so the sweep measures how many electrons lie in the way.';
    });
    UI.slider(s.ctl, { label: 'Dispersion measure', min: 10, max: 2000, value: st.dm, log: true, fmt: function (x) { return Math.round(x) + ' pc/cm³'; }, oninput: function (x) { st.dm = x; } });
    return { stop: stop };
  };

  // ------------------------------------------------------------------------------------------ 8 a gravitational-wave chirp
  X.chirp = function (host) {
    var s = stage(host, 0.48), st = { mc: 28 };
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, Mc = st.mc * A.C.Msun * A.C.G / Math.pow(A.C.c, 3);   // chirp mass in seconds
      var b = box(s, 50, 14, s.W - 70, s.H - 52, [-0.25, 0.01], [-1.15, 1.15], { xl: 'time before merger (s)', yl: 'strain (scaled)' });
      var pts = [], phase = 0, last = -0.25;
      for (var t = -0.25; t < -0.0005; t += 0.0002) {
        var tau = -t, fgw = (1 / Math.PI) * Math.pow(5 / (256 * tau), 3 / 8) * Math.pow(Mc, -5 / 8);
        phase += 2 * Math.PI * fgw * (t - last); last = t;
        if (fgw > 1500) break;
        pts.push([b.x(t), b.y(Math.pow(Math.min(1, fgw / 300), 2 / 3) * Math.cos(phase))]);
      }
      line(c, pts, C.c1, 1.3);
      var f50 = (1 / Math.PI) * Math.pow(5 / (256 * 0.05), 3 / 8) * Math.pow(Mc, -5 / 8);
      s.out.innerHTML = 'Chirp mass ' + fx(st.mc, 1) + ' M☉: 50 ms before merging the waves are at ' + fx(f50, 0) + ' Hz and climbing. Heavier pairs chirp lower and end sooner; two neutron stars (1.2 M☉) sweep through LIGO’s band for a whole minute.';
    }
    UI.slider(s.ctl, { label: 'Chirp mass', min: 1.2, max: 60, value: st.mc, log: true, fmt: function (x) { return x.toFixed(1) + ' M☉'; }, oninput: function (x) { st.mc = x; draw(); } });
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 9 photometric redshifts: the Lyman and 4000 A breaks through filters
  var FILTERS = [['u', 355, 60], ['g', 469, 140], ['r', 617, 140], ['i', 748, 150], ['z', 893, 150]];
  function sed(lamRest, kind) {   // a schematic galaxy spectrum (relative f_lambda)
    if (lamRest < 91.2) return 0;
    if (kind === 'old') return lamRest < 400 ? 0.25 + 0.2 * (lamRest - 91.2) / 308.8 : 1 - 0.0002 * (lamRest - 400);
    var base = Math.pow(lamRest / 150, -1.2); if (lamRest < 121.6) base *= 0.4;   // young: blue continuum, forest bites below Lya
    return base;
  }
  X.photoz = function (host) {
    var s = stage(host, 0.5), st = { z: 0.4, kind: 'old' };
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, b = box(s, 50, 14, s.W - 70, s.H - 52, [300, 1000], [0, 1.25], { xl: 'observed wavelength (nm)', yl: 'brightness' });
      FILTERS.forEach(function (fl, k) { c.fillStyle = [C.c5, C.c1, C.c3, C.c2, C.c4][k]; c.globalAlpha = 0.15; c.fillRect(b.x(fl[1] - fl[2] / 2), b.y(1.25), b.x(fl[1] + fl[2] / 2) - b.x(fl[1] - fl[2] / 2), b.y(0) - b.y(1.25)); c.globalAlpha = 1; text(c, fl[0], b.x(fl[1]), b.y(1.18), C['ink-2'], 'center'); });
      var pts = [], mx = 0, vals = [];
      for (var l = 300; l <= 1000; l += 2) { var v = sed(l / (1 + st.z), st.kind); vals.push([l, v]); mx = Math.max(mx, v); }
      vals.forEach(function (p) { pts.push([b.x(p[0]), b.y(p[1] / mx)]); });
      line(c, pts, C.ink, 2);
      var mags = FILTERS.map(function (fl) { var sm = 0, n = 0; vals.forEach(function (p) { if (Math.abs(p[0] - fl[1]) < fl[2] / 2) { sm += p[1]; n++; } }); return sm / n / mx; });
      mags.forEach(function (m, k) { disc(c, b.x(FILTERS[k][1]), b.y(m), 5, [C.c5, C.c1, C.c3, C.c2, C.c4][k]); });
      var brk = st.kind === 'old' ? 400 * (1 + st.z) : 91.2 * (1 + st.z), drop = mags.findIndex(function (m) { return m > 0.15; });
      s.out.innerHTML = (st.kind === 'old' ? 'An old galaxy: its 4000 Å break' : 'A young galaxy: its Lyman break (91.2 nm; the Lyman-α forest eats into it up to 121.6 nm)') + ' falls at <b>' + Math.round(brk) + ' nm</b> for z = ' + fx(st.z, 2) + '. The five dots are what broad filters measure: where the light jumps between filters tells the redshift without a spectrum. ' + (st.kind === 'young' && drop > 0 ? 'Here it “drops out” of ' + FILTERS.slice(0, drop).map(function (f) { return f[0]; }).join(', ') + ': a dropout galaxy.' : '');
    }
    UI.slider(s.ctl, { label: 'Redshift z', min: 0, max: 6.5, value: st.z, step: 0.01, fmt: function (x) { return x.toFixed(2); }, oninput: function (x) { st.z = x; draw(); } });
    var row = document.createElement('div'); row.className = 'ctl-row';
    [['old', 'Old stars'], ['young', 'Young stars']].forEach(function (k) { var bt = UI.button(row, k[1], function () { st.kind = k[0]; draw(); }, 'small ghost'); void bt; });
    s.ctl.appendChild(row);
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 8 the atmosphere's windows
  X.windows = function (host) {
    var s = stage(host, 0.46), SITES = [['sea level', 0], ['Mitzpe Ramon (0.9 km)', 0.9], ['Mauna Kea (4.2 km)', 4.2], ['SOFIA aircraft (12 km)', 12], ['balloon (40 km)', 40], ['orbit (500 km)', 500]], st = { k: 0 };
    // opacity at sea level by log10 wavelength (m): near-infrared windows (J, H, K, L, M) between water bands
    var OP = [[-12, 1], [-7.4, 1], [-6.62, 1], [-6.55, 0.05], [-6.2, 0.0], [-6.1, 0.15], [-6.04, 0.0], [-5.98, 0.6], [-5.8, 0.05], [-5.72, 0.6], [-5.66, 0.1], [-5.58, 0.9],
      [-5.45, 0.15], [-5.3, 0.9], [-5.0, 0.2], [-4.9, 0.9], [-4.7, 1], [-3.6, 1], [-3.3, 0.6], [-3.1, 0.4], [-2.7, 0.15], [-2.0, 0.0], [1.2, 0.0], [1.5, 1], [3, 1]];
    var BANDS = [['γ', -12, -11], ['X', -11, -8], ['UV', -8, -6.4], ['vis', -6.4, -6.15], ['infrared', -6.15, -3], ['radio', -3, 3]];
    function op0(x) { var k = 0; while (k < OP.length - 2 && OP[k + 1][0] < x) k++; var a = OP[k], q = OP[k + 1]; return a[1] + (q[1] - a[1]) * (x - a[0]) / (q[0] - a[0]); }
    // what still absorbs above a height h (km): water vapour has a 2 km scale height, the air 7 km; ozone (UV) sits at
    // 20-30 km; X- and gamma rays are stopped by the whole column; long radio waves by the ionosphere (100-300 km)
    function blocked(x, h) {
      if (x > 1.45) return h < 300 ? op0(x) : 0;
      var o = op0(x), tau;
      if (x < -8) tau = 1e6 * Math.exp(-h / 7);
      else if (x < -6.55) tau = 3e3 * Math.exp(-Math.max(0, h - 25) / 4) * (h < 25 ? 1 : 1);
      else if (x < -3.0 && x > -6.15) tau = 30 * o * Math.exp(-h / 2);
      else tau = -Math.log(Math.max(1e-3, 1 - o)) * Math.exp(-h / 7);
      return 1 - Math.exp(-tau);
    }
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, h = SITES[st.k][1], b = box(s, 30, 30, s.W - 50, s.H - 74, [-12, 3], [0, 1.05], { xl: 'wavelength (log₁₀ metres)' });
      var pts = []; for (var x = -12; x <= 3; x += 0.02) pts.push([b.x(x), b.y(blocked(x, h))]);
      c.fillStyle = C.dark ? 'rgba(160,170,190,0.35)' : 'rgba(90,100,120,0.3)'; c.beginPath(); c.moveTo(b.x(-12), b.y(0)); pts.forEach(function (p) { c.lineTo(p[0], p[1]); }); c.lineTo(b.x(3), b.y(0)); c.fill();
      BANDS.forEach(function (bd) { text(c, bd[0], b.x((bd[1] + bd[2]) / 2), 20, C['ink-2'], 'center'); });
      [-12, -9, -6, -3, 0, 3].forEach(function (v) { text(c, String(v), b.x(v), b.y(0) + 13, C['ink-3'], 'center'); });
      text(c, 'grey: absorbed above ' + SITES[st.k][0], b.x(-11.8), b.y(0.55), C.ink);
      var msg = ['From sea level only visible light, a few near-infrared windows and radio waves (about 1 mm to 30 m) get through.',
        'Mitzpe Ramon, at 0.9 km in a dry desert, is above a third of the water vapour: the near-infrared windows are a little cleaner.',
        'On Mauna Kea or in the Atacama, above most of the water, the near- and mid-infrared windows open and the submillimetre (ALMA) becomes possible.',
        'In an aircraft at 12 km, 99% of the water vapour is below: the far infrared opens (SOFIA, 2010–2022).',
        'A balloon at 40 km sees the whole infrared and the longer ultraviolet, above the ozone layer; X-rays still need more.',
        'In orbit everything reaches you, from gamma rays to the long radio waves the ionosphere reflects.'][st.k];
      s.out.innerHTML = msg;
    }
    UI.slider(s.ctl, { label: 'Telescope altitude', min: 0, max: SITES.length - 1, value: 0, step: 1, fmt: function (k) { return SITES[Math.round(k)][0]; }, oninput: function (k) { st.k = Math.round(k); draw(); } });
    draw(); return { stop: function () { } };
  };
})(typeof window !== 'undefined' ? window : globalThis);
