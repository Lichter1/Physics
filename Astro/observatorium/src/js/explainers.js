/* Observatorium — explainers: small live diagrams inside the lectures. Each one shows the idea behind a
 * relation (the geometry, the mechanism) with a slider or two; the missions then test it on real data.
 * OBS.EXPLAINERS[id](host, opts) builds one in host and returns {stop}. */
(function (G) {
  'use strict';
  var OBS = G.OBS, A = OBS.astro, UI = OBS.ui;
  var X = OBS.EXPLAINERS = {};

  // ------------------------------------------------------------------------------------------ helpers
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
  /** Run draw(t) every frame while the explainer is on screen; returns stop(). */
  function loop(draw) {
    var on = true, t0 = performance.now(), still = OBS.reducedMotion();
    (function f(now) { if (!on) return; draw(still ? 2.5 : (now - t0) / 1000); if (!still) G.requestAnimationFrame(f); })(t0);
    return function () { on = false; };
  }
  function disc(c, x, y, r, fill, stroke) { c.beginPath(); c.arc(x, y, Math.max(0.5, r), 0, 7); if (fill) { c.fillStyle = fill; c.fill(); } if (stroke) { c.strokeStyle = stroke; c.stroke(); } }
  function line(c, pts, col, w, dash) { c.beginPath(); c.strokeStyle = col; c.lineWidth = w || 1.5; c.setLineDash(dash || []); pts.forEach(function (p, i) { if (i) c.lineTo(p[0], p[1]); else c.moveTo(p[0], p[1]); }); c.stroke(); c.setLineDash([]); c.lineWidth = 1; }
  function text(c, s, x, y, col, align) { c.fillStyle = col; c.textAlign = align || 'left'; c.fillText(s, x, y); c.textAlign = 'left'; }
  function arrow(c, x0, y0, x1, y1, col) {
    line(c, [[x0, y0], [x1, y1]], col, 2);
    var a = Math.atan2(y1 - y0, x1 - x0), h = 7;
    c.beginPath(); c.moveTo(x1, y1); c.lineTo(x1 - h * Math.cos(a - 0.4), y1 - h * Math.sin(a - 0.4)); c.lineTo(x1 - h * Math.cos(a + 0.4), y1 - h * Math.sin(a + 0.4)); c.closePath(); c.fillStyle = col; c.fill();
  }
  function starCol(T, C) { return 'rgb(' + OBS.starRGB(A.bbBpRp(T), 'gaia', true) + ')'; }
  function f(v, s) { return OBS.fmt(v, s || 3); }
  /** A small plot area: returns mapping functions for a box inside the stage. */
  function plotBox(s, x0, y0, w, h, xr, yr, opt) {
    opt = opt || {};
    var lx = opt.logx, ly = opt.logy;
    function tx(v) { v = lx ? Math.log10(v) : v; return x0 + (v - xr[0]) / (xr[1] - xr[0]) * w; }
    function ty(v) { v = ly ? Math.log10(v) : v; return y0 + h - (v - yr[0]) / (yr[1] - yr[0]) * h; }
    var c = s.ctx; c.strokeStyle = s.C.rule; c.strokeRect(x0 + 0.5, y0 + 0.5, w, h);
    if (opt.xl) text(c, opt.xl, x0 + w / 2, y0 + h + 15, s.C['ink-2'], 'center');
    if (opt.yl) { c.save(); c.translate(x0 - 8, y0 + h / 2); c.rotate(-Math.PI / 2); text(c, opt.yl, 0, 0, s.C['ink-2'], 'center'); c.restore(); }
    return { x: tx, y: ty, x0: x0, y0: y0, w: w, h: h };
  }

  // ------------------------------------------------------------------------------------------ 1.1 magnitudes
  X.magnitude = function (host) {
    var s = stage(host, 0.38), st = { r: 0.01 };
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, dm = -2.5 * Math.log10(st.r);
      disc(c, s.W * 0.2, s.H * 0.45, 26, C.dark ? '#fff4ea' : '#3a3020');
      disc(c, s.W * 0.2 + 100, s.H * 0.45, 26 * Math.pow(st.r, 0.25), C.dark ? '#fff4ea' : '#3a3020');
      text(c, 'star A', s.W * 0.2, s.H * 0.45 + 46, C['ink-2'], 'center'); text(c, 'star B', s.W * 0.2 + 100, s.H * 0.45 + 46, C['ink-2'], 'center');
      var x0 = s.W * 0.5, w = s.W * 0.45, y = s.H * 0.45;
      line(c, [[x0, y], [x0 + w, y]], C.rule, 2);
      for (var m = 0; m <= 10; m++) { var x = x0 + w * m / 10; line(c, [[x, y - 5], [x, y + 5]], C['ink-3'], 1); text(c, String(m), x, y + 18, C['ink-3'], 'center'); }
      disc(c, x0, y, 6, C.c1); disc(c, x0 + w * Math.min(10, dm) / 10, y, 6, C.c2);
      text(c, 'magnitude scale (bigger = fainter)', x0 + w / 2, y - 16, C['ink-2'], 'center');
      s.out.innerHTML = 'B sends <b>' + f(st.r, 3) + '</b> of A’s light per second: it is <b>' + OBS.fix(dm, 2) + '</b> magnitudes fainter. Every 5 magnitudes is a factor of exactly 100; every magnitude a factor of 2.512.';
    }
    UI.slider(s.ctl, { label: 'B’s flux relative to A', min: 1e-4, max: 1, value: st.r, log: true, fmt: function (x) { return f(x, 2); }, oninput: function (x) { st.r = x; draw(); } });
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 1.1 filters: B and V
  // Johnson B and V passbands approximated as Gaussians (B: 442 nm, 94 nm wide; V: 540 nm, 88 nm wide); the eye's
  // daylight sensitivity as a Gaussian peaking at 555 nm. Zero point: a 5772 K blackbody gets the Sun's B - V = 0.65.
  function gauss(l, c, w) { var s = w / 2.355; return Math.exp(-0.5 * (l - c) * (l - c) / (s * s)); }
  function bandFlux(T, c, w) { var s = 0; for (var l = 300; l <= 750; l += 2) s += A.planck(l * 1e-9, T) * l * gauss(l, c, w); return s; }
  function rawBV(T) { return -2.5 * Math.log10(bandFlux(T, 442, 94) / bandFlux(T, 540, 88)); }
  var BV0 = null;
  X.filters = function (host) {
    var s = stage(host, 0.5), st = { T: 5772 };
    var PRE = [['Rigel', 12000, -0.03], ['Sun', 5772, 0.65], ['Betelgeuse', 3600, 1.85]];
    if (BV0 === null) BV0 = 0.653 - rawBV(5772);
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, pb = plotBox(s, 36, 12, s.W - 52, s.H - 52, [330, 730], [0, 1.15], {});
      text(c, 'wavelength [nm]', pb.x0 + pb.w, pb.y0 + pb.h + 30, C['ink-2'], 'right');
      for (var l = 380; l < 720; l += 4) { var hue = 270 - (l - 380) / 340 * 270; c.fillStyle = 'hsla(' + hue + ',85%,55%,0.18)'; c.fillRect(pb.x(l), pb.y0 + pb.h - 10, pb.x(l + 4) - pb.x(l), 10); }
      function curve(fn, col, w, dash, fill) {
        var pts = []; for (var x = 330; x <= 730; x += 2) pts.push([pb.x(x), pb.y(fn(x))]);
        if (fill) { c.beginPath(); c.moveTo(pts[0][0], pb.y(0)); pts.forEach(function (p) { c.lineTo(p[0], p[1]); }); c.lineTo(pts[pts.length - 1][0], pb.y(0)); c.closePath(); c.globalAlpha = 0.18; c.fillStyle = col; c.fill(); c.globalAlpha = 1; }
        line(c, pts, col, w, dash);
      }
      curve(function (x) { return gauss(x, 442, 94); }, C.c1, 2, null, true);
      curve(function (x) { return gauss(x, 540, 88); }, C.c3, 2, null, true);
      curve(function (x) { return gauss(x, 555, 100); }, C['ink-2'], 1.5, [4, 4]);
      var pk = 0; for (var x = 330; x <= 730; x += 2) pk = Math.max(pk, A.planck(x * 1e-9, st.T));
      curve(function (x) { return A.planck(x * 1e-9, st.T) / pk; }, C.accent, 2.4);
      [400, 500, 600, 700].forEach(function (t) { text(c, String(t), pb.x(t), pb.y0 + pb.h + 13, C['ink-3'], 'center'); });
      text(c, 'B filter', pb.x(442), pb.y(1.04), C.c1, 'center'); text(c, 'V filter', pb.x(540), pb.y(1.04), C.c3, 'center');
      text(c, 'human eye', pb.x(640), pb.y(gauss(640, 555, 100)) - 6, C['ink-2']);
      var bv = rawBV(st.T) + BV0, fb = bandFlux(st.T, 442, 94), fv = bandFlux(st.T, 540, 88);
      s.out.innerHTML = 'Gold: the spectrum of a <b>' + Math.round(st.T).toLocaleString('en-US') + ' K</b> star (scaled to fit). Through B it gives <b>' + f(fb / fv, 2) + '×</b> the light it gives through V, so B − V = <b>' + OBS.fix(bv, 2) + '</b> (for a perfect blackbody). Bigger B − V is redder. ' +
        'Measured: Rigel −0.03, the Sun 0.65, Betelgeuse 1.85; cool stars come out redder than a blackbody because molecules absorb their blue light (mission 1.4).';
    }
    var row = UI.row(s.ctl), seg = document.createElement('div'); seg.className = 'seg'; row.appendChild(seg);
    var sl = UI.slider(s.ctl, { label: 'Star’s temperature', min: 3000, max: 25000, value: st.T, log: true, unit: 'K', fmt: function (x) { return Math.round(x).toLocaleString('en-US'); }, oninput: function (x) { st.T = x; draw(); } });
    PRE.forEach(function (p) { UI.button(seg, p[0], function () { st.T = p[1]; sl.set(p[1], true); draw(); }, 'seg-b'); });
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 1.2 parallax
  X.parallax = function (host) {
    var s = stage(host, 0.5), st = { d: 3.5 };
    var stop = loop(function (t) {
      s.fit(); var c = s.ctx, C = s.C, cx = s.W * 0.2, cy = s.H * 0.5, R = s.H * 0.28;
      var ang = t * 0.9, ex = cx + R * Math.cos(ang), ey = cy + R * Math.sin(ang);
      var sx = s.W * 0.2 + R * (1.2 + 3.2 * Math.log10(1 + st.d)), sy = cy, wall = s.W - 18;
      disc(c, cx, cy, 9, '#f2c55c'); text(c, 'Sun', cx, cy + 24, C['ink-2'], 'center');
      disc(c, cx, cy, R, null, C.rule); disc(c, ex, ey, 5, C.c1); text(c, 'Earth', ex + 8, ey - 8, C.c1);
      // lines of sight from both sides of the orbit and from now
      var proj = function (yE) { return sy + (sy - yE) * (wall - sx) / (sx - cx); };
      line(c, [[cx, cy - R], [wall, proj(cy - R)]], C['ink-3'], 1, [3, 4]); line(c, [[cx, cy + R], [wall, proj(cy + R)]], C['ink-3'], 1, [3, 4]);
      line(c, [[ex, ey], [wall, sy + (sy - ey) * (wall - sx) / (sx - ex)]], C.c1, 1.2);
      disc(c, sx, sy, 6, C.dark ? '#ffe7c0' : '#7a5a18'); text(c, 'star', sx, sy - 12, C['ink-2'], 'center');
      line(c, [[wall, 10], [wall, s.H - 10]], C.rule, 3); text(c, 'distant stars', wall - 4, 16, C['ink-3'], 'right');
      disc(c, wall, sy + (sy - ey) * (wall - sx) / (sx - ex), 4, C.accent);
      // the angle p, between the line to the Sun and the line from the edge of the orbit
      line(c, [[cx, cy], [sx, sy]], C.c2, 1.2); line(c, [[cx, cy - R], [sx, sy]], C.c2, 1.2);
      text(c, 'p', sx - 34, sy - 8, C.c2);
      text(c, '1 AU', cx + 4, cy - R / 2, C['ink-3']);
      s.out.innerHTML = 'At <b>' + f(st.d) + ' pc</b> the parallax is p = 1/d = <b>' + f(1 / st.d) + '″</b>. The yellow dot on the far wall swings back and forth by 2p each year. (Angles drawn about 100,000 times too large.)';
    });
    UI.slider(s.ctl, { label: 'Distance to the star', min: 1, max: 100, value: st.d, log: true, unit: 'pc', oninput: function (x) { st.d = x; } });
    return { stop: stop };
  };

  // ------------------------------------------------------------------------------------------ 1.3 inverse square
  X.inverse = function (host) {
    var s = stage(host, 0.42), st = { d: 10 };
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, x0 = 40, y = s.H * 0.5, u = (s.W - 120) / 3.4;
      disc(c, x0, y, 10, '#f2c55c');
      [1, 2, 3].forEach(function (k) {
        var x = x0 + u * k, h = 18 * k;
        line(c, [[x0, y], [x, y - h]], C['ink-3'], 1, [2, 3]); line(c, [[x0, y], [x, y + h]], C['ink-3'], 1, [2, 3]);
        for (var i = 0; i < k; i++) for (var j = 0; j < k; j++) { c.strokeStyle = C.c1; c.strokeRect(x - 4 + i * 6 - 3 * k + 3, y - h + j * (2 * h / k), 5, 2 * h / k - 2); }
        text(c, k + 'r: light spread over ' + k * k + '×', x, y + h + 16, C['ink-2'], 'center');
      });
      var m = A.C.MV_sun + 5 * Math.log10(st.d / 10);
      s.out.innerHTML = 'The same light crosses an area that grows as r², so the flux falls as 1/r². A Sun-like star (M = 4.83) at <b>' + f(st.d) + ' pc</b> looks like magnitude <b>' + OBS.fix(m, 1) + '</b>' + (m < 6.5 ? ': visible to the naked eye.' : ': too faint for the naked eye (limit ≈ 6.5).');
    }
    UI.slider(s.ctl, { label: 'Distance of a Sun-like star', min: 1, max: 1000, value: st.d, log: true, unit: 'pc', oninput: function (x) { st.d = x; draw(); } });
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 1.4 Planck
  X.planck = function (host) {
    var s = stage(host, 0.48), st = { T: 5772 };
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, pb = plotBox(s, 46, 12, s.W - 110, s.H - 48, [100, 2000], [0, 1.1], { xl: 'wavelength [nm]', yl: 'brightness (peak = 1)' });
      // visible band
      for (var l = 400; l < 700; l += 5) { var hue = 270 - (l - 400) / 300 * 270; c.fillStyle = 'hsla(' + hue + ',80%,55%,0.25)'; c.fillRect(pb.x(l), pb.y0, pb.x(l + 5) - pb.x(l), pb.h); }
      [3000, 10000].forEach(function (T) {   // reference shapes, each scaled to its own peak
        var pk = A.planck(A.wienPeak(T), T), pts = [];
        for (var x = 100; x <= 2000; x += 10) pts.push([pb.x(x), pb.y(A.planck(x * 1e-9, T) / pk)]);
        line(c, pts, C['ink-3'], 1, [3, 3]); text(c, T.toLocaleString('en-US') + ' K', pb.x(Math.min(1900, A.wienPeak(T) * 1e9)) + 4, pb.y(1) + 12, C['ink-3']);
      });
      var pk = A.planck(A.wienPeak(st.T), st.T), pts = [], x;
      for (x = 100; x <= 2000; x += 5) pts.push([pb.x(x), pb.y(A.planck(x * 1e-9, st.T) / pk)]);
      line(c, pts, C.accent, 2.4);
      var lp = A.wienPeak(st.T) * 1e9; line(c, [[pb.x(Math.min(2000, Math.max(100, lp))), pb.y0], [pb.x(Math.min(2000, Math.max(100, lp))), pb.y0 + pb.h]], C.accent, 1, [4, 3]);
      disc(c, s.W - 32, 40, 22, starCol(st.T, C)); text(c, 'its colour', s.W - 32, 76, C['ink-2'], 'center');
      s.out.innerHTML = 'A <b>' + Math.round(st.T).toLocaleString('en-US') + ' K</b> blackbody peaks at λ = 2.898 mm·K / T = <b>' + Math.round(lp) + ' nm</b>. Each curve is drawn to its own peak, to compare shapes (dashed: 3,000 K and 10,000 K). In absolute terms hotter is brighter at every wavelength: per square metre this body gives out <b>' + f(Math.pow(st.T / 5772, 4), 2) + '×</b> the Sun’s power (σT⁴).';
    }
    UI.slider(s.ctl, { label: 'Temperature', min: 2500, max: 30000, value: st.T, log: true, unit: 'K', fmt: function (x) { return Math.round(x).toLocaleString('en-US'); }, oninput: function (x) { st.T = x; draw(); } });
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 1.5 Stefan–Boltzmann
  X.stefan = function (host) {
    var s = stage(host, 0.42), st = { T: 20000, L: 0.01 };
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, R = A.radius(st.L, st.T), y = s.H * 0.5;
      function px(r) { return 6 + 26 * (Math.log10(r) + 2.5); }   // log scale of size
      disc(c, s.W * 0.18, y, px(0.00917), C.c1); text(c, 'Earth', s.W * 0.18, y + px(0.00917) + 16, C['ink-2'], 'center');
      disc(c, s.W * 0.42, y, px(1), '#f2c55c'); text(c, 'Sun', s.W * 0.42, y + px(1) + 16, C['ink-2'], 'center');
      disc(c, s.W * 0.75, y, px(R), starCol(st.T, C)); text(c, 'your star', s.W * 0.75, Math.min(s.H - 4, y + px(R) + 16), C['ink-2'], 'center');
      text(c, 'sizes on a log scale', 8, 16, C['ink-3']);
      s.out.innerHTML = 'L = 4πR²σT⁴ ⇒ R = √L · (5772 K / T)² = <b>' + f(R) + ' R☉</b> = ' + f(R / 0.00917) + ' Earth radii. Hot and faint means small; cool and bright means huge.';
    }
    UI.slider(s.ctl, { label: 'Temperature', min: 2500, max: 40000, value: st.T, log: true, unit: 'K', fmt: function (x) { return Math.round(x).toLocaleString('en-US'); }, oninput: function (x) { st.T = x; draw(); } });
    UI.slider(s.ctl, { label: 'Luminosity', min: 1e-4, max: 1e6, value: st.L, log: true, unit: 'L☉', oninput: function (x) { st.L = x; draw(); } });
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 1.6 two bodies
  X.orbit2 = function (host) {
    var s = stage(host, 0.5), st = { q: 0.5 }, trace = [];
    var stop = loop(function (t) {
      s.fit(); var c = s.ctx, C = s.C, cx = s.W * 0.27, cy = s.H * 0.48, a = s.H * 0.36, ph = t * 0.8;
      var r1 = a * st.q / (1 + st.q), r2 = a / (1 + st.q);
      var x1 = cx + r1 * Math.cos(ph), y1 = cy + r1 * Math.sin(ph) * 0.35, x2 = cx - r2 * Math.cos(ph), y2 = cy - r2 * Math.sin(ph) * 0.35;
      c.strokeStyle = C.rule; c.beginPath(); c.ellipse(cx, cy, r1, r1 * 0.35, 0, 0, 7); c.stroke(); c.beginPath(); c.ellipse(cx, cy, r2, r2 * 0.35, 0, 0, 7); c.stroke();
      line(c, [[cx - 4, cy], [cx + 4, cy]], C.ink, 1); line(c, [[cx, cy - 4], [cx, cy + 4]], C.ink, 1);
      disc(c, x1, y1, 10 + 6 * (1 - st.q), C.c1); disc(c, x2, y2, 6 + 8 * st.q, C.c2);
      text(c, 'heavier', x1, y1 - 18, C.c1, 'center'); text(c, 'lighter', x2, y2 - 16, C.c2, 'center');
      text(c, 'you are below, looking up', cx, s.H - 6, C['ink-3'], 'center');
      // line-of-sight velocities, traced
      var v1 = Math.cos(ph) * st.q / (1 + st.q), v2 = -Math.cos(ph) / (1 + st.q);
      trace.push([t, v1, v2]); while (trace.length && t - trace[0][0] > 8) trace.shift();
      var pb = plotBox(s, s.W * 0.55, 14, s.W * 0.42, s.H - 46, [t - 8, t], [-1.05, 1.05], { xl: 'time →', yl: 'velocity toward you' });
      line(c, trace.map(function (p) { return [pb.x(p[0]), pb.y(p[1])]; }), C.c1, 2); line(c, trace.map(function (p) { return [pb.x(p[0]), pb.y(p[2])]; }), C.c2, 2);
      s.out.innerHTML = 'Both orbit the centre of mass (+). M₁ r₁ = M₂ r₂, so M₁ v₁ = M₂ v₂: with M₂/M₁ = <b>' + st.q.toFixed(2) + '</b>, the lighter star swings <b>' + (1 / st.q).toFixed(2) + '×</b> faster. The ratio of the two curves’ amplitudes is the mass ratio.';
    });
    UI.slider(s.ctl, { label: 'Mass ratio M₂ / M₁', min: 0.1, max: 1, value: st.q, fmt: function (x) { return x.toFixed(2); }, oninput: function (x) { st.q = x; trace.length = 0; } });
    return { stop: stop };
  };

  // ------------------------------------------------------------------------------------------ 1.7 power laws
  X.powerlaw = function (host) {
    var s = stage(host, 0.45), st = { a: 3.5 };
    function draw() {
      s.fit(); var c = s.ctx, C = s.C;
      var p1 = plotBox(s, 46, 12, s.W * 0.4, s.H - 46, [0, 10], [0, Math.pow(10, st.a)], { xl: 'M (linear)', yl: 'L (linear)' });
      var pts = []; for (var m = 0; m <= 10; m += 0.1) pts.push([p1.x(m), p1.y(Math.pow(m, st.a))]); line(c, pts, C.accent, 2);
      var p2 = plotBox(s, s.W * 0.56, 12, s.W * 0.4, s.H - 46, [-1, 1], [-st.a, st.a], { logx: true, logy: true, xl: 'M (log)', yl: 'L (log)' });
      pts = []; for (var e = -1; e <= 1; e += 0.05) pts.push([p2.x(Math.pow(10, e)), p2.y(Math.pow(Math.pow(10, e), st.a))]); line(c, pts, C.accent, 2);
      text(c, 'slope = ' + st.a.toFixed(2), p2.x0 + 8, p2.y0 + 16, C.accent);
      s.out.innerHTML = 'L ∝ M^a curves steeply on linear axes but is a straight line of slope a on logarithmic ones: log L = a log M + const. With a = <b>' + st.a.toFixed(2) + '</b>, 10× the mass gives <b>' + f(Math.pow(10, st.a)) + '×</b> the light, and the lifetime M/L shrinks by ' + f(Math.pow(10, st.a - 1)) + '×.';
    }
    UI.slider(s.ctl, { label: 'Exponent a', min: 1, max: 5, value: st.a, fmt: function (x) { return x.toFixed(2); }, oninput: function (x) { st.a = x; draw(); } });
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 1.8 / 2.3 sliding sequences
  function msCurve(c) { // a schematic main sequence: absolute G magnitude against BP-RP
    return 2.2 + 3.9 * c + (c > 1.4 ? 1.6 * (c - 1.4) : 0);
  }
  X.cmd = function (host) {
    var s = stage(host, 0.5), st = { mu: 3 }, target = 5.66;
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, pb = plotBox(s, 46, 10, s.W - 70, s.H - 44, [-0.2, 3], [18, -1], { xl: 'colour', yl: 'brightness (up = brighter)' });
      var ref = [], cl = [];
      for (var x = 0; x <= 3; x += 0.05) { ref.push([pb.x(x), pb.y(msCurve(x) + st.mu)]); cl.push([pb.x(x), pb.y(msCurve(x) + target)]); }
      line(c, cl, C.c2, 5); line(c, ref, C.c3, 2, [5, 4]);
      text(c, 'cluster (apparent)', pb.x(2.2), pb.y(msCurve(2.2) + target) + 18, C.c2); text(c, 'nearby stars + shift', pb.x(0.4), pb.y(msCurve(0.4) + st.mu) - 8, C.c3);
      s.out.innerHTML = 'Shift m − M = <b>' + st.mu.toFixed(2) + '</b> mag ⇒ d = 10^((m−M)/5 + 1) = <b>' + f(A.distFromMu(st.mu)) + ' pc</b>' + (Math.abs(st.mu - target) < 0.15 ? '. They line up: this is the cluster’s distance.' : '.');
    }
    UI.slider(s.ctl, { label: 'Shift (distance modulus)', min: 0, max: 12, value: st.mu, fmt: function (x) { return x.toFixed(2); }, unit: 'mag', oninput: function (x) { st.mu = x; draw(); } });
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 2.1 turn-off
  X.turnoff = function (host) {
    var s = stage(host, 0.55), st = { logt: 8.5 };
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, a = A.ML_SLOPE, t = Math.pow(10, st.logt);
      var Mto = Math.pow(t / 1e10, 1 / (1 - a)), Lto = Math.pow(Mto, a), MGto = 4.74 - 2.5 * Math.log10(Lto);
      var pb = plotBox(s, 46, 10, s.W - 70, s.H - 44, [-0.4, 3], [13, -6], { xl: 'colour (blue → red)', yl: 'brightness' });
      // a zero-age main sequence from O stars (M_G ≈ −5.5) to red dwarfs, roughly as Gaia sees it
      var ZAMS = [[-0.35, -5.5], [-0.2, -1.5], [0, 1.0], [0.4, 2.8], [0.8, 4.6], [1.2, 6.3], [1.8, 8.8], [2.5, 11.5], [3.0, 12.8]];
      function zM(x) { for (var k = 1; k < ZAMS.length; k++) if (x <= ZAMS[k][0]) { var u = (x - ZAMS[k - 1][0]) / (ZAMS[k][0] - ZAMS[k - 1][0]); return ZAMS[k - 1][1] + u * (ZAMS[k][1] - ZAMS[k - 1][1]); } return ZAMS[ZAMS.length - 1][1]; }
      var ms = [], dead = [], xt = -0.35;
      for (var x = -0.35; x <= 3; x += 0.01) { var M = zM(x); (M >= MGto ? ms : dead).push([pb.x(x), pb.y(M)]); if (M < MGto) xt = x; }
      line(c, dead, C['ink-3'], 3, [2, 4]); line(c, ms, C.accent, 5);
      var gb = [];
      for (var k = 0; k <= 20; k++) { var u = k / 20; gb.push([pb.x(xt + 0.15 * u + 1.1 * u * u), pb.y(MGto - 0.5 * u - 3.5 * u * u)]); }
      if (MGto > -5.5) line(c, gb, C.c4, 3);
      text(c, 'turn-off', pb.x(xt) + 8, pb.y(MGto) + 4, C.accent);
      s.out.innerHTML = 'At <b>' + (t >= 1e9 ? f(t / 1e9, 2) + ' Gyr' : f(t / 1e6, 2) + ' Myr') + '</b>, stars heavier than <b>' + f(Mto, 2) + ' M☉</b> have left the main sequence (t_MS ≈ 10 Gyr · M^{1−a}, a = ' + a.toFixed(2) + '). The turn-off sits at M_G ≈ ' + OBS.fix(MGto, 1) + '; red: the giants it feeds. Grey: the part of the main sequence that is gone.';
    }
    UI.slider(s.ctl, { label: 'Age of the cluster', min: 7, max: 10.15, value: st.logt, fmt: function (x) { var t = Math.pow(10, x); return t >= 1e9 ? (t / 1e9).toFixed(1) + ' Gyr' : Math.round(t / 1e6) + ' Myr'; }, oninput: function (x) { st.logt = x; draw(); } });
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 2.2 Cepheids
  X.cepheid = function (host) {
    var s = stage(host, 0.48), st = { P: 10 }, trace = [];
    var stop = loop(function (t) {
      s.fit(); var c = s.ctx, C = s.C, ph = (t / (st.P / 6)) % 1;
      var lum = ph < 0.25 ? ph / 0.25 : 1 - (ph - 0.25) / 0.75, rad = 0.85 + 0.15 * Math.sin(2 * Math.PI * (ph - 0.1));
      disc(c, s.W * 0.16, s.H * 0.45, s.H * 0.26 * rad, 'rgba(255,220,150,' + (0.45 + 0.55 * lum) + ')');
      text(c, 'it swells and shrinks', s.W * 0.16, s.H - 8, C['ink-3'], 'center');
      trace.push([t, lum]); while (trace.length && t - trace[0][0] > 6) trace.shift();
      var pb = plotBox(s, s.W * 0.34, 12, s.W * 0.3, s.H - 46, [t - 6, t], [-0.1, 1.1], { xl: 'time', yl: 'brightness' });
      line(c, trace.map(function (p) { return [pb.x(p[0]), pb.y(p[1])]; }), C.accent, 2);
      var pl = plotBox(s, s.W * 0.73, 12, s.W * 0.24, s.H - 46, [0, 2], [-1, -7], { xl: 'log P [d]', yl: 'M_V' });
      line(c, [[pl.x(0), pl.y(-2.43 * (0 - 1) - 4.05)], [pl.x(2), pl.y(-2.43 * (2 - 1) - 4.05)]], C['ink-3'], 1.5);
      var MV = -2.43 * (Math.log10(st.P) - 1) - 4.05; disc(c, pl.x(Math.log10(st.P)), pl.y(MV), 5, C.accent);
      s.out.innerHTML = 'Period <b>' + f(st.P, 2) + ' d</b> (animation sped up): M_V ≈ −2.43 (log P − 1) − 4.05 = <b>' + OBS.fix(MV, 2) + '</b>, i.e. ' + f(Math.pow(10, -0.4 * (MV - 4.83))) + ' L☉ in V. Bigger Cepheids are brighter and pulse more slowly.';
    });
    UI.slider(s.ctl, { label: 'Pulsation period', min: 1, max: 60, value: st.P, log: true, unit: 'd', oninput: function (x) { st.P = x; trace.length = 0; } });
    return { stop: stop };
  };

  // ------------------------------------------------------------------------------------------ the distance ladder
  X.ladder = function (host, o) {
    var s = stage(host, 0.6);
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, pb = plotBox(s, 20, 8, s.W - 40, s.H - 40, [-6, 10.5], [0, 7], { logx: true, xl: 'distance [pc] (log scale)' });
      for (var e = -5; e <= 10; e += 1) { var x = pb.x(Math.pow(10, e)); line(c, [[x, pb.y0 + pb.h], [x, pb.y0 + pb.h - 5]], C['ink-3'], 1); if (e % 2 === 0) text(c, '10^' + e, x, pb.y0 + pb.h - 8, C['ink-3'], 'center'); }
      var R = [['radar, Kepler’s laws', -6, -4.5], ['parallax (Gaia)', 0, 4], ['main-sequence fitting', 1, 4.5], ['Cepheids', 3, 7.6], ['tip of the red giant branch', 4.5, 7.4],
        ['water masers (NGC 4258)', 6.75, 7.0], ['Tully–Fisher, Faber–Jackson, fundamental plane', 6, 8.5], ['type Ia supernovae', 7, 9.8], ['Hubble–Lemaître law', 7.5, 10.4]];
      var hi = o && o.hi !== undefined ? [].concat(o.hi) : [];
      R.forEach(function (r, k) {
        var x0 = pb.x(Math.pow(10, r[1])), x1 = pb.x(Math.pow(10, r[2])), y = pb.y(6.5 - k * 0.74), on = hi.indexOf(k) >= 0;
        c.fillStyle = on ? C.accent : C['ink-3']; c.globalAlpha = on ? 0.9 : 0.45; c.fillRect(x0, y - 7, x1 - x0, 14); c.globalAlpha = 1;
        text(c, r[0], x0 + 4, y - 11, on ? C.accent : C['ink-2']);
      });
      s.out.innerHTML = 'Each method is calibrated where it overlaps the one below. An error in a low rung carries up to every rung above it.';
    }
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 2.4 pulsars
  X.pulsar = function (host) {
    var s = stage(host, 0.48), st = { P: 1 }, pulses = [];
    var last = 0;
    var stop = loop(function (t) {
      s.fit(); var c = s.ctx, C = s.C, cx = s.W * 0.22, cy = s.H * 0.42, slow = st.P * 1.5, ang = 2 * Math.PI * t / slow;
      for (var b = 0; b < 2; b++) {
        var a = ang + b * Math.PI, L = s.W * 0.6;
        c.beginPath(); c.moveTo(cx, cy); c.lineTo(cx + L * Math.cos(a - 0.12), cy + L * Math.sin(a - 0.12)); c.lineTo(cx + L * Math.cos(a + 0.12), cy + L * Math.sin(a + 0.12)); c.closePath();
        c.fillStyle = 'rgba(120,180,255,0.18)'; c.fill();
      }
      disc(c, cx, cy, 9, C.dark ? '#dfe8ff' : '#334');
      var ex = s.W * 0.86; disc(c, ex, cy, 7, C.c1); text(c, 'Earth', ex, cy + 22, C['ink-2'], 'center');
      var hit = Math.cos(ang) > Math.cos(0.12) || Math.cos(ang + Math.PI) > Math.cos(0.12);
      if (hit && t - last > slow * 0.3) { pulses.push(t); last = t; }
      while (pulses.length && t - pulses[0] > 6) pulses.shift();
      var y0 = s.H - 18; line(c, [[20, y0], [s.W - 20, y0]], C.rule, 1);
      pulses.forEach(function (p) { var x = s.W - 20 - (t - p) / 6 * (s.W - 40); line(c, [[x, y0], [x, y0 - 22]], C.accent, 2); });
      text(c, 'pulses received →', 20, y0 - 28, C['ink-3']);
      var Bf = A.pulsarB(st.P, 1e-15);
      s.out.innerHTML = 'A neutron star 20 km across spins once every <b>' + f(st.P) + ' s</b> (drawn ' + f(1.5) + '× slower); its beams sweep past Earth twice per turn here. A dipole slowing by Ṗ = 10⁻¹⁵ s/s has B ≈ 3.2×10¹⁹ √(PṖ) = ' + f(Bf, 2) + ' G and age P/2Ṗ = ' + f(A.pulsarAge(st.P, 1e-15) / 1e6, 2) + ' Myr.';
    });
    UI.slider(s.ctl, { label: 'Spin period', min: 0.3, max: 3, value: st.P, log: true, unit: 's', oninput: function (x) { st.P = x; } });
    return { stop: stop };
  };

  // ------------------------------------------------------------------------------------------ 3.1 transits
  function overlap(d, k) { // area of a disc of radius k overlapping the unit disc, centres d apart, over π
    if (d >= 1 + k) return 0; if (d <= 1 - k) return k * k; if (d <= k - 1) return 1;
    var k0 = Math.acos((k * k + d * d - 1) / (2 * k * d)), k1 = Math.acos((1 - k * k + d * d) / (2 * d));
    return (k * k * k0 + k1 - 0.5 * Math.sqrt(Math.max(0, 4 * d * d - Math.pow(1 + d * d - k * k, 2)))) / Math.PI;
  }
  X.transit = function (host) {
    var s = stage(host, 0.55), st = { k: 0.1, b: 0.2 }, trace = [];
    var stop = loop(function (t) {
      s.fit(); var c = s.ctx, C = s.C, cx = s.W * 0.25, cy = s.H * 0.4, R = s.H * 0.3, u = ((t * 0.18) % 1) * 3.2 - 1.6;
      var g = c.createRadialGradient(cx, cy, 0, cx, cy, R); g.addColorStop(0, '#fff1c8'); g.addColorStop(0.8, '#f2b25a'); g.addColorStop(1, '#c46a2a');
      disc(c, cx, cy, R, g);
      var px = cx + u * R, py = cy + st.b * R; disc(c, px, py, st.k * R, '#05070c');
      var drop = overlap(Math.hypot(u, st.b), st.k); trace.push([u, 1 - drop]); if (trace.length > 400) trace.shift();
      var pb = plotBox(s, s.W * 0.52, 12, s.W * 0.45, s.H - 46, [-1.6, 1.6], [1 - 1.25 * st.k * st.k - 0.002, 1.002], { xl: 'time', yl: 'brightness' });
      line(c, trace.filter(function (p, i) { return i === 0 || p[0] > trace[i - 1][0]; }).map(function (p) { return [pb.x(p[0]), pb.y(p[1])]; }), C.accent, 2);
      s.out.innerHTML = 'Planet/star size ratio k = <b>' + st.k.toFixed(3) + '</b>: the dip is k² = <b>' + (100 * st.k * st.k).toFixed(2) + '%</b> of the light (uniform disc). A Jupiter round the Sun gives 1%, an Earth 0.008%. Raising the impact parameter shortens the transit and, past 1 − k, gives a grazing V shape.';
      if (u > 1.55) trace.length = 0;
    });
    UI.slider(s.ctl, { label: 'R_planet / R_star', min: 0.01, max: 0.3, value: st.k, log: true, fmt: function (x) { return x.toFixed(3); }, oninput: function (x) { st.k = x; trace.length = 0; } });
    UI.slider(s.ctl, { label: 'Impact parameter b', min: 0, max: 1.1, value: st.b, fmt: function (x) { return x.toFixed(2); }, oninput: function (x) { st.b = x; trace.length = 0; } });
    return { stop: stop };
  };

  // ------------------------------------------------------------------------------------------ 3.2 detection biases
  X.selection = function (host) {
    var s = stage(host, 0.5), st = { m: 1 };
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, pb = plotBox(s, 50, 10, s.W - 70, s.H - 44, [-0.5, 3.5], [-3, 2.5], { logx: true, logy: true, xl: 'orbital period [days]', yl: 'signal' });
      var tr = [], rv = [];
      for (var e = -0.5; e <= 3.5; e += 0.05) {
        var P = Math.pow(10, e), a = Math.pow(Math.pow(P / 365.25, 2), 1 / 3);
        tr.push([pb.x(P), pb.y(100 * 0.00465 / a)]);
        rv.push([pb.x(P), pb.y(28.4 * st.m * Math.pow(P / 365.25, -1 / 3))]);
      }
      line(c, tr, C.c1, 2); line(c, rv, C.c2, 2);
      line(c, [[pb.x0, pb.y(1)], [pb.x0 + pb.w, pb.y(1)]], C['ink-3'], 1, [4, 4]);
      text(c, 'transit probability [%]', pb.x(40), pb.y(100 * 0.00465 / Math.pow(Math.pow(40 / 365.25, 2), 1 / 3)) - 8, C.c1);
      text(c, 'Doppler wobble K [m/s]', pb.x(3), pb.y(28.4 * st.m * Math.pow(3 / 365.25, -1 / 3)) - 8, C.c2);
      text(c, '1 m/s, about today’s best', pb.x0 + 6, pb.y(1) - 4, C['ink-3']);
      s.out.innerHTML = 'For a <b>' + f(st.m, 2) + ' Jupiter-mass</b> planet round a Sun-like star: K = 28.4 m/s · (M_p/M_J) · (P/1 yr)^{−1/3}, and a transit is seen by a fraction R★/a of observers. Both favour short periods.';
    }
    UI.slider(s.ctl, { label: 'Planet mass', min: 0.003, max: 10, value: st.m, log: true, unit: 'M_Jup', oninput: function (x) { st.m = x; draw(); } });
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 3.3 lensing
  X.lens = function (host) {
    var s = stage(host, 0.5), st = { u0: 0.3 }, trace = [];
    var stop = loop(function (t) {
      s.fit(); var c = s.ctx, C = s.C, cx = s.W * 0.25, cy = s.H * 0.46, RE = s.H * 0.24, tau = ((t * 0.25) % 1) * 4 - 2;
      var u = Math.hypot(tau, st.u0), ux = tau / u, uy = st.u0 / u;
      disc(c, cx, cy, RE, null, C['ink-3']); text(c, 'Einstein ring', cx, cy - RE - 6, C['ink-3'], 'center');
      disc(c, cx, cy, 6, C.c2); text(c, 'lens', cx + 8, cy + 14, C.c2);
      line(c, [[cx - 2 * RE, cy + st.u0 * RE], [cx + 2 * RE, cy + st.u0 * RE]], C.rule, 1, [3, 3]);
      disc(c, cx + tau * RE, cy + st.u0 * RE, 3, C['ink-2']);
      var tp = (u + Math.sqrt(u * u + 4)) / 2, tm = (u - Math.sqrt(u * u + 4)) / 2, A0 = A.magnification(u);
      var Ap = (A0 + 1) / 2, Am = (A0 - 1) / 2;
      disc(c, cx + tp * ux * RE, cy + tp * uy * RE, 3 + 3 * Math.sqrt(Ap), C.accent); disc(c, cx + tm * ux * RE, cy + tm * uy * RE, 3 + 3 * Math.sqrt(Am), C.accent);
      trace.push([tau, A0]); if (tau < -1.95) trace.length = 0;
      var pb = plotBox(s, s.W * 0.55, 12, s.W * 0.42, s.H - 46, [-2, 2], [0.8, Math.min(12, A.magnification(st.u0) * 1.1)], { xl: 'time / t_E', yl: 'magnification' });
      line(c, trace.map(function (p) { return [pb.x(p[0]), pb.y(p[1])]; }), C.accent, 2);
      s.out.innerHTML = 'Grey: the true source moving behind the lens. Gold: its two images, at θ = (u ± √(u² + 4))/2 Einstein radii, too close to resolve. Together they are A = (u² + 2)/(u√(u² + 4)) = <b>' + OBS.fix(A0, 2) + '</b> times brighter. Closest approach u₀ = ' + st.u0.toFixed(2) + ' gives a peak of ' + OBS.fix(A.magnification(st.u0), 2) + '.';
    });
    UI.slider(s.ctl, { label: 'Closest approach u₀', min: 0.05, max: 1.5, value: st.u0, log: true, unit: 'R_E', fmt: function (x) { return x.toFixed(2); }, oninput: function (x) { st.u0 = x; trace.length = 0; } });
    return { stop: stop };
  };

  // ------------------------------------------------------------------------------------------ 4.1 resonances
  X.resonance = function (host) {
    var s = stage(host, 0.52), st = { a: 2.5 }, marks = [], lastSep = null, T = 0;
    var stop = loop(function (t) {
      s.fit(); var c = s.ctx, C = s.C, cx = s.W * 0.33, cy = s.H * 0.5, sc = s.H * 0.42 / 5.2;
      var dt = 0.04; T += dt;
      var nJ = 2 * Math.PI / Math.pow(5.2026, 1.5), nA = 2 * Math.PI / Math.pow(st.a, 1.5), thJ = nJ * T * 3, thA = nA * T * 3;
      disc(c, cx, cy, 5.2026 * sc, null, C.rule); disc(c, cx, cy, st.a * sc, null, C.rule); disc(c, cx, cy, 6, '#f2c55c');
      var jx = cx + 5.2026 * sc * Math.cos(thJ), jy = cy + 5.2026 * sc * Math.sin(thJ), ax = cx + st.a * sc * Math.cos(thA), ay = cy + st.a * sc * Math.sin(thA);
      disc(c, jx, jy, 7, C.c2); disc(c, ax, ay, 3, C.ink);
      var sep = Math.sin(thA - thJ);
      if (lastSep !== null && lastSep < 0 && sep >= 0 && Math.cos(thA - thJ) > 0) { marks.push(thA); if (marks.length > 40) marks.shift(); }
      lastSep = sep;
      marks.forEach(function (m) { disc(c, cx + st.a * sc * Math.cos(m), cy + st.a * sc * Math.sin(m), 3, C.accent); });
      var ratio = Math.pow(st.a / 5.2026, 1.5);
      text(c, 'Jupiter', jx + 9, jy, C.c2);
      s.out.innerHTML = 'Gold: where the asteroid is each time Jupiter overtakes it. At <b>' + st.a.toFixed(3) + ' AU</b> its year is <b>' + ratio.toFixed(3) + '</b> of Jupiter’s. At a simple fraction (2.502 AU = 1/3, 2.825 AU = 2/5) the kicks land at the same few places and add up; otherwise they spread round the orbit and cancel.';
    });
    UI.slider(s.ctl, { label: 'Asteroid orbit size a', min: 2.0, max: 3.4, value: st.a, step: 0.001, unit: 'AU', fmt: function (x) { return x.toFixed(3); }, oninput: function (x) { st.a = x; marks.length = 0; } });
    return { stop: stop };
  };

  // ------------------------------------------------------------------------------------------ 4.2 Shapley
  X.shapley = function (host) {
    var s = stage(host, 0.5), st = { R0: 8 }, pts = [], seed = 3;
    function rnd() { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5; }
    for (var k = 0; k < 150; k++) { var r = 2 + 10 * Math.abs(rnd() + rnd()), a = Math.PI * 2 * (rnd() + 0.5); pts.push([r * Math.cos(a), r * Math.sin(a)]); }
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, cx = s.W * 0.27, cy = s.H * 0.5, sc = s.H * 0.4 / 15;
      c.strokeStyle = C.rule; c.beginPath(); c.ellipse(cx, cy, 15 * sc, 15 * sc, 0, 0, 7); c.stroke();
      pts.forEach(function (p) { disc(c, cx + p[0] * sc, cy + p[1] * sc, 2, C.c1); });
      disc(c, cx, cy, 4, C.c2); text(c, 'centre', cx + 6, cy - 6, C.c2);
      var sx = cx - st.R0 * sc; disc(c, sx, cy, 5, '#f2c55c'); text(c, 'Sun', sx - 4, cy + 18, C['ink-2'], 'center');
      var bins = new Array(12).fill(0);
      pts.forEach(function (p) { var l = Math.atan2(p[1], p[0] + st.R0); bins[Math.floor(((l / Math.PI / 2) + 0.5) * 12) % 12]++; });
      var pb = plotBox(s, s.W * 0.56, 12, s.W * 0.4, s.H - 46, [-180, 180], [0, 50], { xl: 'direction seen from the Sun [°]', yl: 'clusters' });
      bins.forEach(function (n, i) { var x0 = pb.x(-180 + 30 * i), x1 = pb.x(-150 + 30 * i); c.fillStyle = C.c1; c.fillRect(x0 + 1, pb.y(n), x1 - x0 - 2, pb.y(0) - pb.y(n)); });
      s.out.innerHTML = 'Seen from <b>' + st.R0.toFixed(1) + ' kpc</b> out, the clusters crowd toward the centre (0°). From the centre they would look the same in every direction. How lopsided they look tells you how far out you are.';
    }
    UI.slider(s.ctl, { label: 'Sun’s distance from the centre', min: 0, max: 14, value: st.R0, unit: 'kpc', fmt: function (x) { return x.toFixed(1); }, oninput: function (x) { st.R0 = x; draw(); } });
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 4.3 rotation curves
  X.rotation = function (host) {
    var s = stage(host, 0.5), st = { halo: 0 };
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, pb = plotBox(s, 46, 12, s.W - 70, s.H - 46, [0, 40], [0, 220], { xl: 'distance from the centre [kpc]', yl: 'orbital speed [km/s]' });
      var disk = [], hal = [], tot = [], kep = [];
      for (var r = 0.5; r <= 40; r += 0.5) {
        var y = r / 3, vd = 187 * Math.sqrt((1 - Math.exp(-y) * (1 + y)) / Math.max(y, 0.1));   // a concentrated mass: falls off outside
        var vh = A.vHalo(r, st.halo, 4);
        disk.push([pb.x(r), pb.y(vd)]); hal.push([pb.x(r), pb.y(vh)]); tot.push([pb.x(r), pb.y(Math.sqrt(vd * vd + vh * vh))]);
      }
      line(c, disk, C.c4, 2, [5, 4]); line(c, hal, C.c5, 1.5, [2, 3]); line(c, tot, C.accent, 2.5);
      for (r = 4; r <= 38; r += 3) disc(c, pb.x(r), pb.y(150 + 4 * Math.sin(r)), 3, C.ink);
      text(c, 'stars + gas', pb.x(30), pb.y(70) + 14, C.c4); text(c, 'measured (typical)', pb.x(25), pb.y(150) - 10, C.ink);
      s.out.innerHTML = 'V² = G M(&lt;r) / r. The visible matter is concentrated, so its curve falls like 1/√r outside the disk. Flat measured speeds mean M(&lt;r) ∝ r: add a halo (dotted) to make the total (gold) match. Halo speed far out: <b>' + Math.round(st.halo) + ' km/s</b>.';
    }
    UI.slider(s.ctl, { label: 'Dark halo strength', min: 0, max: 200, value: st.halo, unit: 'km/s', fmt: function (x) { return Math.round(x); }, oninput: function (x) { st.halo = x; draw(); } });
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 5.1 galaxy colours
  X.colourmix = function (host) {
    var s = stage(host, 0.42), st = { t: 0 };
    function draw() {
      s.fit(); var c = s.ctx, C = s.C;
      var young = Math.exp(-st.t / 0.3), ur = 1.2 + 1.4 * (1 - young);
      var mix = OBS.starRGB(-0.2 + 1.5 * (1 - young), 'gaia', true);
      for (var k = 0; k < 600; k++) {
        var r = Math.pow(Math.random(), 0.7) * s.H * 0.38, a = Math.random() * 7, arm = Math.sin(2 * a + r / 18);
        var isY = Math.random() < young * (0.5 + 0.5 * arm);
        disc(c, s.W * 0.25 + r * Math.cos(a), s.H * 0.5 + r * Math.sin(a) * 0.55, 1.3, isY ? '#8fb6ff' : '#ffc38a');
      }
      c.fillStyle = 'rgb(' + mix + ')'; c.fillRect(s.W * 0.6, s.H * 0.25, s.W * 0.12, s.H * 0.5);
      text(c, 'total colour', s.W * 0.66, s.H * 0.25 - 8, C['ink-2'], 'center');
      s.out.innerHTML = '<b>' + st.t.toFixed(1) + ' Gyr</b> after star formation stopped. The hot blue stars live only tens of millions of years, so the blue light fades fast and the galaxy reddens: u − r ≈ <b>' + ur.toFixed(1) + '</b> (schematic). Within about a billion years it joins the red sequence.';
    }
    UI.slider(s.ctl, { label: 'Time since star formation stopped', min: 0, max: 5, value: st.t, unit: 'Gyr', fmt: function (x) { return x.toFixed(1); }, oninput: function (x) { st.t = x; draw(); } });
    draw(); return { stop: function () { } };
  };

  // ------------------------------------------------------------------------------------------ 5.2 bound or not
  X.virial = function (host) {
    var s = stage(host, 0.5), st = { k: 1 }, gal = [], seed = 11;
    function rnd() { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5; }
    function reset() { gal = []; for (var i = 0; i < 40; i++) { var r = 0.3 + Math.abs(rnd()) * 1.4, a = Math.PI * 2 * (rnd() + 0.5), v = 0.9 * (1 + 0.6 * rnd()), b = a + Math.PI / 2 + rnd(); gal.push({ x: r * Math.cos(a), y: r * Math.sin(a), vx: v * Math.cos(b), vy: v * Math.sin(b) }); } }
    reset();
    var stop = loop(function () {
      s.fit(); var c = s.ctx, C = s.C, cx = s.W * 0.5, cy = s.H * 0.5, sc = s.H * 0.18, GM = 1.6 * st.k, dt = 0.012;
      gal.forEach(function (g) {
        for (var k = 0; k < 2; k++) { var r = Math.hypot(g.x, g.y) + 0.15, a = -GM / (r * r * r); g.vx += a * g.x * dt; g.vy += a * g.y * dt; g.x += g.vx * dt; g.y += g.vy * dt; }
        disc(c, cx + g.x * sc, cy + g.y * sc, 2.5, C.dark ? '#ffd9a8' : '#8a5a12');
      });
      disc(c, cx, cy, 1.6 * sc, null, C.rule);
      var escaped = gal.filter(function (g) { return Math.hypot(g.x, g.y) > 4.5; }).length;
      s.out.innerHTML = 'Galaxies moving at the measured speeds, held by <b>' + (st.k < 0.1 ? 'only the mass in stars' : f(st.k, 2) + '× the mass needed') + '</b>. ' + escaped + ' of 40 have flown off. Virial balance, 2K + W = 0, gives M ≈ 3σ²R/G: the mass that just holds them.';
    });
    UI.slider(s.ctl, { label: 'Mass holding the cluster', min: 0.03, max: 1.5, value: st.k, log: true, fmt: function (x) { return x < 0.1 ? 'stars only' : x.toFixed(2) + '×'; }, oninput: function (x) { st.k = x; reset(); } });
    return { stop: stop };
  };

  // ------------------------------------------------------------------------------------------ 5.3 Tully–Fisher
  X.tf = function (host) {
    var s = stage(host, 0.4), st = { v: 200 };
    function draw() {
      s.fit(); var c = s.ctx, C = s.C, ratio = st.v / 100, L = Math.pow(ratio, 4);
      var base = s.H * 0.8, w = s.W * 0.12;
      [['V', ratio], ['M ∝ V²R', ratio * ratio * ratio], ['L ∝ M', L]].forEach(function (b, k) {
        var h = Math.min(base - 20, 18 * Math.pow(b[1], 0.6)), x = s.W * (0.15 + 0.28 * k);
        c.fillStyle = [C.c1, C.c3, C.accent][k]; c.fillRect(x, base - h, w, h);
        text(c, b[0], x + w / 2, base + 16, C['ink-2'], 'center'); text(c, '×' + f(b[1], 3), x + w / 2, base - h - 6, C.ink, 'center');
      });
      s.out.innerHTML = 'Compared with a galaxy spinning at 100 km/s, one at <b>' + Math.round(st.v) + ' km/s</b>: mass M = V²R/G, and if disks have similar surface brightness then R ∝ √L ∝ √M, so M ∝ V⁴ and L ∝ V⁴ = <b>' + f(L, 3) + '×</b>.';
    }
    UI.slider(s.ctl, { label: 'Rotation speed', min: 50, max: 300, value: st.v, unit: 'km/s', fmt: function (x) { return Math.round(x); }, oninput: function (x) { st.v = x; draw(); } });
    draw(); return { stop: function () { } };
  };
})(typeof window !== 'undefined' ? window : globalThis);
