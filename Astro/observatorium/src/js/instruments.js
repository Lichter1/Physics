/* Observatorium — controls and instruments. An instrument is the data lens pointed at something made from
 * real numbers: a Planck curve through Gaia's two filters, a star's parallax ellipse, a binary's fitted
 * velocity curves. Missions build them; the free lens can open them too. */
(function (G) {
  'use strict';
  var OBS = G.OBS, A = OBS.astro;

  // ------------------------------------------------------------------------------------------ controls
  var uid = 0;
  var UI = OBS.ui = {
    clear: function (el) { el.innerHTML = ''; },
    /** A labelled slider. Linear, or logarithmic with log: true. Returns {get, set, el}. */
    slider: function (el, o) {
      var id = o.id || 'ctl-' + (++uid), wrap = document.createElement('div');
      wrap.className = 'ctl ctl-slider';
      // one position per step when the slider has a step, so an arrow key moves it by exactly one step
      var N = o.step && !o.log ? Math.max(1, Math.min(5000, Math.round((o.max - o.min) / o.step))) : 1000, lo = o.log ? Math.log10(o.min) : o.min, hi = o.log ? Math.log10(o.max) : o.max;
      wrap.innerHTML = '<label for="' + id + '">' + o.label + '</label><input type="range" id="' + id + '" min="0" max="' + N + '" step="1"><output for="' + id + '"></output>';
      el.appendChild(wrap);
      var input = wrap.querySelector('input'), out = wrap.querySelector('output');
      function toVal(k) { var a = lo + (hi - lo) * k / N; return o.log ? Math.pow(10, a) : a; }
      function toK(v) { var a = o.log ? Math.log10(v) : v; return Math.round((a - lo) / (hi - lo) * N); }
      var api = {
        el: wrap, value: o.value,
        get: function () { return api.value; },
        set: function (v, silent) { api.value = Math.max(o.min, Math.min(o.max, v)); input.value = toK(api.value); out.textContent = (o.fmt || OBS.fmt)(api.value) + (o.unit ? ' ' + o.unit : ''); if (!silent && o.oninput) o.oninput(api.value); }
      };
      input.addEventListener('input', function () {
        var v = toVal(+input.value);
        if (o.step) v = Math.round(v / o.step) * o.step;
        api.set(v);
      });
      api.set(o.value, true);
      return api;
    },
    button: function (el, label, fn, cls) {
      var b = document.createElement('button'); b.type = 'button'; b.className = 'btn ' + (cls || ''); b.innerHTML = label;
      b.addEventListener('click', fn); el.appendChild(b); return b;
    },
    number: function (el, o) {
      var id = o.id || 'ctl-' + (++uid), wrap = document.createElement('div'); wrap.className = 'ctl ctl-number';
      wrap.innerHTML = '<label for="' + id + '">' + o.label + '</label><input type="number" inputmode="decimal" id="' + id + '" step="any"' + (o.placeholder ? ' placeholder="' + o.placeholder + '"' : '') + '>' + (o.unit ? '<span class="unit">' + o.unit + '</span>' : '');
      el.appendChild(wrap);
      var input = wrap.querySelector('input');
      input.addEventListener('input', function () { if (o.oninput) o.oninput(parseFloat(input.value)); });
      return { el: wrap, get: function () { return parseFloat(input.value); }, set: function (v) { input.value = v; } };
    },
    readout: function (el, cls) { var d = document.createElement('div'); d.className = 'readout ' + (cls || ''); el.appendChild(d); return d; },
    row: function (el) { var d = document.createElement('div'); d.className = 'ctl-row'; el.appendChild(d); return d; }
  };

  // ------------------------------------------------------------------------------------------ pseudo-datasets
  /** A dataset built in memory (an instrument's view), with the same shape as a catalogue dataset. */
  OBS.pseudo = function (d, cols) {
    var names = Object.keys(cols), n = cols[names[0]].length, c = {};
    names.forEach(function (k) { c[k] = Array.isArray(cols[k]) && typeof cols[k][0] === 'string' ? cols[k] : Float64Array.from(cols[k]); });
    d.table = { n: n, names: names, cols: c, computed: {} };
    d.pos = !!d.pos;
    return d;
  };
  function v(id, label, unit, name, o) { o = o || {}; o.id = id; o.label = label; o.unit = unit; o.f = function (c, i) { return c[name][i]; }; o.scale = o.scale || 'lin'; return o; }

  var INS = OBS.instruments = {};

  // ------------------------------------------------------------------------------------------ spectrograph
  /** Planck curve through Gaia's BP and RP bands. stars: [{label, bprp, teff}] (real Gaia values). */
  INS.spectrograph = function (chart, tools, stars, onchange) {
    var lam = []; for (var l = 150; l <= 1500; l += 5) lam.push(l);
    var ds = OBS.pseudo({ id: 'spectrum', title: 'Spectrograph', x: 'lam', y: 'B', vars: [v('lam', 'Wavelength λ', 'nm', 'lam'), v('B', 'Brightness per nm (peak = 1)', '', 'B')] },
      { lam: lam, B: lam.map(function () { return NaN; }) });
    chart.setDataset(ds);
    chart.setView({ x0: 150, x1: 1500, y0: 0, y1: 1.12 });
    var st = { T: 5772, star: 0, limits: false };
    function norm(T) { var pk = A.planck(A.wienPeak(T), T); return function (x) { return A.planck(x * 1e-9, T) / pk; }; }
    function bandCurve(b, lo, hi) { return function (x) { return x >= lo && x <= hi ? b : NaN; }; }
    function redraw() {
      var T = st.T, f = norm(T), C = OBS.colours();
      chart.overlays = [];
      chart.add({ type: 'curve', id: 'bp', f: bandCurve(1.06, 330, 680), colour: C.c1, width: 6, label: 'BP filter', labelAt: 360 });
      chart.add({ type: 'curve', id: 'rp', f: bandCurve(1.0, 630, 1050), colour: C.c4, width: 6, label: 'RP filter', labelAt: 900 });
      chart.add({ type: 'curve', id: 'bb', f: f, colour: C.ink, width: 2, label: 'blackbody, ' + Math.round(T) + ' K', labelAt: Math.min(1400, A.wienPeak(T) * 1e9 * 1.5) });
      if (st.limits) {
        var pk = A.planck(A.wienPeak(T), T);
        chart.add({ type: 'curve', id: 'wien', f: function (x) { var lm = x * 1e-9; return 2 * A.C.h * A.C.c * A.C.c / Math.pow(lm, 5) * Math.exp(-A.C.h * A.C.c / (lm * A.C.k * T)) / pk; }, colour: C.c2, dash: [5, 4], label: 'Wien limit', labelAt: 200 });
        chart.add({ type: 'curve', id: 'rj', f: function (x) { var lm = x * 1e-9; return 2 * A.C.c * A.C.k * T / Math.pow(lm, 4) / pk; }, colour: C.c3, dash: [2, 3], label: 'Rayleigh–Jeans limit', labelAt: 1100 });
      }
      chart.add({ type: 'vline', id: 'peak', value: A.wienPeak(T) * 1e9, drag: false, colour: C['ink-3'], dash: [3, 3], label: 'peak ' + Math.round(A.wienPeak(T) * 1e9) + ' nm' });
      var s = stars[st.star], bb = A.bbBpRp(T);
      out.innerHTML = '<b>' + s.label + '</b>: measured BP − RP = <span class="num">' + OBS.fix(s.bprp, 3) + '</span>' +
        ' · this blackbody gives <span class="num">' + OBS.fix(bb, 3) + '</span>' +
        ' <span class="' + (Math.abs(bb - s.bprp) < 0.03 ? 'ok' : 'muted') + '">(' + (Math.abs(bb - s.bprp) < 0.03 ? 'match' : 'difference ' + OBS.fix(bb - s.bprp, 2)) + ')</span>';
      if (onchange) onchange(st);
    }
    var row = UI.row(tools);
    var pick = document.createElement('div'); pick.className = 'seg'; row.appendChild(pick);
    stars.forEach(function (s, k) {
      var b = UI.button(pick, s.label, function () { st.star = k; [].forEach.call(pick.children, function (x, j) { x.setAttribute('aria-pressed', j === k); }); redraw(); }, 'seg-b');
      b.setAttribute('aria-pressed', k === 0);
    });
    var sl = UI.slider(tools, { label: 'Temperature of the blackbody', min: 2000, max: 40000, log: true, value: st.T, unit: 'K', fmt: function (x) { return Math.round(x).toLocaleString('en-US'); }, oninput: function (T) { st.T = T; redraw(); } });
    var lim = document.createElement('label'); lim.className = 'check'; lim.innerHTML = '<input type="checkbox" id="spec-limits"> Show the Wien and Rayleigh–Jeans limits';
    tools.appendChild(lim);
    lim.querySelector('input').addEventListener('change', function (e) { st.limits = e.target.checked; redraw(); });
    var out = UI.readout(tools);
    redraw();
    return { state: st, slider: sl, stars: stars, redraw: redraw };
  };

  // ------------------------------------------------------------------------------------------ parallax telescope
  /** A star's yearly parallax ellipse against fixed background stars. target: {name, ra, dec, plx [mas]}. */
  INS.parallax = function (chart, tools, target, onchange) {
    var p = target.plx, rng = [], seed = 7;
    function rnd() { seed = (seed * 16807) % 2147483647; return seed / 2147483647; }
    var bx = [], by = [];
    for (var k = 0; k < 26; k++) { bx.push((rnd() - 0.5) * 5 * p); by.push((rnd() - 0.5) * 3.4 * p); }
    var ds = OBS.pseudo({ id: 'scope', title: 'Telescope', x: 'x', y: 'y', vars: [v('x', 'Offset east', 'mas', 'x', { rev: true }), v('y', 'Offset north', 'mas', 'y')] }, { x: bx, y: by });
    chart.setDataset(ds); chart.equal = true; chart.pointColour = null;
    chart.setView({ x0: -2.4 * p, x1: 2.4 * p, y0: -1.6 * p, y1: 1.6 * p });
    var st = { t: 0, marks: [], playing: false, showTrail: true };
    var trail = { type: 'ref', id: 'trail', xs: [], ys: [], under: false, size: 2.2, alpha: 0.8 };
    function posAt(t) { var s = A.parallaxShift(target.ra, target.dec, t); return [s.x * p, s.y * p]; }
    for (var j = 0; j < 120; j++) { var q = posAt(j / 120); trail.xs.push(q[0]); trail.ys.push(q[1]); }
    // The ellipse's long axis (along the ecliptic, length 2p) joins the two dates six months apart that are
    // farthest apart; across it the ellipse is only 2p sin(beta) wide.
    var tLong = 0, best = 0;
    for (j = 0; j < 365; j++) { var a0 = posAt(j / 365.25), a1 = posAt(j / 365.25 + 0.5), dd = Math.hypot(a0[0] - a1[0], a0[1] - a1[1]); if (dd > best) { best = dd; tLong = j / 365.25; } }
    var axisEnds = [posAt(tLong), posAt(tLong + 0.5)];
    st.longAxis = best;
    // Time runs in days from the March equinox (t = 0). The slider moves one day at a time and shows the date.
    var YEAR = 365.25, MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    function wrap(t) { return ((t % 1) + 1) % 1; }
    function date(t) { var d = new Date(Date.UTC(2026, 2, 20) + Math.round(wrap(t) * YEAR) * 864e5); return d.getUTCDate() + ' ' + MON[d.getUTCMonth()]; }
    function days(dt) { return Math.round(wrap(dt) * YEAR); }
    function span(n) { return n + (n === 1 ? ' day' : ' days') + ' (' + (n / YEAR * 12).toFixed(1) + ' months)'; }
    function redraw() {
      var C = OBS.colours(), q = posAt(st.t);
      chart.overlays = [];
      trail.colour = C['ink-3'];
      if (st.showTrail) {
        chart.add(trail);
        chart.add({ type: 'line', id: 'axis', x1: axisEnds[0][0], y1: axisEnds[0][1], x2: axisEnds[1][0], y2: axisEnds[1][1], colour: C['ink-2'], width: 1.2, dash: [2, 4], drag: false, segment: true, label: 'long axis = 2p' });
      }
      st.marks.forEach(function (m, k) { chart.add({ type: 'marker', id: 'm' + k, x: m.x, y: m.y, colour: C.c2, label: m.label, drag: false }); });
      var since = st.marks.length === 1 ? ' · ' + days(st.t - st.marks[0].t) + ' d after mark 1' : '';
      chart.add({ type: 'marker', id: 'star', x: q[0], y: q[1], colour: C.accent, label: target.name + ' · ' + date(st.t) + since, drag: false });
      if (st.marks.length === 2) {
        var a = st.marks[0], b = st.marks[1];
        chart.add({ type: 'line', id: 'ruler', x1: a.x, y1: a.y, x2: b.x, y2: b.y, colour: C.c2, width: 1.5, dash: [4, 3], drag: false, segment: true });
      } else if (st.marks.length === 1) {     // a live ruler from the first mark to where the star is now
        chart.add({ type: 'line', id: 'ruler', x1: st.marks[0].x, y1: st.marks[0].y, x2: q[0], y2: q[1], colour: C.c2, width: 1, dash: [2, 3], drag: false, segment: true });
      }
      chart.draw();
      var sep = st.marks.length === 2 ? Math.hypot(st.marks[0].x - st.marks[1].x, st.marks[0].y - st.marks[1].y) : NaN;
      st.sep = sep;
      var live = st.marks.length === 1 ? Math.hypot(q[0] - st.marks[0].x, q[1] - st.marks[0].y) : NaN;
      clock.innerHTML = st.marks.length === 1 ? 'Since your first mark (' + st.marks[0].label + '): <span class="num">' + span(days(st.t - st.marks[0].t)) + '</span>; the star is now <span class="num">' + OBS.fmt(live / 1000, 3) + '″</span> from it.'
        : st.marks.length === 2 ? 'Between your marks: <span class="num">' + span(days(st.marks[1].t - st.marks[0].t)) + '</span>' : 'Time since your first mark: no mark yet';
      out.innerHTML = st.marks.length === 2 ? 'The two marks are <span class="num">' + OBS.fmt(sep, 3) + ' mas</span> = <span class="num">' + OBS.fmt(sep / 1000, 3) + '″</span> apart (' + st.marks[0].label + ' → ' + st.marks[1].label + '): <span class="num">' + Math.round(100 * sep / best) + '%</span> of the ellipse’s long axis.'
        : st.marks.length === 1 ? 'One mark made. Move the date, or jump ahead exactly 6 months, and mark again.' : 'Move through the year, then mark the star’s position twice. The dotted line is the long axis of the yearly path.';
      if (onchange) onchange(st);
    }
    var time = UI.slider(tools, { label: 'Date', min: 0, max: 364, value: 0, step: 1, fmt: function (d) { return date(d / YEAR) + ' (day ' + Math.round(d) + ')'; }, oninput: function (d) { st.t = d / YEAR; redraw(); } });
    function setT(t) { st.t = wrap(t); time.set(Math.min(364, Math.round(st.t * YEAR)), true); redraw(); }
    var row = UI.row(tools);
    UI.button(row, 'Play a year', function () {
      if (st.playing) return; st.playing = true; var t0 = performance.now(), start = st.t;
      (function step(now) { var u = Math.min(1, (now - t0) / 6000); setT(start + u); if (u < 1) G.requestAnimationFrame(step); else st.playing = false; })(t0);
    });
    UI.button(row, 'Mark position', function () {
      var q = posAt(st.t); if (st.marks.length >= 2) st.marks = [];
      st.marks.push({ x: q[0], y: q[1], t: st.t, label: date(st.t) }); redraw();
    }, 'primary');
    UI.button(row, 'Jump ahead 6 months', function () { setT(st.t + 0.5); });
    UI.button(row, 'Clear marks', function () { st.marks = []; redraw(); });
    var clock = UI.readout(tools);
    var tr = document.createElement('label'); tr.className = 'check'; tr.innerHTML = '<input type="checkbox" id="scope-trail" checked> Show the whole year’s path and its long axis';
    tools.appendChild(tr); tr.querySelector('input').addEventListener('change', function (e) { st.showTrail = e.target.checked; redraw(); });
    var out = UI.readout(tools);
    redraw();
    return { state: st, posAt: posAt };
  };

  // ------------------------------------------------------------------------------------------ binary velocities
  /** The two radial-velocity curves of a Gaia double-lined binary, and rules you drag to read them. */
  INS.binary = function (chart, tools, b, onchange) {
    var ph = [], vv = [], who = [];
    for (var k = 0; k < 160; k++) {
      var f = k / 160;
      ph.push(f); vv.push(A.rv(f, b.e, b.w, b.K1, b.gamma)); who.push('star 1 (brighter)');
      ph.push(f); vv.push(A.rv(f, b.e, b.w + 180, b.K2, b.gamma)); who.push('star 2 (fainter)');
    }
    var ds = OBS.pseudo({ id: 'rvcurve', title: 'Radial velocities', x: 'ph', y: 'v', colourBy: 'who', catOrder: { who: ['star 1 (brighter)', 'star 2 (fainter)'] },
      vars: [v('ph', 'Orbital phase', 'cycles', 'ph'), v('v', 'Velocity toward (−) or away (+)', 'km/s', 'v'), { id: 'who', label: 'Star', cat: true, f: function (c, i) { return c.who[i]; } }] },
      { ph: ph, v: vv, who: who });
    chart.setDataset(ds); if (OBS.app && OBS.app.legend) OBS.app.legend();     // which curve is which star
    var top = Math.max.apply(null, vv), bot = Math.min.apply(null, vv), span = top - bot;
    chart.setView({ x0: 0, x1: 1, y0: bot - 0.15 * span, y1: top + 0.15 * span });
    var C = OBS.colours(), st = {};
    var g = b.gamma;
    var lines = [['t1', 'star 1 top', g + span * 0.25, C.c1], ['b1', 'star 1 bottom', g - span * 0.25, C.c1], ['t2', 'star 2 top', g + span * 0.4, C.c2], ['b2', 'star 2 bottom', g - span * 0.4, C.c2]];
    lines.forEach(function (l) { chart.add({ type: 'hline', id: l[0], value: l[2], label: l[1], colour: l[3], dash: [5, 4] }); });
    var out = UI.readout(tools);
    function read() {
      st.K1 = Math.abs(chart.get('t1').value - chart.get('b1').value) / 2;
      st.K2 = Math.abs(chart.get('t2').value - chart.get('b2').value) / 2;
      st.m = A.binaryMasses(b.period, st.K1, st.K2, b.e, 90);
      out.innerHTML = 'Your readings: K₁ = <span class="num">' + OBS.fix(st.K1, 1) + '</span> km/s, K₂ = <span class="num">' + OBS.fix(st.K2, 1) + '</span> km/s, P = <span class="num">' + OBS.fix(b.period, 4) + '</span> d (eclipsing, so sin i ≈ 1).' +
        (OBS.unlocked(['kepler']) || st.reveal ? '<br>Masses: M₁ = <span class="num">' + OBS.fmt(st.m.M1, 3) + '</span> M☉, M₂ = <span class="num">' + OBS.fmt(st.m.M2, 3) + '</span> M☉; separation a = <span class="num">' + OBS.fmt(st.m.aRsun, 3) + '</span> R☉.' : '');
      if (onchange) onchange(st);
    }
    OBS.onMission('overlay', function (e) { if (e.chart === chart && chart.ds === ds) read(); });
    read();
    return { state: st, read: read };
  };

  // ------------------------------------------------------------------------------------------ folding
  /** Fold a time series on a trial period. Returns {get P, t0}. */
  INS.fold = function (chart, tools, o) {
    var st = { P: o.P0, t0: o.t0 || 0 };
    function apply() { chart.fold = { P: st.P, t0: st.t0 }; chart.autoscale(); if (o.onchange) o.onchange(st); }
    var row = UI.row(tools);
    var on = document.createElement('label'); on.className = 'check'; on.innerHTML = '<input type="checkbox" id="fold-on"' + (o.start ? ' checked' : '') + '> Fold the light curve on a trial period';
    row.appendChild(on);
    var sP = UI.slider(tools, { label: 'Trial period', min: o.min, max: o.max, log: true, value: st.P, unit: 'd', fmt: function (x) { return x.toFixed(x < 10 ? 4 : 3); }, oninput: function (P) { st.P = P + (fine ? fine.get() * o.fine : 0); if (chart.fold) apply(); } });     // keep the fine offset
    var fine = UI.slider(tools, { label: 'Fine tune', min: -1, max: 1, value: 0, fmt: function (x) { return (x >= 0 ? '+' : '−') + Math.abs(x * o.fine).toFixed(5) + ' d'; }, oninput: function (x) { st.P = sP.get() + x * o.fine; if (chart.fold) apply(); } });
    if (o.shift) UI.slider(tools, { label: 'Shift phase', min: 0, max: 1, value: 0, fmt: function (x) { return x.toFixed(2); }, oninput: function (x) { st.t0 = (o.t0 || 0) + x * st.P; if (chart.fold) apply(); } });
    on.querySelector('input').addEventListener('change', function (e) {
      if (e.target.checked) apply(); else { chart.fold = null; chart.autoscale(); if (o.onchange) o.onchange(st); }
    });
    if (o.start) apply();
    st.coarse = sP; st.fine = fine;
    // A periodogram: how sharp the fold is at every trial period. Dips mark periods that line the points up
    // (and their aliases at half and double the period); click one to jump there, then fine-tune.
    if (o.periodogram !== false) {
      var xs = chart.xs(), ys = chart.ys(), t = [], yv = [];
      for (var i = 0; i < xs.length; i++) if (chart.mask[i] && isFinite(xs[i]) && isFinite(ys[i])) { t.push(xs[i]); yv.push(ys[i]); }
      var stride = Math.max(1, Math.floor(t.length / 3000)), tt = [], yy = [];
      for (i = 0; i < t.length; i += stride) { tt.push(t[i]); yy.push(yv[i]); }
      var NP = 1600, Ps = [], S = [], smin = Infinity, smax = 0;
      for (var k = 0; k < NP; k++) { var P = o.min * Math.pow(o.max / o.min, k / (NP - 1)), s = INS.foldScatter(tt, yy, P, 40); Ps.push(P); S.push(s); smin = Math.min(smin, s); smax = Math.max(smax, s); }
      var lab = document.createElement('p'); lab.className = 'small muted'; lab.textContent = 'How sharp the fold is at each trial period (lower is sharper). Click a dip to try that period; use the arrow keys on a slider for single steps.';
      var pg = document.createElement('canvas'); pg.className = 'ins-mini'; pg.style.height = '110px'; pg.style.cursor = 'crosshair';
      tools.appendChild(lab); tools.appendChild(pg);
      var drawPg = function () {
        var r = pg.getBoundingClientRect(), dpr = Math.min(2, G.devicePixelRatio || 1), w = Math.max(200, r.width), h = 110, c = pg.getContext('2d'), C = OBS.colours();
        pg.width = w * dpr; pg.height = h * dpr; c.setTransform(dpr, 0, 0, dpr, 0, 0); c.fillStyle = C.panel; c.fillRect(0, 0, w, h);
        var X = function (P) { return 8 + (w - 16) * Math.log(P / o.min) / Math.log(o.max / o.min); }, Y = function (s) { return 10 + (h - 30) * (s - smin) / (smax - smin || 1); };
        c.strokeStyle = C.c1; c.lineWidth = 1; c.beginPath();
        for (var k = 0; k < NP; k++) { if (k) c.lineTo(X(Ps[k]), Y(S[k])); else c.moveTo(X(Ps[k]), Y(S[k])); }
        c.stroke();
        c.strokeStyle = C.accent; c.beginPath(); c.moveTo(X(st.P), 4); c.lineTo(X(st.P), h - 18); c.stroke();
        c.font = '11px ' + getComputedStyle(document.documentElement).getPropertyValue('--sans'); c.fillStyle = C['ink-3'];
        [1, 2, 5, 10, 20, 50].forEach(function (P) { if (P >= o.min && P <= o.max) c.fillText(P + ' d', X(P) - 6, h - 4); });
      };
      pg.addEventListener('click', function (e) {
        var r = pg.getBoundingClientRect(), u = (e.clientX - r.left - 8) / (r.width - 16), P = o.min * Math.pow(o.max / o.min, Math.max(0, Math.min(1, u)));
        // snap to the sharpest period within a few pixels of the click
        var kb = 0, best = Infinity; for (var k = 0; k < NP; k++) if (Math.abs(Math.log(Ps[k] / P)) < 6 / r.width * Math.log(o.max / o.min) && S[k] < best) { best = S[k]; kb = k; }
        if (best < Infinity) P = Ps[kb];
        sP.set(P, true); fine.set(0, true); st.P = P;
        var cb = on.querySelector('input'); if (!cb.checked) { cb.checked = true; }
        apply(); drawPg();
      });
      var onch = o.onchange; o.onchange = function (s) { drawPg(); if (onch) onch(s); };
      requestAnimationFrame(drawPg);
      var ro = G.ResizeObserver ? new G.ResizeObserver(drawPg) : null; if (ro) { ro.observe(pg); OBS.cleanup(function () { ro.disconnect(); }); }
    }
    return st;
  };
  /** How sharp a fold is: mean scatter of the folded curve around its own running median (lower is better). */
  INS.foldScatter = function (t, y, P, nb) {
    nb = nb || 50; var bins = []; for (var k = 0; k < nb; k++) bins.push([]);
    for (var i = 0; i < t.length; i++) { var ph = ((t[i] / P) % 1 + 1) % 1; bins[Math.floor(ph * nb)].push(y[i]); }
    var s = 0, n = 0;
    bins.forEach(function (b) { if (b.length < 2) return; var m = b.reduce(function (a, x) { return a + x; }, 0) / b.length; b.forEach(function (x) { s += (x - m) * (x - m); n++; }); });
    return Math.sqrt(s / n);
  };

  // ------------------------------------------------------------------------------------------ models by hand
  /** Sliders for a model curve over the data, with a live goodness-of-fit. o: {params:[{id,label,min,max,value,log,unit,fmt}], f(x, p), err(i)} */
  INS.model = function (chart, tools, o) {
    var p = {}, C = OBS.colours(), sliders = {};
    var out = UI.readout(tools);
    function chi() {
      var xs = OBS.column(chart.ds, chart.xid), ys = OBS.column(chart.ds, chart.yid), s = 0, n = 0;
      for (var i = 0; i < xs.length; i++) {
        if (!chart.mask[i] || !isFinite(ys[i])) continue;
        var m = o.f(xs[i], p), e = o.err ? o.err(i) : 1;
        if (isFinite(m)) { s += (ys[i] - m) * (ys[i] - m) / (e * e); n++; }
      }
      return { chi2: s, n: n, red: s / Math.max(1, n - o.params.length) };
    }
    function update() {
      chart.add({ type: 'curve', id: 'model', f: function (x) { return o.f(x, p); }, colour: o.colour || C.c1, width: 2.4, label: o.label });
      (o.extra || []).forEach(function (e) { chart.add({ type: 'curve', id: e.id, f: function (x) { return e.f(x, p); }, colour: e.colour || C['ink-2'], dash: e.dash || [5, 4], width: 1.5, label: e.label }); });
      var q = chi(); p._fit = q;
      out.innerHTML = (o.describe ? o.describe(p, q) : '') + ' <span class="muted">Goodness of fit χ²/N = <span class="num">' + OBS.fmt(q.red, 3) + '</span></span>';
      if (o.onchange) o.onchange(p, q);
    }
    o.params.forEach(function (d) {
      p[d.id] = d.value;
      sliders[d.id] = UI.slider(tools, { label: d.label, min: d.min, max: d.max, value: d.value, log: d.log, unit: d.unit, fmt: d.fmt, step: d.step, oninput: function (x) { p[d.id] = x; update(); } });
    });
    tools.appendChild(out);
    update();
    return { p: p, sliders: sliders, update: update, chi: chi };
  };

  /** A reference population drawn under the data that slides vertically (distance modulus fitting). */
  INS.slideRef = function (chart, tools, o) {
    var C = OBS.colours();
    var ref = chart.add({ type: 'ref', id: 'ref', xs: o.xs, ys: o.ys, dx: 0, dy: o.value, colour: o.colour || C.c3, alpha: o.alpha || 0.55, size: o.size || 1.8, label: o.label, under: o.under });
    var out = UI.readout(tools);
    var s = UI.slider(tools, {
      label: o.label, min: o.min, max: o.max, value: o.value, step: 0.01, fmt: function (x) { return x.toFixed(2); }, unit: 'mag',
      oninput: function (x) { ref.dy = x; chart.draw(); if (o.onchange) o.onchange(x, out); }
    });
    tools.appendChild(out);
    if (o.onchange) o.onchange(o.value, out);
    return s;
  };
})(typeof window !== 'undefined' ? window : globalThis);
