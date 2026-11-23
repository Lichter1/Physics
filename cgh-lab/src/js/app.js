/* cGħ Lab — shell: the three dials, number formatting, tabs, the cube of theories and the Map panel. */
(function () {
  'use strict';
  var P = CGH.physics, REAL = P.REAL, K = P.K, REF = P.REF;
  var $ = function (id) { return document.getElementById(id); };

  /* ---------- numbers people can read ---------- */
  var SUP = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  function sup(n) { return String(n).split('').map(function (ch) { return SUP[ch] || ch; }).join(''); }
  function num(x, sig) {
    sig = sig || 3;
    if (x === 0) return '0';
    if (!isFinite(x)) return isNaN(x) ? '–' : (x > 0 ? '∞' : '−∞');
    var ax = Math.abs(x), e = Math.floor(Math.log10(ax));
    if (e >= -3 && e < 6) {
      var v = Number(x.toPrecision(sig));
      return (Math.abs(v) >= 10000 ? v.toLocaleString('en-US') : String(v)).replace('-', '−');
    }
    var m = x / Math.pow(10, e), ms = m.toFixed(sig - 1);
    if (Math.abs(Number(ms)) >= 10) { e += 1; ms = (m / 10).toFixed(sig - 1); }
    return ms.replace('-', '−') + '×10' + sup(e);
  }
  function unit(x, table, sig) {
    var ax = Math.abs(x);
    for (var i = 0; i < table.length; i++) {
      if (ax < table[i][0]) return num(x / table[i][1], sig) + ' ' + table[i][2];
    }
    var last = table[table.length - 1];
    return num(x / last[1], sig) + ' ' + last[2];
  }
  var LY = 9.4607e15;
  function length(m, sig) {
    return unit(m, [[1e-12, 1, 'm'], [1e-9, 1e-12, 'pm'], [1e-6, 1e-9, 'nm'], [1e-3, 1e-6, 'µm'], [1e-2, 1e-3, 'mm'],
      [1, 1e-2, 'cm'], [1e3, 1, 'm'], [1e9, 1e3, 'km'], [1e10, 1e3, 'km'], [LY / 10, K.AU, 'AU'], [Infinity, LY, 'light-years']], sig);
  }
  function duration(s, sig) {
    return unit(s, [[1e-9, 1, 's'], [1e-6, 1e-9, 'ns'], [1e-3, 1e-6, 'µs'], [1, 1e-3, 'ms'], [120, 1, 's'], [7200, 60, 'min'],
      [172800, 3600, 'h'], [2 * K.year, 86400, 'days'], [Infinity, K.year, 'years']], sig);
  }
  function mass(kg, sig) {
    return unit(kg, [[1e-9, 1, 'kg'], [1e-6, 1e-9, 'µg'], [1e-3, 1e-6, 'mg'], [1, 1e-3, 'g'], [1e3, 1, 'kg'], [1e6, 1e3, 't'], [Infinity, 1, 'kg']], sig);
  }
  function pad2(n) { return (n < 10 ? '0' : '') + n; }
  // A clock face in text: 0:07.3, 1:02:07, 3 d 04:11, 7.02 years
  function clock(s) {
    if (!isFinite(s)) return '–';
    var neg = s < 0; s = Math.abs(s);
    var out;
    if (s < 3600) { var mm = Math.floor(s / 60), ss = s - mm * 60; out = mm + ':' + (ss < 10 ? '0' : '') + ss.toFixed(1); }
    else if (s < 86400) { var h = Math.floor(s / 3600), m2 = Math.floor((s - h * 3600) / 60); out = h + ':' + pad2(m2) + ':' + pad2(Math.floor(s % 60)); }
    else if (s < 2 * K.year) { var d = Math.floor(s / 86400), r = s - d * 86400; out = d + ' d ' + pad2(Math.floor(r / 3600)) + ':' + pad2(Math.floor((r % 3600) / 60)); }
    else out = num(s / K.year, 4) + ' years';
    return (neg ? '−' : '') + out;
  }
  function times(x) { return x >= 1 ? num(x) + '×' : '÷' + num(1 / x); }

  /* ---------- the constants ---------- */
  var state = { c: REAL.c, G: REAL.G, hbar: REAL.hbar };
  var listeners = [];
  function onChange(fn) { listeners.push(fn); }
  function set(patch) {
    var changed = false;
    for (var k in patch) if (state[k] !== patch[k]) { state[k] = patch[k]; changed = true; }
    if (!changed) return;
    try { localStorage.setItem('cgh-dials', JSON.stringify(state)); } catch (e) { /* storage is optional */ }
    listeners.forEach(function (fn) { fn(state); });
  }
  function isReal(k) { return Math.abs(state[k] / REAL[k] - 1) < 1e-9; }

  var DIALS = {
    c: { lo: Math.log10(2), hi: Math.log10(REAL.c), unit: 'm/s' },
    G: { lo: Math.log10(REAL.G) - 3, hi: Math.log10(REAL.G) + 12, unit: 'm³/kg·s²' },
    hbar: { lo: Math.log10(REAL.hbar), hi: 0, unit: 'J·s' }
  };
  function toSlider(k, v) { var d = DIALS[k]; return Math.round(1000 * (Math.log10(v) - d.lo) / (d.hi - d.lo)); }
  function fromSlider(k, s) {
    var d = DIALS[k], realPos = toSlider(k, REAL[k]);
    if (Math.abs(s - realPos) <= 5) return REAL[k];                 // snap to the real value
    var v = Math.pow(10, d.lo + (d.hi - d.lo) * s / 1000);
    return Number(v.toPrecision(3));
  }
  function dialNote(k) {
    if (isReal(k)) return 'the value in our universe';
    var r = state[k] / REAL[k];
    if (k === 'c') return num(1 / r, 2) + ' times slower than real';
    if (k === 'G') return r > 1 ? num(r, 2) + ' times stronger than real' : num(1 / r, 2) + ' times weaker than real';
    return num(r, 2) + ' times larger than real';
  }
  function renderDials() {
    ['c', 'G', 'hbar'].forEach(function (k) {
      $('dial-' + k).value = toSlider(k, state[k]);
      $('val-' + k).textContent = num(state[k], 3) + ' ' + DIALS[k].unit;
      $('note-' + k).textContent = dialNote(k);
      $('reset-' + k).disabled = isReal(k);
    });
  }

  var PRESETS = [
    { name: 'Our universe', v: { c: REAL.c, G: REAL.G, hbar: REAL.hbar } },
    { name: 'Slow light', v: { c: 10, G: REAL.G, hbar: REAL.hbar } },
    { name: 'Big ħ', v: { c: REAL.c, G: REAL.G, hbar: 0.1 } },
    { name: 'Both', v: { c: 10, G: REAL.G, hbar: 0.1 } }
  ];
  function same(a, b) { return Math.abs(a / b - 1) < 1e-6; }
  function matches(v) { return same(state.c, v.c) && same(state.G, v.G) && same(state.hbar, v.hbar); }
  function renderPresets() {
    Array.prototype.forEach.call($('presets').children, function (b, i) {
      b.setAttribute('aria-pressed', matches(PRESETS[i].v) ? 'true' : 'false');
    });
  }

  /* ---------- theme-aware canvas helpers ---------- */
  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  var themeListeners = [];
  function onTheme(fn) { themeListeners.push(fn); }
  function themeChanged() { themeListeners.forEach(function (fn) { fn(); }); }
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    if (mq.addEventListener) mq.addEventListener('change', themeChanged);
  }
  if (window.MutationObserver) new MutationObserver(themeChanged).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  // Match the canvas bitmap to its CSS box. Returns CSS-pixel size and the scale to apply.
  function fit(canvas, maxDpr) {
    var r = canvas.getBoundingClientRect();
    var dpr = Math.min(window.devicePixelRatio || 1, maxDpr || 2);
    var w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    return { w: r.width, h: r.height, dpr: dpr };
  }

  /* ---------- the cube of theories ---------- */
  var THEORIES = [
    { what: 'Nothing but forces and inertia. Billiard balls in deep space.' },
    { what: 'Light is slow enough to race. Scene: Light.' },
    { what: 'Falling, orbits, tides. The world you grew up in.' },
    { what: 'Slow light and gravity together: clocks disagree with height. Scene: Gravity.' },
    { what: 'Footballs spread out and interfere. Scene: Quantum.' },
    { what: 'Fast and fuzzy at once: particles appear and vanish. Not simulated here.' },
    { what: 'A ball that cannot lie on the floor. Scene: Quantum, Drop.' },
    { what: 'All three at once. No accepted theory exists. Scene: Gravity, Black-hole mission, for the one formula people trust.' }
  ];
  function cornerPreset(bits) {
    return { c: bits & 1 ? 10 : REAL.c, G: bits & 2 ? REAL.G : REAL.G * 1e-3, hbar: bits & 4 ? 0.1 : REAL.hbar };
  }
  function cubeGeometry(w, h) {
    var S = Math.min(w, h * 1.12) * 0.5;
    var ox = (w - 1.42 * S) / 2, oy = h - (h - 1.30 * S) / 2 + 0.02 * h;
    function proj(x, y, z) { return [ox + S * (x + 0.42 * z), oy - S * (y + 0.30 * z)]; }
    var corners = [];
    for (var b = 0; b < 8; b++) corners.push(proj(b & 1, (b >> 1) & 1, (b >> 2) & 1));
    return { S: S, proj: proj, corners: corners };
  }
  function drawCube(canvas, big) {
    var f = fit(canvas, 2), ctx = canvas.getContext('2d');
    ctx.setTransform(f.dpr, 0, 0, f.dpr, 0, 0);
    ctx.clearRect(0, 0, f.w, f.h);
    var g = cubeGeometry(f.w, f.h), r = P.regime(state);
    var ink = css('--ink'), muted = css('--muted'), line = css('--line');
    var col = [css('--c'), css('--g'), css('--h')];
    ctx.lineCap = 'round';
    for (var a = 0; a < 8; a++) for (var bit = 0; bit < 3; bit++) {
      var b = a | (1 << bit);
      if (b === a) continue;
      var hidden = a === 4 || b === 4;
      ctx.beginPath(); ctx.moveTo(g.corners[a][0], g.corners[a][1]); ctx.lineTo(g.corners[b][0], g.corners[b][1]);
      ctx.setLineDash(hidden ? [3, 4] : []);
      // the three edges leaving the origin are the axes and carry their constant's colour
      ctx.strokeStyle = a === 0 ? col[bit] : (hidden ? line : muted);
      ctx.lineWidth = a === 0 ? (big ? 3 : 2) : 1;
      ctx.stroke();
    }
    ctx.setLineDash([]);
    if (big) {
      ctx.font = 'italic 600 17px ' + css('--font-math');
      ctx.textBaseline = 'middle';
      var mid = function (i, j) { return [(g.corners[i][0] + g.corners[j][0]) / 2, (g.corners[i][1] + g.corners[j][1]) / 2]; };
      var m;
      ctx.fillStyle = col[0]; ctx.textAlign = 'center'; m = mid(0, 1); ctx.fillText('slower c →', m[0], m[1] + 16);
      ctx.fillStyle = col[1]; ctx.textAlign = 'right'; m = mid(0, 2); ctx.fillText('G on ↑', m[0] - 8, m[1]);
      ctx.fillStyle = col[2]; ctx.textAlign = 'left'; m = mid(1, 5); ctx.fillText('↗ bigger ħ', m[0] + 8, m[1] + 8);
    }
    for (var c = 0; c < 8; c++) {
      var on = c === r.corner;
      ctx.beginPath(); ctx.arc(g.corners[c][0], g.corners[c][1], on ? (big ? 7 : 4.5) : (big ? 4 : 2.5), 0, 6.2832);
      ctx.fillStyle = on ? ink : css('--surface'); ctx.fill();
      ctx.strokeStyle = on ? ink : muted; ctx.lineWidth = 1.5; ctx.stroke();
    }
    // where the dials actually are, between the corners
    var p = g.proj(r.rel, r.grav, r.quant);
    if (big) {
      ctx.strokeStyle = muted; ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
      var foot = g.proj(r.rel, 0, r.quant);
      ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(foot[0], foot[1]); ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.beginPath(); ctx.arc(p[0], p[1], big ? 11 : 7, 0, 6.2832);
    ctx.globalAlpha = 0.18; ctx.fillStyle = ink; ctx.fill(); ctx.globalAlpha = 1;
    ctx.beginPath(); ctx.arc(p[0], p[1], big ? 4.5 : 3, 0, 6.2832);
    ctx.fillStyle = css('--focus'); ctx.fill();
    if (big) {
      var q = g.corners[r.corner];
      ctx.font = '600 14px ' + css('--font-text'); ctx.fillStyle = ink;
      var right = (r.corner & 1) || (r.corner & 4);
      ctx.textAlign = right ? 'right' : 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText(r.name, q[0] + (right ? -10 : 10), q[1] - 9);
    }
    return g;
  }

  function bitsHTML(b) {
    return '<span class="sym-c' + (b & 1 ? '' : ' off') + '">c</span>' +
           '<span class="sym-g' + (b & 2 ? '' : ' off') + '">G</span>' +
           '<span class="sym-h' + (b & 4 ? '' : ' off') + '">ħ</span>';
  }

  /* ---------- the everyday table ---------- */
  var PROTON = 8.4e-16;
  function everyday(k) {
    var rows = [], g = P.earthGravity(k.G);
    var v = P.speedFromEffort(REF.bike, k.c), beta = v / k.c, loss = P.clockLoss(beta) * 3600;
    rows.push({ dial: 'c', what: 'A bike ride, 8 m/s of effort', val: num(v) + ' m/s = ' + num(beta, 2) + ' c', level: P.logLevel(beta, -3, 0),
      verdict: beta > 0.05 ? 'Your watch falls ' + duration(loss, 2) + ' behind the town for every hour you ride.' : 'Your watch falls ' + duration(loss, 2) + ' behind per hour. Nobody notices.' });

    var ex = P.rindlerExcess(g, 30, k.c), horizon = P.rindlerHorizon(g, k.c);
    rows.push({ dial: 'cg', what: 'A clock on the tenth floor, 30 m up', val: '+' + duration(ex * 86400, 2) + ' a day', level: P.logLevel(ex, -4, 0),
      verdict: ex > 0.01 ? 'People upstairs age ' + num(1 + ex, 3) + ' times faster. Clocks stop ' + length(horizon, 2) + ' below the street.' : 'It runs ahead of the street clock, far too little to matter.' });

    var rs = 2 * k.G * K.Mearth / (k.c * k.c);
    rows.push({ dial: 'cg', what: 'Earth’s horizon radius, 2GM/c²', val: length(rs), level: P.logLevel(rs / K.Rearth, -6, 0),
      verdict: rs >= K.Rearth ? 'Bigger than Earth. The planet would be a black hole; the tower scene keeps only its local pull.' : 'Earth is ' + num(K.Rearth / rs, 2) + ' times bigger than that. No horizon.' });

    var lam = P.deBroglie(REF.ball, REF.roll, k.hbar);
    rows.push({ dial: 'h', what: 'Wavelength of a rolling football', val: length(lam), level: P.logLevel(lam, -3, 0),
      verdict: lam >= 0.05 ? 'About the size of a doorway. It diffracts and interferes.' : (lam < PROTON ? num(PROTON / lam, 2) + ' times smaller than a proton. It is a ball.' : 'Far finer than anything it meets. It is a ball.') });

    var hover = (2 / 3) * P.bouncer(REF.ball, g, k.hbar).heights[0];
    rows.push({ dial: 'gh', what: 'Lowest a football can sit above the floor', val: length(hover), level: P.logLevel(hover, -3, 0),
      verdict: hover >= 0.01 ? 'It hovers. Gravity pulls down, the uncertainty principle pushes back.' : 'Indistinguishable from resting on it.' });

    var a0 = P.bohrRadius(k);
    rows.push({ dial: 'h', what: 'Size of an atom, ∝ ħ²', val: length(a0), level: P.clamp01(Math.log10(a0 / 5.29e-11) / 20),
      verdict: a0 > 1e-3 ? 'Atoms that wide would leave no solid matter. The lab ignores this by decree.' : 'Atoms stay atoms.' });

    var al = P.fineStructure(k);
    rows.push({ dial: 'ch', what: 'Fine-structure constant α', val: al < 1 ? '1/' + num(1 / al, 3) : num(al), level: P.clamp01(Math.abs(Math.log10(al * 137.036)) / 6),
      verdict: al >= 1 ? 'Above 1: an electron in hydrogen would have to outrun light. No atoms, no chemistry.' : (Math.abs(al * 137.036 - 1) > 0.01 ? 'Changed, so every atom and molecule is different.' : 'The real value. Chemistry as usual.') });

    var pl = P.planck(k);
    rows.push({ dial: 'all', what: 'Planck length and mass', val: length(pl.length, 2) + ', ' + mass(pl.mass, 2), level: P.logLevel(pl.length, -35, -2),
      verdict: pl.length > PROTON ? 'Squeeze ' + mass(pl.mass, 2) + ' into ' + length(pl.length, 2) + ' and you need quantum gravity. That is ' + num(pl.length / PROTON, 2) + ' times the size of a proton, and nobody knows the rules.' : 'Quantum gravity begins ' + num(Math.log10(PROTON / pl.length), 2) + ' orders of magnitude below a proton.' });
    return rows;
  }
  function renderEveryday() {
    var now = everyday(state), real = everyday(REAL), html = '';
    for (var i = 0; i < now.length; i++) {
      var r = now[i], cls = r.dial.length > 1 ? 'all' : r.dial;
      var syms = r.dial === 'all' ? 'cgh' : r.dial;
      var tag = syms.split('').map(function (s) { return '<span class="sym sym-' + s + '">' + (s === 'h' ? 'ħ' : s === 'g' ? 'G' : 'c') + '</span>'; }).join('');
      html += '<tr class="' + (r.level > 0.5 ? 'hot' : '') + '"><td class="what">' + tag + ' &nbsp;' + r.what + '</td>' +
        '<td class="val" data-label="Now">' + r.val + '<span class="meter ' + cls + '"><i style="width:' + Math.round(r.level * 100) + '%"></i></span></td>' +
        '<td class="val" data-label="In our universe">' + real[i].val + '</td><td class="verdict">' + r.verdict + '</td></tr>';
    }
    $('everyday').innerHTML = html;
  }
  function renderTheories() {
    var r = P.regime(state);
    Array.prototype.forEach.call($('theories').querySelectorAll('.theory'), function (b, i) {
      b.setAttribute('aria-current', i === r.corner ? 'true' : 'false');
    });
  }

  /* ---------- tabs ---------- */
  var tabs = {}, current = null;
  function registerTab(name, hooks) { tabs[name] = hooks; }
  function show(name) {
    if (!document.getElementById('panel-' + name)) name = 'map';
    if (current === name) return;
    if (current && tabs[current] && tabs[current].hide) tabs[current].hide();
    current = name;
    ['map', 'light', 'gravity', 'quantum'].forEach(function (t) {
      $('panel-' + t).hidden = t !== name;
      $('tab-' + t).setAttribute('aria-selected', t === name ? 'true' : 'false');
    });
    if (tabs[name] && tabs[name].show) tabs[name].show();
    try { if (location.hash.slice(1) !== name) history.replaceState(null, '', '#' + name); } catch (e) { location.hash = name; }
  }
  function currentTab() { return current; }

  /* ---------- wiring ---------- */
  function renderShell() {
    renderDials(); renderPresets();
    $('regime-name').textContent = P.regime(state).name;
    drawCube($('mini-cube'), false);
  }
  function renderMap() {
    drawCube($('cube'), true); renderTheories(); renderEveryday();
  }

  function init() {
    try {
      var saved = JSON.parse(localStorage.getItem('cgh-dials') || 'null');
      if (saved && saved.c > 0 && saved.G > 0 && saved.hbar > 0) { state.c = saved.c; state.G = saved.G; state.hbar = saved.hbar; }
    } catch (e) { /* start from the real values */ }

    ['c', 'G', 'hbar'].forEach(function (k) {
      $('dial-' + k).addEventListener('input', function (ev) { var p = {}; p[k] = fromSlider(k, Number(ev.target.value)); set(p); });
      $('reset-' + k).addEventListener('click', function () { var p = {}; p[k] = REAL[k]; set(p); });
    });
    PRESETS.forEach(function (p) {
      var b = document.createElement('button');
      b.className = 'chip'; b.type = 'button'; b.textContent = p.name;
      b.addEventListener('click', function () { set(p.v); });
      $('presets').appendChild(b);
    });
    P.CORNERS.forEach(function (name, bits) {
      var li = document.createElement('li'), b = document.createElement('button');
      b.className = 'theory'; b.type = 'button';
      b.innerHTML = '<span class="bits">' + bitsHTML(bits) + '</span><span class="t-name">' + name + '</span><span class="t-what">' + THEORIES[bits].what + '</span>';
      b.addEventListener('click', function () { set(cornerPreset(bits)); });
      li.appendChild(b); $('theories').appendChild(li);
    });
    $('cube').addEventListener('click', function (ev) {
      var rect = ev.target.getBoundingClientRect(), g = cubeGeometry(rect.width, rect.height);
      var x = ev.clientX - rect.left, y = ev.clientY - rect.top, best = -1, bd = 30 * 30;
      g.corners.forEach(function (c, i) { var d = (c[0] - x) * (c[0] - x) + (c[1] - y) * (c[1] - y); if (d < bd) { bd = d; best = i; } });
      if (best >= 0) set(cornerPreset(best));
    });
    Array.prototype.forEach.call(document.querySelectorAll('.tab'), function (t) {
      t.addEventListener('click', function () { show(t.getAttribute('data-tab')); });
    });
    window.addEventListener('hashchange', function () { show(location.hash.slice(1) || 'map'); });

    registerTab('map', { show: renderMap });
    onChange(function () { renderShell(); if (current === 'map') renderMap(); });
    onTheme(function () { renderShell(); if (current === 'map') renderMap(); });
    window.addEventListener('resize', function () { drawCube($('mini-cube'), false); if (current === 'map') drawCube($('cube'), true); });

  }
  // Called once every scene has registered itself.
  function start() {
    renderShell();
    show(location.hash.slice(1) || 'map');
  }

  CGH.app = {
    state: state, set: set, onChange: onChange, isReal: isReal,
    num: num, length: length, duration: duration, mass: mass, clock: clock, times: times, sup: sup,
    css: css, onTheme: onTheme, fit: fit, registerTab: registerTab, show: show, currentTab: currentTab, init: init, start: start, $: $
  };
})();
