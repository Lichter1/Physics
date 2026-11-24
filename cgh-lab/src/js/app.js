/* cGħ World — the console: four dimensionless dials, number formatting, and the cube of theories. */
(function () {
  'use strict';
  var P = CGH.physics, G = CGH.geo, REAL = P.REAL;
  var $ = function (id) { return document.getElementById(id); };
  var WALK = 1.4, MASS = 70, ROCK = 4 / 3 * Math.PI * Math.pow(G.R, 3) * 3000;   // a planet of ordinary rock

  /* ---------- numbers people can read ---------- */
  var SUP = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  function num(x, sig) {
    sig = sig || 3;
    if (x === 0) return '0';
    if (!isFinite(x)) return '–';
    var e = Math.floor(Math.log10(Math.abs(x)));
    if (e >= -3 && e < 6) { var v = Number(x.toPrecision(sig)); return (Math.abs(v) >= 1e4 ? v.toLocaleString('en-US') : String(v)).replace('-', '−'); }
    var m = (x / Math.pow(10, e)).toFixed(sig - 1);
    if (Math.abs(Number(m)) >= 10) { e += 1; m = (x / Math.pow(10, e)).toFixed(sig - 1); }
    return m.replace('-', '−') + '×10' + String(e).split('').map(function (c) { return SUP[c]; }).join('');
  }
  function unit(x, t, sig) { for (var i = 0; i < t.length; i++) if (Math.abs(x) < t[i][0] || i === t.length - 1) return num(x / t[i][1], sig) + ' ' + t[i][2]; }
  function length(m, sig) { return unit(m, [[1e-9, 1, 'm'], [1e-6, 1e-9, 'nm'], [1e-3, 1e-6, 'µm'], [1e-2, 1e-3, 'mm'], [1, 1e-2, 'cm'], [1e4, 1, 'm'], [1e13, 1e3, 'km'], [Infinity, 9.4607e15, 'light-years']], sig); }
  function duration(s, sig) { return unit(s, [[1e-6, 1e-9, 'ns'], [1e-3, 1e-6, 'µs'], [1, 1e-3, 'ms'], [120, 1, 's'], [7200, 60, 'min'], [172800, 3600, 'h'], [6.3e7, 86400, 'days'], [Infinity, 3.156e7, 'years']], sig); }
  function clock(s) {
    var neg = s < 0; s = Math.abs(s);
    var m = Math.floor(s / 60), r = s - 60 * m, out;
    out = s < 3600 ? m + ':' + (r < 10 ? '0' : '') + r.toFixed(1) : Math.floor(s / 3600) + ':' + String(m % 60).padStart(2, '0') + ':' + String(Math.floor(r)).padStart(2, '0');
    return (neg ? '−' : '') + out;
  }

  /* ---------- the four dials ---------- */
  // cw: light speed in walking speeds · comp: rs/R of the planet · lam: your wavelength at walking pace (m)
  // photons: photons a mid-grey pixel collects per frame
  var KEYS = ['cw', 'comp', 'lam', 'photons', 'alpha', 'cosmo'];
  var REALV = { cw: REAL.c / WALK, comp: 2 * REAL.G * ROCK / (REAL.c * REAL.c * G.R), lam: 2 * Math.PI * REAL.hbar / (MASS * WALK), photons: 3e5, alpha: 1, cosmo: 1.65e26 };
  var state = { cw: REALV.cw, comp: REALV.comp, lam: REALV.lam, photons: REALV.photons, alpha: 1, cosmo: REALV.cosmo };
  var DIAL = {
    cw: { lo: Math.log10(1.2), hi: Math.log10(REALV.cw) },
    comp: { cube: true },
    lam: { lo: -2, hi: Math.log10(8) },
    photons: { lo: Math.log10(0.3), hi: Math.log10(REALV.photons) },
    alpha: { lo: Math.log10(0.25), hi: Math.log10(4) },
    cosmo: { lo: 3, hi: Math.log10(REALV.cosmo) }
  };
  function toSlider(k, v) {
    if (k === 'comp') return v < 1e-9 ? 0 : Math.round(1000 * Math.cbrt(v / 0.95));
    if (k === 'lam' && v < 1e-2) return 0;
    var d = DIAL[k]; return Math.round(1000 * (Math.log10(v) - d.lo) / (d.hi - d.lo));
  }
  function fromSlider(k, s) {
    if (k === 'comp') return s <= 2 ? REALV.comp : 0.95 * Math.pow(s / 1000, 3);
    if (k === 'lam' && s <= 2) return REALV.lam;
    if ((k === 'cw' || k === 'photons' || k === 'cosmo') && s >= 995) return REALV[k];
    if (k === 'alpha' && Math.abs(s - 500) <= 6) return 1;
    var d = DIAL[k]; return Number(Math.pow(10, d.lo + (d.hi - d.lo) * s / 1000).toPrecision(3));
  }
  var listeners = [];
  function onChange(fn) { listeners.push(fn); }
  function set(patch) {
    var changed = false;
    for (var k in patch) if (state[k] !== patch[k]) { state[k] = patch[k]; changed = true; }
    if (!changed) return;
    try { localStorage.setItem('cgh-world', JSON.stringify(state)); } catch (e) { /* optional */ }
    listeners.forEach(function (fn) { fn(state); });
  }
  function isReal(k) { return Math.abs(state[k] / REALV[k] - 1) < 1e-6; }

  // What the dials mean in c, G and ħ.
  function derived() {
    var c = state.cw * WALK, rs = state.comp * G.R, GM = rs * c * c / 2;
    return {
      c: c, rs: rs, GM: GM, g: GM / (G.R * G.R * Math.sqrt(1 - state.comp)), Gratio: GM / (REAL.G * ROCK),
      hbar: state.lam * MASS * WALK / (2 * Math.PI), hbarLight: REAL.hbar * REALV.photons / state.photons,
      alpha2: state.alpha * state.alpha, lam: 1 / (state.cosmo * state.cosmo)
    };
  }

  function note(k) {
    var d = derived(), v = state[k];
    if (k === 'cw') return isReal(k) ? 'our universe: c = 3.00×10⁸ m/s' : 'c = ' + num(d.c) + ' m/s; a walk is ' + num(1 / v, 2) + ' c';
    if (k === 'comp') {
      if (isReal(k)) return 'a 382 m rock in our universe';
      return v < 0.5 ? 'GM = ' + num(d.GM) + ' m³/s² at this c; a neutron star is 0.4–0.5' : v < 0.66 ? 'near 2/3, light orbits at your eye' : v < 0.68 ? 'light orbits at your eye' : 'inside the photon orbit: the sky shrinks overhead';
    }
    if (k === 'lam') return isReal(k) ? 'our universe: ħ = 1.05×10⁻³⁴ J·s' : 'ħ = ' + num(d.hbar) + ' J·s' + (v < 0.5 ? '; too fine to show here' : '');
    if (k === 'alpha') return isReal(k) ? 'our universe: α = 1/137.036' : 'atomic colours shift ×' + num(v * v, 3) + ' in frequency; atoms ' + num(1 / v, 2) + '× as wide; no element beyond Z ≈ ' + Math.floor(137 / v);
    if (k === 'cosmo') {
      if (isReal(k)) return 'our universe: about 17 billion light-years';
      var d = derived(), rf = Math.cbrt(d.rs * v * v / 2);
      return 'higher clocks slowed by Λ; ' + (d.rs > 0 && rf - G.R < 5000 ? 'a ball would float ' + length(rf - G.R, 2) + ' up' : 'its push is too weak here');
    }
    return isReal(k) ? 'roughly what an eye gets in daylight' : 'as if ħ were ' + num(d.hbarLight / REAL.hbar, 2) + '× larger, light kept as bright';
  }
  function show(k) {
    var v = state[k];
    return k === 'cw' ? num(v) + ' walking speeds' : k === 'comp' ? 'rs/R = ' + num(v) : k === 'lam' ? 'λ = ' + length(v) :
      k === 'alpha' ? num(v, 3) + ' × today’s α' : k === 'cosmo' ? 'horizon at ' + length(v, 3) : num(v) + ' per pixel';
  }
  var locked = null;                 // during a mission: the only dials you may turn
  function renderDials() {
    KEYS.forEach(function (k) {
      var off = !!locked && locked.indexOf(k) < 0;
      $('dial-' + k).value = toSlider(k, state[k]);
      $('dial-' + k).disabled = off;
      $('dial-' + k).closest('.dial').classList.toggle('locked', off);
      $('val-' + k).textContent = show(k);
      $('note-' + k).textContent = off ? 'set by the mission' : note(k);
      $('reset-' + k).disabled = off || isReal(k);
    });
    Array.prototype.forEach.call($('presets').children, function (b) { b.disabled = !!locked; });
  }
  function lock(keys) { locked = keys; renderDials(); }

  var PRESETS = [
    { name: 'Our universe', v: REALV },
    { name: 'Slow light', v: { cw: 7, comp: REALV.comp, lam: REALV.lam, photons: REALV.photons } },
    { name: 'Heavy planet', v: { cw: 7, comp: 0.45, lam: REALV.lam, photons: REALV.photons } },
    { name: 'Light orbits', v: { cw: 7, comp: 0.667, lam: REALV.lam, photons: REALV.photons } },
    { name: 'You are a wave', v: { cw: REALV.cw, comp: REALV.comp, lam: 2, photons: REALV.photons }, pose: 'doors' },
    { name: 'Everything', v: { cw: 7, comp: 0.45, lam: 1.2, photons: 150 } }
  ];
  PRESETS.forEach(function (p) { p.v = Object.assign({ alpha: 1, cosmo: REALV.cosmo }, p.v); });
  function renderPresets() {
    Array.prototype.forEach.call($('presets').children, function (b, i) {
      var v = PRESETS[i].v, on = KEYS.every(function (k) { return Math.abs(state[k] / v[k] - 1) < 1e-3; });
      b.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
  }

  /* ---------- cube of theories ---------- */
  function regime() {
    var rel = P.logLevel(1 / state.cw, -3, -0.5), grav = P.logLevel(Math.max(state.comp, 1e-30), -3, -0.7), quant = P.logLevel(state.lam, -2, -0.3);
    var bits = (rel > 0.5 ? 1 : 0) | (grav > 0.5 ? 2 : 0) | (quant > 0.5 ? 4 : 0);
    return { rel: rel, grav: grav, quant: quant, corner: bits, name: P.CORNERS[bits] };
  }
  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function fit(canvas, maxDpr) {
    var r = canvas.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, maxDpr || 2);
    var w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    return { w: r.width, h: r.height, dpr: dpr };
  }
  function drawCube() {
    var cv = $('mini-cube'), f = fit(cv, 2), ctx = cv.getContext('2d'), r = regime();
    ctx.setTransform(f.dpr, 0, 0, f.dpr, 0, 0); ctx.clearRect(0, 0, f.w, f.h);
    var S = Math.min(f.w, f.h) * 0.56, ox = 6, oy = f.h - 6;
    function pr(x, y, z) { return [ox + S * (x + 0.42 * z), oy - S * (y + 0.32 * z)]; }
    var col = [css('--c'), css('--g'), css('--h')];
    for (var a = 0; a < 8; a++) for (var bit = 0; bit < 3; bit++) {
      var b = a | (1 << bit); if (b === a) continue;
      var p = pr(a & 1, (a >> 1) & 1, (a >> 2) & 1), q = pr(b & 1, (b >> 1) & 1, (b >> 2) & 1);
      ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]);
      ctx.strokeStyle = a === 0 ? col[bit] : css('--line'); ctx.lineWidth = a === 0 ? 2 : 1; ctx.stroke();
    }
    var d = pr(r.rel, r.grav, r.quant);
    ctx.beginPath(); ctx.arc(d[0], d[1], 4, 0, 6.2832); ctx.fillStyle = css('--focus'); ctx.fill();
    $('regime-name').textContent = r.name;
  }

  function render() { renderDials(); renderPresets(); drawCube(); }
  function init() {
    try {
      var s = JSON.parse(localStorage.getItem('cgh-world') || 'null');
      if (s && ['cw', 'comp', 'lam', 'photons'].every(function (k) { return s[k] > 0; })) state = Object.assign(state, s);
      CGH.app.state = state;
    } catch (e) { /* start from our universe */ }
    KEYS.forEach(function (k) {
      $('dial-' + k).addEventListener('input', function (ev) { var p = {}; p[k] = fromSlider(k, Number(ev.target.value)); set(p); });
      $('reset-' + k).addEventListener('click', function () { var p = {}; p[k] = REALV[k]; set(p); });
    });
    PRESETS.forEach(function (p) {
      var b = document.createElement('button');
      b.className = 'chip'; b.type = 'button'; b.textContent = p.name;
      b.addEventListener('click', function () {
        set(Object.assign({}, p.v));
        if (p.pose) document.dispatchEvent(new CustomEvent('cgh-pose', { detail: p.pose }));
      });
      $('presets').appendChild(b);
    });
    onChange(render);
    window.addEventListener('resize', drawCube);
    render();
  }

  CGH.app = {
    state: state, KEYS: KEYS, REAL: REALV, WALK: WALK, MASS: MASS, set: set, lock: lock, onChange: onChange, isReal: isReal, derived: derived, regime: regime,
    num: num, length: length, duration: duration, clock: clock, css: css, fit: fit, init: init, $: $
  };
})();
