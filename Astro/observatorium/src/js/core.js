/* Observatorium — core: namespace, small helpers, storage, events, CSV parsing and data loading. */
(function (G) {
  'use strict';
  var OBS = G.OBS = G.OBS || {};

  OBS.$ = function (id) { return document.getElementById(id); };
  OBS.esc = function (s) {
    return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; });
  };
  OBS.clamp = function (v, a, b) { return v < a ? a : v > b ? b : v; };

  /** A number with `sig` significant figures; powers of ten written as ×10ⁿ outside 1e-3 … 1e5. */
  var SUP = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  OBS.fmt = function (v, sig) {
    if (v === null || v === undefined || !isFinite(v)) return '–';
    sig = sig || 3;
    var a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) {
      var e = Math.floor(Math.log10(a)), m = v / Math.pow(10, e);
      if (Math.abs(+m.toPrecision(sig)) >= 10) { m /= 10; e += 1; }
      return (+m.toPrecision(sig)) + '×10' + String(e).split('').map(function (c) { return SUP[c]; }).join('');
    }
    var s = (+v.toPrecision(sig)).toString();
    if (a >= 1000) s = Math.round(v).toLocaleString('en-US');
    return s.replace('-', '−');
  };
  /** Fixed decimals, with a proper minus sign. */
  OBS.fix = function (v, d) { return isFinite(v) ? v.toFixed(d).replace('-', '−') : '–'; };

  // ---------------------------------------------------------------- per-viewer storage (may be unavailable)
  OBS.store = {
    get: function (k, def) {
      try { var v = G.localStorage.getItem('observatorium.' + k); return v === null ? def : JSON.parse(v); } catch (e) { return def; }
    },
    set: function (k, v) { try { G.localStorage.setItem('observatorium.' + k, JSON.stringify(v)); } catch (e) { /* private mode */ } }
  };

  // ---------------------------------------------------------------- events
  var handlers = {};
  OBS.on = function (ev, fn) { (handlers[ev] = handlers[ev] || []).push(fn); };
  OBS.emit = function (ev, data) { (handlers[ev] || []).slice().forEach(function (fn) { fn(data); }); };

  // ---------------------------------------------------------------- CSV
  function splitLine(line) {
    if (line.indexOf('"') < 0) return line.split(',');
    var out = [], cur = '', q = false;
    for (var i = 0; i < line.length; i++) {
      var ch = line[i];
      if (q) {
        if (ch === '"' && line[i + 1] === '"') { cur += '"'; i++; } else if (ch === '"') q = false; else cur += ch;
      } else if (ch === '"') q = true; else if (ch === ',') { out.push(cur); cur = ''; } else cur += ch;
    }
    out.push(cur);
    return out;
  }
  /** Parse CSV text into columns. A column whose every non-empty value is a number becomes a Float64Array
   *  (empty cells are NaN); anything else stays an array of strings. */
  OBS.parseCSV = function (text) {
    var lines = text.split(/\r?\n/).filter(function (l) { return l.length; });
    var names = splitLine(lines[0]), n = lines.length - 1, rows = new Array(n);
    for (var i = 0; i < n; i++) rows[i] = splitLine(lines[i + 1]);
    var cols = {};
    names.forEach(function (name, j) {
      var numeric = true, k;
      for (k = 0; k < n; k++) {
        var s = rows[k][j];
        // not a number, or an identifier too long for a double (Gaia source_id has 19 digits): keep as text
        if (s !== '' && s !== undefined && (!isFinite(+s) || /^\d{16,}$/.test(s))) { numeric = false; break; }
      }
      if (numeric) {
        var a = new Float64Array(n);
        for (k = 0; k < n; k++) { var v = rows[k][j]; a[k] = v === '' || v === undefined ? NaN : +v; }
        cols[name] = a;
      } else {
        var b = new Array(n);
        for (k = 0; k < n; k++) b[k] = rows[k][j] || '';
        cols[name] = b;
      }
    });
    return { n: n, names: names, cols: cols };
  };

  // ---------------------------------------------------------------- data loading
  // The page ships either with the data inlined (OBS_INLINE, the stand-alone file) or beside it in data/.
  var cache = {};
  function getText(file) {
    var inline = G.OBS_INLINE && G.OBS_INLINE[file];
    if (inline !== undefined) return Promise.resolve(inline);
    return fetch('data/' + file).then(function (r) {
      if (!r.ok) throw new Error('Could not load data/' + file + ' (' + r.status + ')');
      return r.text();
    });
  }
  OBS.loadTable = function (name) {
    if (!cache[name]) cache[name] = getText(name + '.csv').then(OBS.parseCSV);
    return cache[name];
  };
  OBS.loadJSON = function (name) {
    if (!cache['json:' + name]) cache['json:' + name] = getText(name + '.json').then(JSON.parse);
    return cache['json:' + name];
  };

  OBS.reducedMotion = function () {
    try { return G.matchMedia('(prefers-reduced-motion: reduce)').matches; } catch (e) { return false; }
  };
  /** Theme colours from CSS tokens, re-read whenever the theme may have changed. */
  OBS.colours = function () {
    var cs = G.getComputedStyle(document.documentElement), o = {};
    ['--bg', '--panel', '--ink', '--ink-2', '--ink-3', '--rule', '--accent', '--accent-2', '--sel', '--warn', '--ok',
      '--sky', '--sky-star', '--sky-glow', '--grid', '--c1', '--c2', '--c3', '--c4', '--c5', '--c6', '--c7', '--c8'].forEach(function (k) {
      o[k.slice(2)] = cs.getPropertyValue(k).trim();
    });
    o.dark = cs.getPropertyValue('--is-dark').trim() === '1';
    return o;
  };
})(typeof window !== 'undefined' ? window : globalThis);
