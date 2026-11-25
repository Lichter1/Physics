/* Observatorium — the data lens: a scatter plot (or histogram) of any two variables of the open dataset,
 * with brushing, hover, zoom and the hand-fitting tools missions use: lines, markers, horizontal and vertical
 * rules, reference populations that slide, model curves and period folding. */
(function (G) {
  'use strict';
  var OBS = G.OBS;
  var PAD = { l: 58, r: 14, t: 12, b: 42 };

  function Chart(canvas) {
    this.cv = canvas;
    this.ctx = canvas.getContext('2d');
    this.ds = null; this.sub = null; this.filter = null;
    this.xid = null; this.yid = null;
    this.sel = null;
    this.overlays = [];
    this.fold = null;              // {P, t0} when folding a time series
    this.bins = 40;
    this.tool = 'select';
    this.colourBy = 'auto';
    this.view = null;              // {x0, x1, y0, y1} in axis space
    this.hover = -1;
    this.drag = null;
    this.listeners = [];
    this._bind();
  }
  var P = Chart.prototype;
  OBS.Chart = Chart;

  // ------------------------------------------------------------------------------------------ data
  P.setDataset = function (ds, sub) {
    this.ds = ds; this.sub = sub === undefined ? (ds.subset ? ds.subset.def : null) : sub;
    this.filter = null; this.fold = null; this.overlays = [];
    this.sel = new Uint8Array(ds.table.n);
    this.colourBy = 'auto';
    this.xid = ds.x; this.yid = ds.y;
    this._mask(); this.autoscale();
  };
  P.setSubset = function (sub) { this.sub = sub; this.sel = new Uint8Array(this.ds.table.n); this._mask(); this.autoscale(); };
  P.setFilter = function (fn) { this.filter = fn; this._mask(); this.autoscale(); };
  P._mask = function () {
    var ds = this.ds, t = ds.table, n = t.n, m = new Uint8Array(n), c = t.cols;
    var subcol = ds.subset && this.sub && this.sub !== '*' ? c[ds.subset.col] : null;
    for (var i = 0; i < n; i++) m[i] = (!subcol || subcol[i] === this.sub) && (!this.filter || this.filter(c, i)) ? 1 : 0;
    this.mask = m;
  };
  P.setAxes = function (xid, yid) {
    if (xid) this.xid = xid;
    if (yid) this.yid = yid;
    this.autoscale();
  };
  P.hist = function () { return this.yid === '#'; };
  P.xvar = function () { return this.fold ? { id: 'phase', label: 'Phase (cycles)', unit: '', scale: 'lin' } : OBS.varOf(this.ds, this.xid); };
  P.yvar = function () { return this.hist() ? { id: '#', label: 'Number of objects', unit: '', scale: 'lin' } : OBS.varOf(this.ds, this.yid); };
  /** Raw x values; when folding, phase in [0, 1). */
  P.xs = function () {
    var x = OBS.column(this.ds, this.xid);
    if (!this.fold) return x;
    var out = new Float64Array(x.length), f = this.fold;
    for (var i = 0; i < x.length; i++) { var ph = ((x[i] - f.t0) / f.P) % 1; out[i] = ph < 0 ? ph + 1 : ph; }
    return out;
  };
  P.ys = function () { return this.hist() ? null : OBS.column(this.ds, this.yid); };
  function ax(v, log) { return log ? (v > 0 ? Math.log10(v) : NaN) : v; }
  function unax(a, log) { return log ? Math.pow(10, a) : a; }
  P.logX = function () { return !this.fold && this.xvar().scale === 'log'; };
  P.logY = function () { return !this.hist() && this.yvar().scale === 'log'; };

  P.autoscale = function () {
    var xs = this.xs(), ys = this.ys(), lx = this.logX(), ly = this.logY(), m = this.mask, a = [], b = [];
    for (var i = 0; i < xs.length; i++) {
      if (!m[i]) continue;
      var X = ax(xs[i], lx), Y = ys ? ax(ys[i], ly) : 0;
      if (isFinite(X) && isFinite(Y)) { a.push(X); b.push(Y); }
    }
    function rng(v, lo, hi) {
      if (!v.length) return [0, 1];
      v.sort(function (p, q) { return p - q; });
      var x0 = v[Math.floor(lo * (v.length - 1))], x1 = v[Math.ceil(hi * (v.length - 1))];
      if (x1 - x0 < 1e-9) { x0 -= 0.5; x1 += 0.5; }
      var pad = 0.04 * (x1 - x0);
      return [x0 - pad, x1 + pad];
    }
    var big = a.length > 3000, q = big ? 0.002 : 0;
    var rx = this.fold ? [0, 2] : rng(a, q, 1 - q), ry = ys ? rng(b, q, 1 - q) : [0, 1];
    this.view = { x0: rx[0], x1: rx[1], y0: ry[0], y1: ry[1] };
    if (this.hist()) this._histMax();
    this.draw();
  };
  P.setView = function (v) {
    var lx = this.logX(), ly = this.logY();
    this.view = { x0: ax(v.x0, lx), x1: ax(v.x1, lx), y0: v.y0 === undefined ? this.view.y0 : ax(v.y0, ly), y1: v.y1 === undefined ? this.view.y1 : ax(v.y1, ly) };
    if (this.hist()) this._histMax();
    this.draw();
  };
  P._histMax = function () {
    var h = this.histogram(); var mx = 0;
    h.all.forEach(function (c) { mx = Math.max(mx, c); });
    this.view.y0 = 0; this.view.y1 = mx * 1.08 || 1;
  };
  /** Counts per bin over the visible x range, for all shown objects and the selected ones. */
  P.histogram = function () {
    var xs = this.xs(), lx = this.logX(), v = this.view, nb = this.bins, all = new Array(nb).fill(0), sel = new Array(nb).fill(0);
    var w = (v.x1 - v.x0) / nb;
    for (var i = 0; i < xs.length; i++) {
      if (!this.mask[i]) continue;
      var X = ax(xs[i], lx), k = Math.floor((X - v.x0) / w);
      if (k >= 0 && k < nb) { all[k]++; if (this.sel[i]) sel[k]++; }
    }
    return { all: all, sel: sel, x0: v.x0, w: w, log: lx };
  };

  // ------------------------------------------------------------------------------------------ geometry
  P.size = function () {
    var r = this.cv.getBoundingClientRect(), dpr = Math.min(2, G.devicePixelRatio || 1);
    var w = Math.max(200, Math.round(r.width)), h = Math.max(160, Math.round(r.height));
    if (this.cv.width !== w * dpr || this.cv.height !== h * dpr) { this.cv.width = w * dpr; this.cv.height = h * dpr; }
    this.W = w; this.H = h; this.dpr = dpr;
  };
  P.px = function (X) {
    var v = this.view, rev = !this.fold && this.xvar().rev;
    var f = (X - v.x0) / (v.x1 - v.x0); if (rev) f = 1 - f;
    return PAD.l + f * (this.W - PAD.l - PAD.r);
  };
  P.py = function (Y) {
    var v = this.view, rev = !this.hist() && this.yvar().rev;
    var f = (Y - v.y0) / (v.y1 - v.y0); if (rev) f = 1 - f;
    return this.H - PAD.b - f * (this.H - PAD.t - PAD.b);
  };
  P.ix = function (px) {
    var v = this.view, rev = !this.fold && this.xvar().rev;
    var f = (px - PAD.l) / (this.W - PAD.l - PAD.r); if (rev) f = 1 - f;
    return v.x0 + f * (v.x1 - v.x0);
  };
  P.iy = function (py) {
    var v = this.view, rev = !this.hist() && this.yvar().rev;
    var f = (this.H - PAD.b - py) / (this.H - PAD.t - PAD.b); if (rev) f = 1 - f;
    return v.y0 + f * (v.y1 - v.y0);
  };
  /** Data value (not axis space) under a pixel. */
  P.dataAt = function (px, py) { return { x: unax(this.ix(px), this.logX()), y: unax(this.iy(py), this.logY()) }; };
  P.toPx = function (x, y) { return [this.px(ax(x, this.logX())), this.py(ax(y, this.logY()))]; };

  // ------------------------------------------------------------------------------------------ colours
  /** Approximate colour a star of this BP-RP (or B-V) colour has, for drawing. */
  OBS.starRGB = function (c, kind, dark) {
    var t = kind === 'bv' ? (c + 0.3) / 2.3 : (c + 0.2) / 3.4;   // 0 = hot blue, 1 = cool red
    t = isFinite(t) ? Math.max(0, Math.min(1, t)) : 0.4;          // unknown colour: draw it white-yellow
    var stops = dark ? [[155, 176, 255], [202, 215, 255], [255, 244, 234], [255, 210, 161], [255, 160, 96], [255, 110, 70]]
      : [[40, 80, 200], [70, 110, 200], [120, 110, 90], [175, 115, 40], [190, 80, 30], [170, 40, 30]];
    var s = t * (stops.length - 1), k = Math.min(stops.length - 2, Math.floor(s)), f = s - k;
    var a = stops[k], b = stops[k + 1];
    return [Math.round(a[0] + f * (b[0] - a[0])), Math.round(a[1] + f * (b[1] - a[1])), Math.round(a[2] + f * (b[2] - a[2]))];
  };
  P.colourKey = function () {
    var ds = this.ds;
    if (this.colourBy === 'none') return null;
    if (this.colourBy !== 'auto') return this.colourBy;
    if (ds.colourBy) return ds.colourBy;
    if (ds.gaiaColour) return '@gaia';
    if (ds.bv) return '@bv';
    return null;
  };
  /** For categorical colouring: the categories in order of size, and a palette lookup. */
  P.categories = function () {
    var key = this.colourKey();
    if (!key || key[0] === '@') return null;
    var ck = this.ds.id + '|' + this.sub + '|' + key + '|' + (this.filter ? this.filter.toString().length : 0);
    if (this._cats && this._catsKey === ck) return this._cats;
    var col = OBS.column(this.ds, key), cnt = {};
    for (var i = 0; i < col.length; i++) if (this.mask[i]) cnt[col[i]] = (cnt[col[i]] || 0) + 1;
    var order = this.ds.catOrder && this.ds.catOrder[key];
    var cats = order ? order.filter(function (c) { return cnt[c]; }) : Object.keys(cnt).sort(function (a, b) { return cnt[b] - cnt[a]; });
    this._catsKey = ck;
    return (this._cats = { col: col, cats: cats, cnt: cnt });
  };
  /** Category colours: the theme's palette, unless a figure sets its own (chart.palette). */
  P.pal = function (C) { return this.palette || [C.c1, C.c2, C.c3, C.c4, C.c5, C.c6, C.c7, C.c8]; };
  /** The colour a point is drawn in (shared with the sky). */
  P.colourOf = function (i, C) {
    var key = this.colourKey();
    if (key === '@gaia' || key === '@bv') return 'rgb(' + OBS.starRGB(OBS.column(this.ds, key === '@gaia' ? 'bp_rp' : 'ci')[i], key === '@bv' ? 'bv' : 'gaia', C.dark) + ')';
    var cats = this.categories();
    if (cats) return this.pal(C)[Math.min(7, cats.cats.indexOf(cats.col[i]))];
    return C.accent;
  };

  // ------------------------------------------------------------------------------------------ drawing
  P.draw = function () {
    var self = this;
    if (this._raf) return;
    this._raf = G.requestAnimationFrame(function () { self._raf = 0; self._draw(); });
  };
  P._draw = function () {
    if (!this.ds) return;
    this.size();
    var ctx = this.ctx, W = this.W, H = this.H, C = OBS.colours(), dpr = this.dpr;
    if (this.equal) {   // same scale on both axes (sky offsets)
      var v = this.view, sx = (v.x1 - v.x0) / (W - PAD.l - PAD.r), sy = (v.y1 - v.y0) / (H - PAD.t - PAD.b), s = Math.max(sx, sy);
      var mx = (v.x0 + v.x1) / 2, my = (v.y0 + v.y1) / 2, hx = s * (W - PAD.l - PAD.r) / 2, hy = s * (H - PAD.t - PAD.b) / 2;
      this.view = { x0: mx - hx, x1: mx + hx, y0: my - hy, y1: my + hy };
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = C.panel; ctx.fillRect(0, 0, W, H);
    this._axes(ctx, C);
    ctx.save();
    ctx.beginPath(); ctx.rect(PAD.l, PAD.t, W - PAD.l - PAD.r, H - PAD.t - PAD.b); ctx.clip();
    this.overlays.forEach(function (o) { if ((o.type === 'ref' || o.type === 'span') && o.under !== false) this._overlay(ctx, o, C); }, this);
    if (this.hist()) this._bars(ctx, C); else this._points(ctx, C);
    this.overlays.forEach(function (o) { if (!((o.type === 'ref' || o.type === 'span') && o.under !== false)) this._overlay(ctx, o, C); }, this);
    if (this.drag && this.drag.kind === 'brush') {
      var d = this.drag; ctx.strokeStyle = C.sel; ctx.setLineDash([4, 3]); ctx.lineWidth = 1;
      ctx.fillStyle = C.sel;
      var bx = Math.min(d.x0, d.x1), by = this.hist() ? PAD.t : Math.min(d.y0, d.y1);
      var bw = Math.abs(d.x1 - d.x0), bh = this.hist() ? H - PAD.t - PAD.b : Math.abs(d.y1 - d.y0);
      ctx.globalAlpha = 0.12; ctx.fillRect(bx, by, bw, bh); ctx.globalAlpha = 1;
      ctx.strokeRect(bx, by, bw, bh); ctx.setLineDash([]);
    }
    ctx.restore();
    if (this.hover >= 0 && !this.hist()) this._hover(ctx, C);
    this.listeners.forEach(function (fn) { fn('draw'); });
  };
  P._points = function (ctx, C) {
    var xs = this.xs(), ys = this.ys(), m = this.mask, lx = this.logX(), ly = this.logY(), n = xs.length;
    var shown = 0; for (var i = 0; i < n; i++) shown += m[i];
    var s = shown > 20000 ? 1.4 : shown > 5000 ? 1.9 : shown > 800 ? 2.6 : 3.6, h = s / 2;
    var key = this.colourKey(), cats = this.categories(), dark = C.dark, pal = this.pal(C);
    var ccol = key === '@gaia' ? OBS.column(this.ds, 'bp_rp') : key === '@bv' ? OBS.column(this.ds, 'ci') : null;
    var cycles = this.fold ? 2 : 1, anySel = false;
    for (i = 0; i < n; i++) if (this.sel[i]) { anySel = true; break; }
    ctx.globalAlpha = shown > 20000 ? 0.55 : shown > 5000 ? 0.7 : 0.9;
    // error bars: ds.err is a column for Vobs (rotation curves) or a map {y variable: error column}
    var ecol = !this.ds.err || this.fold ? null : typeof this.ds.err === 'string' ? (this.yid === 'Vobs' ? this.ds.err : null) : this.ds.err[this.yid];
    var err = ecol ? this.ds.table.cols[ecol] : null;
    if (this.ds.connect && !this.fold && !anySel) {   // a spectrum or a time series: join the points, one line per category
      ctx.globalAlpha = 1; ctx.lineWidth = 1.6; var prev = null, pc = null;
      for (i = 0; i < n; i++) {
        if (!m[i]) { prev = null; continue; }
        var XX = ax(xs[i], lx), YY = ax(ys[i], ly), cc = cats ? cats.col[i] : null;
        if (!isFinite(XX) || !isFinite(YY)) { prev = null; continue; }
        var pt = [this.px(XX), this.py(YY)];
        if (prev && cc === pc && XX >= prev[2]) {
          ctx.strokeStyle = cats ? pal[Math.min(7, cats.cats.indexOf(cc))] : (this.pointColour || C.accent);
          ctx.beginPath(); ctx.moveTo(prev[0], prev[1]); ctx.lineTo(pt[0], pt[1]); ctx.stroke();
        }
        prev = [pt[0], pt[1], XX]; pc = cc;
      }
      ctx.lineWidth = 1;
      return;
    }
    for (var pass = 0; pass < 2; pass++) {
      for (i = 0; i < n; i++) {
        if (!m[i] || (pass === 1) !== !!this.sel[i]) continue;
        var X = ax(xs[i], lx), Y = ax(ys[i], ly);
        if (!isFinite(X) || !isFinite(Y)) continue;
        if (pass === 0) {
          if (anySel) ctx.fillStyle = C['ink-3'];
          else if (ccol) { var rgb = OBS.starRGB(ccol[i], key === '@bv' ? 'bv' : 'gaia', dark); ctx.fillStyle = 'rgb(' + rgb + ')'; }
          else if (cats) ctx.fillStyle = pal[Math.min(7, cats.cats.indexOf(cats.col[i]))];
          else ctx.fillStyle = this.pointColour || C.accent;
        } else ctx.fillStyle = C.sel;
        for (var k = 0; k < cycles; k++) {
          var x = this.px(X + k), y = this.py(Y);
          if (err && err[i] > 0) {
            if (shown > 400) ctx.globalAlpha = 0.25;   // many points: faint bars, so the points still show
            var ylo = ly ? ax(Math.max(ys[i] - err[i], ys[i] * 1e-3), ly) : Y - err[i], yhi = ax(ys[i] + err[i], ly);
            ctx.fillRect(x - 0.5, this.py(yhi), 1, this.py(ylo) - this.py(yhi));
            if (shown > 400) ctx.globalAlpha = pass === 0 ? (shown > 20000 ? 0.55 : shown > 5000 ? 0.7 : 0.9) : 1;
          }
          ctx.fillRect(x - h, y - h, s, s);
        }
      }
      if (pass === 0) ctx.globalAlpha = 1;
    }
    ctx.globalAlpha = 1;
  };
  P._bars = function (ctx, C) {
    var hg = this.histogram(), y0 = this.py(0);
    for (var k = 0; k < hg.all.length; k++) {
      var xa = this.px(hg.x0 + k * hg.w), xb = this.px(hg.x0 + (k + 1) * hg.w), l = Math.min(xa, xb), w = Math.abs(xb - xa) - 1;
      ctx.fillStyle = C['ink-3']; ctx.fillRect(l, this.py(hg.all[k]), w, y0 - this.py(hg.all[k]));
      if (hg.sel[k]) { ctx.fillStyle = C.sel; ctx.fillRect(l, this.py(hg.sel[k]), w, y0 - this.py(hg.sel[k])); }
    }
  };
  function niceTicks(a, b, n) {
    var span = b - a, step = Math.pow(10, Math.floor(Math.log10(span / n))), err = span / n / step;
    step *= err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1;
    var out = [];
    for (var v = Math.ceil(a / step) * step; v <= b + step * 1e-9; v += step) out.push(+v.toPrecision(12));
    return out;
  }
  function tickLabel(v, log, plain) {
    if (plain) return String(Math.round(v)).replace('-', '−');   // years: 1850, not 1,850
    if (!log) return OBS.fmt(Math.abs(v) < 1e-9 ? 0 : v, 4);   // a tick at 5.6e-17 is zero
    var p = Math.round(v), x = Math.pow(10, v);
    if (Math.abs(v - p) > 1e-6) return OBS.fmt(x, 2);
    if (p >= -3 && p <= 4) return OBS.fmt(x, 2);
    return OBS.fmt(x, 1);
  }
  P._axes = function (ctx, C) {
    var W = this.W, H = this.H, v = this.view, lx = this.logX(), ly = this.logY();
    ctx.font = '11px ' + getComputedStyle(document.documentElement).getPropertyValue('--mono');
    ctx.strokeStyle = C.grid; ctx.fillStyle = C['ink-2']; ctx.lineWidth = 1;
    var self = this;
    function ticks(lo, hi, log, n) {
      if (!log || hi - lo < 1.2) return niceTicks(Math.min(lo, hi), Math.max(lo, hi), n).map(function (t) { return { v: t, major: true }; });
      var out = [];
      for (var p = Math.floor(Math.min(lo, hi)); p <= Math.max(lo, hi); p++) {
        out.push({ v: p, major: true });
        if (hi - lo < 5) [2, 5].forEach(function (m) { out.push({ v: p + Math.log10(m), major: hi - lo < 2.5 }); });
      }
      return out;
    }
    var xt = ticks(v.x0, v.x1, lx, Math.max(3, Math.floor((W - PAD.l) / 90)));
    var yt = ticks(v.y0, v.y1, ly, Math.max(3, Math.floor((H - PAD.b) / 50)));
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    xt.forEach(function (t) {
      if (t.v < Math.min(v.x0, v.x1) || t.v > Math.max(v.x0, v.x1)) return;
      var x = Math.round(self.px(t.v)) + 0.5;
      ctx.globalAlpha = t.major ? 1 : 0.45;
      ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, H - PAD.b); ctx.stroke();
      if (t.major) ctx.fillText(self.fold ? OBS.fmt(t.v, 3) : tickLabel(t.v, lx, !self.fold && self.xvar().plain), x, H - PAD.b + 5);
    });
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    yt.forEach(function (t) {
      if (t.v < Math.min(v.y0, v.y1) || t.v > Math.max(v.y0, v.y1)) return;
      var y = Math.round(self.py(t.v)) + 0.5;
      ctx.globalAlpha = t.major ? 1 : 0.45;
      ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(W - PAD.r, y); ctx.stroke();
      if (t.major) ctx.fillText(tickLabel(t.v, ly, !self.hist() && self.yvar().plain), PAD.l - 6, y);
    });
    ctx.globalAlpha = 1;
    ctx.strokeStyle = C.rule; ctx.strokeRect(PAD.l + 0.5, PAD.t + 0.5, W - PAD.l - PAD.r - 1, H - PAD.t - PAD.b - 1);
    var xv = this.xvar(), yv = this.yvar();
    ctx.fillStyle = C.ink; ctx.font = '12px ' + getComputedStyle(document.documentElement).getPropertyValue('--sans');
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText(xv.label + (xv.unit ? ' [' + xv.unit + ']' : '') + (xv.rev ? '  ← larger' : ''), PAD.l + (W - PAD.l - PAD.r) / 2, H - 4);
    ctx.save(); ctx.translate(13, PAD.t + (H - PAD.t - PAD.b) / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = 'middle';
    ctx.fillText(yv.label + (yv.unit ? ' [' + yv.unit + ']' : '') + (yv.rev ? '  (up = brighter)' : ''), 0, 0);
    ctx.restore();
  };
  P._overlay = function (ctx, o, C) {
    var lx = this.logX(), ly = this.logY(), self = this, col = o.colour || C.accent;
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = o.width || 2;
    ctx.font = '12px ' + getComputedStyle(document.documentElement).getPropertyValue('--sans');
    if (o.dash) ctx.setLineDash(o.dash);
    if (o.type === 'line') {
      var a = this.toPx(o.x1, o.y1), b = this.toPx(o.x2, o.y2), dx = b[0] - a[0], dy = b[1] - a[1];
      var t0 = o.segment ? 0 : -10, t1 = o.segment ? 1 : 10;  // a fit line extends across the plot; a ruler stops at its ends
      ctx.beginPath(); ctx.moveTo(a[0] + t0 * dx, a[1] + t0 * dy); ctx.lineTo(a[0] + t1 * dx, a[1] + t1 * dy); ctx.stroke();
      ctx.setLineDash([]);
      if (o.drag !== false) [a, b].forEach(function (p) { ctx.beginPath(); ctx.arc(p[0], p[1], 6, 0, 7); ctx.fillStyle = C.panel; ctx.fill(); ctx.stroke(); });
      if (o.label) { ctx.fillStyle = col; ctx.textAlign = o.segment ? 'center' : 'left'; ctx.fillText(o.label, o.segment ? (a[0] + b[0]) / 2 : b[0] + 9, o.segment ? (a[1] + b[1]) / 2 - 8 : b[1] - 9); ctx.textAlign = 'left'; }
    } else if (o.type === 'hline' || o.type === 'vline') {
      var p = o.type === 'hline' ? this.py(ax(o.value, ly)) : this.px(ax(o.value, lx));
      ctx.beginPath();
      if (o.type === 'hline') { ctx.moveTo(PAD.l, p); ctx.lineTo(this.W - PAD.r, p); } else { ctx.moveTo(p, PAD.t); ctx.lineTo(p, this.H - PAD.b); }
      ctx.stroke(); ctx.setLineDash([]);
      ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      if (o.label) ctx.fillText(o.label, o.type === 'hline' ? PAD.l + 6 : p + 5, o.type === 'hline' ? p - 3 : PAD.t + 14);
      if (o.drag !== false) {
        var hx = o.type === 'hline' ? this.W - PAD.r - 12 : p, hy = o.type === 'hline' ? p : this.H - PAD.b - 12;
        ctx.beginPath(); ctx.arc(hx, hy, 6, 0, 7); ctx.fillStyle = C.panel; ctx.fill(); ctx.stroke();
      }
    } else if (o.type === 'point' || o.type === 'marker') {
      var q = this.toPx(o.x, o.y);
      ctx.setLineDash([]);
      if (o.type === 'point') {
        ctx.beginPath(); ctx.arc(q[0], q[1], 8, 0, 7); ctx.fillStyle = C.panel; ctx.fill(); ctx.lineWidth = 2; ctx.stroke();
        ctx.beginPath(); ctx.moveTo(q[0] - 13, q[1]); ctx.lineTo(q[0] + 13, q[1]); ctx.moveTo(q[0], q[1] - 13); ctx.lineTo(q[0], q[1] + 13); ctx.stroke();
      } else {
        ctx.beginPath(); ctx.arc(q[0], q[1], 5, 0, 7); ctx.lineWidth = 1.5; ctx.stroke();
        ctx.beginPath(); ctx.arc(q[0], q[1], 1.4, 0, 7); ctx.fill();
      }
      ctx.fillStyle = col; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      if (o.label) ctx.fillText(o.label, q[0] + 9, q[1] - 6);
    } else if (o.type === 'curve') {
      var v = this.view, N = 240, started = false;
      ctx.beginPath();
      for (var i = 0; i <= N; i++) {
        var X = v.x0 + (v.x1 - v.x0) * i / N, x = unax(X, lx);
        if (this.fold) x = X;
        var y = o.f(x);
        if (!isFinite(y) || (ly && y <= 0)) { started = false; continue; }
        var sx = this.px(X), sy = this.py(ax(y, ly));
        if (!started) { ctx.moveTo(sx, sy); started = true; } else ctx.lineTo(sx, sy);
      }
      ctx.stroke();
      if (o.label) {
        var xl = o.labelAt !== undefined ? o.labelAt : unax(v.x0 + 0.82 * (v.x1 - v.x0), lx), yl = o.f(xl);
        if (isFinite(yl)) { var lp = this.toPx(xl, yl); ctx.fillStyle = col; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText(o.label, lp[0] + 4, lp[1] - 4); }
      }
    } else if (o.type === 'span') {   // a shaded band between two x values
      var sa = this.px(ax(o.x0, lx)), sb = this.px(ax(o.x1, lx));
      ctx.globalAlpha = o.alpha || 0.13; ctx.fillRect(Math.min(sa, sb), PAD.t, Math.abs(sb - sa), this.H - PAD.t - PAD.b); ctx.globalAlpha = 1;
      if (o.label) { ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(o.label, (sa + sb) / 2, PAD.t + 4); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; }
    } else if (o.type === 'ref') {
      ctx.globalAlpha = o.alpha || 0.5;
      var s = o.size || 1.6;
      for (var j = 0; j < o.xs.length; j++) {
        var RX = ax(o.xs[j], lx) + (o.dx || 0), RY = ax(o.ys[j], ly) + (o.dy || 0);
        if (isFinite(RX) && isFinite(RY)) ctx.fillRect(this.px(RX) - s / 2, this.py(RY) - s / 2, s, s);
      }
      ctx.globalAlpha = 1;
    }
    ctx.setLineDash([]); ctx.lineWidth = 1;
  };
  P._hover = function (ctx, C) {
    var i = this.hover, xs = this.xs(), ys = this.ys();
    var X = ax(xs[i], this.logX()), Y = ax(ys[i], this.logY()), x = this.px(X), y = this.py(Y);
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x, y, 6, 0, 7); ctx.stroke();
  };

  // ------------------------------------------------------------------------------------------ overlays API
  P.add = function (o) { this.remove(o.id); this.overlays.push(o); this.draw(); return o; };
  P.remove = function (id) { this.overlays = this.overlays.filter(function (o) { return o.id !== id; }); this.draw(); };
  P.get = function (id) { for (var i = 0; i < this.overlays.length; i++) if (this.overlays[i].id === id) return this.overlays[i]; return null; };
  /** Slope and intercept of a line overlay in axis space (log10 where the axis is logarithmic). */
  P.lineFit = function (id) {
    var o = this.get(id); if (!o) return null;
    var lx = this.logX(), ly = this.logY();
    var X1 = ax(o.x1, lx), X2 = ax(o.x2, lx), Y1 = ax(o.y1, ly), Y2 = ax(o.y2, ly);
    var m = (Y2 - Y1) / (X2 - X1);
    return { slope: m, at: function (X) { return Y1 + m * (X - X1); }, intercept: Y1 - m * X1 };
  };
  /** rms vertical distance (axis space) of the shown (or selected) points from a line overlay. */
  P.lineRms = function (id, useSel) {
    var L = this.lineFit(id); if (!L) return NaN;
    var xs = this.xs(), ys = this.ys(), lx = this.logX(), ly = this.logY(), s = 0, n = 0;
    for (var i = 0; i < xs.length; i++) {
      if (!this.mask[i] || (useSel && !this.sel[i])) continue;
      var X = ax(xs[i], lx), Y = ax(ys[i], ly);
      if (isFinite(X) && isFinite(Y)) { var d = Y - L.at(X); s += d * d; n++; }
    }
    return n ? Math.sqrt(s / n) : NaN;
  };

  // ------------------------------------------------------------------------------------------ selection
  P.selected = function () { var out = []; for (var i = 0; i < this.sel.length; i++) if (this.sel[i] && this.mask[i]) out.push(i); return out; };
  P.shown = function () { var out = []; for (var i = 0; i < this.mask.length; i++) if (this.mask[i]) out.push(i); return out; };
  P.clearSel = function () { this.sel.fill(0); this._changed(); };
  P.setSel = function (idx, add) { if (!add) this.sel.fill(0); idx.forEach(function (i) { this.sel[i] = 1; }, this); this._changed(); };
  P._changed = function () { this.draw(); OBS.emit('selection', { ds: this.ds, chart: this }); };
  P._brush = function (d, add) {
    var xs = this.xs(), ys = this.ys(), lx = this.logX(), ly = this.logY();
    var xa = Math.min(d.x0, d.x1), xb = Math.max(d.x0, d.x1), ya = Math.min(d.y0, d.y1), yb = Math.max(d.y0, d.y1);
    if (!add) this.sel.fill(0);
    var cycles = this.fold ? 2 : 1;
    for (var i = 0; i < xs.length; i++) {
      if (!this.mask[i]) continue;
      var X = ax(xs[i], lx);
      for (var k = 0; k < cycles; k++) {
        var x = this.px(X + k);
        if (x < xa || x > xb) continue;
        if (!this.hist()) { var y = this.py(ax(ys[i], ly)); if (!(y >= ya && y <= yb)) continue; }
        this.sel[i] = 1;
      }
    }
    this._changed();
  };
  P.nearest = function (px, py, r) {
    if (this.hist()) return -1;
    var xs = this.xs(), ys = this.ys(), lx = this.logX(), ly = this.logY(), best = -1, bd = (r || 10) * (r || 10);
    var cycles = this.fold ? 2 : 1;
    for (var i = 0; i < xs.length; i++) {
      if (!this.mask[i]) continue;
      for (var k = 0; k < cycles; k++) {
        var dx = this.px(ax(xs[i], lx) + k) - px, dy = this.py(ax(ys[i], ly)) - py, d = dx * dx + dy * dy;
        if (d < bd) { bd = d; best = i; }
      }
    }
    return best;
  };

  // ------------------------------------------------------------------------------------------ interaction
  P._handleAt = function (x, y) {
    for (var i = this.overlays.length - 1; i >= 0; i--) {
      var o = this.overlays[i]; if (o.drag === false) continue;
      if (o.type === 'line') {
        var a = this.toPx(o.x1, o.y1), b = this.toPx(o.x2, o.y2);
        if (Math.hypot(a[0] - x, a[1] - y) < 12) return { o: o, end: 1 };
        if (Math.hypot(b[0] - x, b[1] - y) < 12) return { o: o, end: 2 };
      } else if (o.type === 'hline') {
        if (Math.abs(this.py(ax(o.value, this.logY())) - y) < 9) return { o: o };
      } else if (o.type === 'vline') {
        if (Math.abs(this.px(ax(o.value, this.logX())) - x) < 9) return { o: o };
      } else if (o.type === 'point') {
        var q = this.toPx(o.x, o.y); if (Math.hypot(q[0] - x, q[1] - y) < 14) return { o: o };
      }
    }
    return null;
  };
  P._bind = function () {
    var self = this, cv = this.cv;
    function pos(e) { var r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
    cv.addEventListener('pointerdown', function (e) {
      if (!self.ds) return;
      var p = pos(e), h = self._handleAt(p[0], p[1]);
      cv.setPointerCapture(e.pointerId);
      if (h) self.drag = { kind: 'handle', h: h };
      else if (self.tool === 'pan' || e.button === 1) self.drag = { kind: 'pan', x: p[0], y: p[1], v: Object.assign({}, self.view) };
      else self.drag = { kind: 'brush', x0: p[0], y0: p[1], x1: p[0], y1: p[1], add: e.shiftKey };
      e.preventDefault();
    });
    cv.addEventListener('pointermove', function (e) {
      if (!self.ds) return;
      var p = pos(e), d = self.drag;
      if (!d) {
        var h = self._handleAt(p[0], p[1]);
        cv.style.cursor = h ? (h.o.type === 'hline' ? 'ns-resize' : h.o.type === 'vline' ? 'ew-resize' : 'grab') : self.tool === 'pan' ? 'move' : 'crosshair';
        var i = self.nearest(p[0], p[1], 8);
        if (i !== self.hover) { self.hover = i; self.draw(); self.listeners.forEach(function (fn) { fn('hover', i); }); }
        return;
      }
      if (d.kind === 'brush') { d.x1 = p[0]; d.y1 = p[1]; self.draw(); }
      else if (d.kind === 'pan') {
        var v = d.v, fx = (v.x1 - v.x0) / (self.W - PAD.l - PAD.r), fy = (v.y1 - v.y0) / (self.H - PAD.t - PAD.b);
        var sx = self.xvar().rev ? -1 : 1, sy = self.yvar().rev ? -1 : 1;
        var ddx = (p[0] - d.x) * fx * sx, ddy = (p[1] - d.y) * fy * sy;
        self.view = { x0: v.x0 - ddx, x1: v.x1 - ddx, y0: v.y0 + ddy, y1: v.y1 + ddy };
        self.draw();
      } else if (d.kind === 'handle') {
        var o = d.h.o, val = self.dataAt(p[0], p[1]);
        if (o.type === 'line') { if (d.h.end === 1) { o.x1 = val.x; o.y1 = val.y; } else { o.x2 = val.x; o.y2 = val.y; } }
        else if (o.type === 'hline') o.value = val.y;
        else if (o.type === 'vline') o.value = self.fold ? self.ix(p[0]) : val.x;
        else if (o.type === 'point') { o.x = val.x; o.y = val.y; }
        if (o.onmove) o.onmove(o);
        self.draw(); OBS.emit('overlay', { chart: self, o: o });
      }
    });
    function end(e) {
      var d = self.drag; self.drag = null;
      if (!d) return;
      if (d.kind === 'brush') {
        if (Math.abs(d.x1 - d.x0) < 4 && Math.abs(d.y1 - d.y0) < 4) {
          var i = self.nearest(d.x0, d.y0, 10);
          if (i >= 0) OBS.emit('pick', { ds: self.ds, i: i });
          else if (!d.add) self.clearSel();
          self.draw();
        } else self._brush(d, d.add);
      } else self.draw();
    }
    cv.addEventListener('pointerup', end);
    cv.addEventListener('pointercancel', end);
    cv.addEventListener('pointerleave', function () { if (self.hover >= 0 && !self.drag) { self.hover = -1; self.draw(); self.listeners.forEach(function (fn) { fn('hover', -1); }); } });
    cv.addEventListener('wheel', function (e) {
      if (!self.ds) return;
      e.preventDefault();
      var p = pos(e), k = Math.exp(e.deltaY * 0.0015), v = self.view;
      var X = self.ix(p[0]), Y = self.iy(p[1]);
      var zy = !self.hist() && !e.shiftKey, zx = !e.altKey;
      self.view = {
        x0: zx ? X + (v.x0 - X) * k : v.x0, x1: zx ? X + (v.x1 - X) * k : v.x1,
        y0: zy ? Y + (v.y0 - Y) * k : v.y0, y1: zy ? Y + (v.y1 - Y) * k : v.y1
      };
      if (self.hist()) self._histMax();
      self.draw();
    }, { passive: false });
    cv.addEventListener('dblclick', function () { self.autoscale(); });
    if (G.ResizeObserver) new G.ResizeObserver(function () { self.draw(); }).observe(cv);
  };
})(typeof window !== 'undefined' ? window : globalThis);
