/* Observatorium — the sky: a dome you can turn and zoom (stereographic) or the whole sky at once (Hammer),
 * in equatorial or Galactic coordinates. The glow is Gaia's star counts; the stars are real catalogue stars.
 * Whatever the data lens shows is drawn here too, and a selection in one is a selection in the other. */
(function (G) {
  'use strict';
  var OBS = G.OBS, A = OBS.astro, R2D = 180 / Math.PI, D2R = Math.PI / 180;

  function Sky(canvas) {
    this.cv = canvas; this.ctx = canvas.getContext('2d');
    this.mode = 'dome'; this.frame = 'equ';
    this.lon0 = 83; this.lat0 = 5; this.fov = 110;     // looking at Orion
    this.layers = { milkyway: true, constellations: true, grid: false, labels: true };
    this.markers = [];
    this.chart = null; this.ds = null;
    this.mw = null; this.bg = null; this.con = null;
    this.drag = null; this.anim = null; this.listeners = [];
    this._mwCanvas = document.createElement('canvas');
    this._bind();
  }
  var P = Sky.prototype;
  OBS.Sky = Sky;

  P.init = function () {
    var self = this;
    return Promise.all([OBS.loadJSON('milkyway'), OBS.open('sky'), OBS.loadJSON('constellations')]).then(function (r) {
      var mw = r[0], bin = atob(mw.data), img = new Uint8Array(bin.length);
      for (var i = 0; i < bin.length; i++) img[i] = bin.charCodeAt(i);
      self.mw = { w: mw.w, h: mw.h, img: img };
      self.bg = r[1]; self.con = r[2];
      self.draw(true);
    });
  };
  P.setDataset = function (ds, chart) { this.ds = ds; this.chart = chart; this._pos = null; this.draw(); };

  // ------------------------------------------------------------------------------------- coordinates
  /** Frame coordinates (lon, lat) of an equatorial position. */
  P.toFrame = function (ra, dec) { return this.frame === 'gal' ? A.equToGal(ra, dec) : [ra, dec]; };
  P.fromFrame = function (lon, lat) { return this.frame === 'gal' ? A.galToEqu(lon, lat) : [lon, lat]; };
  P._posOf = function (ds) {
    // frame coordinates of every object in a dataset, cached per frame
    var key = ds.id + ':' + this.frame;
    var t = ds.table; t.computed.__sky = t.computed.__sky || {};
    if (t.computed.__sky[key]) return t.computed.__sky[key];
    var ra = OBS.column(ds, 'ra'), de = OBS.column(ds, 'dec'), n = ra.length, lo = new Float64Array(n), la = new Float64Array(n);
    for (var i = 0; i < n; i++) { var p = this.toFrame(ra[i], de[i]); lo[i] = p[0]; la[i] = p[1]; }
    return (t.computed.__sky[key] = { lon: lo, lat: la });
  };
  P.size = function () {
    var r = this.cv.getBoundingClientRect(), dpr = Math.min(2, G.devicePixelRatio || 1);
    var w = Math.max(200, Math.round(r.width)), h = Math.max(160, Math.round(r.height));
    if (this.cv.width !== w * dpr || this.cv.height !== h * dpr) { this.cv.width = w * dpr; this.cv.height = h * dpr; this._mwKey = null; }
    this.W = w; this.H = h; this.dpr = dpr; this.cx = w / 2; this.cy = h / 2;
    this.scale = this.mode === 'dome' ? (w / 2) / (2 * Math.tan(this.fov * D2R / 4)) : Math.min(w / (4 * Math.SQRT2), h / (2 * Math.SQRT2)) * 0.96;
  };
  /** Project frame (lon, lat) in degrees to screen; null when behind you. */
  P.project = function (lon, lat) {
    var l = lon * D2R, b = lat * D2R, l0 = this.lon0 * D2R, b0 = this.lat0 * D2R, x, y;
    if (this.mode === 'dome') {
      var cb = Math.cos(b), sb = Math.sin(b), dl = l - l0, cd = Math.cos(dl);
      var cosc = Math.sin(b0) * sb + Math.cos(b0) * cb * cd;
      if (cosc < -0.3) return null;
      var k = 2 / (1 + cosc);
      x = k * cb * Math.sin(dl); y = k * (Math.cos(b0) * sb - Math.sin(b0) * cb * cd);
    } else {
      var d = l - l0; d = Math.atan2(Math.sin(d), Math.cos(d));
      var z = Math.sqrt(1 + Math.cos(b) * Math.cos(d / 2));
      x = 2 * Math.SQRT2 * Math.cos(b) * Math.sin(d / 2) / z; y = Math.SQRT2 * Math.sin(b) / z;
    }
    return [this.cx - x * this.scale, this.cy - y * this.scale];
  };
  /** Screen -> frame (lon, lat), or null outside the map. */
  P.unproject = function (X, Y) {
    var x = (this.cx - X) / this.scale, y = (this.cy - Y) / this.scale, l0 = this.lon0 * D2R, b0 = this.lat0 * D2R, lon, lat;
    if (this.mode === 'dome') {
      var rho = Math.hypot(x, y);
      if (rho < 1e-12) return [this.lon0, this.lat0];
      var c = 2 * Math.atan(rho / 2), sc = Math.sin(c), cc = Math.cos(c);
      lat = Math.asin(cc * Math.sin(b0) + y * sc * Math.cos(b0) / rho);
      lon = l0 + Math.atan2(x * sc, rho * Math.cos(b0) * cc - y * Math.sin(b0) * sc);
    } else {
      if (x * x / 8 + y * y / 2 > 1) return null;
      var z = Math.sqrt(1 - x * x / 16 - y * y / 4);
      lon = l0 + 2 * Math.atan2(z * x, 2 * (2 * z * z - 1)); lat = Math.asin(z * y);
    }
    lon *= R2D; lon = ((lon % 360) + 360) % 360;
    return [lon, lat * R2D];
  };

  // ------------------------------------------------------------------------------------- drawing
  P.draw = function (full) {
    var self = this; this._full = this._full || full;
    if (this._raf) return;
    this._raf = G.requestAnimationFrame(function () { self._raf = 0; var f = self._full; self._full = false; self._draw(f); });
  };
  P._draw = function (full) {
    if (!this.bg) return;
    this.size();
    var ctx = this.ctx, C = OBS.colours(), dpr = this.dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, this.W, this.H);
    if (this.mode === 'allsky') {
      ctx.beginPath(); ctx.ellipse(this.cx, this.cy, 2 * Math.SQRT2 * this.scale, Math.SQRT2 * this.scale, 0, 0, 7);
      ctx.fillStyle = C.sky; ctx.fill();
    } else { ctx.fillStyle = C.sky; ctx.fillRect(0, 0, this.W, this.H); }
    if (this.layers.milkyway) this._milkyWay(ctx, C, full || !this.drag);
    if (this.layers.grid) this._grid(ctx, C);
    var fg = this.ds && this.ds.pos && this.ds.id !== 'sky';
    if (this.layers.constellations && (!fg || this.mode === 'dome')) this._constellations(ctx, C);
    this._stars(ctx, C, fg);
    if (fg) this._dataset(ctx, C);
    this._markers(ctx, C);
    if (this.drag && this.drag.kind === 'circle' && this.drag.r) {
      var d = this.drag; ctx.strokeStyle = C.sel; ctx.setLineDash([4, 3]); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(d.x0, d.y0, d.r, 0, 7); ctx.stroke(); ctx.setLineDash([]);
    }
    if (this.hoverItem) this._hoverRing(ctx, C);
    this.listeners.forEach(function (fn) { fn('draw'); });
  };
  function parseRGB(s) {
    var m = /#([0-9a-f]{6})/i.exec(s); if (m) { var n = parseInt(m[1], 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
    m = /rgba?\(([^)]+)\)/.exec(s); if (m) return m[1].split(',').slice(0, 3).map(Number);
    return [200, 210, 255];
  }
  P._milkyWay = function (ctx, C, fine) {
    var f = fine ? 3 : 6, w = Math.ceil(this.W / f), h = Math.ceil(this.H / f);
    var key = [this.mode, this.frame, this.lon0.toFixed(3), this.lat0.toFixed(3), this.fov.toFixed(3), w, h, C.dark].join();
    var mc = this._mwCanvas;
    if (this._mwKey !== key) {
      mc.width = w; mc.height = h;
      var mctx = mc.getContext('2d'), im = mctx.createImageData(w, h), px = im.data, M = this.mw, rgb = parseRGB(C['sky-glow']);
      var gamma = C.dark ? 1.6 : 1.3, amax = C.dark ? 0.85 : 0.55;
      for (var j = 0; j < h; j++) for (var i = 0; i < w; i++) {
        var p = this.unproject((i + 0.5) * f, (j + 0.5) * f);
        if (!p) continue;
        var e = this.frame === 'gal' ? A.galToEqu(p[0], p[1]) : p;
        var u = e[0] / 360 * M.w - 0.5, v = (90 - e[1]) / 180 * M.h - 0.5;
        var u0 = Math.floor(u), v0 = Math.max(0, Math.min(M.h - 2, Math.floor(v))), fu = u - u0, fv = Math.max(0, Math.min(1, v - v0));
        var a = (u0 % M.w + M.w) % M.w, b = (a + 1) % M.w;
        var val = (M.img[v0 * M.w + a] * (1 - fu) + M.img[v0 * M.w + b] * fu) * (1 - fv) + (M.img[(v0 + 1) * M.w + a] * (1 - fu) + M.img[(v0 + 1) * M.w + b] * fu) * fv;
        var al = Math.pow(val / 255, gamma) * amax, k = 4 * (j * w + i);
        px[k] = rgb[0]; px[k + 1] = rgb[1]; px[k + 2] = rgb[2]; px[k + 3] = al * 255;
      }
      mctx.putImageData(im, 0, 0);
      this._mwKey = key;
    }
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(mc, 0, 0, w * f, h * f);
  };
  P._grid = function (ctx, C) {
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1; ctx.globalAlpha = 0.8;
    var self = this;
    function poly(pts) {
      ctx.beginPath(); var on = false, last = null;
      pts.forEach(function (p) {
        var q = self.project(p[0], p[1]);
        if (!q || (last && Math.hypot(q[0] - last[0], q[1] - last[1]) > self.W / 3)) { on = false; last = q; return; }
        if (!on) { ctx.moveTo(q[0], q[1]); on = true; } else ctx.lineTo(q[0], q[1]);
        last = q;
      });
      ctx.stroke();
    }
    for (var lon = 0; lon < 360; lon += 30) { var pts = []; for (var lat = -88; lat <= 88; lat += 2) pts.push([lon, lat]); poly(pts); }
    for (lat = -60; lat <= 60; lat += 30) { pts = []; for (lon = 0; lon <= 360; lon += 2) pts.push([lon, lat]); poly(pts); }
    ctx.globalAlpha = 1;
    ctx.fillStyle = C['ink-3']; ctx.font = '10px ' + getComputedStyle(document.documentElement).getPropertyValue('--mono');
    for (lon = 0; lon < 360; lon += 30) {
      var q = this.project(lon, 0);
      if (q) ctx.fillText(this.frame === 'gal' ? 'l ' + lon + '°' : (lon / 15) + 'h', q[0] + 3, q[1] - 3);
    }
  };
  P._constellations = function (ctx, C) {
    var c = this.bg.table.cols, ra = c.ra, de = c.dec, self = this, pos = this._posOf(this.bg);
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1; ctx.globalAlpha = 0.9;
    ctx.beginPath();
    this.con.lines.forEach(function (s) {
      var a = self.project(pos.lon[s[0]], pos.lat[s[0]]), b = self.project(pos.lon[s[1]], pos.lat[s[1]]);
      if (a && b && Math.hypot(a[0] - b[0], a[1] - b[1]) < self.W / 2) { ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); }
    });
    ctx.stroke(); ctx.globalAlpha = 1;
    if (this.layers.labels && this.mode === 'dome' && this.fov > 25 && this.fov < 160) {
      ctx.fillStyle = C['ink-3']; ctx.font = '11px ' + getComputedStyle(document.documentElement).getPropertyValue('--sans'); ctx.textAlign = 'center';
      this.con.labels.forEach(function (l) {
        var q = self.project(pos.lon[l[1]], pos.lat[l[1]]);
        if (q && q[0] > 0 && q[0] < self.W && q[1] > 0 && q[1] < self.H) ctx.fillText(l[0].toUpperCase(), q[0], q[1] + 16);
      });
      ctx.textAlign = 'left';
    }
  };
  /** Naked-eye stars: the backdrop, or the dataset itself when the sky is the dataset. */
  P._stars = function (ctx, C, faint) {
    var t = this.bg.table, c = t.cols, pos = this._posOf(this.bg), n = t.n;
    var zoom = this.mode === 'dome' ? Math.max(1, Math.pow(60 / this.fov, 0.35)) : 0.8;
    var isDs = this.ds && this.ds.id === 'sky' && this.chart, sel = isDs ? this.chart.sel : null, mask = isDs ? this.chart.mask : null;
    var anySel = false; if (sel) for (var k = 0; k < n; k++) if (sel[k]) { anySel = true; break; }
    var named = [];
    for (var i = 0; i < n; i++) {
      var q = this.project(pos.lon[i], pos.lat[i]); if (!q || q[0] < -5 || q[0] > this.W + 5 || q[1] < -5 || q[1] > this.H + 5) continue;
      var r = Math.max(0.45, (6.8 - c.mag[i]) * 0.42) * zoom;
      if (faint) r *= 0.6;
      var rgb = OBS.starRGB(c.ci[i], 'bv', C.dark);
      ctx.globalAlpha = faint ? 0.45 : (anySel && !sel[i] ? 0.35 : 1);
      if (!C.dark) { ctx.fillStyle = 'rgb(' + rgb.map(function (x) { return Math.round(x * 0.55); }) + ')'; }
      else ctx.fillStyle = 'rgb(' + rgb + ')';
      if (sel && sel[i] && mask[i]) ctx.fillStyle = C.sel;
      ctx.beginPath(); ctx.arc(q[0], q[1], r, 0, 7); ctx.fill();
      if (c.name[i] && !faint) named.push([c.mag[i], c.name[i], q, r]);
    }
    ctx.globalAlpha = 1;
    if (!this.layers.labels || faint) return;
    var lim = this.mode === 'dome' ? (this.fov > 90 ? 1.0 : this.fov > 40 ? 2.2 : this.fov > 15 ? 3.5 : 5.5) : 0.6;
    ctx.fillStyle = C['ink-2']; ctx.font = '11px ' + getComputedStyle(document.documentElement).getPropertyValue('--sans');
    named.forEach(function (s) { if (s[0] < lim) ctx.fillText(s[1], s[2][0] + s[3] + 3, s[2][1] + 3); });
  };
  /** The open dataset's objects, coloured as the lens colours them; selected ones highlighted. */
  P._dataset = function (ctx, C) {
    var ds = this.ds, ch = this.chart, pos = this._posOf(ds), n = ds.table.n, m = ch.mask, sel = ch.sel;
    var anySel = false; for (var k = 0; k < n; k++) if (sel[k] && m[k]) { anySel = true; break; }
    var shown = 0; for (k = 0; k < n; k++) shown += m[k];
    var s = shown > 20000 ? 1.3 : shown > 3000 ? 1.8 : 2.6;
    if (this.mode === 'dome' && this.fov < 20) s *= 1.4;
    for (var pass = 0; pass < 2; pass++) {
      for (var i = 0; i < n; i++) {
        if (!m[i] || (pass === 1) !== !!sel[i]) continue;
        var q = this.project(pos.lon[i], pos.lat[i]);
        if (!q || q[0] < -3 || q[0] > this.W + 3 || q[1] < -3 || q[1] > this.H + 3) continue;
        ctx.globalAlpha = pass ? 1 : anySel ? 0.35 : 0.85;
        ctx.fillStyle = pass ? C.sel : anySel ? C['ink-3'] : ch.colourOf(i, C);
        ctx.fillRect(q[0] - s / 2, q[1] - s / 2, s * (pass ? 1.6 : 1), s * (pass ? 1.6 : 1));
      }
    }
    ctx.globalAlpha = 1;
  };
  P._markers = function (ctx, C) {
    var self = this;
    ctx.font = '12px ' + getComputedStyle(document.documentElement).getPropertyValue('--sans');
    this.markers.forEach(function (mk) {
      var f = self.toFrame(mk.ra, mk.dec), q = self.project(f[0], f[1]);
      if (!q) return;
      var r = mk.r ? mk.r * self.scale * D2R * (self.mode === 'dome' ? 1 : 0.9) : 9;
      ctx.strokeStyle = mk.colour || C.accent; ctx.fillStyle = mk.colour || C.accent; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(q[0], q[1], Math.max(6, r), 0, 7); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(q[0] - Math.max(6, r) - 6, q[1]); ctx.lineTo(q[0] - Math.max(6, r) + 2, q[1]); ctx.stroke();
      if (mk.label) ctx.fillText(mk.label, q[0] + Math.max(6, r) + 5, q[1] - 5);
    });
  };
  P._hoverRing = function (ctx, C) {
    var h = this.hoverItem, q = this.project(h.lon, h.lat); if (!q) return;
    ctx.strokeStyle = C.ink; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(q[0], q[1], 7, 0, 7); ctx.stroke();
  };

  // ------------------------------------------------------------------------------------- moving around
  P.goTo = function (ra, dec, fov, opts) {
    opts = opts || {};
    if (opts.mode) this.mode = opts.mode;
    if (opts.frame) { this.frame = opts.frame; this._mwKey = null; }
    var f = this.toFrame(ra, dec), self = this;
    var from = { lon: this.lon0, lat: this.lat0, fov: this.fov }, to = { lon: f[0], lat: f[1], fov: fov || this.fov };
    var dl = ((to.lon - from.lon + 540) % 360) - 180;
    if (OBS.reducedMotion() || opts.instant) { this.lon0 = to.lon; this.lat0 = to.lat; this.fov = to.fov; this.draw(true); return; }
    var t0 = performance.now(), dur = 1400;
    if (this.anim) G.cancelAnimationFrame(this.anim);
    (function step(now) {
      var u = Math.min(1, (now - t0) / dur), e = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
      // zoom out a little in the middle of long trips, like swinging a telescope
      var hop = Math.min(1, (Math.abs(dl) + Math.abs(to.lat - from.lat)) / 90), wide = Math.max(from.fov, to.fov) * (1 + 0.6 * hop);
      var lf = Math.log(from.fov) + (Math.log(to.fov) - Math.log(from.fov)) * e + Math.sin(Math.PI * e) * (Math.log(wide) - Math.max(Math.log(from.fov), Math.log(to.fov))) * hop;
      self.lon0 = (from.lon + dl * e + 360) % 360; self.lat0 = from.lat + (to.lat - from.lat) * e; self.fov = Math.exp(lf);
      self._draw(u >= 1);
      if (u < 1) self.anim = G.requestAnimationFrame(step); else self.anim = null;
    })(t0);
  };
  P.setMode = function (m) { this.mode = m; if (m === 'allsky') this.lat0 = 0; this._mwKey = null; this.draw(true); };
  P.setFrame = function (f) {
    if (f === this.frame) return;
    var e = this.fromFrame(this.lon0, this.lat0); this.frame = f; var n = this.toFrame(e[0], e[1]);
    this.lon0 = n[0]; this.lat0 = this.mode === 'allsky' ? 0 : n[1]; this._mwKey = null; this.draw(true);
  };
  P.centre = function () { return this.fromFrame(this.lon0, this.lat0); };

  // ------------------------------------------------------------------------------------- interaction
  P._nearest = function (X, Y) {
    var best = null, bd = 100, self = this;
    function scan(ds, mask) {
      var pos = self._posOf(ds), n = ds.table.n;
      for (var i = 0; i < n; i++) {
        if (mask && !mask[i]) continue;
        var q = self.project(pos.lon[i], pos.lat[i]); if (!q) continue;
        var d = (q[0] - X) * (q[0] - X) + (q[1] - Y) * (q[1] - Y);
        if (d < bd) { bd = d; best = { ds: ds, i: i, lon: pos.lon[i], lat: pos.lat[i] }; }
      }
    }
    if (this.ds && this.ds.pos && this.ds.id !== 'sky' && this.chart) scan(this.ds, this.chart.mask);
    if (!best) scan(this.bg, null);
    return best;
  };
  P._circleSelect = function (d) {
    if (!this.ds || !this.ds.pos || !this.chart) return;
    var c = this.unproject(d.x0, d.y0), e = this.unproject(d.x0 + d.r, d.y0);
    if (!c || !e) return;
    var ce = this.fromFrame(c[0], c[1]), ee = this.fromFrame(e[0], e[1]), rad = A.sep(ce[0], ce[1], ee[0], ee[1]);
    var ra = OBS.column(this.ds, 'ra'), de = OBS.column(this.ds, 'dec'), idx = [];
    for (var i = 0; i < ra.length; i++) if (this.chart.mask[i] && A.sep(ra[i], de[i], ce[0], ce[1]) < rad) idx.push(i);
    this.chart.setSel(idx, d.add);
  };
  P._bind = function () {
    var self = this, cv = this.cv;
    function pos(e) { var r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
    cv.addEventListener('pointerdown', function (e) {
      var p = pos(e); cv.setPointerCapture(e.pointerId);
      if (self.anim) { G.cancelAnimationFrame(self.anim); self.anim = null; }
      self.drag = (e.shiftKey || self.tool === 'circle') ? { kind: 'circle', x0: p[0], y0: p[1], r: 0, add: e.shiftKey && self.tool === 'circle' }
        : { kind: 'turn', x: p[0], y: p[1], lon: self.lon0, lat: self.lat0, moved: false };
      e.preventDefault();
    });
    cv.addEventListener('pointermove', function (e) {
      var p = pos(e), d = self.drag;
      if (!d) {
        var h = self._nearest(p[0], p[1]);
        if ((h && h.i) !== (self.hoverItem && self.hoverItem.i) || (h && h.ds) !== (self.hoverItem && self.hoverItem.ds)) {
          self.hoverItem = h; self.draw(); self.listeners.forEach(function (fn) { fn('hover', h); });
        }
        return;
      }
      if (d.kind === 'circle') { d.r = Math.hypot(p[0] - d.x0, p[1] - d.y0); self.draw(); return; }
      var dx = p[0] - d.x, dy = p[1] - d.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true;
      if (self.mode === 'dome') {
        self.lat0 = Math.max(-89.9, Math.min(89.9, d.lat + dy / self.scale * R2D));
        self.lon0 = ((d.lon + dx / self.scale * R2D / Math.max(0.15, Math.cos(self.lat0 * D2R))) % 360 + 360) % 360;
      } else self.lon0 = ((d.lon + dx / self.scale * R2D * 0.7) % 360 + 360) % 360;
      self.draw();
    });
    function end(e) {
      var d = self.drag; self.drag = null; if (!d) return;
      if (d.kind === 'circle') { if (d.r > 4) self._circleSelect(d); }
      else if (!d.moved) {
        var p = pos(e), h = self._nearest(p[0], p[1]);
        if (h) OBS.emit('pick', { ds: h.ds, i: h.i, from: 'sky' });
      }
      self.draw(true);
    }
    cv.addEventListener('pointerup', end);
    cv.addEventListener('pointercancel', end);
    cv.addEventListener('pointerleave', function () { if (self.hoverItem) { self.hoverItem = null; self.draw(); self.listeners.forEach(function (fn) { fn('hover', null); }); } });
    cv.addEventListener('wheel', function (e) {
      e.preventDefault();
      if (self.mode === 'allsky') {
        if (e.deltaY < 0) { var p = pos(e), q = self.unproject(p[0], p[1]); if (q) { self.mode = 'dome'; self.lon0 = q[0]; self.lat0 = q[1]; self.fov = 140; self._mwKey = null; OBS.emit('skymode'); } }
      } else {
        self.fov = Math.max(0.2, Math.min(200, self.fov * Math.exp(e.deltaY * 0.0015)));
        if (self.fov >= 199 && e.deltaY > 0) { self.mode = 'allsky'; self.lat0 = 0; self._mwKey = null; OBS.emit('skymode'); }
      }
      self.draw();
      clearTimeout(self._wt); self._wt = setTimeout(function () { self.draw(true); }, 150);
    }, { passive: false });
    if (G.ResizeObserver) new G.ResizeObserver(function () { self.draw(true); }).observe(cv);
  };
})(typeof window !== 'undefined' ? window : globalThis);
