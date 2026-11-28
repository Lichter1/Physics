/* Observatorium — the sky: a dome you can turn and zoom (stereographic) or the whole sky at once (Hammer),
 * in equatorial, Galactic or horizon coordinates (the sky over the Wise Observatory at a chosen moment). Two looks:
 *   drawn  catalogue stars as glowing points over Gaia's star-count glow
 *   photo  real photographs: ESO's all-sky Milky Way panorama, with Digitized Sky Survey and Legacy Survey
 *          close-ups blended in at full resolution wherever they exist; or the whole sky at another wavelength
 *          (radio, 21 cm, microwave, far infrared, H-alpha, X-ray, gamma rays)
 * Whatever the data lens shows is drawn here too, and a selection in one is a selection in the other. */
(function (G) {
  'use strict';
  var OBS = G.OBS, A = OBS.astro, R2D = 180 / Math.PI, D2R = Math.PI / 180;
  // ICRS -> Galactic (same matrix as astro.js)
  var RG = [[-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
    [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
    [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669]];

  function Sky(canvas) {
    this.cv = canvas; this.ctx = canvas.getContext('2d');
    this.mode = 'dome'; this.frame = 'equ';
    this.lon0 = 83; this.lat0 = 5; this.fov = 110;     // looking at Orion
    this.layers = { milkyway: true, constellations: true, grid: false, labels: true, photo: false };
    this.band = 'visible';                               // which panorama the photo look shows
    this.obs = { lat: A.WISE.lat, lon: A.WISE.lon, jd: A.jd(Date.now()) };   // for the horizon frame
    this.markers = [];
    this.chart = null; this.ds = null;
    this.mw = null; this.bg = null; this.con = null; this.images = null;
    this.drag = null; this.anim = null; this.listeners = [];
    this._mwCanvas = document.createElement('canvas');
    this._pix = {};           // decoded photographs: id -> {w, h, data}
    this._loading = {};
    this._bind();
  }
  var P = Sky.prototype;
  OBS.Sky = Sky;

  P.init = function () {
    var self = this;
    OBS.loadJSON('images').then(function (m) { self.images = m; OBS.emit('skyimages', m); }, function () { self.images = null; });
    return Promise.all([OBS.loadJSON('milkyway'), OBS.open('sky'), OBS.loadJSON('constellations')]).then(function (r) {
      var mw = r[0], bin = atob(mw.data), img = new Uint8Array(bin.length);
      for (var i = 0; i < bin.length; i++) img[i] = bin.charCodeAt(i);
      self.mw = { w: mw.w, h: mw.h, img: img };
      self.bg = r[1]; self.con = r[2];
      self.draw(true);
    });
  };
  P.setDataset = function (ds, chart) { this.ds = ds; this.chart = chart; this.draw(); };

  // ------------------------------------------------------------------------------------- coordinates
  // the horizon frame: lon = 360 - azimuth (so east is to the left, as on every other sky map), lat = altitude
  P.lst = function () { return A.lst(this.obs.jd, this.obs.lon); };
  P.toFrame = function (ra, dec) {
    if (this.frame === 'gal') return A.equToGal(ra, dec);
    if (this.frame === 'hor') { var h = A.equToHor(ra, dec, this.lst(), this.obs.lat); return [(360 - h[0]) % 360, h[1]]; }
    return [ra, dec];
  };
  P.fromFrame = function (lon, lat) {
    if (this.frame === 'gal') return A.galToEqu(lon, lat);
    if (this.frame === 'hor') return A.horToEqu((360 - lon) % 360, lat, this.lst(), this.obs.lat);
    return [lon, lat];
  };
  /** 3x3 matrix taking a unit vector in the view's frame to the equatorial frame (columns = frame axes). */
  P._toEqu = function () {
    var self = this;
    return [0, 1, 2].map(function (k) { var p = [[0, 0], [90, 0], [0, 90]][k], e = self.fromFrame(p[0], p[1]); return A.vec(e[0], e[1]); });
  };
  /** Set the moment shown in the horizon frame (JavaScript time in ms, UTC). */
  P.setTime = function (ms) { this.obs.jd = A.jd(ms); this._mwKey = null; this.draw(true); };
  P._posOf = function (ds) {
    var hor = this.frame === 'hor', key = ds.id + ':' + this.frame + (hor ? ':' + this.obs.jd.toFixed(6) : ''), t = ds.table;
    t.computed.__sky = t.computed.__sky || {};
    var slot = hor ? ds.id + ':hor' : key;                 // the horizon frame keeps only its latest moment
    if (t.computed.__sky[slot] && t.computed.__sky[slot].key === key) return t.computed.__sky[slot];
    var ra = OBS.column(ds, 'ra'), de = OBS.column(ds, 'dec'), n = ra.length, lo = new Float64Array(n), la = new Float64Array(n);
    for (var i = 0; i < n; i++) { var p = this.toFrame(ra[i], de[i]); lo[i] = p[0]; la[i] = p[1]; }
    return (t.computed.__sky[slot] = { lon: lo, lat: la, key: key });
  };
  P.size = function () {
    var r = this.cv.getBoundingClientRect(), dpr = Math.min(2, G.devicePixelRatio || 1);
    var w = Math.max(200, Math.round(r.width)), h = Math.max(160, Math.round(r.height));
    if (this.cv.width !== w * dpr || this.cv.height !== h * dpr) { this.cv.width = w * dpr; this.cv.height = h * dpr; this._mwKey = null; }
    this.W = w; this.H = h; this.dpr = dpr; this.cx = w / 2; this.cy = h / 2;
    this.scale = this.mode === 'dome' ? (w / 2) / (2 * Math.tan(this.fov * D2R / 4)) : Math.min(w / (4 * Math.SQRT2), h / (2 * Math.SQRT2)) * 0.96;
  };
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
  /** Unit vectors (in the view's frame) of the centre, east and north there: a fast inverse projection. */
  P._basis = function () {
    var l = this.lon0 * D2R, b = this.lat0 * D2R, cl = Math.cos(l), sl = Math.sin(l), cb = Math.cos(b), sb = Math.sin(b);
    return { c: [cb * cl, cb * sl, sb], e: [-sl, cl, 0], n: [-sb * cl, -sb * sl, cb] };
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
    var ctx = this.ctx, C = OBS.colours(), dpr = this.dpr, photo = this.layers.photo && this.images;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, this.W, this.H);
    var sky = photo ? '#000' : C.sky;
    if (this.mode === 'allsky') {
      ctx.beginPath(); ctx.ellipse(this.cx, this.cy, 2 * Math.SQRT2 * this.scale, Math.SQRT2 * this.scale, 0, 0, 7);
      ctx.fillStyle = sky; ctx.fill();
    } else { ctx.fillStyle = sky; ctx.fillRect(0, 0, this.W, this.H); }
    if (photo) this._photo(ctx, full || !this.drag);
    else if (this.layers.milkyway) this._milkyWay(ctx, C, full || !this.drag);
    if (this.layers.grid) this._grid(ctx, C, photo);
    var fg = this.ds && this.ds.pos && this.ds.id !== 'sky';
    if (this.layers.constellations && (!fg || this.mode === 'dome')) this._constellations(ctx, C, photo);
    if (photo) { if (this.band === 'visible') this._starLabels(ctx, C, true); } else this._stars(ctx, C, fg);
    if (fg) this._dataset(ctx, C, photo);
    else if (photo && this.ds && this.ds.id === 'sky') this._skySelection(ctx, C);
    if (this.frame === 'hor') this._horizon(ctx, C);
    this._markers(ctx, C, photo);
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
    var key = ['mw', this.mode, this.frame, this.frame === 'hor' ? this.obs.jd.toFixed(5) : '', this.lon0.toFixed(3), this.lat0.toFixed(3), this.fov.toFixed(3), w, h, C.dark].join();
    var mc = this._mwCanvas;
    if (this._mwKey !== key) {
      mc.width = w; mc.height = h;
      var mctx = mc.getContext('2d'), im = mctx.createImageData(w, h), px = im.data, M = this.mw, rgb = parseRGB(C['sky-glow']);
      var gamma = C.dark ? 1.6 : 1.3, amax = C.dark ? 0.85 : 0.55;
      for (var j = 0; j < h; j++) for (var i = 0; i < w; i++) {
        var p = this.unproject((i + 0.5) * f, (j + 0.5) * f);
        if (!p) continue;
        var e = this.frame === 'equ' ? p : this.fromFrame(p[0], p[1]);
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

  // ------------------------------------------------------------------------------------- photographs
  /** Decode a photograph into pixels (once; at most a few close-ups are kept in memory). */
  P._pixels = function (key, file) {
    var self = this;
    if (this._pix[key]) { this._pix[key].used = performance.now(); return this._pix[key]; }
    if (this._loading[key]) return null;
    this._loading[key] = true;
    var img = new Image();
    img.onload = function () {
      var c = document.createElement('canvas'); c.width = img.naturalWidth; c.height = img.naturalHeight;
      var x = c.getContext('2d'); x.drawImage(img, 0, 0);
      var d;
      try { d = x.getImageData(0, 0, c.width, c.height).data; } catch (e) { self._loading[key] = false; return; }
      self._pix[key] = { w: c.width, h: c.height, data: d, used: performance.now() };
      var keys = Object.keys(self._pix).filter(function (k) { return k !== 'allsky' && k.indexOf('band:') !== 0; });
      if (keys.length > 8) { keys.sort(function (a, b) { return self._pix[a].used - self._pix[b].used; }); delete self._pix[keys[0]]; }
      self._loading[key] = false; self._mwKey = null; self.draw(true);
    };
    img.onerror = function () { self._loading[key] = false; };
    img.src = (G.OBS_INLINE && G.OBS_INLINE[file]) || ('data/' + file);
    return null;
  };
  /** The close-ups worth drawing at this view: near the centre, and not far smaller than a pixel. */
  P._visibleCutouts = function () {
    if (!this.images) return [];
    var c = this.fromFrame(this.lon0, this.lat0), view = this.mode === 'dome' ? this.fov : 360, self = this;
    return this.images.cutouts.filter(function (k) {
      return k.fov > view / 400 && A.sep(c[0], c[1], k.ra, k.dec) < view * 0.75 + k.fov;
    }).sort(function (a, b) { return b.fov - a.fov; }).slice(0, 6).map(function (k) {   // never more than the pixel cache holds
      var ra = k.ra * D2R, de = k.dec * D2R;
      return { k: k, c: [Math.cos(de) * Math.cos(ra), Math.cos(de) * Math.sin(ra), Math.sin(de)], e: [-Math.sin(ra), Math.cos(ra), 0], n: [-Math.sin(de) * Math.cos(ra), -Math.sin(de) * Math.sin(ra), Math.cos(de)], s: k.fov * D2R / k.w, px: self._pixels(k.id, k.file) };
    }).filter(function (q) { return q.px; });
  };
  /** The panorama for the current wavelength (the visible-light photograph unless another band is chosen). */
  P.bandInfo = function () {
    var b = this.images && this.band !== 'visible' && (this.images.bands || []).filter(function (x) { return x.id === this.band; }, this)[0];
    return b || (this.images ? this.images.allsky : null);
  };
  P.setBand = function (id) { this.band = id || 'visible'; if (this.band !== 'visible') this.layers.photo = true; this._mwKey = null; OBS.emit('skymode'); this.draw(true); };
  P._photo = function (ctx, fine) {
    var pano = this.bandInfo(), vis = this.band === 'visible';
    var all = this._pixels(vis ? 'allsky' : 'band:' + this.band, pano.file);
    var f = fine ? 1.25 : 4, w = Math.ceil(this.W / f), h = Math.ceil(this.H / f);
    var cuts = vis ? this._visibleCutouts() : [];
    var key = ['ph', this.band, this.mode, this.frame, this.frame === 'hor' ? this.obs.jd.toFixed(5) : '', this.lon0.toFixed(4), this.lat0.toFixed(4), this.fov.toFixed(4), w, h, !!all, cuts.map(function (q) { return q.k.id; }).join('+')].join();
    var mc = this._mwCanvas;
    if (this._mwKey !== key) {
      mc.width = w; mc.height = h;
      var mctx = mc.getContext('2d'), im = mctx.createImageData(w, h), px = im.data;
      var B = this._basis(), M = this._toEqu(), dome = this.mode === 'dome';
      // the panorama has 3.5' pixels (the other wavelengths 15'): when zoomed far past that, dim it rather than show big blurry blocks
      var fade = dome ? Math.max(0.35, Math.min(1, this.fov / (vis ? 12 : 40))) : 1;
      var AW = all ? all.w : 0, AH = all ? all.h : 0, AD = all ? all.data : null;
      for (var j = 0; j < h; j++) {
        for (var i = 0; i < w; i++) {
          var X = (i + 0.5) * f, Y = (j + 0.5) * f, v0, v1, v2;
          if (dome) {
            var x = (this.cx - X) / this.scale, y = (this.cy - Y) / this.scale, r2 = x * x + y * y, q = 4 + r2;
            var dc = (4 - r2) / q, de = 4 * x / q, dn = 4 * y / q;
            v0 = dc * B.c[0] + de * B.e[0] + dn * B.n[0]; v1 = dc * B.c[1] + de * B.e[1] + dn * B.n[1]; v2 = dc * B.c[2] + de * B.e[2] + dn * B.n[2];
          } else {
            var p = this.unproject(X, Y); if (!p) continue;
            var cl = Math.cos(p[1] * D2R); v0 = cl * Math.cos(p[0] * D2R); v1 = cl * Math.sin(p[0] * D2R); v2 = Math.sin(p[1] * D2R);
          }
          // e = equatorial vector (the frame's axes are the columns of M), g = Galactic vector
          var e0 = M[0][0] * v0 + M[1][0] * v1 + M[2][0] * v2, e1 = M[0][1] * v0 + M[1][1] * v1 + M[2][1] * v2, e2 = M[0][2] * v0 + M[1][2] * v1 + M[2][2] * v2;
          var g0 = RG[0][0] * e0 + RG[0][1] * e1 + RG[0][2] * e2, g1 = RG[1][0] * e0 + RG[1][1] * e1 + RG[1][2] * e2, g2 = RG[2][0] * e0 + RG[2][1] * e1 + RG[2][2] * e2;
          var r = 0, gg = 0, bb = 0, wt = 0, k = 4 * (j * w + i);
          for (var c = 0; c < cuts.length && wt < 1; c++) {
            var Q = cuts[c], cosc = e0 * Q.c[0] + e1 * Q.c[1] + e2 * Q.c[2];
            if (cosc < 0.5) continue;
            var xi = (e0 * Q.e[0] + e1 * Q.e[1] + e2 * Q.e[2]) / cosc, et = (e0 * Q.n[0] + e1 * Q.n[1] + e2 * Q.n[2]) / cosc;
            var cxp = Q.px.w / 2 - xi / Q.s, cyp = Q.px.h / 2 - et / Q.s;
            if (cxp < 0 || cyp < 0 || cxp >= Q.px.w - 1 || cyp >= Q.px.h - 1) continue;
            // feather the edges so a close-up fades into the panorama
            var edge = Math.min(cxp, cyp, Q.px.w - 1 - cxp, Q.px.h - 1 - cyp) / (0.06 * Q.px.w), a = Math.min(1, edge) * (1 - wt);
            var o = 4 * ((cyp | 0) * Q.px.w + (cxp | 0)), d = Q.px.data;
            r += a * d[o]; gg += a * d[o + 1]; bb += a * d[o + 2]; wt += a;
          }
          if (wt < 1 && AD) {
            var l = Math.atan2(g1, g0) * R2D, bgl = Math.asin(Math.max(-1, Math.min(1, g2))) * R2D;
            var ax = (0.5 - l / 360) * AW - 0.5, ay = Math.max(0, Math.min(AH - 1.001, (90 - bgl) / 180 * AH - 0.5));
            ax = ((ax % AW) + AW) % AW;
            var ix = ax | 0, iy = ay | 0, fx = ax - ix, fy = ay - iy, ix2 = (ix + 1) % AW;
            var o00 = 4 * (iy * AW + ix), o01 = 4 * (iy * AW + ix2), o10 = o00 + 4 * AW, o11 = o01 + 4 * AW, rest = (1 - wt) * fade;
            for (var ch = 0; ch < 3; ch++) {
              var vv = (AD[o00 + ch] * (1 - fx) + AD[o01 + ch] * fx) * (1 - fy) + (AD[o10 + ch] * (1 - fx) + AD[o11 + ch] * fx) * fy;
              if (ch === 0) r += rest * vv; else if (ch === 1) gg += rest * vv; else bb += rest * vv;
            }
          }
          px[k] = r; px[k + 1] = gg; px[k + 2] = bb; px[k + 3] = 255;
        }
      }
      mctx.putImageData(im, 0, 0);
      this._mwKey = key;
    }
    ctx.save();
    if (this.mode === 'allsky') { ctx.beginPath(); ctx.ellipse(this.cx, this.cy, 2 * Math.SQRT2 * this.scale, Math.SQRT2 * this.scale, 0, 0, 7); ctx.clip(); }
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(mc, 0, 0, w * f, h * f);
    ctx.restore();
  };
  /** Which photographs are on screen, for the credit line. */
  P.credits = function () {
    if (!this.layers.photo || !this.images) return '';
    if (this.band !== 'visible') { var b = this.bandInfo(); return b.label + ': ' + b.credit + '.'; }
    var cuts = this._visibleCutouts().map(function (q) { return q.k; });
    var out = cuts.length ? cuts.map(function (k) { return k.label; }).join(', ') + ': ' + cuts[0].credit + '. ' : '';
    return out + 'All-sky: ' + this.images.allsky.credit + '.';
  };

  // ------------------------------------------------------------------------------------- overlays
  P._grid = function (ctx, C, photo) {
    ctx.strokeStyle = photo ? 'rgba(170,190,255,0.22)' : C.grid; ctx.lineWidth = 1; ctx.globalAlpha = 0.8;
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
    ctx.fillStyle = photo ? 'rgba(200,210,240,0.7)' : C['ink-3']; ctx.font = '10px ' + getComputedStyle(document.documentElement).getPropertyValue('--mono');
    for (lon = 0; lon < 360; lon += 30) {
      var q = this.project(lon, 0);
      if (q) ctx.fillText(this.frame === 'gal' ? 'l ' + lon + '°' : this.frame === 'hor' ? 'az ' + ((360 - lon) % 360) + '°' : (lon / 15) + 'h', q[0] + 3, q[1] - 3);
    }
  };
  P._constellations = function (ctx, C, photo) {
    var self = this, pos = this._posOf(this.bg);
    ctx.strokeStyle = photo ? 'rgba(160,185,255,0.28)' : C.grid; ctx.lineWidth = 1; ctx.globalAlpha = 0.9;
    ctx.beginPath();
    this.con.lines.forEach(function (s) {
      var a = self.project(pos.lon[s[0]], pos.lat[s[0]]), b = self.project(pos.lon[s[1]], pos.lat[s[1]]);
      if (a && b && Math.hypot(a[0] - b[0], a[1] - b[1]) < self.W / 2) { ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); }
    });
    ctx.stroke(); ctx.globalAlpha = 1;
    if (this.layers.labels && this.mode === 'dome' && this.fov > 25 && this.fov < 140) {
      ctx.fillStyle = photo ? 'rgba(190,205,240,0.55)' : C['ink-3']; ctx.font = '10.5px ' + getComputedStyle(document.documentElement).getPropertyValue('--sans'); ctx.textAlign = 'center';
      this.con.labels.forEach(function (l) {
        var q = self.project(pos.lon[l[1]], pos.lat[l[1]]);
        if (q && q[0] > 0 && q[0] < self.W && q[1] > 0 && q[1] < self.H) ctx.fillText(l[0].toUpperCase(), q[0], q[1] + 16);
      });
      ctx.textAlign = 'left';
    }
  };
  // Soft star images: a white-hot core, a coloured halo, and spikes for the brightest (as a telescope draws them).
  var SPRITES = {};
  function sprite(rgb, dark, spikes) {
    var key = rgb.join() + (dark ? 'd' : 'l') + (spikes ? 's' : '');
    if (SPRITES[key]) return SPRITES[key];
    var S = 64, c = document.createElement('canvas'); c.width = c.height = S;
    var x = c.getContext('2d'), g = x.createRadialGradient(S / 2, S / 2, 0, S / 2, S / 2, S / 2), col = 'rgba(' + rgb.join(',') + ',';
    if (dark) {
      g.addColorStop(0, 'rgba(255,255,255,1)'); g.addColorStop(0.12, col + '1)'); g.addColorStop(0.3, col + '0.35)'); g.addColorStop(0.6, col + '0.08)'); g.addColorStop(1, col + '0)');
    } else {
      g.addColorStop(0, col + '1)'); g.addColorStop(0.18, col + '0.9)'); g.addColorStop(0.4, col + '0.18)'); g.addColorStop(1, col + '0)');
    }
    x.fillStyle = g; x.fillRect(0, 0, S, S);
    if (spikes) {
      x.globalCompositeOperation = 'lighter'; x.strokeStyle = col + (dark ? '0.55)' : '0.4)'); x.lineWidth = 1;
      x.beginPath(); x.moveTo(S / 2, 2); x.lineTo(S / 2, S - 2); x.moveTo(2, S / 2); x.lineTo(S - 2, S / 2); x.stroke();
    }
    return (SPRITES[key] = c);
  }
  /** Naked-eye stars: the backdrop, or the dataset itself when the sky is the dataset. */
  P._stars = function (ctx, C, faint) {
    var t = this.bg.table, c = t.cols, pos = this._posOf(this.bg), n = t.n;
    var zoom = this.mode === 'dome' ? Math.max(1, Math.pow(60 / this.fov, 0.35)) : 0.8;
    var isDs = this.ds && this.ds.id === 'sky' && this.chart, sel = isDs ? this.chart.sel : null, mask = isDs ? this.chart.mask : null;
    var anySel = false; if (sel) for (var k = 0; k < n; k++) if (sel[k]) { anySel = true; break; }
    var named = [], dark = C.dark;
    for (var i = 0; i < n; i++) {
      var q = this.project(pos.lon[i], pos.lat[i]); if (!q || q[0] < -9 || q[0] > this.W + 9 || q[1] < -9 || q[1] > this.H + 9) continue;
      var r = Math.max(0.45, (6.8 - c.mag[i]) * 0.42) * zoom;
      if (faint) r *= 0.6;
      ctx.globalAlpha = faint ? 0.45 : (anySel && !sel[i] ? 0.35 : 1);
      if (sel && sel[i] && mask[i]) { ctx.fillStyle = C.sel; ctx.beginPath(); ctx.arc(q[0], q[1], Math.max(2, r), 0, 7); ctx.fill(); }
      else {
        var rgb = OBS.starRGB(c.ci[i], 'bv', dark);
        if (!dark) rgb = rgb.map(function (x) { return Math.round(x * 0.55); });
        if (r < 1.1) { ctx.fillStyle = 'rgb(' + rgb + ')'; ctx.fillRect(q[0] - r, q[1] - r, 2 * r, 2 * r); }
        else { var s = r * (dark ? 4.2 : 2.6); ctx.drawImage(sprite(rgb, dark, !faint && c.mag[i] < 1.2 && dark), q[0] - s, q[1] - s, 2 * s, 2 * s); }
      }
      if (c.name[i] && !faint) named.push([c.mag[i], c.name[i], q, r]);
    }
    ctx.globalAlpha = 1;
    if (this.layers.labels && !faint) this._labels(ctx, C, named, false);
  };
  P._labels = function (ctx, C, named, photo) {
    var lim = this.mode === 'dome' ? (this.fov > 90 ? 1.0 : this.fov > 40 ? 2.2 : this.fov > 15 ? 3.5 : 5.5) : 0.6, self = this;
    var mk = this.markers.map(function (m) { var f = self.toFrame(m.ra, m.dec); return self.project(f[0], f[1]); }).filter(Boolean);
    named = named.filter(function (s) { return !mk.some(function (q) { return Math.hypot(q[0] - s[2][0], q[1] - s[2][1]) < 14; }); });
    ctx.font = '11px ' + getComputedStyle(document.documentElement).getPropertyValue('--sans');
    named.forEach(function (s) {
      if (s[0] >= lim) return;
      var x = s[2][0] + s[3] + 4, y = s[2][1] + 3;
      if (photo) { ctx.fillStyle = 'rgba(0,0,0,0.55)'; ctx.fillText(s[1], x + 1, y + 1); ctx.fillStyle = 'rgba(235,240,255,0.92)'; }
      else ctx.fillStyle = C['ink-2'];
      ctx.fillText(s[1], x, y);
    });
  };
  P._starLabels = function (ctx, C) {
    if (!this.layers.labels) return;
    var c = this.bg.table.cols, pos = this._posOf(this.bg), named = [];
    for (var i = 0; i < this.bg.table.n; i++) {
      if (!c.name[i]) continue;
      var q = this.project(pos.lon[i], pos.lat[i]); if (!q || q[0] < 0 || q[0] > this.W || q[1] < 0 || q[1] > this.H) continue;
      named.push([c.mag[i], c.name[i], q, 3]);
    }
    this._labels(ctx, C, named, true);
  };
  P._skySelection = function (ctx, C) {
    var t = this.bg.table, pos = this._posOf(this.bg), sel = this.chart.sel;
    ctx.strokeStyle = C.sel; ctx.lineWidth = 1.5;
    for (var i = 0; i < t.n; i++) {
      if (!sel[i]) continue;
      var q = this.project(pos.lon[i], pos.lat[i]); if (!q) continue;
      ctx.beginPath(); ctx.arc(q[0], q[1], 7, 0, 7); ctx.stroke();
    }
  };
  /** The open dataset's objects, coloured as the lens colours them; selected ones highlighted. On a photograph
   *  only small datasets and selections are drawn, as rings, so the picture stays visible. */
  P._dataset = function (ctx, C, photo) {
    var ds = this.ds, ch = this.chart, pos = this._posOf(ds), n = ds.table.n, m = ch.mask, sel = ch.sel;
    var anySel = false; for (var k = 0; k < n; k++) if (sel[k] && m[k]) { anySel = true; break; }
    var shown = 0; for (k = 0; k < n; k++) shown += m[k];
    var s = shown > 20000 ? 1.3 : shown > 3000 ? 1.8 : 2.6;
    if (this.mode === 'dome' && this.fov < 20) s *= 1.4;
    for (var pass = 0; pass < 2; pass++) {
      if (photo && pass === 0 && shown > 1500) continue;
      for (var i = 0; i < n; i++) {
        if (!m[i] || (pass === 1) !== !!sel[i]) continue;
        var q = this.project(pos.lon[i], pos.lat[i]);
        if (!q || q[0] < -3 || q[0] > this.W + 3 || q[1] < -3 || q[1] > this.H + 3) continue;
        if (photo) {
          ctx.globalAlpha = pass ? 1 : 0.75; ctx.strokeStyle = pass ? C.sel : ch.colourOf(i, C); ctx.lineWidth = 1.2;
          ctx.beginPath(); ctx.arc(q[0], q[1], pass ? 4 : 3, 0, 7); ctx.stroke();
          continue;
        }
        ctx.globalAlpha = pass ? 1 : anySel ? 0.35 : 0.85;
        ctx.fillStyle = pass ? C.sel : anySel ? C['ink-3'] : ch.colourOf(i, C);
        ctx.fillRect(q[0] - s / 2, q[1] - s / 2, s * (pass ? 1.6 : 1), s * (pass ? 1.6 : 1));
      }
    }
    ctx.globalAlpha = 1;
  };
  P._markers = function (ctx, C, photo) {
    var self = this;
    ctx.font = '12px ' + getComputedStyle(document.documentElement).getPropertyValue('--sans');
    this.markers.forEach(function (mk) {
      var f = self.toFrame(mk.ra, mk.dec), q = self.project(f[0], f[1]);
      if (!q) return;
      var r = Math.max(7, mk.r ? mk.r * self.scale * D2R : 9), col = mk.colour || (photo ? '#f1d08f' : C.accent);
      ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(q[0], q[1], r, 0, 7); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(q[0] - r - 7, q[1]); ctx.lineTo(q[0] - r + 2, q[1]); ctx.moveTo(q[0] + r + 7, q[1]); ctx.lineTo(q[0] + r - 2, q[1]); ctx.stroke();
      if (mk.label) {
        if (photo) { ctx.fillStyle = 'rgba(0,0,0,0.6)'; ctx.fillText(mk.label, q[0] + r + 6, q[1] - 4); ctx.fillStyle = col; }
        ctx.fillText(mk.label, q[0] + r + 5, q[1] - 5);
      }
    });
  };
  /** Ground below the horizon, the daylight or twilight the Sun's altitude gives, compass points and the Sun. */
  P.sunAlt = function () { var s = A.sunPos(this.obs.jd); return A.equToHor(s[0], s[1], this.lst(), this.obs.lat)[1]; };
  P._horizon = function (ctx, C) {
    var f = 4, w = Math.ceil(this.W / f), h = Math.ceil(this.H / f), c = this._hz || (this._hz = document.createElement('canvas'));
    var sun = A.sunPos(this.obs.jd), sh = A.equToHor(sun[0], sun[1], this.lst(), this.obs.lat), sAlt = sh[1];
    // sky brightness: full day above the horizon, fading through civil, nautical and astronomical twilight to -18 deg
    var day = Math.max(0, Math.min(1, (sAlt + 18) / 18)), dayA = Math.pow(day, 1.6) * 0.93;
    var key = [this.mode, this.lon0.toFixed(3), this.lat0.toFixed(3), this.fov.toFixed(3), w, h, dayA.toFixed(3), C.dark].join();
    if (this._hzKey !== key) {
      c.width = w; c.height = h;
      var x = c.getContext('2d'), im = x.createImageData(w, h), px = im.data;
      var ground = C.dark ? [16, 14, 12] : [96, 88, 76], blue = sAlt > 0 ? [112, 156, 210] : [30, 52, 96];
      for (var j = 0; j < h; j++) for (var i = 0; i < w; i++) {
        var p = this.unproject((i + 0.5) * f, (j + 0.5) * f), k = 4 * (j * w + i);
        if (!p) continue;
        if (p[1] < 0) { px[k] = ground[0]; px[k + 1] = ground[1]; px[k + 2] = ground[2]; px[k + 3] = 242; }
        else if (dayA > 0) { var t = 1 - p[1] / 90 * 0.35; px[k] = blue[0] * t; px[k + 1] = blue[1] * t; px[k + 2] = blue[2]; px[k + 3] = dayA * 255; }
      }
      x.putImageData(im, 0, 0); this._hzKey = key;
    }
    ctx.imageSmoothingEnabled = true; ctx.drawImage(c, 0, 0, w * f, h * f);
    // the horizon line and the compass
    ctx.strokeStyle = C.dark ? 'rgba(220,200,160,0.6)' : 'rgba(60,50,40,0.6)'; ctx.lineWidth = 1.2; ctx.beginPath();
    var on = false, last = null;
    for (var az = 0; az <= 360; az += 1) {
      var q = this.project((360 - az) % 360, 0);
      if (!q || (last && Math.abs(q[0] - last[0]) > this.W / 3)) { on = false; last = q; continue; }
      if (!on) { ctx.moveTo(q[0], q[1]); on = true; } else ctx.lineTo(q[0], q[1]);
      last = q;
    }
    ctx.stroke();
    ctx.font = '600 13px ' + getComputedStyle(document.documentElement).getPropertyValue('--sans'); ctx.textAlign = 'center';
    var self = this;
    [['N', 0], ['NE', 45], ['E', 90], ['SE', 135], ['S', 180], ['SW', 225], ['W', 270], ['NW', 315]].forEach(function (d) {
      var q = self.project((360 - d[1]) % 360, -2.5);
      if (q && q[0] > 0 && q[0] < self.W && q[1] > 0 && q[1] < self.H) { ctx.fillStyle = C.dark ? 'rgba(240,220,180,0.9)' : 'rgba(40,30,20,0.85)'; ctx.fillText(d[0], q[0], q[1] + 12); }
    });
    ctx.textAlign = 'left';
    if (sAlt > -1) {
      var sq = this.project((360 - sh[0]) % 360, sAlt);
      if (sq) { ctx.fillStyle = '#ffd34d'; ctx.beginPath(); ctx.arc(sq[0], sq[1], Math.max(6, 0.27 * this.scale * D2R * 2), 0, 7); ctx.fill(); }
    }
  };
  P._hoverRing = function (ctx, C) {
    var h = this.hoverItem, q = this.project(h.lon, h.lat); if (!q) return;
    ctx.strokeStyle = this.layers.photo ? '#fff' : C.ink; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(q[0], q[1], 7, 0, 7); ctx.stroke();
  };

  // ------------------------------------------------------------------------------------- moving around
  /** Swing to a direction. o: {mode, frame, instant, photo}. */
  P.goTo = function (ra, dec, fov, opts) {
    opts = opts || {};
    if (opts.mode && opts.mode !== this.mode) { this.mode = opts.mode; if (this.mode === 'allsky') this.lat0 = 0; }
    if (opts.frame && opts.frame !== this.frame) { var e = this.fromFrame(this.lon0, this.lat0); this.frame = opts.frame; var nf = this.toFrame(e[0], e[1]); this.lon0 = nf[0]; this.lat0 = nf[1]; }
    if (opts.photo !== undefined && this.images) this.layers.photo = opts.photo;
    if (opts.band !== undefined) { this.band = opts.band; if (opts.band !== 'visible' && this.images) this.layers.photo = true; }
    else if (opts.photo === true || opts.photo === false) this.band = 'visible';   // photographs, or the drawn sky: both in visible light
    if (opts.time !== undefined) this.obs.jd = A.jd(opts.time);
    this._mwKey = null; OBS.emit('skymode');
    var f = this.toFrame(ra, dec), self = this;
    var from = { lon: this.lon0, lat: this.lat0, fov: this.fov }, to = { lon: f[0], lat: this.mode === 'allsky' ? 0 : f[1], fov: fov || this.fov };
    var dl = ((to.lon - from.lon + 540) % 360) - 180;
    if (this.anim) { G.cancelAnimationFrame(this.anim); this.anim = null; }
    if (OBS.reducedMotion() || opts.instant) { this.lon0 = to.lon; this.lat0 = to.lat; this.fov = to.fov; this.draw(true); return; }
    var t0 = performance.now(), dur = opts.duration || 1300;
    (function step(now) {
      var u = Math.min(1, (now - t0) / dur), e = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
      // zoom out a little in the middle of long trips, like swinging a telescope
      var hop = Math.min(1, (Math.abs(dl) + Math.abs(to.lat - from.lat)) / 90), wide = Math.max(from.fov, to.fov) * (1 + 0.6 * hop);
      var lf = Math.log(from.fov) + (Math.log(to.fov) - Math.log(from.fov)) * e + Math.sin(Math.PI * e) * (Math.log(wide) - Math.max(Math.log(from.fov), Math.log(to.fov))) * hop;
      self.lon0 = (from.lon + dl * e + 360) % 360; self.lat0 = from.lat + (to.lat - from.lat) * e; self.fov = Math.exp(lf);
      self.drag = u < 1 ? { kind: 'anim' } : null;    // coarse rendering while moving
      self._draw(u >= 1);
      if (u < 1) self.anim = G.requestAnimationFrame(step); else { self.anim = null; self.drag = null; self.draw(true); }
    })(t0);
  };
  P.setMode = function (m) { this.mode = m; if (m === 'allsky') this.lat0 = 0; this._mwKey = null; this.draw(true); };
  P.setFrame = function (f) {
    if (f === this.frame) return;
    var e = this.fromFrame(this.lon0, this.lat0); this.frame = f; var n = this.toFrame(e[0], e[1]);
    this.lon0 = n[0]; this.lat0 = this.mode === 'allsky' ? 0 : n[1];
    if (this.mode === 'allsky' && f === 'gal') this.lon0 = 0;    // a whole-sky Galactic map has the centre of the Galaxy in the middle
    this._mwKey = null; this.draw(true);
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
    if (!best && (!this.band || this.band === 'visible')) scan(this.bg, null);
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
    var touches = {};
    function spread() { var k = Object.keys(touches); return k.length < 2 ? 0 : Math.hypot(touches[k[0]][0] - touches[k[1]][0], touches[k[0]][1] - touches[k[1]][1]); }
    cv.addEventListener('pointerdown', function (e) {
      var p = pos(e);
      if (self.guard && self.guard.active()) { self.drag = { kind: 'tap', x: p[0], y: p[1] }; return; }     // let the page scroll
      touches[e.pointerId] = p;
      if (Object.keys(touches).length === 2) { self.drag = { kind: 'pinch', d0: spread(), fov: self.fov }; e.preventDefault(); return; }
      cv.setPointerCapture(e.pointerId);
      if (self.anim) { G.cancelAnimationFrame(self.anim); self.anim = null; }
      // as in the lens: with Select circle, a drag replaces the selection and shift-drag adds to it; with Turn, shift-drag draws a
      // circle that replaces it and shift with Ctrl, Alt or ⌘ adds
      self.drag = (e.shiftKey || self.tool === 'circle') ? { kind: 'circle', x0: p[0], y0: p[1], r: 0, add: self.tool === 'circle' ? e.shiftKey : (e.altKey || e.ctrlKey || e.metaKey) }
        : { kind: 'turn', x: p[0], y: p[1], lon: self.lon0, lat: self.lat0, moved: false };
      e.preventDefault();
    });
    cv.addEventListener('pointermove', function (e) {
      var p = pos(e), d = self.drag;
      if (touches[e.pointerId]) touches[e.pointerId] = p;
      if (d && d.kind === 'tap') { if (Math.hypot(p[0] - d.x, p[1] - d.y) > 8) self.drag = null; return; }
      if (d && d.kind === 'pinch') {
        var r = spread() / (d.d0 || 1); if (!r) return;
        if (self.mode === 'allsky') { if (r > 1.25) { var k = Object.keys(touches), mx = (touches[k[0]][0] + touches[k[1]][0]) / 2, my = (touches[k[0]][1] + touches[k[1]][1]) / 2, q = self.unproject(mx, my); if (q) { self.mode = 'dome'; self.lon0 = q[0]; self.lat0 = q[1]; self.fov = 140; d.fov = 140 * r; self._mwKey = null; OBS.emit('skymode'); } } }
        else { self.fov = Math.max(0.05, Math.min(200, d.fov / r)); if (self.fov >= 199 && r < 1) { self.mode = 'allsky'; self.lat0 = 0; self._mwKey = null; OBS.emit('skymode'); } }
        self.draw(); return;
      }
      if (!d || d.kind === 'anim') {
        if (d) return;
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
      delete touches[e.pointerId];
      var d = self.drag;
      if (d && d.kind === 'tap') { self.drag = null; if (e.type === 'pointerup') self.guard.arm(); return; }
      if (d && d.kind === 'pinch') { if (!Object.keys(touches).length) { self.drag = null; self.draw(true); } else self.drag = { kind: 'anim' }; return; }
      if (d && d.kind === 'anim' && Object.keys(touches).length) return;
      self.drag = null; if (!d || d.kind === 'anim') return;
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
      if (self.anim) { G.cancelAnimationFrame(self.anim); self.anim = null; self.drag = null; }
      if (self.mode === 'allsky') {
        if (e.deltaY < 0) { var p = pos(e), q = self.unproject(p[0], p[1]); if (q) { self.mode = 'dome'; self.lon0 = q[0]; self.lat0 = q[1]; self.fov = 140; self._mwKey = null; OBS.emit('skymode'); } }
      } else {
        self.fov = Math.max(0.05, Math.min(200, self.fov * Math.exp(e.deltaY * 0.0015)));
        if (self.fov >= 199 && e.deltaY > 0) { self.mode = 'allsky'; self.lat0 = 0; self._mwKey = null; OBS.emit('skymode'); }
      }
      self.drag = { kind: 'anim' }; self.draw();
      clearTimeout(self._wt); self._wt = setTimeout(function () { self.drag = null; self.draw(true); }, 180);
    }, { passive: false });
    if (G.ResizeObserver) new G.ResizeObserver(function () { self.draw(true); }).observe(cv);
  };
})(typeof window !== 'undefined' ? window : globalThis);
