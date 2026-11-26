/* Observatorium — instruments for chapters 6-9: a redshift template for spectra, aperture photometry on a real
 * image, a night planner for the horizon sky, an isotropy meter for the whole sky, and a Markov chain that explores
 * a chi-squared map. Each works on real data; the missions set them up and read their state. */
(function (G) {
  'use strict';
  var OBS = G.OBS, A = OBS.astro, UI = OBS.ui, INS = OBS.instruments;
  function f(v, s) { return '<span class="num">' + OBS.fmt(v, s || 3) + '</span>'; }
  function fx(v, d) { return '<span class="num">' + OBS.fix(v, d) + '</span>'; }

  // ------------------------------------------------------------------------------------------ line lists (vacuum, nm)
  var LINES = OBS.LINES = {
    emission: [['[O II]', 372.8, 'em'], ['Hβ', 486.3, 'em'], ['[O III]', 500.8, 'em'], ['Hα', 656.5, 'em'], ['[N II]', 658.5, 'em'], ['[S II]', 671.8, 'em']],
    absorption: [['Ca K', 393.4, 'abs'], ['Ca H', 397.0, 'abs'], ['G band', 430.5, 'abs'], ['Mg b', 517.4, 'abs'], ['Na D', 589.5, 'abs']],
    quasar: [['Lyα', 121.6, 'em'], ['N V', 124.0, 'em'], ['Si IV', 139.8, 'em'], ['C IV', 154.9, 'em'], ['C III]', 190.9, 'em'], ['Mg II', 279.8, 'em']]
  };

  // ------------------------------------------------------------------------------------------ redshift template
  /** Lines at rest wavelengths slid by (1+z) over a spectrum. o: {lines, z, min, max, forest, onchange} */
  INS.zfit = function (chart, tools, o) {
    var st = { z: o.z || 0 }, C = OBS.colours(), out = UI.readout(tools);
    function redraw() {
      chart.overlays = chart.overlays.filter(function (x) { return String(x.id).indexOf('zl-') !== 0; });
      if (o.forest) chart.overlays.push({ id: 'zl-forest', type: 'span', x0: 91.2 * (1 + st.z), x1: 121.6 * (1 + st.z), colour: C.c1, label: 'Lyman-α forest?', drag: false });
      o.lines.forEach(function (l, k) {
        chart.overlays.push({ id: 'zl-' + k, type: 'vline', value: l[1] * (1 + st.z), label: l[0], drag: false, dash: l[2] === 'abs' ? [2, 3] : [6, 3], width: 1.3, colour: l[2] === 'abs' ? C.c2 : C.c4 });
      });
      chart.draw();
      var main = o.lines[o.main || 0], v = st.z < 0.1 ? st.z * 299792.458 : NaN;
      out.innerHTML = 'Trial redshift z = ' + fx(st.z, 4) + ': ' + main[0] + ' (rest ' + fx(main[1], 1) + ' nm) would appear at ' + fx(main[1] * (1 + st.z), 1) + ' nm.' +
        (isFinite(v) ? ' Recession speed cz ≈ ' + f(v, 3) + ' km/s, so d ≈ cz/H₀ ≈ ' + f(v / 70, 3) + ' Mpc.' : '');
      if (o.onchange) o.onchange(st);
    }
    var coarse = UI.slider(tools, { label: 'Redshift z', min: o.min || 0, max: o.max || 1, value: st.z, fmt: function (x) { return x.toFixed(3); }, oninput: function (x) { st.z = x + fine.get(); redraw(); } });
    var fine = UI.slider(tools, { label: 'Fine tune', min: -0.01, max: 0.01, value: 0, fmt: function (x) { return (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(4); }, oninput: function (x) { st.z = coarse.get() + x; redraw(); } });
    tools.appendChild(out);
    st.set = function (z) { coarse.set(z, true); fine.set(0, true); st.z = z; redraw(); };
    st.coarse = coarse; st.fine = fine;
    redraw();
    return st;
  };

  // ------------------------------------------------------------------------------------------ aperture photometry
  /** Measure stars in a calibrated image (nanomaggies, AB zero point 22.5). field: photfield.json. */
  INS.photometry = function (app, tools, field, o) {
    o = o || {};
    var cv = app.imageView(true), ctx = cv.getContext('2d'), W = field.w, H = field.h, ps = field.pixscale;
    var I = new Float32Array(W * H); for (var k = 0; k < I.length; k++) I[k] = field.data[k] / field.scale;
    var st = { r: o.r || 2, sel: -1, done: {}, fwhm: NaN, last: null };     // a too-small aperture to begin with: the curve of growth is the lesson
    // display: asinh stretch
    var sorted = Array.prototype.slice.call(I).sort(function (a, b) { return a - b; }), lo = sorted[Math.floor(0.02 * I.length)], hi = sorted[Math.floor(0.998 * I.length)];
    var img = document.createElement('canvas'); img.width = W; img.height = H;
    (function () {
      var ic = img.getContext('2d'), d = ic.createImageData(W, H), a = 0.03, top = Math.asinh((hi - lo) / a);
      for (var i = 0; i < I.length; i++) { var v = Math.max(0, Math.min(1, Math.asinh(Math.max(0, I[i] - lo) / a) / top)), g = Math.round(255 * Math.pow(v, 0.9)); d.data[4 * i] = g; d.data[4 * i + 1] = g; d.data[4 * i + 2] = g; d.data[4 * i + 3] = 255; }
      ic.putImageData(d, 0, 0);
    })();
    function median(a) { a = a.slice().sort(function (p, q) { return p - q; }); return a.length ? a[a.length >> 1] : 0; }
    function sky(x, y, r) {
      var vals = [], rin = r + 4, rout = r + 11;
      for (var j = Math.max(0, Math.floor(y - rout)); j <= Math.min(H - 1, Math.ceil(y + rout)); j++)
        for (var i = Math.max(0, Math.floor(x - rout)); i <= Math.min(W - 1, Math.ceil(x + rout)); i++) {
          var d = Math.hypot(i - x, j - y); if (d >= rin && d <= rout) vals.push(I[j * W + i]);
        }
      var m = median(vals), mad = median(vals.map(function (v) { return Math.abs(v - m); }));
      return { level: m, sigma: 1.4826 * mad, n: vals.length };
    }
    /** Flux inside radius r (4x4 sub-pixels for the edge), minus the sky. */
    function measure(x, y, r) {
      var s = sky(x, y, r), F = 0, area = 0;
      for (var j = Math.max(0, Math.floor(y - r - 1)); j <= Math.min(H - 1, Math.ceil(y + r + 1)); j++)
        for (var i = Math.max(0, Math.floor(x - r - 1)); i <= Math.min(W - 1, Math.ceil(x + r + 1)); i++) {
          var frac = 0;
          for (var a = 0; a < 4; a++) for (var b = 0; b < 4; b++) if (Math.hypot(i - 0.375 + a * 0.25 - x, j - 0.375 + b * 0.25 - y) <= r) frac += 1 / 16;
          if (frac) { F += frac * (I[j * W + i] - s.level); area += frac; }
        }
      var err = s.sigma * Math.sqrt(area * (1 + area / Math.max(1, s.n)));
      return { F: F, err: err, snr: F / err, m: F > 0 ? 22.5 - 2.5 * Math.log10(F) : NaN, area: area, sky: s };
    }
    function centroid(x, y) {
      for (var it = 0; it < 3; it++) {
        var sx = 0, sy = 0, s = 0, b = sky(x, y, 4).level;
        for (var j = Math.max(0, Math.round(y) - 4); j <= Math.min(H - 1, Math.round(y) + 4); j++)
          for (var i = Math.max(0, Math.round(x) - 4); i <= Math.min(W - 1, Math.round(x) + 4); i++) {
            var w = Math.max(0, I[j * W + i] - b); if (Math.hypot(i - x, j - y) <= 4) { sx += w * i; sy += w * j; s += w; }
          }
        if (s > 0) { x = sx / s; y = sy / s; }
      }
      return [x, y];
    }
    function fwhm(x, y) {
      var b = sky(x, y, 6).level, s = 0, r2 = 0;
      for (var j = Math.max(0, Math.round(y) - 6); j <= Math.min(H - 1, Math.round(y) + 6); j++)
        for (var i = Math.max(0, Math.round(x) - 6); i <= Math.min(W - 1, Math.round(x) + 6); i++) {
          var w = I[j * W + i] - b, d2 = (i - x) * (i - x) + (j - y) * (j - y); if (d2 <= 36 && w > 0) { s += w; r2 += w * d2; }
        }
      return 2.3548 * Math.sqrt(r2 / (2 * s)) * ps;
    }
    // layout
    var view = { s: 1, ox: 0, oy: 0 };
    function draw() {
      var r = cv.getBoundingClientRect(), dpr = Math.min(2, G.devicePixelRatio || 1), w = Math.max(200, r.width), h = Math.max(160, r.height);
      if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.fillStyle = '#05070c'; ctx.fillRect(0, 0, w, h);
      var sc = Math.min((w - 20) / W, (h - 20) / H); view.s = sc; view.ox = (w - W * sc) / 2; view.oy = (h - H * sc) / 2;
      ctx.imageSmoothingEnabled = false; ctx.drawImage(img, view.ox, view.oy, W * sc, H * sc);
      ctx.font = '12px ' + getComputedStyle(document.documentElement).getPropertyValue('--sans');
      field.stars.forEach(function (s, k) {
        var X = view.ox + (s.x + 0.5) * sc, Y = view.oy + (s.y + 0.5) * sc, done = st.done[k];
        ctx.strokeStyle = done ? '#7ed39d' : 'rgba(241,208,143,0.85)'; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(X - 14, Y); ctx.lineTo(X - 8, Y); ctx.moveTo(X + 8, Y); ctx.lineTo(X + 14, Y); ctx.moveTo(X, Y - 14); ctx.lineTo(X, Y - 8); ctx.stroke();
        ctx.fillStyle = ctx.strokeStyle; ctx.fillText(String.fromCharCode(65 + k), X + 10, Y - 9);
      });
      if (st.last) {
        var X = view.ox + (st.last.x + 0.5) * sc, Y = view.oy + (st.last.y + 0.5) * sc;
        ctx.strokeStyle = '#59e3c9'; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.arc(X, Y, st.r * sc, 0, 7); ctx.stroke();
        ctx.setLineDash([4, 3]); ctx.strokeStyle = 'rgba(89,227,201,0.6)'; ctx.beginPath(); ctx.arc(X, Y, (st.r + 4) * sc, 0, 7); ctx.stroke(); ctx.beginPath(); ctx.arc(X, Y, (st.r + 11) * sc, 0, 7); ctx.stroke(); ctx.setLineDash([]);
      }
      ctx.fillStyle = 'rgba(235,240,255,0.8)';
      ctx.fillText('Field ' + (W * ps).toFixed(0) + '″ across · north up, east left · r band', 12, h - 10);
    }
    var out = UI.readout(tools), mini = document.createElement('canvas'); mini.className = 'ins-mini';
    var slider = UI.slider(tools, { label: 'Aperture radius', min: 1, max: 15, value: st.r, step: 0.5, fmt: function (x) { return x.toFixed(1) + ' px = ' + (x * ps).toFixed(2) + '″'; }, oninput: function (x) { st.r = x; update(); } });
    tools.appendChild(out); tools.appendChild(mini);
    var table = document.createElement('div'); tools.appendChild(table);
    function curve() {
      var r = mini.getBoundingClientRect(), dpr = Math.min(2, G.devicePixelRatio || 1), w = Math.max(200, r.width), h = 130, c = mini.getContext('2d'), C = OBS.colours();
      mini.width = w * dpr; mini.height = h * dpr; c.setTransform(dpr, 0, 0, dpr, 0, 0); c.fillStyle = C.panel; c.fillRect(0, 0, w, h);
      c.font = '11px ' + getComputedStyle(document.documentElement).getPropertyValue('--sans');
      if (!st.last) { c.fillStyle = C['ink-3']; c.fillText('Click a star to see how flux and noise grow with the aperture.', 10, h / 2); return; }
      var rs = [], Fs = [], Ss = []; for (var rr = 1; rr <= 15; rr += 0.5) { var m = measure(st.last.x, st.last.y, rr); rs.push(rr); Fs.push(m.F); Ss.push(m.snr); }
      var Fmax = Math.max.apply(null, Fs), Smax = Math.max.apply(null, Ss), X = function (r) { return 34 + (r - 1) / 14 * (w - 44); };
      c.strokeStyle = C.rule; c.strokeRect(34.5, 8.5, w - 44, h - 30);
      function plot(vals, mx, col) { c.beginPath(); vals.forEach(function (v, i) { var y = 8 + (h - 30) * (1 - Math.max(0, v) / mx); if (i) c.lineTo(X(rs[i]), y); else c.moveTo(X(rs[i]), y); }); c.strokeStyle = col; c.lineWidth = 1.8; c.stroke(); }
      plot(Fs, Fmax * 1.05, C.c1); plot(Ss, Smax * 1.05, C.c2);
      c.strokeStyle = C.sel; c.beginPath(); c.moveTo(X(st.r), 8); c.lineTo(X(st.r), h - 22); c.stroke();
      c.fillStyle = C.c1; c.fillText('flux (curve of growth)', 40, 20); c.fillStyle = C.c2; c.fillText('signal-to-noise', 40, 34);
      c.fillStyle = C['ink-3']; c.fillText('aperture radius →', w - 110, h - 8);
    }
    function row() {
      var ks = Object.keys(st.done);
      table.innerHTML = ks.length ? '<table class="ins-table"><thead><tr><th>Star</th><th>SDSS r</th><th>Your m</th><th>Δ</th><th>S/N</th></tr></thead><tbody>' + ks.map(function (k) {
        var d = st.done[k], s = field.stars[k], dm = d.m - s.r;
        return '<tr><td>' + String.fromCharCode(65 + (+k)) + '</td><td>' + s.r.toFixed(2) + '</td><td>' + d.m.toFixed(2) + '</td><td class="' + (Math.abs(dm) < 0.15 ? 'good' : '') + '">' + (dm >= 0 ? '+' : '−') + Math.abs(dm).toFixed(2) + '</td><td>' + d.snr.toFixed(0) + '</td></tr>';
      }).join('') + '</tbody></table>' : '';
    }
    function update() {
      draw();
      if (!st.last) { out.innerHTML = 'Click a star (the marked ones are in the SDSS catalogue) to put an aperture on it.'; curve(); return; }
      var m = measure(st.last.x, st.last.y, st.r), s = st.sel >= 0 ? field.stars[st.sel] : null;
      st.last.m = m;
      if (st.sel >= 0 && isFinite(m.m)) st.done[st.sel] = { m: m.m, snr: m.snr, r: st.r };
      out.innerHTML = (s ? 'Star ' + String.fromCharCode(65 + st.sel) + ' (SDSS r = ' + s.r.toFixed(2) + '): ' : 'Unlisted object: ') +
        'flux F = ' + f(m.F, 3) + ' nanomaggies in ' + fx(m.area, 0) + ' pixels, sky σ = ' + f(m.sky.sigma, 2) + ' per pixel. ' +
        'm = 22.5 − 2.5 log₁₀ F = ' + fx(m.m, 2) + ' ± ' + fx(1.0857 / m.snr, 2) + ' (S/N = ' + fx(m.snr, 0) + '). Star width (FWHM) ' + fx(st.fwhm, 2) + '″.';
      row(); curve();
      if (o.onchange) o.onchange(st);
    }
    cv.onclick = function (e) {
      var r = cv.getBoundingClientRect(), x = (e.clientX - r.left - view.ox) / view.s - 0.5, y = (e.clientY - r.top - view.oy) / view.s - 0.5;
      if (x < 0 || y < 0 || x >= W || y >= H) return;
      var best = -1, bd = 10 / Math.max(0.5, view.s) + 3;
      field.stars.forEach(function (s, k) { var d = Math.hypot(s.x - x, s.y - y); if (d < bd) { bd = d; best = k; } });
      var c = centroid(best >= 0 ? field.stars[best].x : x, best >= 0 ? field.stars[best].y : y);
      st.sel = best; st.last = { x: c[0], y: c[1] }; st.fwhm = fwhm(c[0], c[1]);
      update();
    };
    var ro = G.ResizeObserver ? new G.ResizeObserver(function () { if (!cv.hidden) draw(); }) : null; if (ro) ro.observe(cv);
    st.measure = measure; st.pick = function (k) { var s = field.stars[k], c = centroid(s.x, s.y); st.sel = k; st.last = { x: c[0], y: c[1] }; st.fwhm = fwhm(c[0], c[1]); update(); };
    st.slider = slider; st.stop = function () { if (ro) ro.disconnect(); cv.onclick = null; }; OBS.cleanup(st.stop);
    update();
    return st;
  };

  // ------------------------------------------------------------------------------------------ planning a night
  /** The horizon sky over the Wise Observatory: a date, a time and a target. o: {targets, target, onchange} */
  INS.night = function (app, tools, o) {
    var sky = app.sky, st = { day: o.day || 0, hour: o.hour || 22, t: 0 }, base = Date.UTC(2026, 6, 1);   // the season: 1 July 2026 to 30 June 2027
    var cv = app.imageView(true), ctx = cv.getContext('2d');
    /** The observer's plot: altitude of the target and the Sun through the chosen night. */
    function plot(t, d0) {
      var r = cv.getBoundingClientRect(), dpr = Math.min(2, G.devicePixelRatio || 1), w = Math.max(220, r.width), h = Math.max(160, r.height), C = OBS.colours();
      if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.fillStyle = C.panel; ctx.fillRect(0, 0, w, h);
      ctx.font = '11px ' + getComputedStyle(document.documentElement).getPropertyValue('--sans');
      var L = 48, R = w - 16, T = 16, B = h - 36;
      function X(hr) { return L + (hr - 16) / 16 * (R - L); } function Y(alt) { return B - (alt + 20) / 110 * (B - T); }
      var lat = sky.obs.lat, lon = sky.obs.lon, tgt = [], sun = [];
      for (var hr = 16; hr <= 32.001; hr += 0.125) {
        var jd = A.jd(d0 + (hr - 2) * 3600000), lst = A.lst(jd, lon), sp = A.sunPos(jd);
        tgt.push([hr, A.equToHor(t.ra, t.dec, lst, lat)[1]]); sun.push([hr, A.equToHor(sp[0], sp[1], lst, lat)[1]]);
      }
      // dark hours shaded
      ctx.fillStyle = C.dark ? 'rgba(90,110,180,0.18)' : 'rgba(40,60,140,0.12)';
      var d0x = null; sun.concat([[33, 99]]).forEach(function (p) { if (p[1] < -18 && d0x === null) d0x = p[0]; else if (p[1] >= -18 && d0x !== null) { ctx.fillRect(X(d0x), T, X(p[0]) - X(d0x), B - T); d0x = null; } });
      ctx.strokeStyle = C.rule; ctx.strokeRect(L + 0.5, T + 0.5, R - L, B - T);
      for (var a = 0; a <= 90; a += 30) { ctx.fillStyle = C['ink-3']; ctx.textAlign = 'right'; ctx.fillText(a + '°', L - 6, Y(a) + 4); ctx.strokeStyle = C.rule; ctx.beginPath(); ctx.moveTo(L, Y(a)); ctx.lineTo(R, Y(a)); ctx.stroke(); }
      for (hr = 16; hr <= 32; hr += 2) { ctx.textAlign = 'center'; ctx.fillText((hr % 24 < 10 ? '0' : '') + (hr % 24) + ':00', X(hr), B + 14); }
      ctx.textAlign = 'center'; ctx.fillText('local time (UTC+2)', (L + R) / 2, B + 28); ctx.textAlign = 'left';
      ctx.setLineDash([4, 4]); ctx.strokeStyle = C['ink-3']; ctx.beginPath(); ctx.moveTo(L, Y(30)); ctx.lineTo(R, Y(30)); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = C['ink-3']; ctx.fillText('airmass 2', L + 4, Y(30) - 4);
      function curve(pts, col, wd) { ctx.strokeStyle = col; ctx.lineWidth = wd; ctx.beginPath(); pts.forEach(function (p, k) { var x = X(p[0]), y = Y(Math.max(-20, p[1])); if (k) ctx.lineTo(x, y); else ctx.moveTo(x, y); }); ctx.stroke(); ctx.lineWidth = 1; }
      curve(sun, '#ffd34d', 1.5); curve(tgt, C.accent, 2.5);
      ctx.fillStyle = C.accent; ctx.fillText(t.name, L + 6, T + 14); ctx.fillStyle = '#d9a400'; ctx.fillText('the Sun · shaded: dark night (Sun below −18°)', L + 6, T + 28);
      ctx.strokeStyle = C.sel; ctx.beginPath(); ctx.moveTo(X(st.hour), T); ctx.lineTo(X(st.hour), B); ctx.stroke();
    }
    var sel = document.createElement('label'); sel.className = 'sel'; sel.innerHTML = '<span>Target</span><select>' + o.targets.map(function (t, k) { return '<option value="' + k + '">' + t.name + '</option>'; }).join('') + '</select>';
    tools.appendChild(sel); var pick = sel.querySelector('select'); pick.value = o.target || 0;
    var out = UI.readout(tools);
    function date() { return new Date(base + st.day * 86400000); }
    function dayLabel(d) { var x = new Date(base + d * 86400000); return x.toUTCString().slice(5, 16); }
    var sd = UI.slider(tools, { label: 'Date', min: 0, max: 364, value: st.day, step: 1, fmt: function (d) { return dayLabel(d); }, oninput: function (x) { st.day = x; update(); } });
    var sh = UI.slider(tools, { label: 'Local time (UTC+2, standard time; summer clocks read 1 h more)', min: 16, max: 32, value: st.hour, step: 0.25, fmt: function (h) { var H = Math.floor(h) % 24, m = Math.round((h % 1) * 60); return (H < 10 ? '0' : '') + H + ':' + (m < 10 ? '0' : '') + m + (h >= 24 ? ' (next day)' : ''); }, oninput: function (x) { st.hour = x; update(); } });
    tools.appendChild(out);
    function update() {
      var t = o.targets[+pick.value], d = date(), ms = Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()) + (st.hour - 2) * 3600000;
      st.t = ms; st.target = t;
      if (sky.frame !== 'hor') { sky.setMode('dome'); sky.setFrame('hor'); }
      sky.obs.jd = A.jd(ms);
      var lst = A.lst(sky.obs.jd, sky.obs.lon), h = A.equToHor(t.ra, t.dec, lst, sky.obs.lat), sun = A.sunPos(sky.obs.jd);
      var sAlt = A.equToHor(sun[0], sun[1], lst, sky.obs.lat)[1], X = A.airmass(h[1]), max = 90 - Math.abs(sky.obs.lat - t.dec);
      // when the target crosses the meridian (local sidereal time = its RA), nearest to this moment
      var dH = ((t.ra - lst + 540) % 360) - 180, transit = st.hour + dH / 15 / 1.0027379;
      st.alt = h[1]; st.az = h[0]; st.sunAlt = sAlt; st.max = max; st.X = X; st.transit = transit;
      plot(t, Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()));
      sky.markers = [{ ra: t.ra, dec: t.dec, label: t.name }];
      sky.lon0 = (360 - h[0]) % 360; sky.lat0 = Math.max(18, Math.min(60, h[1])); sky.fov = 110; sky._mwKey = null; sky.draw(true);
      if (app.timeUI) app.timeUI();
      var dark = sAlt < -18 ? 'dark night' : sAlt < -12 ? 'astronomical twilight' : sAlt < -6 ? 'nautical twilight' : sAlt < 0 ? 'civil twilight' : 'daytime';
      var tt = ((transit % 24) + 24) % 24, TH = Math.floor(tt), TM = Math.round((tt % 1) * 60); if (TM === 60) { TH = (TH + 1) % 24; TM = 0; }
      out.innerHTML = t.name + ': altitude ' + fx(h[1], 1) + '°, azimuth ' + fx(h[0], 0) + '°' + (h[1] > 0 ? ', airmass ' + fx(X, 2) + ' (dimmed by ' + fx(0.21 * X, 2) + ' mag at 0.9 km)' : ', below the horizon') + '. ' +
        'Highest possible: ' + fx(max, 1) + '°, when it crosses the meridian at ' + (TH < 10 ? '0' : '') + TH + ':' + (TM < 10 ? '0' : '') + TM + '. The Sun is at ' + fx(sAlt, 0) + '° (' + dark + ').';
      if (o.onchange) o.onchange(st);
    }
    pick.addEventListener('change', update);
    // the sky's own date box and Now button move the planner too: evenings and the small hours belong to one night
    OBS.onMission('skytime', function (ms) {
      var loc = new Date(ms + 2 * 3600000), hr = loc.getUTCHours() + loc.getUTCMinutes() / 60, d0 = Date.UTC(loc.getUTCFullYear(), loc.getUTCMonth(), loc.getUTCDate());
      if (hr < 12) { hr += 24; d0 -= 86400000; }
      var day = Math.round((d0 - base) / 86400000); day = ((day % 365) + 365) % 365;
      st.day = day; st.hour = Math.max(16, Math.min(32, Math.round(hr * 4) / 4)); sd.set(st.day, true); sh.set(st.hour, true); update();
    });
    var ro = G.ResizeObserver ? new G.ResizeObserver(function () { if (!cv.hidden && st.target) update(); }) : null; if (ro) ro.observe(cv);
    st.update = update; st.sd = sd; st.sh = sh; st.pick = pick; st.stop = function () { if (ro) ro.disconnect(); }; OBS.cleanup(st.stop);
    update();
    return st;
  };

  // ------------------------------------------------------------------------------------------ isotropy
  /** How a sky population is spread: share near the Galactic plane, and the two classic statistics. */
  OBS.isotropy = function (ds, mask) {
    var ra = OBS.column(ds, 'ra'), de = OBS.column(ds, 'dec'), n = 0, plane = 0, cgc = 0, q = 0;
    for (var i = 0; i < ra.length; i++) {
      if (mask && !mask[i]) continue;
      if (!isFinite(ra[i]) || !isFinite(de[i])) continue;
      var g = A.equToGal(ra[i], de[i]), b = g[1] * Math.PI / 180, l = g[0] * Math.PI / 180;
      n++; if (Math.abs(g[1]) < 10) plane++;
      cgc += Math.cos(b) * Math.cos(l); q += Math.sin(b) * Math.sin(b) - 1 / 3;
    }
    return { n: n, plane: plane / n, cosgc: cgc / n, sinb2: q / n, err: 1 / Math.sqrt(3 * n) };
  };

  // ------------------------------------------------------------------------------------------ a Markov chain over a chi-squared map
  /** Supernova chi-squared over (Om, OL), with the offset (H0, M) fitted analytically at every point. */
  OBS.snChi2 = function (sn) {
    var n = sn.z.length, zmin = Infinity, zmax = 0;
    for (var i = 0; i < n; i++) { zmin = Math.min(zmin, sn.z[i]); zmax = Math.max(zmax, sn.z[i]); }
    var NG = 48, lz0 = Math.log(zmin), lz1 = Math.log(zmax), zg = [];
    for (var k = 0; k < NG; k++) zg.push(Math.exp(lz0 + (lz1 - lz0) * k / (NG - 1)));
    var idx = new Int32Array(n), wt = new Float64Array(n), w = new Float64Array(n), sw = 0;
    for (i = 0; i < n; i++) {
      var u = (Math.log(sn.z[i]) - lz0) / (lz1 - lz0) * (NG - 1), k0 = Math.min(NG - 2, Math.max(0, Math.floor(u)));
      idx[i] = k0; wt[i] = u - k0; w[i] = 1 / (sn.emu[i] * sn.emu[i]); sw += w[i];
    }
    return function (Om, OL) {
      // no Big Bang (a bounce) if E^2 turns negative before z = 3; reject those universes
      for (var zz = 0; zz <= 3; zz += 0.1) { var a = 1 + zz; if (Om * a * a * a + (1 - Om - OL) * a * a + OL <= 0) return { chi2: Infinity }; }
      if (Om < 0) return { chi2: Infinity };
      var mg = zg.map(function (z) { return 5 * Math.log10((1 + z) * A.dM(z, Om, OL)); });
      var m = new Float64Array(n), s = 0;
      for (var i = 0; i < n; i++) { m[i] = mg[idx[i]] * (1 - wt[i]) + mg[idx[i] + 1] * wt[i]; s += w[i] * (sn.mu[i] - m[i]); }
      var off = s / sw, c2 = 0;
      for (i = 0; i < n; i++) { var r = sn.mu[i] - m[i] - off; c2 += w[i] * r * r; }
      return { chi2: c2, off: off, H0: 299792.458 * 1e5 / Math.pow(10, off / 5) };
    };
  };
  INS.mcmc = function (app, tools, sn, o) {
    o = o || {};
    var cv = app.imageView(true), ctx = cv.getContext('2d'), chi = OBS.snChi2(sn);
    var X0 = 0, X1 = 1, Y0 = 0, Y1 = 1.4, NX = 50, NY = 56, grid = [], gmin = Infinity, best = null;
    for (var j = 0; j < NY; j++) for (var i = 0; i < NX; i++) {
      var Om = X0 + (X1 - X0) * (i + 0.5) / NX, OL = Y0 + (Y1 - Y0) * (j + 0.5) / NY, q = chi(Om, OL);
      grid.push(q.chi2); if (q.chi2 < gmin) { gmin = q.chi2; best = { Om: Om, OL: OL }; }
    }
    var st = { chain: [], acc: 0, tries: 0, run: false, step: o.step || 0.004, burn: 0, evals: NX * NY, gmin: gmin, best: best, inAt: null };
    var map = document.createElement('canvas'); map.width = NX; map.height = NY;
    (function () {
      var mc = map.getContext('2d'), d = mc.createImageData(NX, NY);
      for (var j = 0; j < NY; j++) for (var i = 0; i < NX; i++) {
        var dc = grid[j * NX + i] - gmin, k = 4 * ((NY - 1 - j) * NX + i), v = isFinite(dc) ? Math.exp(-dc / 40) : 0;
        var col = !isFinite(dc) ? [40, 40, 46] : dc < 2.3 ? [241, 208, 143] : dc < 6.17 ? [217, 174, 95] : [Math.round(30 + 90 * v), Math.round(40 + 110 * v), Math.round(80 + 150 * v)];
        d.data[k] = col[0]; d.data[k + 1] = col[1]; d.data[k + 2] = col[2]; d.data[k + 3] = 255;
      }
      mc.putImageData(d, 0, 0);
    })();
    var box = { x: 0, y: 0, w: 1, h: 1 };
    function P(Om, OL) { return [box.x + (Om - X0) / (X1 - X0) * box.w, box.y + box.h - (OL - Y0) / (Y1 - Y0) * box.h]; }
    function draw() {
      var r = cv.getBoundingClientRect(), dpr = Math.min(2, G.devicePixelRatio || 1), w = Math.max(220, r.width), h = Math.max(180, r.height), C = OBS.colours();
      if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.fillStyle = C.panel; ctx.fillRect(0, 0, w, h);
      var side = Math.min(w - 80, (h - 60) * (X1 - X0) / (Y1 - Y0) * 1.0);
      box = { x: 58, y: 14, w: side, h: side * (Y1 - Y0) / (X1 - X0) };
      if (box.h > h - 56) { box.h = h - 56; box.w = box.h * (X1 - X0) / (Y1 - Y0); }
      ctx.imageSmoothingEnabled = false; ctx.drawImage(map, box.x, box.y, box.w, box.h);
      ctx.strokeStyle = C.rule; ctx.strokeRect(box.x + 0.5, box.y + 0.5, box.w, box.h);
      ctx.font = '11px ' + getComputedStyle(document.documentElement).getPropertyValue('--sans'); ctx.fillStyle = C['ink-2'];
      for (var t = 0; t <= 1.001; t += 0.2) { var p = P(t, 0); ctx.textAlign = 'center'; ctx.fillText(t.toFixed(1), p[0], p[1] + 14); }
      for (t = 0; t <= 1.401; t += 0.2) { p = P(0, t); ctx.textAlign = 'right'; ctx.fillText(t.toFixed(1), p[0] - 5, p[1] + 4); }
      ctx.textAlign = 'center'; ctx.fillText('Ωm, the density of matter', box.x + box.w / 2, box.y + box.h + 30);
      ctx.save(); ctx.translate(box.x - 40, box.y + box.h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('ΩΛ, dark energy', 0, 0); ctx.restore(); ctx.textAlign = 'left';
      // flat universes, and the line between slowing down and speeding up
      function seg(a, b, col, lab, dash) { var p = P(a[0], a[1]), q = P(b[0], b[1]); ctx.strokeStyle = col; ctx.setLineDash(dash || []); ctx.lineWidth = 1.3; ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = col; if (lab) ctx.fillText(lab, q[0] - 90, q[1] - 6); }
      seg([0, 1], [1, 0], C.ink, 'flat (Ωm + ΩΛ = 1)', [6, 4]);
      seg([0, 0], [1, 0.5], C.c4, 'accelerating ↑ / slowing ↓', [3, 3]);
      // the chain
      var ch = st.chain, from = Math.max(0, ch.length - 3000);
      ctx.strokeStyle = 'rgba(89,227,201,0.55)'; ctx.lineWidth = 1; ctx.beginPath();
      for (var k = from; k < ch.length; k++) { var q = P(ch[k][0], ch[k][1]); if (k === from) ctx.moveTo(q[0], q[1]); else ctx.lineTo(q[0], q[1]); }
      ctx.stroke();
      if (ch.length) { q = P(ch[ch.length - 1][0], ch[ch.length - 1][1]); ctx.fillStyle = C.sel; ctx.beginPath(); ctx.arc(q[0], q[1], 4, 0, 7); ctx.fill(); }
      ctx.fillStyle = C['ink-3']; ctx.fillText('gold: 68% and 95% regions from the grid', box.x + 6, box.y + 14);
    }
    var out = UI.readout(tools), row = UI.row(tools);
    var runB = UI.button(row, 'Run the chain', function () { st.run = !st.run; runB.textContent = st.run ? 'Pause' : 'Run the chain'; if (st.run) loop(); }, 'primary');
    UI.button(row, 'Start again', function () { st.reset(); draw(); report(); });
    var sStep = UI.slider(tools, { label: 'Step size', min: 0.003, max: 0.4, value: st.step, log: true, fmt: function (x) { return x.toFixed(3); }, oninput: function (x) { st.step = x; } });
    var sBurn = UI.slider(tools, { label: 'Burn-in (steps dropped)', min: 0, max: 2000, value: st.burn, step: 50, fmt: function (x) { return Math.round(x); }, oninput: function (x) { st.burn = x; report(); } });
    tools.appendChild(out);
    var mini = document.createElement('canvas'); mini.className = 'ins-mini'; tools.appendChild(mini);
    var cur = null;
    function gauss() { var u = 1 - Math.random(), v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
    st.stepOnce = function () {
      // every chain starts far from the answer, in a decelerating universe, so the walk-in is there to see
      if (!cur) { var s = [0.85 + 0.1 * Math.random(), 0.1 + 0.1 * Math.random()]; cur = { p: s, c: chi(s[0], s[1]).chi2 }; if (!isFinite(cur.c)) { cur = null; return; } st.chain.push(cur.p); }
      var t = [cur.p[0] + st.step * gauss(), cur.p[1] + st.step * gauss()];
      st.tries++;
      var c = t[0] < 0 || t[0] > 1.2 || t[1] < -0.2 || t[1] > 1.6 ? Infinity : chi(t[0], t[1]).chi2;
      if (isFinite(c) && (c <= cur.c || Math.random() < Math.exp(-(c - cur.c) / 2))) { cur = { p: t, c: c }; st.acc++; }
      st.chain.push(cur.p);
      if (st.inAt === null && cur.c - gmin < 6.17) st.inAt = st.chain.length;     // the step at which the chain reached the 95% region
    };
    st.reset = function () { cur = null; st.chain = []; st.acc = 0; st.tries = 0; st.inAt = null; };
    function stats() {
      var s = st.chain.slice(Math.min(st.chain.length, Math.round(st.burn)));
      if (s.length < 20) return null;
      function ms(k) { var m = 0; s.forEach(function (p) { m += p[k]; }); m /= s.length; var v = 0; s.forEach(function (p) { v += (p[k] - m) * (p[k] - m); }); return [m, Math.sqrt(v / s.length)]; }
      var a = ms(0), b = ms(1), acc = s.filter(function (p) { return p[1] > p[0] / 2; }).length / s.length;
      return { n: s.length, Om: a, OL: b, pacc: acc, samples: s };
    }
    function hist(S) {
      var r = mini.getBoundingClientRect(), dpr = Math.min(2, G.devicePixelRatio || 1), w = Math.max(200, r.width), H = 230, c = mini.getContext('2d'), C = OBS.colours();
      mini.width = w * dpr; mini.height = H * dpr; mini.style.height = H + 'px'; c.setTransform(dpr, 0, 0, dpr, 0, 0); c.fillStyle = C.panel; c.fillRect(0, 0, w, H);
      c.font = '11px ' + getComputedStyle(document.documentElement).getPropertyValue('--sans');
      // the trace: ΩΛ against step number, with the burn-in marked; the walk in from the start is what burn-in drops
      var ch = st.chain, th = 90, n = ch.length;
      c.fillStyle = C['ink-2']; c.fillText('ΩΛ at each step (trace)', 12, 13);
      if (n > 1) {
        var stride = Math.max(1, Math.floor(n / (w - 20)));
        c.strokeStyle = C.c2; c.lineWidth = 1; c.beginPath();
        for (var k = 0; k < n; k += stride) { var X = 10 + (w - 20) * k / (n - 1), Yv = 18 + th - th * (ch[k][1] - Y0) / (Y1 - Y0); if (k === 0) c.moveTo(X, Yv); else c.lineTo(X, Yv); }
        c.stroke();
        var bx = 10 + (w - 20) * Math.min(1, st.burn / (n - 1));
        c.fillStyle = 'rgba(128,128,128,0.25)'; c.fillRect(10, 18, bx - 10, th);
        c.strokeStyle = C.accent; c.beginPath(); c.moveTo(bx, 16); c.lineTo(bx, 20 + th); c.stroke();
        c.fillStyle = C['ink-3']; c.fillText('burn-in ' + Math.round(st.burn), Math.min(w - 80, bx + 4), 30);
        if (st.inAt !== null) { var ix = 10 + (w - 20) * Math.min(1, st.inAt / (n - 1)); c.setLineDash([2, 3]); c.strokeStyle = C['ink-3']; c.beginPath(); c.moveTo(ix, 18); c.lineTo(ix, 18 + th); c.stroke(); c.setLineDash([]); }
      } else { c.fillStyle = C['ink-3']; c.fillText('Run the chain to see its path.', 12, 60); }
      c.translate(0, th + 28); var h = H - th - 28;
      if (!S) { c.fillStyle = C['ink-3']; c.fillText('The histogram of ΩΛ appears once the chain has run past the burn-in.', 10, h / 2); c.setTransform(dpr, 0, 0, dpr, 0, 0); return; }
      var nb = 28, bins = new Array(nb).fill(0); S.samples.forEach(function (p) { var k = Math.floor((p[1] - Y0) / (Y1 - Y0) * nb); if (k >= 0 && k < nb) bins[k]++; });
      var mx = Math.max.apply(null, bins), bw = (w - 20) / nb;
      c.fillStyle = C.c1; bins.forEach(function (v, k) { var bh = (h - 30) * v / mx; c.fillRect(10 + k * bw, h - 20 - bh, bw - 1, bh); });
      c.fillStyle = C['ink-2']; c.fillText('ΩΛ from the chain: ' + S.OL[0].toFixed(2) + ' ± ' + S.OL[1].toFixed(2), 12, 14);
      c.fillStyle = C['ink-3']; c.fillText('0', 8, h - 6); c.fillText('1.4', w - 24, h - 6);
      c.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    function report() {
      var S = stats(); st.stats = S;
      out.innerHTML = 'Grid map: ' + f(st.evals, 4) + ' χ² evaluations; best Ωm = ' + fx(best.Om, 2) + ', ΩΛ = ' + fx(best.OL, 2) + ' (χ² = ' + f(gmin, 4) + ' for ' + f(sn.z.length, 4) + ' supernovae). ' +
        'Chain: ' + f(st.chain.length, 4) + ' steps, acceptance ' + fx(st.tries ? 100 * st.acc / st.tries : 0, 0) + '%.' +
        (S ? ' After burn-in: Ωm = ' + fx(S.Om[0], 2) + ' ± ' + fx(S.Om[1], 2) + ', ΩΛ = ' + fx(S.OL[0], 2) + ' ± ' + fx(S.OL[1], 2) + '; ' + fx(100 * S.pacc, 0) + '% of the samples are accelerating universes.' : '');
      hist(S);
      if (o.onchange) o.onchange(st);
    }
    var raf = 0, last = 0;
    function loop() {
      if (!st.run) return;
      if (cv.hidden) { st.run = false; return; }   // the mission was left
      for (var k = 0; k < 40; k++) st.stepOnce();
      var now = performance.now(); if (now - last > 120) { draw(); report(); last = now; }
      raf = G.requestAnimationFrame(loop);
      if (st.chain.length > 20000) { st.run = false; runB.textContent = 'Run the chain'; draw(); report(); }
    }
    var ro = G.ResizeObserver ? new G.ResizeObserver(function () { if (!cv.hidden) draw(); }) : null; if (ro) ro.observe(cv);
    st.report = report; st.draw = draw; st.sStep = sStep; st.sBurn = sBurn;
    st.stop = function () { st.run = false; G.cancelAnimationFrame(raf); if (ro) ro.disconnect(); }; OBS.cleanup(st.stop);
    draw(); report();
    return st;
  };
})(typeof window !== 'undefined' ? window : globalThis);
