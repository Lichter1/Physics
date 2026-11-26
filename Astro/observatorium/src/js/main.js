/* Observatorium — wiring: the app object missions talk to, the lens and sky controls, the scale ladder,
 * object cards, readouts and the theme. */
(function (G) {
  'use strict';
  var OBS = G.OBS, $ = OBS.$, esc = OBS.esc, A = OBS.astro;
  var chart, sky, app;

  // ------------------------------------------------------------------------------------------ the app API
  function buildApp() {
    chart = new OBS.Chart($('chart'));
    sky = new OBS.Sky($('sky'));
    app = OBS.app = {
      chart: chart, sky: sky, tools: $('tools'),
      /** Open a dataset in the lens (and on the sky if it has positions). */
      open: function (id, o) {
        o = o || {};
        return OBS.open(id).then(function (ds) {
          lensMode('data'); app.imageView(false);
          chart.equal = false; chart.pointColour = null; chart.bins = 40;
          chart.setDataset(ds, o.sub);
          if (o.filter) chart.setFilter(o.filter);
          if (o.colourBy) chart.colourBy = o.colourBy;
          // a dataset's default axis may still be locked: fall back to the first quantity the student has
          function open1(id) { var v = OBS.varOf(ds, id); return !v || OBS.unlocked(v.needs); }
          function firstOpen(not) { var v = ds.vars.filter(function (w) { return !w.cat && w.id !== not && OBS.unlocked(w.needs); })[0]; return v ? v.id : null; }
          var dx = o.x || (open1(ds.x) ? ds.x : firstOpen(ds.y)), dy = o.y || (ds.y === '#' || open1(ds.y) ? ds.y : firstOpen(dx));
          if (o.x || o.y || dx !== ds.x || dy !== ds.y) chart.setAxes(dx || ds.x, dy || ds.y);
          sky.setDataset(ds.pos ? ds : null, chart);
          setLevel(ds.level);
          refreshLens(); OBS.emit('dataset', ds);
          return chart;
        });
      },
      /** Point the lens at an instrument: returns the chart and the tools panel for its controls. */
      instrument: function (title) {
        lensMode('instrument', title);
        chart.equal = false; chart.pointColour = null; chart.fold = null;
        app.tools.innerHTML = '';
        return { chart: chart, tools: app.tools };
      },
      reset: function () { OBS.runCleanups(); var iv = $('imgview'); iv.onclick = iv.onpointerdown = iv.onpointermove = iv.onpointerup = null; app.tools.innerHTML = ''; app.note(''); sky.markers = []; chart.fold = null; chart.equal = false; app.imageView(false); sky.draw(); },
      /** Show (or hide) the instrument's own canvas in place of the chart, for images and parameter maps. */
      imageView: function (on) { var cv = $('imgview'); cv.hidden = !on; $('chart').style.visibility = on ? 'hidden' : ''; $('legend').hidden = !!on; return cv; },
      lectureMode: function () { lensMode('lecture'); },
      lensMode: function () { return mode; },
      note: function (html) { var n = $('lens-note'); n.innerHTML = html || ''; n.hidden = !html; },
      markers: function (list) { sky.markers = list || []; sky.draw(); },
      refresh: function () { refreshLens(); applyLock(); },
      legend: function () { legend(); }
    };
    // a mission may call setAxes directly; keep the menus in step
    var setAxes = chart.setAxes;
    chart.setAxes = function (x, y) { setAxes.call(chart, x, y); refreshLens(); };
  }

  // ------------------------------------------------------------------------------------------ the lens
  var mode = 'data';
  function lensMode(m, title) {
    if (mode === 'lecture' && m !== 'lecture' && OBS.lecture) OBS.lecture.stop();
    mode = m;
    $('pane-lens').classList.toggle('lecturing', m === 'lecture');
    $('lens-title').textContent = m === 'lecture' ? 'Lecture' : 'Data lens';
    $('lecture').hidden = m !== 'lecture';
    $('lens-ctl').hidden = m !== 'data';
    $('lens-inst').hidden = m !== 'instrument';
    $('lens-inst').textContent = title || '';
    $('legend').innerHTML = '';
    if (m === 'instrument') { sky.setDataset(null, chart); $('lens-sub').textContent = ''; $('lens-read').innerHTML = ''; $('lens-hover').textContent = ''; }
  }
  function datasetOptions() {
    var html = '';
    OBS.LEVELS.forEach(function (L) {
      var ds = Object.keys(OBS.DATASETS).map(function (k) { return OBS.DATASETS[k]; }).filter(function (d) { return d.level === L.id; });
      if (ds.length) html += '<optgroup label="' + esc(L.label) + '">' + ds.map(function (d) { return '<option value="' + d.id + '">' + esc(d.title) + '</option>'; }).join('') + '</optgroup>';
    });
    $('ds-sel').innerHTML = html;
  }
  function varOptions(ds, sel, allowHist) {
    var opts = ds.vars.filter(function (v) { return !v.cat; }).map(function (v) {
      var ok = OBS.unlocked(v.needs);
      return '<option value="' + v.id + '"' + (ok ? '' : ' disabled') + (v.id === sel ? ' selected' : '') + '>' + esc(v.label) + (ok ? '' : ' (locked)') + '</option>';
    }).join('');
    if (allowHist) opts += '<option value="#"' + (sel === '#' ? ' selected' : '') + '>Number of objects (histogram)</option>';
    return opts;
  }
  function refreshLens() {
    var ds = chart.ds; if (!ds || mode !== 'data') return;
    $('ds-sel').value = ds.id;
    $('lens-sub').textContent = ds.sub || '';
    var sub = $('sub-sel'), subw = $('sub-wrap');
    if (ds.subset) {
      var vals = [], seen = {}, col = ds.table.cols[ds.subset.col];
      for (var i = 0; i < col.length; i++) if (!seen[col[i]]) { seen[col[i]] = 1; vals.push(col[i]); }
      if (ds.subset.names && ds.subset.names['*']) vals.push('*');
      sub.innerHTML = vals.map(function (v) { return '<option value="' + esc(v) + '">' + esc((ds.subset.names && ds.subset.names[v]) || v) + '</option>'; }).join('');
      sub.value = chart.sub; $('sub-label').textContent = ds.subset.label; subw.hidden = false;
    } else subw.hidden = true;
    $('x-sel').innerHTML = varOptions(ds, chart.xid, false);
    $('y-sel').innerHTML = varOptions(ds, chart.yid, true);
    applyLock();
    var lb = OBS.logbook && OBS.logbook.current(), playing = lb && lb.m && lb.step === 2 ? lb.m.id : null;
    var cats = ds.vars.filter(function (v) { return v.cat && OBS.unlocked(v.needs) && v.spoils !== playing; });
    $('c-sel').innerHTML = '<option value="auto">Automatic</option><option value="none">One colour</option>' + cats.map(function (v) { return '<option value="' + v.id + '">' + esc(v.label) + '</option>'; }).join('');
    $('c-sel').value = chart.colourBy === 'auto' || chart.colourBy === 'none' ? chart.colourBy : chart.colourBy;
    $('bins-wrap').hidden = !chart.hist();
    $('bins').value = chart.bins;
    legend(); readSel();
  }
  function applyLock() {
    var lk = OBS.state.lockLens, why = 'Fixed during this exercise; after it, choose freely (Keep exploring).';
    if (chart.ds && chart.logOv) ['x', 'y'].forEach(function (a) {
      var b = $('log' + a), on = a === 'x' ? chart.logX() : chart.logY(), ok = !(lk && lk.axes) && !(a === 'y' && chart.hist()) && (on || chart.canLog(a));
      b.setAttribute('aria-pressed', on); b.disabled = !ok;
      b.title = lk && lk.axes ? why : !ok ? 'This quantity has zero or negative values: a log scale would hide them' : on ? 'Switch to a linear scale' : 'Switch to a logarithmic scale';
    });
    [['ds-sel', lk && lk.ds], ['sub-sel', lk && lk.ds], ['x-sel', lk && lk.axes], ['y-sel', lk && lk.axes]].forEach(function (p) { var e = $(p[0]); e.disabled = !!p[1]; e.title = p[1] ? why : ''; });
    [].forEach.call(document.querySelectorAll('.lv'), function (b) { var L = OBS.LEVELS.filter(function (x) { return x.id === b.dataset.lv; })[0]; b.disabled = !!(L && L.locked) || !!(lk && lk.ds); b.title = lk && lk.ds ? why : (L ? L.note : ''); });
  }
  function legend() {
    var cats = chart.categories(), el = $('legend'), C = OBS.colours(), pal = [C.c1, C.c2, C.c3, C.c4, C.c5, C.c6, C.c7, C.c8];
    if (!cats || chart.hist()) { el.innerHTML = ''; return; }
    el.innerHTML = cats.cats.slice(0, 8).map(function (c, k) { return '<span class="lg"><i style="background:' + pal[k] + '"></i>' + esc(c) + ' <span class="num muted">' + cats.cnt[c].toLocaleString('en-US') + '</span></span>'; }).join('');
  }
  function readSel() {
    if (!chart.ds) return;
    var sel = chart.selected(), shown = chart.shown().length, el = $('lens-read');
    if (mode !== 'data') { el.innerHTML = ''; return; }
    if (!sel.length) { el.innerHTML = '<span class="num">' + shown.toLocaleString('en-US') + '</span> shown. Drag a box to select; shift-drag adds; click a point for its card; scroll to zoom; double-click to fit.'; return; }
    var xs = chart.xs(), ys = chart.ys(), xv = chart.xvar(), yv = chart.yvar();
    function med(a) { var v = sel.map(function (i) { return a[i]; }).filter(isFinite).sort(function (p, q) { return p - q; }); return v[v.length >> 1]; }
    var txt = '<span class="num">' + sel.length.toLocaleString('en-US') + '</span> of ' + shown.toLocaleString('en-US') + ' selected · median ' + esc(xv.label) + ' <span class="num">' + OBS.fmt(med(xs), 3) + '</span>';
    if (ys) txt += ', ' + esc(yv.label) + ' <span class="num">' + OBS.fmt(med(ys), 3) + '</span>';
    var cats = chart.categories();
    if (cats) {
      var cnt = {}; sel.forEach(function (i) { cnt[cats.col[i]] = (cnt[cats.col[i]] || 0) + 1; });
      txt += ' · ' + Object.keys(cnt).sort(function (a, b) { return cnt[b] - cnt[a]; }).slice(0, 3).map(function (k) { return esc(k) + ' ' + Math.round(100 * cnt[k] / sel.length) + '%'; }).join(', ');
    }
    el.innerHTML = txt + ' <button type="button" class="btn small ghost" id="sel-clear">Clear</button>';
    $('sel-clear').addEventListener('click', function () { chart.clearSel(); });
  }
  function bindLens() {
    datasetOptions();
    $('ds-sel').addEventListener('change', function (e) { app.open(e.target.value); });
    $('sub-sel').addEventListener('change', function (e) { chart.setSubset(e.target.value); sky.draw(); refreshLens(); });
    $('x-sel').addEventListener('change', function (e) { chart.fold = null; chart.setAxes(e.target.value, null); });
    $('y-sel').addEventListener('change', function (e) { chart.setAxes(null, e.target.value); });
    ['x', 'y'].forEach(function (a) {
      $('log' + a).addEventListener('click', function () { chart.logOv[a] = !(a === 'x' ? chart.logX() : chart.logY()); chart.autoscale(); refreshLens(); });
    });
    $('c-sel').addEventListener('change', function (e) {
      chart.colourBy = e.target.value; chart._cats = null;
      // outside an exercise, colouring by a category means you want to see it: a selection would grey the rest out
      var lb = OBS.logbook.current(); if (e.target.value !== 'auto' && e.target.value !== 'none' && chart.selected().length && !(lb.m && lb.step === 2)) chart.clearSel();
      chart.draw(); sky.draw(); legend();
    });
    $('bins').addEventListener('input', function (e) { chart.bins = +e.target.value; chart._histMax(); chart.draw(); });
    [].forEach.call(document.querySelectorAll('[data-ltool]'), function (b) {
      b.addEventListener('click', function () { chart.tool = b.dataset.ltool; [].forEach.call(document.querySelectorAll('[data-ltool]'), function (x) { x.setAttribute('aria-pressed', x === b); }); });
    });
    $('fit').addEventListener('click', function () { chart.autoscale(); });
    chart.listeners.push(function (ev, i) {
      if (ev !== 'hover') return;
      var h = $('lens-hover');
      if (i < 0 || !chart.ds) { h.textContent = ''; return; }
      var ds = chart.ds, nm = ds.name ? ds.name(ds.table.cols, i) : '';
      var xv = chart.xvar(), yv = chart.yvar(), xs = chart.xs(), ys = chart.ys();
      h.innerHTML = (nm ? '<b>' + esc(nm) + '</b> · ' : '') + esc(xv.label) + ' <span class="num">' + OBS.fmt(xs[i], 4) + '</span>' + (ys ? ' · ' + esc(yv.label) + ' <span class="num">' + OBS.fmt(ys[i], 4) + '</span>' : '');
    });
    var pending = false;
    OBS.on('selection', function () { if (pending) return; pending = true; G.requestAnimationFrame(function () { pending = false; readSel(); sky.draw(); }); });
    OBS.on('unlock', function () { refreshLens(); });
  }

  // ------------------------------------------------------------------------------------------ the sky
  function setLevel(id) {
    [].forEach.call(document.querySelectorAll('.lv'), function (b) { b.setAttribute('aria-current', b.dataset.lv === id ? 'true' : 'false'); });
  }
  var LEVEL_GO = {
    sky: function () { return app.open('sky').then(function () { sky.goTo(88, 0, 110, { mode: 'dome', frame: 'equ' }); }); },
    solar: function () { return app.open('smallbodies').then(function () { chart.setView({ x0: 0.5, x1: 6, y0: 0, y1: 0.5 }); sky.setMode('allsky'); }); },     // the asteroid belt and Jupiter's Trojans; Fit view shows the Kuiper belt
    stars: function () { return app.open('nearby').then(function () { sky.goTo(0, 0, 200, { mode: 'allsky', frame: 'equ' }); }); },
    clusters: function () { return app.open('clusters', { sub: 'pleiades' }).then(function () { sky.goTo(56.75, 24.12, 9, { mode: 'dome', frame: 'equ' }); }); },
    galaxy: function () { return app.open('globulars').then(function () { sky.goTo(266.4, -28.9, 200, { mode: 'allsky', frame: 'gal' }); }); },
    galaxies: function () { return app.open('galaxies').then(function () { sky.goTo(180, 30, 200, { mode: 'allsky', frame: 'equ' }); }); },
    universe: function () { return app.open('supernovae').then(function () { sky.goTo(266.4, -28.9, 200, { mode: 'allsky', frame: 'gal', band: 'cmb' }); }); }
  };
  function bindSky() {
    $('ladder').innerHTML = OBS.LEVELS.map(function (L) {
      return '<li><button type="button" class="lv" data-lv="' + L.id + '"' + (L.locked ? ' disabled' : '') + ' title="' + esc(L.note) + '"><span class="lv-s">' + esc(L.scale) + '</span><span class="lv-l">' + esc(L.label) + '</span></button></li>';
    }).join('');
    [].forEach.call(document.querySelectorAll('.lv'), function (b) { b.addEventListener('click', function () { if (LEVEL_GO[b.dataset.lv]) LEVEL_GO[b.dataset.lv](); }); });
    function seg(attr, fn) {
      [].forEach.call(document.querySelectorAll('[' + attr + ']'), function (b) {
        b.addEventListener('click', function () { fn(b.getAttribute(attr)); syncSky(); });
      });
    }
    seg('data-smode', function (m) { sky.setMode(m); if (m === 'dome' && sky.fov > 150) sky.fov = 110; });
    seg('data-sframe', function (f) {
      if (f === 'hor' && sky.mode !== 'dome') sky.setMode('dome');
      sky.setFrame(f);
      if (f === 'hor') { sky.lon0 = 180; sky.lat0 = 35; sky.fov = Math.max(sky.fov, 100); sky.draw(true); }
      timeUI();
    });
    // the moment shown in the horizon frame, in the observatory's standard time (UTC+2)
    function pad(n) { return (n < 10 ? '0' : '') + n; }
    function timeUI() {
      $('sky-time-wrap').hidden = sky.frame !== 'hor';
      var d = new Date(Math.round(((sky.obs.jd - 2440587.5) * 86400000 + 2 * 3600000) / 60000) * 60000);
      $('sky-time').value = d.getUTCFullYear() + '-' + pad(d.getUTCMonth() + 1) + '-' + pad(d.getUTCDate()) + 'T' + pad(d.getUTCHours()) + ':' + pad(d.getUTCMinutes());
    }
    app.timeUI = timeUI;
    $('sky-time').addEventListener('change', function (e) {
      var m = /^(\d+)-(\d+)-(\d+)T(\d+):(\d+)/.exec(e.target.value); if (!m) return;
      var ms = Date.UTC(+m[1], m[2] - 1, +m[3], +m[4], +m[5]) - 2 * 3600000;
      sky.setTime(ms); OBS.emit('skytime', ms);       // an instrument that plans the night follows the sky's clock
    });
    $('sky-now').addEventListener('click', function () { var ms = Date.now(); sky.setTime(ms); timeUI(); OBS.emit('skytime', ms); });
    $('sky-band').addEventListener('change', function (e) { sky.setBand(e.target.value); syncSky(); });
    seg('data-stool', function (t) { sky.tool = t === 'circle' ? 'circle' : null; });
    seg('data-slook', function (v) { sky.layers.photo = v === 'photo' && !!sky.images; if (v === 'drawn') sky.band = 'visible'; sky._mwKey = null; sky.draw(true); });
    OBS.on('skyimages', function (m) {
      [].forEach.call(document.querySelectorAll('[data-slook="photo"]'), function (b) { b.disabled = false; b.title = 'Real photographs of the sky'; });
      $('sky-band').innerHTML = '<option value="visible">Visible</option>' + (m.bands || []).map(function (b) { return '<option value="' + b.id + '">' + esc(b.label) + '</option>'; }).join('');
      $('sky-band').value = sky.band;
    });
    ['milkyway', 'constellations', 'grid', 'labels'].forEach(function (k) {
      var b = $('layer-' + k); b.setAttribute('aria-pressed', sky.layers[k]);
      b.addEventListener('click', function () { sky.layers[k] = !sky.layers[k]; b.setAttribute('aria-pressed', sky.layers[k]); sky.draw(true); });
    });
    OBS.on('skymode', syncSky);
    sky.listeners.push(function (ev, h) {
      if (ev === 'draw') {
        var c = sky.centre(); $('sky-read').textContent = sky.frame === 'hor' ? horRead() : (sky.mode === 'dome' ? 'Looking at α ' + raStr(c[0]) + ', δ ' + decStr(c[1]) + ' · field ' + OBS.fmt(sky.fov, 2) + '°' : 'The whole sky · ' + (sky.frame === 'gal' ? 'Galactic coordinates, centre of the Galaxy in the middle' : 'equatorial coordinates'));
        var cr = sky.credits(); $('sky-credit').textContent = cr ? 'Photographs: ' + cr : ''; $('sky-credit').hidden = !cr;
        syncSky();
      }
      if (ev === 'hover') {
        var el = $('sky-hover');
        if (!h) { el.textContent = ''; return; }
        var nm = h.ds.name ? h.ds.name(h.ds.table.cols, h.i) : '';
        var mag = h.ds.mag ? OBS.column(h.ds, h.ds.mag === 'mag' ? 'mag' : h.ds.mag === 'G' ? 'G' : h.ds.mag)[h.i] : NaN;
        el.innerHTML = '<b>' + esc(nm || 'object') + '</b>' + (isFinite(mag) ? ' · <span class="num">' + OBS.fix(mag, 2) + '</span> mag' : '');
      }
    });
  }
  function syncSky() {
    [].forEach.call(document.querySelectorAll('[data-smode]'), function (b) { b.setAttribute('aria-pressed', b.dataset.smode === sky.mode); });
    [].forEach.call(document.querySelectorAll('[data-sframe]'), function (b) { b.setAttribute('aria-pressed', b.dataset.sframe === sky.frame); });
    [].forEach.call(document.querySelectorAll('[data-stool]'), function (b) { b.setAttribute('aria-pressed', (b.dataset.stool === 'circle') === (sky.tool === 'circle')); });
    [].forEach.call(document.querySelectorAll('[data-slook]'), function (b) { b.setAttribute('aria-pressed', (b.dataset.slook === 'photo') === !!sky.layers.photo); });
    var g = $('layer-grid'); if (g) g.setAttribute('aria-pressed', !!sky.layers.grid);
    var bs = $('sky-band'); if (bs && bs.value !== sky.band) bs.value = sky.band;
    if ($('sky-time-wrap')) $('sky-time-wrap').hidden = sky.frame !== 'hor';
    if (sky.frame === 'hor' && app.timeUI && document.activeElement !== $('sky-time')) app.timeUI();
  }
  function horRead() {
    var d = new Date(Math.round(((sky.obs.jd - 2440587.5) * 86400000 + 2 * 3600000) / 60000) * 60000), s = sky.sunAlt();
    var when = d.toISOString().slice(0, 16).replace('T', ' ');
    return 'Mitzpe Ramon, ' + when + ' (UTC+2) · looking at azimuth ' + Math.round((360 - sky.lon0) % 360) + '°, altitude ' + Math.round(sky.lat0) + '° · Sun ' + (s > 0 ? 'up, ' : 'down, ') + Math.round(s) + '°' +
      (s < -18 ? ' (dark night)' : s < -12 ? ' (astronomical twilight)' : s < -6 ? ' (nautical twilight)' : s < 0 ? ' (civil twilight)' : '');
  }
  function raStr(ra) { var h = ra / 15, H = Math.floor(h), m = Math.round((h - H) * 60); if (m === 60) { H++; m = 0; } return H % 24 + 'h ' + (m < 10 ? '0' : '') + m + 'm'; }
  function decStr(d) { return (d < 0 ? '−' : '+') + Math.abs(d).toFixed(1) + '°'; }

  // ------------------------------------------------------------------------------------------ object card
  function card(e) {
    var ds = e.ds, i = e.i, c = ds.table.cols, el = $('pop');
    var nm = ds.name ? ds.name(c, i) : ds.title;
    var rows = ds.vars.map(function (v) {
      var ok = OBS.unlocked(v.needs), x = OBS.column(ds, v.id)[i];
      if (v.cat) return ok ? '<tr><th>' + esc(v.label) + '</th><td>' + esc(x) + '</td></tr>' : '';
      if (!isFinite(x)) return '';
      return '<tr' + (ok ? '' : ' class="locked"') + '><th>' + esc(v.label) + '</th><td>' + (ok ? '<span class="num">' + OBS.fmt(x, 4) + '</span> ' + esc(v.unit) : 'locked') + '</td></tr>';
    }).join('');
    var ext = ds.id === 'sky' && c.spect[i] ? '<tr><th>Spectral type</th><td>' + esc(c.spect[i]) + '</td></tr>' : '';
    el.innerHTML = '<button type="button" class="x" id="pop-x" aria-label="Close">×</button><p class="eyebrow">' + esc(ds.title) + '</p><h3>' + esc(nm) + '</h3><table class="kv">' + ext + rows + '</table>' +
      '<div class="actions">' + (ds.pos ? '<button type="button" class="btn small" id="pop-go">Centre the sky on it</button>' : '') + (ds === chart.ds ? '<button type="button" class="btn small ghost" id="pop-sel">Select it</button>' : '') + '</div>';
    el.hidden = false;
    $('pop-x').addEventListener('click', function () { el.hidden = true; });
    if ($('pop-go')) $('pop-go').addEventListener('click', function () { sky.goTo(c.ra[i], c.dec[i], Math.min(sky.mode === 'dome' ? sky.fov : 30, 30), { mode: 'dome' }); });
    if ($('pop-sel')) $('pop-sel').addEventListener('click', function () { chart.setSel([i], false); });
  }

  // ------------------------------------------------------------------------------------------ theme
  function bindTheme() {
    var root = document.documentElement, b = $('theme');
    function label() {
      var dark = OBS.colours().dark;
      b.textContent = dark ? 'Plate view' : 'Night view';
      b.title = dark ? 'Show the sky as a photographic plate (dark stars on glass), the way Leavitt saw it' : 'Show the sky as it looks at night';
    }
    b.addEventListener('click', function () {
      root.setAttribute('data-theme', OBS.colours().dark ? 'light' : 'dark');
      OBS.store.set('theme', root.getAttribute('data-theme'));
      label(); chart._cats = null; chart.draw(); sky._mwKey = null; sky.draw(true); legend();
      OBS.emit('theme');      // lecture charts and explainers redraw in the new colours
    });
    var saved = OBS.store.get('theme', null); if (saved) root.setAttribute('data-theme', saved);
    try { G.matchMedia('(prefers-color-scheme: light)').addEventListener('change', function () { label(); chart.draw(); sky._mwKey = null; sky.draw(true); }); } catch (e) { /* old browsers */ }
    label();
  }
  function bindTabs() {
    [].forEach.call(document.querySelectorAll('[data-tab]'), function (b) { b.addEventListener('click', function () { OBS.logbook.tab(b.dataset.tab); }); });
  }

  // ------------------------------------------------------------------------------------------ layout
  // The divider sets how the width (or, when stacked, the height) is shared between the sky and the lens;
  // any pane can fold into a slim tab. Remembered per browser.
  var LAY = OBS.store.get('layout', null) || { f: 0.52, fold: {} };
  function layoutMode() { var w = G.innerWidth; return w <= 860 ? 'narrow' : w <= 1280 ? 'medium' : 'wide'; }
  function applyLayout() {
    var g = document.querySelector('.grid'), m = layoutMode(), F = LAY.fold, f = LAY.f, both = !F.sky && !F.lens;
    ['wide', 'medium', 'narrow'].forEach(function (k) { g.classList.toggle(k, k === m); });
    [['logbook', 'pane-logbook'], ['sky', 'pane-sky'], ['lens', 'pane-lens']].forEach(function (p) {
      $(p[1]).classList.toggle('folded', !!F[p[0]]);
    });
    $('split').classList.toggle('off', !both);
    $('split').setAttribute('aria-orientation', m === 'wide' ? 'vertical' : 'horizontal');
    $('split').setAttribute('aria-valuenow', Math.round(100 * f));
    if (m === 'wide') {
      g.style.gridTemplateColumns = (F.logbook ? '56px' : 'minmax(330px, 390px)') + ' ' +
        (F.sky ? '44px' : 'minmax(0, ' + (F.lens ? 1 : f) + 'fr)') + ' 12px ' + (F.lens ? '44px' : 'minmax(0, ' + (F.sky ? 1 : 1 - f) + 'fr)');
      g.style.gridTemplateRows = '';
    } else if (m === 'medium') {
      g.style.gridTemplateColumns = (F.logbook ? '56px' : 'minmax(320px, 370px)') + ' minmax(0, 1fr)';
      g.style.gridTemplateRows = (F.sky ? 'auto' : 'minmax(240px, ' + (F.lens ? 1 : f) + 'fr)') + ' 12px ' + (F.lens ? 'auto' : 'minmax(260px, ' + (F.sky ? 1 : 1 - f) + 'fr)');
    } else { g.style.gridTemplateColumns = ''; g.style.gridTemplateRows = ''; }
    OBS.store.set('layout', LAY);
    if (chart) chart.draw();
    if (sky) sky.draw(true);
  }
  function bindLayout() {
    [].forEach.call(document.querySelectorAll('[data-fold]'), function (b) { b.addEventListener('click', function () { LAY.fold[b.dataset.fold] = true; applyLayout(); }); });
    [].forEach.call(document.querySelectorAll('[data-unfold]'), function (b) { b.addEventListener('click', function () { LAY.fold[b.dataset.unfold] = false; applyLayout(); }); });
    var s = $('split'), drag = false;
    s.addEventListener('pointerdown', function (e) { drag = true; s.setPointerCapture(e.pointerId); s.classList.add('dragging'); e.preventDefault(); });
    s.addEventListener('pointermove', function (e) {
      if (!drag) return;
      var a = $('pane-sky').getBoundingClientRect(), b = $('pane-lens').getBoundingClientRect();
      var f = layoutMode() === 'wide' ? (e.clientX - a.left) / (b.right - a.left) : (e.clientY - a.top) / (b.bottom - a.top);
      LAY.f = OBS.clamp(f, 0.15, 0.85); applyLayout();
    });
    function end() { drag = false; s.classList.remove('dragging'); }
    s.addEventListener('pointerup', end); s.addEventListener('pointercancel', end);
    s.addEventListener('dblclick', function () { LAY.f = 0.52; applyLayout(); });
    s.addEventListener('keydown', function (e) {
      var d = { ArrowLeft: -0.05, ArrowUp: -0.05, ArrowRight: 0.05, ArrowDown: 0.05 }[e.key];
      if (d) { LAY.f = OBS.clamp(LAY.f + d, 0.15, 0.85); applyLayout(); e.preventDefault(); }
    });
    var rt; G.addEventListener('resize', function () { clearTimeout(rt); rt = setTimeout(applyLayout, 120); });
    applyLayout();
  }

  // ------------------------------------------------------------------------------------------ start
  function start() {
    buildApp(); bindTheme(); bindLens(); bindSky(); bindTabs(); bindLayout();
    [$('tour-btn')].concat([].slice.call(document.querySelectorAll('[data-tour]'))).forEach(function (b) { b.addEventListener('click', function () { OBS.tour.start(); }); });
    OBS.on('pick', card);
    OBS.logbook.renderChapters(); OBS.logbook.renderList();
    $('boot').hidden = false;
    sky.init().then(function () { return app.open('sky'); }).then(function () {
      $('boot').hidden = true;
      sky.goTo(88, 0, 110, { mode: 'dome', frame: 'equ', instant: true });
      if (OBS.store.get('toured', false)) OBS.logbook.resume(); else OBS.tour.maybe();
    }).catch(function (e) { $('boot').textContent = 'Could not load the star data: ' + e.message; });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})(typeof window !== 'undefined' ? window : globalThis);
