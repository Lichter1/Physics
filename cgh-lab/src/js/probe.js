/* cGħ World — the light probe. For the pixel under the crosshair it re-traces the light ray in JavaScript
 * (the same steps the GPU takes), draws the path in a slice through the planet, and reads off where the
 * light came from, when it left, how much it bent and how its colour changed on the way. */
(function () {
  'use strict';
  var G = CGH.geo, A = CGH.app, $ = A.$;
  var SODIUM = 589, GREEN = 550;
  var last = null;

  function rgbOf(nm) {                                  // rough visible-spectrum colour for the scale
    var h = nm < 440 ? 270 - (nm - 380) / 60 * 30 : nm < 490 ? 240 - (nm - 440) / 50 * 60 : nm < 580 ? 180 - (nm - 490) / 90 * 120 : 60 - Math.min(60, (nm - 580) / 70 * 60);
    var light = nm < 420 ? 30 + (nm - 380) / 40 * 20 : nm > 700 ? 50 - (nm - 700) / 60 * 25 : 50;
    return 'hsl(' + h.toFixed(0) + ',90%,' + light.toFixed(0) + '%)';
  }
  function name(m) {
    if (m.kind !== 1) return 'the sky: starlight from far away';
    if (m.mat === 0) { var p = G.fromWorld(m.x); return Math.abs(p.l) < 7.7 ? 'the street' : 'the ground'; }
    return G.MATS[m.mat] || 'something';
  }

  function update() {
    var S = CGH.world.state, q = S.q, cam = S.cam;
    if (!q || !cam || $('probe').hidden) return;
    var fx = S.fx, rsRay = fx.bend ? q.rs : 0, rsS = fx.gshift ? q.rs : 0, rsC = fx.rates ? q.rs : 0;
    var nb = G.dot(cam.fwd, cam.beta);
    var ab = fx.aberr ? G.deaberrate(cam.fwd, cam.beta) : { n: cam.fwd, D: 1 / (q.gamma * (1 - nb)) };
    var re = G.len(cam.eye), m = G.march(cam.eye, ab.n, rsRay, true), r = G.len(m.x), hit = m.kind === 1;
    var Dg = hit ? Math.sqrt((1 - rsS / r) / (1 - rsS / re)) : 1 / Math.sqrt(1 - rsS / re);
    var D = fx.doppler ? ab.D * Dg : 1, t = fx.delay ? m.ct / q.c : 0;
    var chord = [m.x[0] - cam.eye[0], m.x[1] - cam.eye[1], m.x[2] - cam.eye[2]], cl = G.len(chord);
    var ref = hit ? chord.map(function (c) { return c / (cl || 1); }) : m.v.map(function (c) { return c / G.len(m.v); });
    var bend = Math.acos(Math.max(-1, Math.min(1, G.dot(ab.n, ref)))) * 180 / Math.PI;
    var lamp = hit && m.mat === 4, nm0 = lamp ? SODIUM : GREEN, nm = nm0 / D;
    last = { m: m, n: ab.n, eye: cam.eye, re: re, rs: rsRay, hit: hit };
    CGH.probe.last = { kind: m.kind, mat: m.mat, s: m.s, D: D, Dg: Dg, h: hit ? r - G.R : null, eyeH: re - G.R };

    var n = A.num, rows = [];
    rows.push(['Looking at', name(m) + (hit ? ', ' + n(m.s, 3) + ' m along the light’s path' + (Math.abs(m.s - cl) > 0.01 * cl + 0.05 ? ' (' + n(cl, 3) + ' m straight)' : '') : '')]);
    rows.push(['Its light left', fx.delay ? A.duration(t, 3) + ' ago, far-away time' : 'just now (light delay switched off)']);
    rows.push(['Bent on the way by', bend < 0.05 ? 'nothing you could see' : n(bend, 3) + '°']);
    rows.push(['Gravity shifts it', Math.abs(Dg - 1) < 1e-4 ? 'not at all' : '×' + Dg.toFixed(4) + (Dg > 1 ? ' (bluer: the light fell to you)' : ' (redder: the light climbed to you)')]);
    rows.push(['Your motion shifts it', Math.abs(ab.D - 1) < 1e-4 ? 'not at all' : '×' + ab.D.toFixed(4) + (ab.D > 1 ? ' (you move toward where it came from)' : ' (you move away)')]);
    rows.push(['Together', fx.doppler ? '×' + D.toFixed(4) + ' in frequency, ×' + (fx.beam ? (D * D * D).toFixed(3) : '1') + ' in brightness' : 'colour shift switched off']);
    if (hit && (m.mat === 9 || m.mat === 10 || m.mat === 4)) {
      var tau = Math.sqrt(1 - rsC / r) * (S.T - (fx.delay ? m.ct / q.c : 0));
      rows.push(['Its clock read, when the light left', A.clock(tau) + ' (yours now: ' + A.clock(S.tau) + ')']);
    }
    $('probe-info').innerHTML = rows.map(function (rw) { return '<div><dt>' + rw[0] + '</dt><dd>' + rw[1] + '</dd></div>'; }).join('');
    drawPath();
    drawSpectrum(nm0, nm, lamp);
  }

  // A slice through the planet containing you, its centre and the ray (light paths here stay in one plane).
  function drawPath() {
    if (!last) return;
    var cv = $('probe-path'), f = A.fit(cv, 2), ctx = cv.getContext('2d'), w = f.w, h = f.h;
    ctx.setTransform(f.dpr, 0, 0, f.dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    var er = last.eye.map(function (c) { return c / last.re; }), nn = last.n, nr = G.dot(nn, er);
    var et = [nn[0] - nr * er[0], nn[1] - nr * er[1], nn[2] - nr * er[2]], tl = G.len(et);
    if (tl < 1e-6) { var fr = G.frame(0, 0).along; et = fr.slice(); tl = 1; }
    et = et.map(function (c) { return c / tl; });
    var pts = last.m.path.map(function (p) { return [G.dot(p, et), G.dot(p, er)]; });
    var eye = [0, last.re], span = 3 * G.R;
    pts = pts.filter(function (p) { return Math.hypot(p[0] - eye[0], p[1] - eye[1]) < span; });
    var xs = pts.map(function (p) { return p[0]; }).concat([eye[0]]), ys = pts.map(function (p) { return p[1]; }).concat([eye[1]]);
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs), y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    var size = Math.max(x1 - x0, (y1 - y0) * w / h, 12), cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, sc = 0.86 * w / size;
    function P(x, y) { return [w / 2 + (x - cx) * sc, h / 2 - (y - cy) * sc]; }
    var ink = A.css('--ink'), muted = A.css('--muted'), line = A.css('--line'), sunk = A.css('--sunk'), accent = A.css('--c');
    var c0 = P(0, 0);
    ctx.beginPath(); ctx.arc(c0[0], c0[1], G.R * sc, 0, 6.2832); ctx.fillStyle = sunk; ctx.fill(); ctx.strokeStyle = muted; ctx.lineWidth = 1.5; ctx.stroke();
    if (last.rs > 0) {
      ctx.setLineDash([4, 4]); ctx.strokeStyle = A.css('--g');
      ctx.beginPath(); ctx.arc(c0[0], c0[1], 1.5 * last.rs * sc, 0, 6.2832); ctx.stroke(); ctx.setLineDash([]);
    }
    var e = P(eye[0], eye[1]), straight = Math.max(size, 20), sd = [G.dot(nn, et), G.dot(nn, er)];
    ctx.setLineDash([5, 4]); ctx.strokeStyle = line; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(e[0], e[1]); var se = P(eye[0] + sd[0] * straight, eye[1] + sd[1] * straight); ctx.lineTo(se[0], se[1]); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = accent; ctx.lineWidth = 2.5; ctx.beginPath();
    pts.forEach(function (p, i) { var s = P(p[0], p[1]); if (i) ctx.lineTo(s[0], s[1]); else ctx.moveTo(s[0], s[1]); });
    ctx.stroke();
    ctx.fillStyle = ink; ctx.beginPath(); ctx.arc(e[0], e[1], 4, 0, 6.2832); ctx.fill();
    ctx.font = '12px ' + A.css('--font-text'); ctx.textBaseline = 'bottom'; ctx.textAlign = 'left';
    ctx.fillText('you', e[0] + 6, e[1] - 4);
    if (last.hit && pts.length) {
      var hp = P(pts[pts.length - 1][0], pts[pts.length - 1][1]);
      ctx.strokeStyle = ink; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(hp[0], hp[1], 5, 0, 6.2832); ctx.stroke();
    }
    ctx.fillStyle = muted; ctx.textBaseline = 'top';
    ctx.fillText('slice through the planet · ' + A.length(size, 2) + ' across', 6, 6);
    ctx.textBaseline = 'bottom';
    ctx.fillText('solid: the light’s path · dashed: a straight line' + (last.rs > 0 ? ' · green: photon orbit' : ''), 6, h - 4);
  }

  function drawSpectrum(nm0, nm, lamp) {
    var cv = $('probe-spec'), f = A.fit(cv, 2), ctx = cv.getContext('2d'), w = f.w, h = f.h, lo = 380, hi = 760;
    ctx.setTransform(f.dpr, 0, 0, f.dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    var x0 = 14, x1 = w - 14, by = 18, bh = 16;
    for (var x = x0; x < x1; x++) { ctx.fillStyle = rgbOf(lo + (hi - lo) * (x - x0) / (x1 - x0)); ctx.fillRect(x, by, 1, bh); }
    function X(v) { return x0 + (x1 - x0) * (v - lo) / (hi - lo); }
    var ink = A.css('--ink'), muted = A.css('--muted');
    ctx.font = '12px ' + A.css('--font-text'); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = muted;
    [400, 500, 600, 700].forEach(function (v) { ctx.fillText(v + ' nm', X(v), by + bh + 4); });
    ctx.strokeStyle = muted; ctx.lineWidth = 2; ctx.setLineDash([3, 2]);
    ctx.beginPath(); ctx.moveTo(X(nm0), by - 4); ctx.lineTo(X(nm0), by + bh + 2); ctx.stroke(); ctx.setLineDash([]);
    ctx.textBaseline = 'bottom'; ctx.fillText(lamp ? 'sodium lamp sends 589' : 'green 550 sent', X(nm0), by - 4);
    ctx.fillStyle = ink; ctx.strokeStyle = ink; ctx.lineWidth = 3;
    var xr = X(Math.max(lo, Math.min(hi, nm)));
    ctx.beginPath(); ctx.moveTo(xr, by - 2); ctx.lineTo(xr, by + bh + 2); ctx.stroke();
    ctx.textBaseline = 'top';
    var label = 'you see ' + A.num(nm, 3) + ' nm' + (nm < lo ? ' ← ultraviolet' : nm > hi ? ' infrared →' : '');
    ctx.textAlign = xr < w / 3 ? 'left' : xr > 2 * w / 3 ? 'right' : 'center';
    ctx.fillText(label, xr, by + bh + 18);
  }

  CGH.probe = { update: update };
})();
