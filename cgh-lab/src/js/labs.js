/* cGħ World — three small labs beside the world.
 *   Orbits      a ball thrown round the planet: Newton's ellipse against Schwarzschild's rosette
 *   Hover       your height as a quantum wave above the floor, under the planet's gravity
 *   Staircases  your wave climbs two staircases and rejoins; your watch records which way you went
 * The physics functions have no DOM and are tested in Node; the drawing below them uses the page.
 */
(function (root) {
  'use strict';

  /* ---------- physics ---------- */
  // Timelike orbit in the Schwarzschild field, by proper time:  x'' = -(GM/r³)(1 + 3L²/(c²r²)) x,  L = |x × x'|.
  // With rel = false it is Newton's  x'' = -GM x / r³. Returns the path and the angles of each closest approach.
  function orbit(GM, c, r0, v0, rel, opts) {
    opts = opts || {};
    var x = [r0, 0], v = [0, v0], L = r0 * v0, rmin = opts.floor || 0, path = [x.slice()], peri = [], turns = opts.turns || 4;
    var k3 = rel ? 3 * L * L / (c * c) : 0, dt = opts.dt || 2e-3 * Math.pow(r0, 1.5) / Math.sqrt(GM), phi = 0, lastPhi = 0, prevR = r0, prev2 = r0;
    function acc(p) { var r = Math.hypot(p[0], p[1]), f = -GM / (r * r * r) * (1 + k3 / (r * r)); return [f * p[0], f * p[1]]; }
    var a = acc(x), crashed = false;
    for (var i = 0; i < (opts.steps || 400000); i++) {
      x = [x[0] + v[0] * dt + 0.5 * a[0] * dt * dt, x[1] + v[1] * dt + 0.5 * a[1] * dt * dt];
      var an = acc(x);
      v = [v[0] + 0.5 * (a[0] + an[0]) * dt, v[1] + 0.5 * (a[1] + an[1]) * dt]; a = an;
      var r = Math.hypot(x[0], x[1]), ang = Math.atan2(x[1], x[0]);
      var d = ang - lastPhi; if (d < -Math.PI) d += 2 * Math.PI; if (d > Math.PI) d -= 2 * Math.PI;
      phi += d; lastPhi = ang;
      if (i % 20 === 0) path.push(x.slice());
      if (prevR < prev2 && prevR <= r) peri.push(phi - d / 2);         // passed a closest approach
      prev2 = prevR; prevR = r;
      if (r < rmin) { crashed = true; break; }
      if (phi > 2 * Math.PI * turns + 0.5) break;
    }
    var precess = peri.length > 1 ? (peri[peri.length - 1] - peri[0]) / (peri.length - 1) - 2 * Math.PI : 0;
    return { path: path, peri: peri, precess: precess, crashed: crashed };
  }
  // Speed of a circular orbit (proper speed dx/dτ for the Schwarzschild case).
  function circularSpeed(GM, c, r, rel) { return rel ? Math.sqrt(GM / (r * (1 - 3 * GM / (c * c * r)))) : Math.sqrt(GM / r); }

  // Two staircases: proper-time difference, the phase that shifts the stripes, and what survives of them.
  // fTop, fBottom: g_tt factors at the two heights; T: far-away time spent apart; m, hbar, c; tick: clock frequency (Hz).
  function staircases(fTop, fBottom, T, m, hbar, c, tick) {
    var dtau = T * (Math.sqrt(fTop) - Math.sqrt(fBottom));
    var w = 2 * Math.PI * tick;
    return {
      dtau: dtau,
      phase: (m * c * c * dtau / hbar) % (2 * Math.PI),        // rest energy × proper-time difference
      V: Math.abs(Math.cos(w * dtau / 2)),                     // overlap of the two clock readings
      D: Math.abs(Math.sin(w * dtau / 2))                      // how well the clock tells the paths apart
    };
  }

  var api = { orbit: orbit, circularSpeed: circularSpeed, staircases: staircases };
  root.CGH = root.CGH || {};
  root.CGH.labs = api;
  if (typeof module !== 'undefined' && module.exports) { module.exports = api; return; }

  /* ---------- the page ---------- */
  var G = CGH.geo, A = CGH.app, QC = CGH.qcore, $ = A.$;
  var tab = 'orbits', O = { last: null, thrown: 0 }, H = { sim: null, on: false, settled: 0, z: 0, key: '' }, T2 = { revived: false, faded: false };

  function dial() { var d = A.derived(); return { GM: d.rs * d.c * d.c / 2, c: d.c, rs: d.rs, hbar: d.hbar, lam: d.lam }; }
  function surfaceG() { var q = dial(); return q.GM / (G.R * G.R * Math.sqrt(Math.max(1e-9, 1 - q.rs / G.R))); }

  function show(name) {
    tab = name;
    ['orbits', 'hover', 'stairs'].forEach(function (t) {
      $('lab-' + t).hidden = t !== name;
      $('labtab-' + t).setAttribute('aria-pressed', t === name ? 'true' : 'false');
    });
    draw();
  }

  /* Orbits */
  function throwBall() {
    var q = dial(), r0 = G.R + Number($('orb-h').value), k = Number($('orb-k').value);
    if (q.GM <= 0) { $('orb-note').textContent = 'Turn the G dial up: with no gravity nothing orbits.'; return; }
    var vE = k * circularSpeed(q.GM, q.c, r0, true), vN = k * circularSpeed(q.GM, q.c, r0, false);
    var per = 2 * Math.PI * Math.pow(r0, 1.5) / Math.sqrt(q.GM);
    O.last = { E: orbit(q.GM, q.c, r0, vE, true, { floor: G.R, turns: 5, dt: per / 4000 }), N: orbit(q.GM, q.c, r0, vN, false, { floor: G.R, turns: 5, dt: per / 4000 }), rs: q.rs, r0: r0 };
    O.thrown++;
    var E = O.last.E, p = E.path, rp = Infinity, ra = 0;
    p.forEach(function (pt) { var r = Math.hypot(pt[0], pt[1]); rp = Math.min(rp, r); ra = Math.max(ra, r); });
    var semi = 2 * rp * ra / (rp + ra), formula = 3 * Math.PI * q.rs / semi;
    $('orb-note').textContent = E.crashed ? 'Einstein’s ball spiralled into the ground: close to a compact planet, orbits need more than Newton’s speed.' :
      'Einstein’s orbit turns by ' + A.num(E.precess * 180 / Math.PI, 3) + '° each time round (weak-field formula 3π rs / p: ' + A.num(formula * 180 / Math.PI, 3) + '°). Newton’s returns to the same place.';
    draw();
  }
  function drawOrbits() {
    var cv = $('orb-canvas'), f = A.fit(cv, 2), ctx = cv.getContext('2d'), w = f.w, h = f.h;
    ctx.setTransform(f.dpr, 0, 0, f.dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    var span = O.last ? Math.max.apply(null, O.last.E.path.concat(O.last.N.path).map(function (p) { return Math.hypot(p[0], p[1]); })) : 2 * G.R;
    var sc = 0.46 * Math.min(w, h) / Math.max(span, G.R * 1.1), cx = w / 2, cy = h / 2;
    ctx.beginPath(); ctx.arc(cx, cy, G.R * sc, 0, 6.2832); ctx.fillStyle = A.css('--sunk'); ctx.fill(); ctx.strokeStyle = A.css('--muted'); ctx.stroke();
    if (!O.last) { ctx.fillStyle = A.css('--muted'); ctx.font = '13px ' + A.css('--font-text'); ctx.textAlign = 'center'; ctx.fillText('Throw a ball to see its orbit', cx, cy); return; }
    function line(path, color, dash) {
      ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.setLineDash(dash || []); ctx.beginPath();
      path.forEach(function (p, i) { var X = cx + p[0] * sc, Y = cy - p[1] * sc; if (i) ctx.lineTo(X, Y); else ctx.moveTo(X, Y); });
      ctx.stroke(); ctx.setLineDash([]);
    }
    line(O.last.N.path, A.css('--muted'), [4, 4]);
    line(O.last.E.path, A.css('--g'));
    ctx.fillStyle = A.css('--muted'); ctx.font = '12px ' + A.css('--font-text'); ctx.textAlign = 'left';
    ctx.fillText('dashed: Newton · solid: Einstein · seen from above the orbit', 8, h - 8);
  }

  /* Hover */
  function hoverStart(mode) {
    var q = dial(), g = surfaceG(), m = A.MASS;
    if (!(A.state.lam >= 0.5) || g < 0.05) { $('hov-note').textContent = 'Needs you to be a wave (third dial at 0.5 m or more) and real gravity (try the Hover mission).'; return; }
    var b = CGH.physics.bouncer(m, g, q.hbar), L = Math.max(4, 3 * b.heights[5] + (mode === 'drop' ? 2.6 : 0)), N = 512;
    H.sim = new QC.Sim1D({ N: N, L: L, m: m, hbar: q.hbar });
    H.sim.build(function (z) { return { V: m * g * z, wall: z < 0.02 }; });
    if (mode === 'drop') H.sim.set(function (z) { return [Math.exp(-(z - 2) * (z - 2) / (4 * 0.15 * 0.15)), 0]; });
    else {
      var zs = []; for (var i = 0; i < N; i++) zs.push(H.sim.z(i) / b.z0 - CGH.physics.AIRY_ZEROS[0]);
      var ai = CGH.qcore.airyShape(zs);
      H.sim.set(function (z) { var k = Math.round(z / H.sim.dz - 0.5); return [z > 0.02 ? ai[k] : 0, 0]; });
    }
    H.b = b; H.g = g; H.on = true; H.mode = mode; H.settled = 0; H.dt = Math.min(0.02 * q.hbar / (m * g * Math.max(2, b.heights[0])), 2e-3);
    $('hov-note').textContent = mode === 'drop' ? 'Dropped from 2 m: watch it bounce and smear.' : 'The lowest state: ' + A.length(2 / 3 * b.heights[0], 2) + ' above the floor on average, and it stays there.';
  }
  function hoverStep(dt) {
    if (!H.on || !H.sim) return 0;
    var n = Math.min(400, Math.ceil(dt / H.dt));
    for (var i = 0; i < n; i++) H.sim.step(dt / n);
    if (H.mode === 'settle') H.settled += dt;
    // where your feet are, drawn from |ψ(z)|²
    var re = H.sim.re, im = H.sim.im, tot = 0, q;
    for (q = 0; q < re.length; q++) tot += re[q] * re[q] + im[q] * im[q];
    var t = Math.random() * tot, acc = 0;
    for (q = 0; q < re.length; q++) { acc += re[q] * re[q] + im[q] * im[q]; if (acc >= t) break; }
    H.z = H.sim.z(Math.min(q, re.length - 1));
    return H.z;
  }
  function drawHover() {
    var cv = $('hov-canvas'), f = A.fit(cv, 2), ctx = cv.getContext('2d'), w = f.w, h = f.h;
    ctx.setTransform(f.dpr, 0, 0, f.dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    var floorY = h - 22, top = 10, ink = A.css('--ink'), muted = A.css('--muted');
    ctx.fillStyle = A.css('--sunk'); ctx.fillRect(0, floorY, w, h - floorY);
    if (!H.sim) { ctx.fillStyle = muted; ctx.font = '13px ' + A.css('--font-text'); ctx.textAlign = 'center'; ctx.fillText('Drop yourself, or settle into the lowest state', w / 2, h / 2); return; }
    var L = H.sim.L, sc = (floorY - top) / L, re = H.sim.re, im = H.sim.im, mx = 0, i;
    for (i = 0; i < re.length; i++) mx = Math.max(mx, re[i] * re[i] + im[i] * im[i]);
    ctx.fillStyle = A.css('--h');
    for (i = 0; i < re.length; i++) { var p = (re[i] * re[i] + im[i] * im[i]) / (mx || 1), y = floorY - H.sim.z(i) * sc; ctx.fillRect(w * 0.18, y - Math.max(1, sc * H.sim.dz), w * 0.5 * p, Math.max(1, sc * H.sim.dz) + 0.5); }
    ctx.font = '12px ' + A.css('--font-text'); ctx.textAlign = 'left';
    H.b.heights.forEach(function (z, k) {
      var y = floorY - z * sc; if (y < top) return;
      ctx.strokeStyle = muted; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(w * 0.15, y); ctx.lineTo(w * 0.75, y); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = muted; ctx.fillText('level ' + (k + 1) + ': ' + A.length(z, 2), w * 0.76, y + 4);
    });
    ctx.fillStyle = ink; ctx.fillText('your feet, as a wave (g = ' + A.num(H.g, 3) + ' m/s²)', 8, top + 4);
    ctx.fillStyle = muted; ctx.fillText('floor', 8, floorY + 15);
  }

  /* Staircases */
  function stairs() {
    var q = dial(), dh = Number($('st-h').value), Tup = Number($('st-t').value), tick = Number($('st-f').value);
    var f = function (r) { return 1 - q.rs / r - q.lam * r * r; };
    return { r: staircases(f(G.R + dh), f(G.R), Tup, A.MASS, q.hbar, q.c, tick), dh: dh, T: Tup, tick: tick };
  }
  function drawStairs() {
    var s = stairs(), cv = $('st-canvas'), f = A.fit(cv, 2), ctx = cv.getContext('2d'), w = f.w, h = f.h;
    ctx.setTransform(f.dpr, 0, 0, f.dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    var muted = A.css('--muted'), base = h - 24, amp = (h - 50) / 2, mid = base - amp;
    function curve(V, ph, color, dash) {
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dash || []); ctx.beginPath();
      for (var x = 0; x <= w; x += 2) { var I = 1 + V * Math.cos(2 * Math.PI * 3 * x / w + ph); var y = base - I * amp; if (x) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
      ctx.stroke(); ctx.setLineDash([]);
    }
    curve(1, 0, muted, [4, 4]);
    curve(1, s.r.phase, A.css('--g'), [2, 3]);
    curve(s.r.V, s.r.phase + Math.PI * s.tick * s.r.dtau, A.css('--h'));
    ctx.fillStyle = muted; ctx.font = '12px ' + A.css('--font-text'); ctx.textAlign = 'left';
    ctx.fillText('dashed grey: no height difference · dotted: no clock · solid: carrying your watch', 8, h - 6);
    void mid;
    if (s.r.V < 0.1 && Math.abs(s.r.dtau) > 0) T2.faded = true;
    if (T2.faded && s.r.V > 0.9 && Math.abs(s.r.dtau) * s.tick > 0.5) T2.revived = true;
    $('st-read').innerHTML = '<div><dt>Your watches differ by</dt><dd>' + A.duration(Math.abs(s.r.dtau), 3) + ' (' + A.num(Math.abs(s.r.dtau) * s.tick, 3) + ' ticks)</dd></div>' +
      '<div><dt>Stripes shift by</dt><dd>' + A.num(s.r.phase, 3) + ' rad</dd></div>' +
      '<div><dt>Stripe strength V</dt><dd>' + s.r.V.toFixed(3) + '</dd></div>' +
      '<div><dt>Which-path record D</dt><dd>' + s.r.D.toFixed(3) + ' (D² + V² = ' + (s.r.D * s.r.D + s.r.V * s.r.V).toFixed(3) + ')</dd></div>';
    $('st-h-out').textContent = s.dh + ' m'; $('st-t-out').textContent = s.T + ' s'; $('st-f-out').textContent = s.tick + ' per s';
  }

  function draw() {
    if (tab === 'orbits') drawOrbits(); else if (tab === 'hover') drawHover(); else drawStairs();
  }
  CGH.labs.init = function () {
    ['orbits', 'hover', 'stairs'].forEach(function (t) { $('labtab-' + t).addEventListener('click', function () { show(t); }); });
    $('orb-throw').addEventListener('click', throwBall);
    ['orb-h', 'orb-k'].forEach(function (id) { $(id).addEventListener('input', function () { $(id + '-out').textContent = id === 'orb-h' ? $(id).value + ' m' : '×' + Number($(id).value).toFixed(2); }); });
    $('hov-drop').addEventListener('click', function () { hoverStart('drop'); });
    $('hov-settle').addEventListener('click', function () { hoverStart('settle'); });
    $('hov-stop').addEventListener('click', function () { H.on = false; H.z = 0; $('hov-note').textContent = 'Stopped: you stand on the floor again.'; });
    ['st-h', 'st-t', 'st-f'].forEach(function (id) { $(id).addEventListener('input', drawStairs); });
    A.onChange(function () { H.on = false; H.z = 0; draw(); });
    show('orbits');
    setInterval(function () { if (tab === 'hover' && H.on) drawHover(); }, 120);
  };
  CGH.labs.hoverZ = function () { return H.on ? H.z : 0; };
  CGH.labs.step = hoverStep;
  CGH.labs.state = { O: O, H: H, T2: T2, show: show, throwBall: throwBall, hoverStart: hoverStart, stairs: stairs };
})(typeof window !== 'undefined' ? window : globalThis);
