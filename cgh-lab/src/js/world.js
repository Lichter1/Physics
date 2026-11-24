/* cGħ World — the first-person engine: walking, the camera, three GPU passes and the head-up display. */
(function () {
  'use strict';
  var G = CGH.geo, P = CGH.physics, A = CGH.app, SH = CGH.shaders, WV = CGH.wave, $ = A.$;
  var EFFORT_MAX = 20, FOV = 1.2, CLOCK_TOP = 46;
  var canvas, gl, prog = {}, fmt, scene = null, acc = [], size = [0, 0], vao, raf = 0, last = 0, ok = false;
  var S = {
    a: G.START.a, l: G.START.l, heading: 0, lookYaw: 0, lookPitch: 0.04, effort: 0, T: 0, tau: 0,
    street: 0, top: 0, newton: false, doppler: true, beam: true, exposure: 1, keys: {}, scale: 0.6, fixedScale: 0, ms: 16,
    view: { a: G.START.a, l: G.START.l }, frame: 0, fresh: true, watched: false, lookTimer: 0, hint: 12, found: null
  };

  /* ---------- GL plumbing ---------- */
  function compile(type, src) {
    var s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  function program(fs, names) {
    var p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, SH.VS)); gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    var u = {}; names.forEach(function (n) { u[n] = gl.getUniformLocation(p, n); });
    return { p: p, u: u };
  }
  function target(w, h) {
    var t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, fmt.internal, w, h, 0, gl.RGBA, fmt.type, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    var f = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
    return { t: t, f: f };
  }
  function free(tg) { if (tg) { gl.deleteTexture(tg.t); gl.deleteFramebuffer(tg.f); } }
  function setup() {
    canvas = $('world-canvas');
    gl = canvas.getContext('webgl2', { antialias: false, alpha: false });
    if (!gl) return false;
    fmt = gl.getExtension('EXT_color_buffer_float') ? { internal: gl.RGBA16F, type: gl.HALF_FLOAT } : { internal: gl.RGBA8, type: gl.UNSIGNED_BYTE };
    prog.march = program(SH.MARCH, ['uEye', 'uFwd', 'uRight', 'uUp', 'uBeta', 'uTan', 'uRs', 'uC', 'uT', 'uExposure', 'uDoppler', 'uBeam']);
    prog.accum = program(SH.ACCUM, ['uNew', 'uOld', 'uK']);
    prog.show = program(SH.SHOW, ['uImg', 'uPhotons', 'uSeed']);
    vao = gl.createVertexArray();
    return true;
  }

  /* ---------- moving ---------- */
  function dials() {
    var d = A.derived(), st = A.state;
    return { c: S.newton ? 1e12 : d.c, rs: S.newton ? 0 : d.rs, hbar: d.hbar, quantum: st.lam >= 0.5, photons: st.photons };
  }
  function update(dtau) {
    var k = S.keys, q = dials(), re = G.R + G.EYE;
    if (k.faster) { S.effort = Math.min(EFFORT_MAX, S.effort + 6 * dtau); syncEffort(); }
    if (k.slower) { S.effort = Math.max(0, S.effort - 8 * dtau); syncEffort(); }
    if (k.left) S.heading -= 1.3 * dtau;
    if (k.right) S.heading += 1.3 * dtau;
    var v = P.speedFromEffort(S.effort, q.c), va = v * Math.cos(S.heading), vl = v * Math.sin(S.heading);
    q.lapse = Math.sqrt(1 - q.rs / re);
    if (q.quantum) {
      var W = WV.W, c = S.newton ? 0 : q.c;
      if (!W.on || W.hbar !== q.hbar || W.c !== c) { WV.start(S.a, S.l, va, vl, q.hbar, c, 0.6); S.fresh = true; }
      var top = WV.topSpeed();
      if (v > top) { va *= top / v; vl *= top / v; }
      q.top = top;
    } else if (WV.W.on) { WV.stop(); S.fresh = true; }
    // your own velocity relative to the ground decides the clocks and the aberration
    var vel = q.quantum ? WV.W.vel || [0, 0] : [va, vl];
    var sp = Math.hypot(vel[0], vel[1]);
    q.v = sp; q.beta = sp / q.c; q.gamma = 1 / Math.sqrt(1 - q.beta * q.beta); q.vel = vel;
    var dtLocal = q.gamma * dtau;                      // time for someone standing where you are
    var dT = dtLocal / q.lapse;                         // far-away (coordinate) time
    S.T += dT; S.tau += dtau;
    S.street += dT * Math.sqrt(1 - q.rs / G.R); S.top += dT * Math.sqrt(1 - q.rs / (G.R + CLOCK_TOP));
    if (q.quantum) {
      WV.step(dtLocal, va, vl);
      if (S.watched) { S.lookTimer -= dtau; if (S.lookTimer <= 0) { S.lookTimer = 0.25; WV.look(0.5, Math.random); } }
      S.a = WV.W.mean.a; S.l = WV.W.mean.l;
      S.view = WV.sample(Math.random);
    } else {
      var na = G.wrapA(S.a + va * dtLocal), nl = S.l + vl * dtLocal;
      if (G.footprint(na, nl) > 0.25) { S.a = na; S.l = nl; }
      else if (G.footprint(na, S.l) > 0.25) S.a = na;
      else if (G.footprint(S.a, nl) > 0.25) S.l = nl;
      S.view = { a: S.a, l: S.l };
    }
    if (S.hint > 0) S.hint -= dtau;
    return q;
  }
  function camera(q) {
    var f = G.frame(S.view.a, S.view.l), y = S.heading + S.lookYaw, p = S.lookPitch;
    var cy = Math.cos(y), sy = Math.sin(y), cp = Math.cos(p), sp = Math.sin(p), fwd = [], right = [], beta = [];
    for (var i = 0; i < 3; i++) {
      fwd.push(cp * (cy * f.along[i] + sy * f.across[i]) + sp * f.up[i]);
      right.push(cy * f.across[i] - sy * f.along[i]);
      beta.push(S.newton ? 0 : (q.vel[0] * f.along[i] + q.vel[1] * f.across[i]) / q.c);
    }
    return { eye: G.toWorld(S.view.a, S.view.l, G.EYE), fwd: fwd, right: right, up: G.cross(right, fwd), beta: beta };
  }

  /* ---------- drawing ---------- */
  function draw(q, cam) {
    var f = A.fit(canvas, 1.5), w = canvas.width, h = canvas.height;
    var sc = S.fixedScale || S.scale, rw = Math.max(64, Math.round(w * sc)), rh = Math.max(36, Math.round(h * sc));
    if (rw !== size[0] || rh !== size[1]) {
      free(scene); free(acc[0]); free(acc[1]);
      scene = target(rw, rh); acc = [target(rw, rh), target(rw, rh)]; size = [rw, rh]; S.fresh = true;
    }
    var Dc = 1 / (q.gamma * (1 - G.dot(cam.fwd, cam.beta)));
    var aim = S.beam && !S.newton ? Math.max(0.02, Math.min(60, Math.pow(Dc, -2.7))) : 1;
    S.exposure += (aim - S.exposure) * 0.15;
    gl.bindVertexArray(vao);

    var m = prog.march;
    gl.bindFramebuffer(gl.FRAMEBUFFER, scene.f); gl.viewport(0, 0, rw, rh); gl.useProgram(m.p);
    gl.uniform3fv(m.u.uEye, cam.eye); gl.uniform3fv(m.u.uFwd, cam.fwd); gl.uniform3fv(m.u.uRight, cam.right);
    gl.uniform3fv(m.u.uUp, cam.up); gl.uniform3fv(m.u.uBeta, cam.beta);
    gl.uniform2f(m.u.uTan, Math.tan(FOV / 2) * f.w / f.h, Math.tan(FOV / 2));
    gl.uniform1f(m.u.uRs, q.rs); gl.uniform1f(m.u.uC, q.c); gl.uniform1f(m.u.uT, S.T % 1e5);
    gl.uniform1f(m.u.uExposure, S.exposure);
    gl.uniform1i(m.u.uDoppler, S.doppler && !S.newton ? 1 : 0); gl.uniform1i(m.u.uBeam, S.beam && !S.newton ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    var k = S.fresh ? 1 : q.quantum ? (WV.W.spread > 0.12 ? 0.08 : 0.35) : 1;
    var src = acc[S.frame % 2], dst = acc[(S.frame + 1) % 2], a = prog.accum;
    gl.bindFramebuffer(gl.FRAMEBUFFER, dst.f); gl.useProgram(a.p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, scene.t);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, src.t);
    gl.uniform1i(a.u.uNew, 0); gl.uniform1i(a.u.uOld, 1); gl.uniform1f(a.u.uK, k);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    var s = prog.show;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, w, h); gl.useProgram(s.p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, dst.t);
    gl.uniform1i(s.u.uImg, 0); gl.uniform1f(s.u.uPhotons, q.photons); gl.uniform1ui(s.u.uSeed, S.frame & 0xffff);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    S.frame++; S.fresh = false;
  }

  function waveMap(q) {
    var el = $('wave-map');
    el.hidden = !q.quantum;
    if (!q.quantum || !WV.W.sim) return;
    var N = WV.N, sim = WV.W.sim, ctx = el.getContext('2d');
    if (el.width !== N) { el.width = N; el.height = N; }
    var img = ctx.createImageData(N, N), d = img.data, mx = 0, n, re = sim.re, im = sim.im;
    for (n = 0; n < N * N; n++) mx = Math.max(mx, re[n] * re[n] + im[n] * im[n]);
    for (n = 0; n < N * N; n++) {
      var p = Math.sqrt((re[n] * re[n] + im[n] * im[n]) / (mx || 1)), o = 4 * n;
      if (sim.wall[n]) { d[o] = d[o + 1] = d[o + 2] = 150; } else { d[o] = 14 + 218 * p; d[o + 1] = 18 + 102 * p; d[o + 2] = 26 + 168 * p; }
      d[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    var x = (S.view.l - WV.W.cl) / WV.dx + N / 2, y = N / 2 - G.wrapA(S.view.a - WV.W.ca) / WV.dx;
    ctx.fillStyle = '#ffd27a'; ctx.fillRect(x - 1.5, y - 1.5, 3, 3);
    ctx.strokeStyle = '#ffd27a'; ctx.beginPath(); ctx.moveTo(x, y);
    ctx.lineTo(x + 9 * Math.sin(S.heading + S.lookYaw), y - 9 * Math.cos(S.heading + S.lookYaw)); ctx.stroke();
  }

  function hud(q) {
    var n = A.num, rs = q.rs, re = G.R + G.EYE;
    $('w-hud-l').innerHTML = 'speed <b>' + n(q.v) + ' m/s</b> = ' + (q.beta < 1e-3 ? n(q.beta, 2) : q.beta.toFixed(3)) + ' c<br>γ = ' + (q.gamma < 1.0001 ? '1.000' : q.gamma.toFixed(3));
    $('w-hud-r').innerHTML = 'your watch <b>' + A.clock(S.tau) + '</b><br>street clocks <b>' + A.clock(S.street) + '</b><br>tower top <b>' + A.clock(S.top) + '</b>';
    var msg = S.newton ? 'Newton’s view: light instant and straight' : '';
    if (!msg && q.quantum) msg = WV.W.spread > 0.5 ? 'You are spread over ' + n(WV.W.spread, 2) + ' m. Every place you might be is overlaid. Press F to look.' : 'You are a narrow wave. Walk, and watch the map.';
    if (!msg) msg = 'Drag to look around';
    if (S.hint > 0 && canvas.clientWidth > 640) msg += '<br>W S effort · A D steer · F look · Space stop';
    $('w-hud-b').innerHTML = msg;

    var el = G.horizon(rs, G.R, re) * 180 / Math.PI, rows = [];
    rows.push([el < 0 ? 'Horizon, below your eye level' : 'Horizon, above your eye level', Math.abs(el).toFixed(1) + '°' + (el > 0 ? ' (sky is a cone ' + (2 * (90 - el)).toFixed(0) + '° wide)' : '')]);
    rows.push(['Tower-top clock runs, against yours', '×' + G.clockRate(rs, CLOCK_TOP).toFixed(rs > 0.01 * G.R ? 3 : 6)]);
    rows.push(['Light crosses the planet in', A.duration(2 * G.R / q.c * (1 + rs / G.R), 2)]);
    if (q.quantum) rows.push(['Your spread / top speed here', n(WV.W.spread, 2) + ' m / ' + n(q.top, 2) + ' m/s']);
    else rows.push(['Your wavelength now', A.length(2 * Math.PI * q.hbar / (A.MASS * Math.max(q.v, A.WALK) * q.gamma), 2)]);
    rows.push(['Photons per pixel per frame', n(q.photons, 2)]);
    $('w-readouts').innerHTML = rows.map(function (r) { return '<div class="readout"><dt>' + r[0] + '</dt><dd>' + r[1] + '</dd></div>'; }).join('');
  }

  var hudT = 0;
  function frame(now) {
    raf = requestAnimationFrame(frame);
    var dt = Math.min(0.05, (now - last) / 1000 || 0);
    S.ms += ((now - last) - S.ms) * 0.05; last = now;
    if (!S.fixedScale && S.frame % 30 === 29) {
      if (S.ms > 42 && S.scale > 0.3) S.scale *= 0.85; else if (S.ms < 22 && S.scale < 1) S.scale = Math.min(1, S.scale * 1.1);
    }
    var q = update(dt), cam = camera(q);
    draw(q, cam);
    hudT += dt;
    if (hudT > 0.12) { hudT = 0; hud(q); waveMap(q); }
  }

  /* ---------- input ---------- */
  function syncEffort() { $('effort').value = S.effort; $('effort-out').textContent = S.effort.toFixed(1) + ' m/s'; }
  function look() {
    if (!WV.W.on) return;
    S.found = WV.look(0.45, Math.random); S.fresh = true;
  }
  function restartClocks() { S.T = 0; S.tau = 0; S.street = 0; S.top = 0; }
  function place(a, l, heading) {
    S.a = a; S.l = l; S.view = { a: a, l: l }; S.heading = heading; S.lookYaw = 0; S.lookPitch = 0.04;
    S.effort = 0; syncEffort(); WV.stop(); restartClocks(); S.fresh = true;
  }
  function bind() {
    $('effort').addEventListener('input', function (e) { S.effort = Number(e.target.value); syncEffort(); });
    syncEffort();
    $('w-stop').addEventListener('click', function () { S.effort = 0; syncEffort(); });
    $('w-turn').addEventListener('click', function () { S.heading += Math.PI; S.lookYaw = 0; });
    $('w-home').addEventListener('click', function () { place(G.START.a, G.START.l, 0); });
    document.addEventListener('cgh-pose', function (e) { if (e.detail === 'doors') place(G.DOORS.a, G.DOORS.l, G.DOORS.heading); });
    $('w-look').addEventListener('click', look);
    $('w-watched').addEventListener('change', function (e) { S.watched = e.target.checked; });
    $('w-newton').addEventListener('change', function (e) { S.newton = e.target.checked; S.fresh = true; });
    $('w-doppler').addEventListener('change', function (e) { S.doppler = e.target.checked; });
    $('w-beam').addEventListener('change', function (e) { S.beam = e.target.checked; });

    var KEYS = { w: 'faster', ArrowUp: 'faster', s: 'slower', ArrowDown: 'slower', a: 'left', ArrowLeft: 'left', d: 'right', ArrowRight: 'right' };
    function key(ev, down) {
      if (!ev.key) return;
      var t = ev.target, onCanvas = t === canvas;
      if (!onCanvas && t && /^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(t.tagName)) return;
      if (down && (ev.key === 'f' || ev.key === 'F')) { look(); return; }
      if (down && ev.key === ' ' && onCanvas) { S.effort = 0; syncEffort(); ev.preventDefault(); return; }
      var act = KEYS[ev.key.length === 1 ? ev.key.toLowerCase() : ev.key];
      if (!act) return;
      S.keys[act] = down;
      if (onCanvas) ev.preventDefault();
    }
    window.addEventListener('keydown', function (e) { key(e, true); });
    window.addEventListener('keyup', function (e) { key(e, false); });
    window.addEventListener('blur', function () { S.keys = {}; });
    var drag = null;
    canvas.addEventListener('pointerdown', function (e) { drag = [e.clientX, e.clientY]; try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* fine */ } canvas.focus({ preventScroll: true }); });
    canvas.addEventListener('pointermove', function (e) {
      if (!drag) return;
      S.lookYaw += (e.clientX - drag[0]) * 0.005;
      S.lookPitch = Math.max(-1.3, Math.min(1.4, S.lookPitch - (e.clientY - drag[1]) * 0.005));
      drag = [e.clientX, e.clientY];
    });
    canvas.addEventListener('pointerup', function () { drag = null; });
    canvas.addEventListener('pointercancel', function () { drag = null; });
    canvas.addEventListener('dblclick', function () { S.lookYaw = 0; S.lookPitch = 0.04; });
    Array.prototype.forEach.call(document.querySelectorAll('#world-view .pad button'), function (b) {
      var act = b.getAttribute('data-act');
      function on(e) { if (act === 'look') look(); else S.keys[act] = true; e.preventDefault(); }
      function off() { S.keys[act] = false; }
      b.addEventListener('pointerdown', on); b.addEventListener('pointerup', off);
      b.addEventListener('pointerleave', off); b.addEventListener('pointercancel', off);
    });
  }

  CGH.world = {
    init: function () {
      canvas = $('world-canvas');
      bind();
      try { ok = setup(); } catch (e) { ok = false; console.error(e); }
      if (!ok) { $('w-hud-b').textContent = 'This view needs WebGL 2, which this browser did not provide.'; return; }
      A.onChange(function () { S.fresh = true; restartClocks(); });   // a new world: every clock starts together
      document.addEventListener('visibilitychange', function () {
        cancelAnimationFrame(raf);
        if (!document.hidden) { last = performance.now(); raf = requestAnimationFrame(frame); }
      });
      last = performance.now(); raf = requestAnimationFrame(frame);
    },
    state: S
  };
})();
