/* cGħ World — the first-person engine: walking and flying, the camera, the GPU passes, the head-up display and the map. */
(function () {
  'use strict';
  var G = CGH.geo, P = CGH.physics, A = CGH.app, SH = CGH.shaders, WV = CGH.wave, $ = A.$;
  var EFFORT_MAX = 20, FOV = 1.2, CLOCK_TOP = 46, CLIMB = 3, TOP_EYE = 80;
  var FX = ['bend', 'delay', 'aberr', 'doppler', 'gshift', 'beam', 'rates'];
  var canvas, gl, prog = {}, fmt, scene = null, acc = [], size = [0, 0], vao, raf = 0, last = 0, ok = false;
  var S = {
    a: G.START.a, l: G.START.l, h: G.EYE, heading: 0, lookYaw: 0, lookPitch: 0.04, effort: 0,
    T: 0, tau: 0, street: 0, top: 0, exposure: 1, keys: {}, scale: 0.6, fixedScale: 0, ms: 16,
    fx: { bend: true, delay: true, aberr: true, doppler: true, gshift: true, beam: true, rates: true },
    view: { a: G.START.a, l: G.START.l }, frame: 0, fresh: true, envRate: 0, envTimer: 0, sharp: 0.5, screen: false, hint: 12, q: null, cam: null
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
    [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER].forEach(function (k) { gl.texParameteri(gl.TEXTURE_2D, k, gl.LINEAR); });
    [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T].forEach(function (k) { gl.texParameteri(gl.TEXTURE_2D, k, gl.CLAMP_TO_EDGE); });
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
    prog.march = program(SH.MARCH, ['uEye', 'uFwd', 'uRight', 'uUp', 'uBeta', 'uTan', 'uRsRay', 'uRsShift', 'uRsClock', 'uCdelay', 'uT',
      'uExposure', 'uDoppler', 'uBeam', 'uAberr', 'uBell']);
    prog.accum = program(SH.ACCUM, ['uNew', 'uOld', 'uK']);
    prog.show = program(SH.SHOW, ['uImg', 'uPhotons', 'uSeed']);
    vao = gl.createVertexArray();
    return true;
  }

  /* ---------- moving ---------- */
  function dials() {
    var d = A.derived(), st = A.state;
    return { c: d.c, rs: d.rs, hbar: d.hbar, quantum: st.lam >= 0.5, photons: st.photons };
  }
  // Is there room for a body with its eyes at (a, l, h)? Checks at eye and knee height.
  function room(a, l, h) {
    return G.scene(G.toWorld(a, l, h)).d > 0.25 && G.scene(G.toWorld(a, l, Math.max(h - 1.2, 0.5))).d > 0.25;
  }
  function update(dtau) {
    var k = S.keys, q = dials();
    if (k.faster) { S.effort = Math.min(EFFORT_MAX, S.effort + 6 * dtau); syncEffort(); }
    if (k.slower) { S.effort = Math.max(0, S.effort - 8 * dtau); syncEffort(); }
    if (k.left) S.heading -= 1.3 * dtau;
    if (k.right) S.heading += 1.3 * dtau;
    if (q.quantum) S.h = G.EYE;                               // the wave lives on the ground
    q.vz = q.quantum ? 0 : CLIMB * ((k.up ? 1 : 0) - (k.down ? 1 : 0));
    if ((S.h <= G.EYE && q.vz < 0) || (S.h >= TOP_EYE && q.vz > 0)) q.vz = 0;
    var v = P.speedFromEffort(S.effort, q.c), va = v * Math.cos(S.heading), vl = v * Math.sin(S.heading);
    q.re = G.R + S.h; q.lapse = Math.sqrt(1 - q.rs / q.re);
    if (q.quantum) {
      var W = WV.W;
      if (!W.on || W.hbar !== q.hbar || W.c !== q.c) { WV.start(S.a, S.l, va, vl, q.hbar, q.c, 0.6); S.fresh = true; }
      var top = WV.topSpeed(); if (v > top) { va *= top / v; vl *= top / v; }
      q.top = top;
    } else if (WV.W.on) { WV.stop(); S.fresh = true; }
    var vel = q.quantum ? WV.W.vel || [0, 0] : [va, vl];
    var sp = Math.min(Math.hypot(vel[0], vel[1], q.vz), 0.9999 * q.c);
    q.v = sp; q.beta = sp / q.c; q.gamma = 1 / Math.sqrt(1 - q.beta * q.beta); q.vel = vel;
    var dtLocal = q.gamma * dtau, dT = dtLocal / q.lapse;    // time for someone standing here; far-away time
    S.T += dT;
    if (S.fx.rates) { S.tau += dtau; S.street += dT * Math.sqrt(1 - q.rs / G.R); S.top += dT * Math.sqrt(1 - q.rs / (G.R + CLOCK_TOP)); }
    else { S.tau += dT; S.street += dT; S.top += dT; }
    if (q.quantum) {
      WV.step(dtLocal, va, vl);
      if (S.envRate) { S.envTimer -= dtau; if (S.envTimer <= 0) { S.envTimer = 1 / S.envRate; WV.look(0.5, Math.random); } }
      S.a = WV.W.mean.a; S.l = WV.W.mean.l;
      S.view = WV.sample(Math.random);
    } else {
      var na = G.wrapA(S.a + va * dtLocal), nl = S.l + vl * dtLocal, nh = Math.max(G.EYE, Math.min(TOP_EYE, S.h + q.vz * dtLocal));
      if (room(na, nl, S.h)) { S.a = na; S.l = nl; } else if (room(na, S.l, S.h)) S.a = na; else if (room(S.a, nl, S.h)) S.l = nl;
      if (nh !== S.h && room(S.a, S.l, nh)) S.h = nh;
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
      beta.push((q.vel[0] * f.along[i] + q.vel[1] * f.across[i] + q.vz * f.up[i]) / q.c);
    }
    return { eye: G.toWorld(S.view.a, S.view.l, S.h), fwd: fwd, right: right, up: G.cross(right, fwd), beta: beta };
  }

  /* ---------- drawing ---------- */
  function draw(q, cam) {
    var f = A.fit(canvas, 1.5), w = canvas.width, h = canvas.height, fx = S.fx;
    var sc = S.fixedScale || S.scale, rw = Math.max(64, Math.round(w * sc)), rh = Math.max(36, Math.round(h * sc));
    if (rw !== size[0] || rh !== size[1]) {
      free(scene); free(acc[0]); free(acc[1]);
      scene = target(rw, rh); acc = [target(rw, rh), target(rw, rh)]; size = [rw, rh]; S.fresh = true;
    }
    var Dc = 1 / (q.gamma * (1 - G.dot(cam.fwd, cam.beta)));
    var aim = fx.beam ? Math.max(0.02, Math.min(60, Math.pow(Dc, -2.7))) : 1;
    S.exposure += (aim - S.exposure) * 0.15;
    gl.bindVertexArray(vao);

    var m = prog.march, u = m.u;
    gl.bindFramebuffer(gl.FRAMEBUFFER, scene.f); gl.viewport(0, 0, rw, rh); gl.useProgram(m.p);
    gl.uniform3fv(u.uEye, cam.eye); gl.uniform3fv(u.uFwd, cam.fwd); gl.uniform3fv(u.uRight, cam.right);
    gl.uniform3fv(u.uUp, cam.up); gl.uniform3fv(u.uBeta, cam.beta);
    gl.uniform2f(u.uTan, Math.tan(FOV / 2) * f.w / f.h, Math.tan(FOV / 2));
    gl.uniform1f(u.uRsRay, fx.bend ? q.rs : 0); gl.uniform1f(u.uRsShift, fx.gshift ? q.rs : 0); gl.uniform1f(u.uRsClock, fx.rates ? q.rs : 0);
    gl.uniform1f(u.uCdelay, fx.delay ? q.c : 1e12); gl.uniform1f(u.uT, S.T % 1e5); gl.uniform1f(u.uExposure, S.exposure);
    gl.uniform1i(u.uDoppler, fx.doppler ? 1 : 0); gl.uniform1i(u.uBeam, fx.beam ? 1 : 0); gl.uniform1i(u.uAberr, fx.aberr ? 1 : 0);
    gl.uniform1f(u.uBell, q.quantum && WV.W.bell > 0 ? 1 : 0);
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

  // Top-down map of your wave: magenta where the bell is silent, cyan where it rang, yellow dots for past looks.
  function waveMap(q) {
    var el = $('wave-map');
    el.hidden = !q.quantum;
    if (!q.quantum || !WV.W.sim) return;
    var N = WV.N, a = WV.W.sim, b = WV.W.sim1, ctx = el.getContext('2d');
    if (el.width !== 2 * N) { el.width = 2 * N; el.height = 2 * N; }
    var img = ctx.createImageData(N, N), d = img.data, mx = 0, n;
    for (n = 0; n < N * N; n++) mx = Math.max(mx, a.re[n] * a.re[n] + a.im[n] * a.im[n] + (b ? b.re[n] * b.re[n] + b.im[n] * b.im[n] : 0));
    for (n = 0; n < N * N; n++) {
      var p0 = Math.sqrt((a.re[n] * a.re[n] + a.im[n] * a.im[n]) / (mx || 1)), p1 = b ? Math.sqrt((b.re[n] * b.re[n] + b.im[n] * b.im[n]) / (mx || 1)) : 0, o = 4 * n;
      if (a.wall[n]) { d[o] = d[o + 1] = d[o + 2] = 150; }
      else { d[o] = Math.min(255, 14 + 218 * p0 + 40 * p1); d[o + 1] = Math.min(255, 18 + 102 * p0 + 200 * p1); d[o + 2] = Math.min(255, 26 + 168 * p0 + 230 * p1); }
      d[o + 3] = 255;
    }
    var tmp = waveMap.tmp || (waveMap.tmp = document.createElement('canvas'));
    tmp.width = N; tmp.height = N; tmp.getContext('2d').putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false; ctx.drawImage(tmp, 0, 0, 2 * N, 2 * N);
    function px(p) { return [2 * (p.l - WV.W.cl + WV.L / 2) / WV.dx, 2 * (WV.L / 2 - G.wrapA(p.a - WV.W.ca)) / WV.dx]; }
    if (WV.W.bell > 0) {
      var bz = px({ a: WV.BELL.a + WV.BELL.ha, l: WV.BELL.l - WV.BELL.hl });
      ctx.strokeStyle = '#5fd7ff'; ctx.lineWidth = 1.5; ctx.strokeRect(bz[0], bz[1], 4 * WV.BELL.hl / WV.dx, 4 * WV.BELL.ha / WV.dx);
    }
    ctx.fillStyle = '#ffd27a';
    WV.W.marks.forEach(function (mk) { var p = px(mk); ctx.fillRect(p[0] - 1.5, p[1] - 1.5, 3, 3); });
    var me = px(S.view);
    ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(me[0], me[1], 4, 0, 6.2832); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(me[0], me[1]);
    ctx.lineTo(me[0] + 16 * Math.sin(S.heading + S.lookYaw), me[1] - 16 * Math.cos(S.heading + S.lookYaw)); ctx.stroke();
  }

  function row(dt, dd) { return '<div class="readout"><dt>' + dt + '</dt><dd>' + dd + '</dd></div>'; }
  function hud(q) {
    var n = A.num, rs = q.rs;
    $('w-hud-l').innerHTML = 'speed <b>' + n(q.v) + ' m/s</b> = ' + (q.beta < 1e-3 ? n(q.beta, 2) : q.beta.toFixed(3)) + ' c<br>γ = ' +
      (q.gamma < 1.0001 ? '1.000' : q.gamma.toFixed(3)) + '<br>eyes <b>' + n(S.h, 3) + ' m</b> up';
    $('w-hud-r').innerHTML = 'your watch <b>' + A.clock(S.tau) + '</b><br>street clocks <b>' + A.clock(S.street) + '</b><br>tower top <b>' + A.clock(S.top) + '</b>';
    var msg = '';
    if (q.quantum) msg = WV.W.spread > 0.5 ? 'You are spread over ' + n(WV.W.spread, 2) + ' m: every place you might be is overlaid. F looks.' : 'You are a narrow wave. Walk, and watch the map.';
    else if (S.h > G.EYE + 0.5) msg = 'Q to go down, E to go up';
    else msg = 'Drag to look around · E to fly up';
    if (S.hint > 0 && canvas.clientWidth > 640) msg += '<br>W S effort · A D steer · E Q up and down · F look · Space stop';
    $('w-hud-b').innerHTML = msg;
    var el = G.horizon(rs, G.R, q.re) * 180 / Math.PI;
    $('w-readouts').innerHTML = [
      row(el < 0 ? 'Horizon, below your eye level' : 'Horizon, above your eye level', Math.abs(el).toFixed(1) + '°' + (el > 0 ? ' (sky is a cone ' + (2 * (90 - el)).toFixed(0) + '° wide)' : '')),
      row('Tower-top clock against yours', '×' + Math.sqrt((1 - rs / (G.R + CLOCK_TOP)) / (1 - rs / q.re)).toFixed(rs > 0.01 * G.R ? 3 : 6)),
      row('Light crosses the planet in', A.duration(2 * G.R / q.c * (1 + rs / G.R), 2)),
      q.quantum ? row('Your spread / top speed here', n(WV.W.spread, 2) + ' m / ' + n(q.top, 2) + ' m/s')
                : row('Your wavelength now', A.length(2 * Math.PI * q.hbar / (A.MASS * Math.max(q.v, A.WALK) * q.gamma), 2)),
      row('Photons per pixel per frame', n(q.photons, 2))
    ].join('');
    measurePanel(q);
  }
  function measurePanel(q) {
    $('m-off').hidden = q.quantum; $('m-on').hidden = !q.quantum;
    if (!q.quantum || !WV.W.sim) return;
    var W = WV.W, ks = WV.kstats(), hb = W.hbar, m = W.pos;
    var px = m.sx * hb * ks.dk[0] / (hb / 2), pa = m.sy * hb * ks.dk[1] / (hb / 2);
    $('m-dx').textContent = A.num(m.sx, 2) + ' m across · ' + A.num(m.sy, 2) + ' m along';
    $('m-dp').textContent = A.num(hb * ks.dk[0] / A.MASS, 2) + ' m/s across · ' + A.num(hb * ks.dk[1] / A.MASS, 2) + ' m/s along';
    $('m-prod').textContent = A.num(px, 3) + ' × ħ/2 across · ' + A.num(pa, 3) + ' × ħ/2 along';
    $('m-bar').style.width = Math.min(100, 100 / Math.max(1, Math.min(px, pa))) + '%';
    $('m-rang').textContent = W.bell > 0 ? Math.round(100 * W.rang) + '% of you rang it' : 'no bell';
    $('m-marks').textContent = W.marks.length + (W.marks.length === 1 ? ' look' : ' looks') + ' so far';
    $('m-behind').textContent = Math.round(100 * WV.chanceBehind()) + '%';
  }

  var hudT = 0, probeT = 0;
  function frame(now) {
    raf = requestAnimationFrame(frame);
    var dt = Math.min(0.05, (now - last) / 1000 || 0);
    S.ms += ((now - last) - S.ms) * 0.05; last = now;
    if (!S.fixedScale && S.frame % 30 === 29) {
      if (S.ms > 42 && S.scale > 0.3) S.scale *= 0.85; else if (S.ms < 22 && S.scale < 1) S.scale = Math.min(1, S.scale * 1.1);
    }
    var q = update(dt), cam = camera(q);
    S.q = q; S.cam = cam;
    draw(q, cam);
    hudT += dt; probeT += dt;
    if (hudT > 0.12) { hudT = 0; hud(q); waveMap(q); }
    if (probeT > 0.25 && CGH.probe) { probeT = 0; CGH.probe.update(); }
  }

  /* ---------- input ---------- */
  function syncEffort() { $('effort').value = S.effort; $('effort-out').textContent = S.effort.toFixed(1) + ' m/s'; }
  function restartClocks() { S.T = 0; S.tau = 0; S.street = 0; S.top = 0; }
  function place(a, l, heading, h) {
    S.a = a; S.l = l; S.view = { a: a, l: l }; S.h = h || G.EYE; S.heading = heading; S.lookYaw = 0; S.lookPitch = 0.04;
    S.effort = 0; syncEffort(); WV.stop(); WV.W.marks = []; restartClocks(); S.fresh = true;
  }
  function look() {
    if (!WV.W.on) return;
    if (WV.look(S.sharp, Math.random, S.screen)) S.fresh = true;
    else $('m-note').textContent = 'None of you has reached the yard behind the wall yet.';
  }
  function setFx(all) { FX.forEach(function (k) { S.fx[k] = all; $('fx-' + k).checked = all; }); S.fresh = true; }
  function bind() {
    $('effort').addEventListener('input', function (e) { S.effort = Number(e.target.value); syncEffort(); });
    syncEffort();
    $('w-stop').addEventListener('click', function () { S.effort = 0; syncEffort(); });
    $('w-turn').addEventListener('click', function () { S.heading += Math.PI; S.lookYaw = 0; });
    $('w-home').addEventListener('click', function () { place(G.START.a, G.START.l, 0); });
    $('w-tower').addEventListener('click', function () { place(G.TOWER.a - 4, 0, 0, CLOCK_TOP); });
    document.addEventListener('cgh-pose', function (e) { if (e.detail === 'doors') place(G.DOORS.a, G.DOORS.l, G.DOORS.heading); });
    FX.forEach(function (k) { $('fx-' + k).addEventListener('change', function (e) { S.fx[k] = e.target.checked; S.fresh = true; }); });
    $('fx-none').addEventListener('click', function () { setFx(false); });
    $('fx-all').addEventListener('click', function () { setFx(true); });
    $('m-look').addEventListener('click', look);
    $('m-relook').addEventListener('click', function () { if (WV.W.on && WV.relook(S.sharp, Math.random, S.screen)) S.fresh = true; });
    $('m-screen').addEventListener('change', function (e) { S.screen = e.target.checked; });
    $('m-sharp').addEventListener('change', function (e) { S.sharp = Number(e.target.value); });
    $('m-env').addEventListener('change', function (e) { S.envRate = Number(e.target.value); S.envTimer = 0; });
    $('m-bell').addEventListener('input', function (e) { WV.W.bell = Number(e.target.value) / 100; $('m-bell-out').textContent = e.target.value + '%'; });
    $('m-erase').addEventListener('click', function () { if (WV.W.on && WV.erase(Math.random)) S.fresh = true; });
    $('m-clear').addEventListener('click', function () { WV.W.marks = []; });

    var KEYS = { w: 'faster', ArrowUp: 'faster', s: 'slower', ArrowDown: 'slower', a: 'left', ArrowLeft: 'left', d: 'right', ArrowRight: 'right', e: 'up', q: 'down' };
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
      S.lookPitch = Math.max(-1.5, Math.min(1.5, S.lookPitch - (e.clientY - drag[1]) * 0.005));
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
      A.onChange(function () { S.fresh = true; restartClocks(); });
      document.addEventListener('visibilitychange', function () {
        cancelAnimationFrame(raf);
        if (!document.hidden) { last = performance.now(); raf = requestAnimationFrame(frame); }
      });
      last = performance.now(); raf = requestAnimationFrame(frame);
    },
    state: S, place: place, setFx: setFx
  };
})();
