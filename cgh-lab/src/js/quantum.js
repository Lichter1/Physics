/* cGħ Lab — Quantum: a football in a yard, with ħ large enough to see.
 *   Two doors  2-D Schrödinger equation, a wall with two openings, hits recorded on the back wall
 *   Fence      2-D, a barrier slightly higher than the ball's energy (tunnelling)
 *   Drop       a ball over a floor under gravity; separable, so the vertical part is solved in 1-D
 */
(function () {
  'use strict';
  var P = CGH.physics, QC = CGH.qcore, A = CGH.app, $ = A.$;
  var M = P.REF.ball, N2 = 128, L2 = 8, N1 = 512, L1 = 4, ZF = 0.4, ABS = 0.75;
  var WX0 = 3.5, WX1 = 3.75, SEP = 1.5, DOOR = 0.5, FX = 4.0, BALL_R = 0.11;
  var canvas, ctx, off, offCtx, img, raf = 0, last = 0;
  var Q = {
    scene: 'doors', speed: 2, eta: 1.2, thick: 0.25, drop: 2, phase: false, paused: false, ground: false,
    sim: null, classical: false, t: 0, pr: null, accum: null, runs: 0, landed: 0, back: 0, hits: [],
    marker: null, slow: 1, ref: 0, latX: 2, latS: 0.5, latT: 0, done: false, wait: 0
  };

  function params() {
    var hbar = A.state.hbar, g = P.earthGravity(A.state.G), p = { hbar: hbar, g: g };
    if (Q.scene === 'drop') {
      var b = P.bouncer(M, g, hbar), dz = L1 / N1;
      p.b = b; p.vmax = Math.sqrt(2 * g * Q.drop);
      p.lam = P.deBroglie(M, p.vmax, hbar);
      p.Eref = Q.ground ? 4 * b.energies[0] : Math.max(M * g * Q.drop, 2 * b.energies[0]);
      p.dt = Math.min(0.02 * hbar / p.Eref, 4 * M * dz * dz / (Math.PI * Math.PI * hbar));
      p.classical = Q.ground ? b.z0 < 6 * dz : p.lam < 4 * dz;
      p.period = 2 * Math.sqrt(2 * Q.drop / g);
    } else {
      var dx = L2 / N2;
      p.E = 0.5 * M * Q.speed * Q.speed;
      p.lam = P.deBroglie(M, Q.speed, hbar);
      p.Eref = p.E * (Q.scene === 'fence' ? Math.max(1, Q.eta) : 1);
      p.dt = Math.min(0.1 * hbar / p.Eref, 4 * M * dx * dx / (Math.PI * Math.PI * hbar));
      p.classical = p.lam < 4 * dx;
      p.V0 = Q.eta * p.E;
    }
    return p;
  }

  function absorber(x, y, g0) {
    var d = Math.min(x, L2 - x, y, L2 - y);
    if (d >= ABS) return 0;
    var s = (ABS - d) / ABS;
    return g0 * s * s;
  }
  function isWall(x, y) { return x > WX0 && x < WX1 && Math.abs(Math.abs(y - L2 / 2) - SEP / 2) > DOOR / 2; }

  function launch() {
    var p = Q.pr;
    Q.t = 0; Q.done = false; Q.wait = 0; Q.marker = null; Q.ref = 0;
    if (p.classical || !Q.sim) return;
    if (Q.scene === 'doors') Q.sim.gaussian(1.9, L2 / 2, 0.45, 1.0, Q.speed, 0);
    else if (Q.scene === 'fence') Q.sim.gaussian(2.0, L2 / 2, 0.5, 1.3, Q.speed, 0);
    else {
      Q.latX = L1 / 2; Q.latS = 0.5; Q.latT = 0;
      if (Q.ground) {
        var zs = []; for (var i = 0; i < N1; i++) zs.push((Q.sim.z(i) - ZF) / p.b.z0 - P.AIRY_ZEROS[0]);
        var ai = QC.airyShape(zs);
        Q.sim.set(function (z) { var k = Math.round(z / Q.sim.dz - 0.5); return [z > ZF ? ai[k] : 0, 0]; });
      } else {
        var zc = ZF + Q.drop;
        Q.sim.set(function (z) { return [z > ZF ? Math.exp(-(z - zc) * (z - zc) / (4 * 0.18 * 0.18)) : 0, 0]; });
      }
    }
  }

  // Rebuild the solver for the current scene, sliders and dials.
  function reset() {
    var p = Q.pr = params();
    Q.accum = new Float64Array(N2); Q.runs = 0; Q.landed = 0; Q.back = 0; Q.hits = []; Q.sim = null;
    if (!p.classical) {
      if (Q.scene === 'drop') {
        Q.sim = new QC.Sim1D({ N: N1, L: L1, m: M, hbar: p.hbar });
        Q.sim.build(function (z) { return { V: M * p.g * (z - ZF), wall: z < ZF }; });
      } else {
        var g0 = 14 * Q.speed / ABS;
        Q.sim = new QC.Sim({ N: N2, L: L2, m: M, hbar: p.hbar });
        Q.sim.screenFrom = Math.round((L2 - ABS) / Q.sim.dx);
        if (Q.scene === 'doors') Q.sim.build(function (x, y) { return { wall: isWall(x, y), damp: absorber(x, y, g0) }; });
        else Q.sim.build(function (x, y) { return { V: x > FX && x < FX + Q.thick ? p.V0 : 0, damp: absorber(x, y, g0) }; });
      }
    }
    launch(); nudge(); labels();
  }

  // One kick is over: add what reached the back wall to the tally and record where this ball landed.
  function finish() {
    if (Q.done) return;
    Q.done = true; Q.wait = 0.5;
    if (Q.scene === 'drop' || Q.pr.classical) return;
    var s = Q.sim.screen, tot = 0, j;
    for (j = 0; j < N2; j++) { Q.accum[j] += s[j]; tot += s[j]; }
    Q.runs++;
    sampleHits(1);
  }
  function sampleHits(n) {
    var tot = 0, j; for (j = 0; j < N2; j++) tot += Q.accum[j];
    if (!Q.runs) return;
    var pLand = tot / Q.runs;
    for (var k = 0; k < n; k++) {
      if (Math.random() > pLand) { Q.back++; continue; }
      var target = Math.random() * tot, acc = 0;
      for (j = 0; j < N2; j++) { acc += Q.accum[j]; if (acc >= target) break; }
      Q.landed++;
      if (Q.hits.length < 4000) Q.hits.push([Math.min(j, N2 - 1) + Math.random(), Math.random()]);
    }
  }

  function classicalBall(t) {            // where Newton's ball is, in yard coordinates; null once it has left
    if (Q.scene === 'drop') {
      var g = Q.pr.g, T = Q.pr.period, u = ((t % T) + T) % T, half = T / 2;
      var tt = u < half ? u : T - u;
      return [L1 / 2, ZF + BALL_R + Math.max(0, Q.drop - 0.5 * g * tt * tt)];
    }
    var v = Q.speed, x0 = Q.scene === 'doors' ? 1.9 : 2.0, y = L2 / 2, x;
    var stop = Q.scene === 'doors' ? WX0 - BALL_R : FX - BALL_R;
    var tHit = (stop - x0) / v;
    if (t < tHit) x = x0 + v * t;
    else if (Q.scene === 'fence' && Q.eta < 1) {
      var vin = v * Math.sqrt(1 - Q.eta), tIn = (Q.thick + 2 * BALL_R) / vin;
      x = t < tHit + tIn ? stop + vin * (t - tHit) : stop + Q.thick + 2 * BALL_R + v * (t - tHit - tIn);
    } else x = stop - v * (t - tHit);
    return x < -0.3 || x > L2 + 0.3 ? null : [x, y];
  }

  function advance(dtReal) {
    var p = Q.pr;
    if (Q.paused) return;
    if (Q.done) { Q.wait -= dtReal; if (Q.wait <= 0 && Q.scene !== 'drop') launch(); return; }
    if (p.classical) {
      Q.t += dtReal;
      if (Q.scene !== 'drop' && !classicalBall(Q.t)) { Q.done = true; Q.wait = 0.4; }
      return;
    }
    var want = Math.min(dtReal, 0.05);
    if (want <= 0) return;
    Q.acc = (Q.acc || 0) + want;                       // run in real time, in whole solver steps
    var steps = Math.floor(Q.acc / p.dt), t0 = performance.now(), done = 0, capped = false;
    for (var i = 0; i < steps; i++) { Q.sim.step(p.dt); done++; if (performance.now() - t0 > 11) { capped = done < steps; break; } }
    var adv = done * p.dt;
    Q.acc = capped ? 0 : Q.acc - adv;                  // if the machine cannot keep up, play in slow motion
    Q.t += adv; Q.latT += adv;
    Q.slow += ((capped ? adv / want : 1) - Q.slow) * 0.1;
    if (Q.scene !== 'drop' && (Q.t > 1.7 * L2 / Q.speed || (Q.t > 1 && Q.sim.absorbed > 0.985))) finish();
    if (Q.marker) { Q.marker.life -= dtReal; if (Q.marker.life <= 0) Q.marker = null; }
  }

  /* ---------- drawing ---------- */
  function hsv(hh, v) {                  // hue in turns, full saturation
    var k = function (n) { var a = (n + hh * 6) % 6; return v * (1 - Math.max(0, Math.min(a, 4 - a, 1))); };
    return [k(5), k(3), k(1)];
  }
  function paint(W, Hh, density, phase) {
    if (!off || off.width !== W || off.height !== Hh) {
      off = document.createElement('canvas'); off.width = W; off.height = Hh; offCtx = off.getContext('2d'); img = offCtx.createImageData(W, Hh);
    }
    var d = img.data, n = W * Hh, mx = 0, q;
    for (q = 0; q < n; q++) if (density[q] > mx) mx = density[q];
    if (mx > Q.ref) Q.ref = mx;
    var ref = Math.max(mx, 0.03 * Q.ref) || 1;
    for (q = 0; q < n; q++) {
      var v = Math.pow(Math.min(1, density[q] / ref), 0.5), o = q * 4, r, g, b;
      if (phase) { var c = hsv(phase[q], v); r = c[0]; g = c[1]; b = c[2]; }
      else { var w = v * v * v; r = 0.91 * v + 0.09 * w; g = 0.47 * v + 0.53 * w; b = 0.76 * v + 0.24 * w; }
      d[o] = 10 + 245 * r; d[o + 1] = 15 + 240 * g; d[o + 2] = 22 + 233 * b; d[o + 3] = 255;
    }
    offCtx.putImageData(img, 0, 0);
  }

  var dens2 = new Float32Array(N2 * N2), ph2 = new Float32Array(N2 * N2);
  var dens1 = new Float32Array(128 * 256);

  function draw() {
    var s = A.fit(canvas, 2), w = s.w, h = s.h, p = Q.pr;
    ctx.setTransform(s.dpr, 0, 0, s.dpr, 0, 0);
    ctx.fillStyle = '#0a0f15'; ctx.fillRect(0, 0, w, h);
    var strip = w < 520 ? 66 : 120, info = w > 820 ? 210 : 0;
    var side = Math.min(h - 8, w - strip - info - 8), ox = 4, oy = Math.max(4, (h - side) / 2);
    if (w < 520) oy = 4;
    var Lw = Q.scene === 'drop' ? L1 : L2, sc = side / Lw;
    function X(x) { return ox + x * sc; }
    function Y(y) { return oy + (Lw - y) * sc; }
    var q, i, j;

    if (!p.classical && Q.sim) {
      if (Q.scene === 'drop') {
        var sx = P.spread(Q.latS, Q.latT, M, p.hbar), re = Q.sim.re, im = Q.sim.im;
        for (j = 0; j < 256; j++) {
          var k = N1 - 2 - 2 * j, pz = re[k] * re[k] + im[k] * im[k] + re[k + 1] * re[k + 1] + im[k + 1] * im[k + 1];
          for (i = 0; i < 128; i++) { var xx = (i + 0.5) * L1 / 128 - Q.latX; dens1[j * 128 + i] = pz * Math.exp(-xx * xx / (2 * sx * sx)) / sx; }
        }
        paint(128, 256, dens1, null);
      } else {
        var r2 = Q.sim.re, i2 = Q.sim.im;
        for (q = 0; q < N2 * N2; q++) {
          dens2[q] = r2[q] * r2[q] + i2[q] * i2[q];
          if (Q.phase) ph2[q] = Math.atan2(i2[q], r2[q]) / 6.2831853 + 0.5;
        }
        paint(N2, N2, dens2, Q.phase ? ph2 : null);
      }
      ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(off, ox, oy, side, side);
    } else { ctx.fillStyle = '#0d141c'; ctx.fillRect(ox, oy, side, side); }

    ctx.strokeStyle = '#3a4756'; ctx.lineWidth = 1; ctx.strokeRect(ox + 0.5, oy + 0.5, side - 1, side - 1);
    ctx.font = '12px ' + A.css('--font-text'); ctx.textBaseline = 'middle';

    // furniture
    if (Q.scene === 'doors') {
      ctx.fillStyle = '#c9d1d9';
      var e1 = L2 / 2 - SEP / 2 - DOOR / 2, e2 = L2 / 2 - SEP / 2 + DOOR / 2, e3 = L2 / 2 + SEP / 2 - DOOR / 2, e4 = L2 / 2 + SEP / 2 + DOOR / 2;
      [[0, e1], [e2, e3], [e4, L2]].forEach(function (seg) { ctx.fillRect(X(WX0), Y(seg[1]), (WX1 - WX0) * sc, (seg[1] - seg[0]) * sc); });
    } else if (Q.scene === 'fence') {
      ctx.fillStyle = 'rgba(201,209,217,0.38)'; ctx.fillRect(X(FX), Y(L2), Q.thick * sc, side);
      ctx.strokeStyle = '#c9d1d9'; ctx.strokeRect(X(FX) + 0.5, Y(L2) + 0.5, Q.thick * sc - 1, side - 1);
    } else {
      ctx.fillStyle = '#566473'; ctx.fillRect(ox, Y(ZF), side, ZF * sc - 1);
    }
    if (Q.scene !== 'drop') {
      var xs = X(L2 - ABS);
      ctx.strokeStyle = '#8b98a5'; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(xs, oy); ctx.lineTo(xs, oy + side); ctx.stroke(); ctx.setLineDash([]);
      if (Q.scene === 'doors') {
        ctx.fillStyle = '#ffd27a';
        for (i = 0; i < Q.hits.length; i++) ctx.fillRect(xs + 3 + Q.hits[i][1] * (ABS * sc - 8), oy + Q.hits[i][0] * side / N2 - 1, 2, 2);
      }
    }
    // Newton's ball, for comparison
    var cb = classicalBall(Q.t);
    if (cb && !(Q.scene === 'drop' && Q.ground)) {
      ctx.beginPath(); ctx.arc(X(cb[0]), Y(cb[1]), Math.max(4, BALL_R * sc), 0, 6.2832);
      if (p.classical) { ctx.fillStyle = '#f4f7fa'; ctx.fill(); }
      ctx.strokeStyle = '#0a0f15'; ctx.lineWidth = 4; ctx.stroke();                  // dark halo so the ring shows on the glow
      ctx.strokeStyle = '#ffd27a'; ctx.lineWidth = 1.75; ctx.setLineDash(p.classical ? [] : [3, 3]); ctx.stroke(); ctx.setLineDash([]);
    }
    if (Q.marker) {
      ctx.beginPath(); ctx.arc(X(Q.marker.x), Y(Q.marker.y), 10 + 26 * (1 - Q.marker.life), 0, 6.2832);
      ctx.strokeStyle = 'rgba(255,210,122,' + Q.marker.life + ')'; ctx.lineWidth = 2; ctx.stroke();
    }
    // scale: one metre, and the ball's wavelength if it fits
    var by = oy + 34;
    ctx.strokeStyle = '#f4f7fa'; ctx.fillStyle = '#f4f7fa'; ctx.lineWidth = 2; ctx.textAlign = 'left';
    ctx.beginPath(); ctx.moveTo(ox + 10, by); ctx.lineTo(ox + 10 + sc, by); ctx.stroke();
    ctx.fillText('1 m', ox + 14 + sc, by);
    if (!p.classical && p.lam < Lw * 0.8) {
      ctx.strokeStyle = '#e878c2'; ctx.fillStyle = '#e878c2';
      ctx.beginPath(); ctx.moveTo(ox + 10, by - 14); ctx.lineTo(ox + 10 + p.lam * sc, by - 14); ctx.stroke();
      ctx.fillText('λ', ox + 14 + p.lam * sc, by - 14);
    }
    if (!p.classical && Q.slow < 0.8) { ctx.fillStyle = '#8b98a5'; ctx.textAlign = 'right'; ctx.fillText('slow motion ×' + Q.slow.toFixed(2), ox + side - 8, oy + side - 12); }

    // the strip beside the yard
    var px = ox + side + 8, pw = strip - 12;
    ctx.textAlign = 'left'; ctx.fillStyle = '#8b98a5';
    if (Q.scene === 'doors') {
      ctx.fillText('back wall', px, oy + 8);
      var mxA = 0; for (j = 0; j < N2; j++) if (Q.accum[j] > mxA) mxA = Q.accum[j];
      var live = Q.sim && !p.classical ? Q.sim.screen : null, mxL = 0;
      if (live) for (j = 0; j < N2; j++) if (live[j] > mxL) mxL = live[j];
      var src = mxA > 0 ? Q.accum : live, mxS = mxA > 0 ? mxA : mxL;
      if (src && mxS > 0) {
        ctx.fillStyle = '#e878c2';
        for (j = 0; j < N2; j++) ctx.fillRect(px, oy + j * side / N2, pw * src[j] / mxS, side / N2 + 0.5);
      }
    } else if (Q.scene === 'fence') {
      var through = Q.sim && !p.classical ? Q.sim.probRightOf(FX + Q.thick) + sum(Q.sim.screen) : (Q.eta < 1 ? 1 : 0);
      var formula = P.barrierT(p.E, p.V0, Q.thick, M, p.hbar);
      ctx.fillText('got through', px, oy + 8);
      var bh = side - 60, bx = px + 6, bw = Math.min(34, pw - 12), by0 = oy + 26;
      ctx.strokeStyle = '#3a4756'; ctx.lineWidth = 1; ctx.strokeRect(bx + 0.5, by0 + 0.5, bw, bh);
      ctx.fillStyle = '#e878c2'; ctx.fillRect(bx, by0 + bh * (1 - through), bw, bh * through);
      if (!p.classical) {
        var fy = by0 + bh * (1 - formula);
        ctx.strokeStyle = '#ffd27a'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(bx - 4, fy); ctx.lineTo(bx + bw + 4, fy); ctx.stroke();
        ctx.fillStyle = '#ffd27a'; if (pw > 80) ctx.fillText('formula', bx + bw + 8, fy);
      }
      ctx.fillStyle = '#f4f7fa'; ctx.fillText((100 * through).toFixed(1) + ' %', px, by0 + bh + 16);
    } else {
      ctx.fillText(pw > 80 ? 'allowed heights' : 'allowed', px, oy + 8);
      var hs = p.b.heights;
      for (i = 0; i < hs.length; i++) {
        var yy = Y(ZF + hs[i]); if (yy < oy + 18) break;
        ctx.strokeStyle = '#e878c2'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(px, yy); ctx.lineTo(px + pw * 0.5, yy); ctx.stroke();
        ctx.fillStyle = '#e878c2'; ctx.fillText(String(i + 1), px + pw * 0.5 + 5, yy);
      }
      if (Q.sim && !p.classical) {
        var my = Y(Q.sim.mean());
        ctx.fillStyle = '#f4f7fa'; ctx.beginPath(); ctx.moveTo(px - 2, my); ctx.lineTo(px - 9, my - 5); ctx.lineTo(px - 9, my + 5); ctx.fill();
      }
    }
    // legend, where there is room
    if (info) {
      var lx = ox + side + strip + 12, ly = oy + 8, lh = 20;
      ctx.fillStyle = '#8b98a5'; ctx.textAlign = 'left';
      var lines = ['glow: where the ball may', 'be found, |ψ|²', '', 'yellow ring: the ball', 'Newton predicts'];
      if (Q.scene === 'doors') lines.push('', 'yellow dots: where each', 'kicked ball landed');
      if (Q.scene === 'fence') lines.push('', 'yellow line: textbook', 'tunnelling formula');
      if (Q.scene === 'drop') lines.push('', 'numbered marks: how high', 'each energy level reaches');
      lines.forEach(function (t, n) { ctx.fillText(t, lx, ly + n * lh); });
    }
  }
  function sum(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s; }

  /* ---------- text around the picture ---------- */
  var TEXT = {
    doors: {
      title: 'Two doors',
      premise: 'A football is rolled at a wall with two open doors. Newton’s ball hits the wall between them and comes back. Watch what the real one does, and where each kick lands on the back wall.',
      explain: '<div><h3>What to try</h3><ul>' +
        '<li>Let a few kicks finish, then press “100 kicks”. Each ball lands in one spot; the stripes only exist in the tally.</li>' +
        '<li>Press “Look at it” just after the wave passes the doors. Finding the ball near one door wipes out the other path, and that kick lands without stripes.</li>' +
        '<li>Lower <span class="sym sym-h">ħ</span>. The stripes crowd together until the grid cannot draw them and the ball simply bounces off the wall.</li></ul></div>' +
        '<div><h3>The arithmetic</h3><p class="formula"><i>λ</i> = 2π<i>ħ</i>/<i>mv</i> &nbsp; stripe spacing ≈ <i>λD</i>/<i>d</i></p>' +
        '<p class="note">Doors 0.5 m wide, 1.5 m apart; back wall 3.5 m behind them. The yard’s edges absorb the wave. One isolated particle, no air, no warmth: a real football loses its wave nature to its surroundings in far less than any time you could name.</p></div>'
    },
    fence: {
      title: 'Fence',
      premise: 'The ball rolls at a low fence that would take a little more energy to clear than the ball has. Newton’s ball bounces back every time.',
      explain: '<div><h3>What to try</h3><ul>' +
        '<li>Set the fence above 1 and watch part of the wave leak through. That fraction is the chance a given ball ends up on the far side.</li>' +
        '<li>Make the fence thicker. The leak falls off exponentially.</li>' +
        '<li>Set the fence a little below 1. Newton’s ball always clears it; the real one still sometimes comes back.</li></ul></div>' +
        '<div><h3>The arithmetic</h3><p class="formula"><i>T</i> = [1 + <i>V</i>₀² sinh²(<i>κw</i>) / 4<i>E</i>(<i>V</i>₀ − <i>E</i>)]⁻¹ &nbsp; <i>κ</i> = √(2<i>m</i>(<i>V</i>₀ − <i>E</i>))/<i>ħ</i></p>' +
        '<p class="note">The yellow line is this plane-wave formula. The simulated ball is a packet with a spread of energies, so the two agree closely, not exactly.</p></div>'
    },
    drop: {
      title: 'Drop',
      premise: 'Side view. The ball is released above a hard floor and falls under gravity. This scene uses two dials at once: ħ and G.',
      explain: '<div><h3>What to try</h3><ul>' +
        '<li>Watch a few bounces. The wave reflects off the floor and interferes with itself; the average height still follows Newton for a while.</li>' +
        '<li>Press “Rest it on the floor”. The lowest state hovers: squeezing the ball onto the floor would cost more energy than lifting it.</li>' +
        '<li>Turn <span class="sym sym-g">G</span> or <span class="sym sym-h">ħ</span> and watch the allowed heights move. Neutrons do exactly this over a mirror, with rungs about 14 µm high.</li></ul></div>' +
        '<div><h3>The arithmetic</h3><p class="formula"><i>z</i>₀ = (<i>ħ</i>²/2<i>m</i>²<i>g</i>)<sup>1/3</sup> &nbsp; <i>E<sub>n</sub></i> = <i>mgz</i>₀<i>a<sub>n</sub></i>, &nbsp; <i>a<sub>n</sub></i> = 2.34, 4.09, 5.52 …</p>' +
        '<p class="note">The <i>a<sub>n</sub></i> are the zeros of the Airy function. Gravity here is Earth’s pull under your <span class="sym">G</span>, treated as a fixed classical field. Whether gravity itself has quantum states is the open question this scene does not touch.</p></div>'
    }
  };

  function labels() {
    var p = Q.pr, n = A.num, rows;
    $('q-speed-out').textContent = Q.speed.toFixed(1) + ' m/s';
    $('q-height-out').textContent = Q.eta.toFixed(2) + ' × ball energy';
    $('q-thick-out').textContent = A.length(Q.thick, 3);
    $('q-drop-out').textContent = Q.drop.toFixed(1) + ' m';
    if (Q.scene === 'drop') {
      rows = [
        ['Gravity here, g', n(p.g) + ' m/s²'],
        ['Height scale z₀', A.length(p.b.z0, 3)],
        ['Lowest state hovers at', A.length(2 / 3 * p.b.heights[0], 3)],
        ['First three rungs reach', p.b.heights.slice(0, 3).map(function (x) { return A.length(x, 2); }).join(', ')],
        ['Shortest wavelength in the fall', A.length(p.lam, 3)]
      ];
    } else {
      rows = [
        ['Ball', '0.43 kg at ' + Q.speed.toFixed(1) + ' m/s'],
        ['Wavelength λ', A.length(p.lam, 3)],
        ['Energy', n(p.E) + ' J']
      ];
      if (Q.scene === 'doors') {
        rows.push(['Stripe spacing on the back wall', A.length(p.lam * (L2 - ABS - WX1) / SEP, 2)]);
        rows.push(['Kicks: landed, came back', Q.landed + ', ' + Q.back]);
      } else {
        rows.push(['Textbook chance of getting through', (100 * P.barrierT(p.E, p.V0, Q.thick, M, p.hbar)).toFixed(1) + ' %']);
        rows.push(['Newton says', Q.eta > 1 ? 'never' : 'always']);
      }
    }
    $('q-readouts').innerHTML = rows.map(function (r) { return '<div class="readout"><dt>' + r[0] + '</dt><dd>' + r[1] + '</dd></div>'; }).join('');
  }
  function nudge() {
    var p = Q.pr, el = $('q-nudge');
    el.hidden = !p.classical;
    if (p.classical) {
      $('q-nudge-text').textContent = 'With ħ = ' + A.num(p.hbar) + ' J·s this ball’s wavelength is ' + A.length(p.lam, 2) +
        (p.lam < 8.4e-16 ? ', ' + A.num(8.4e-16 / p.lam, 2) + ' times smaller than a proton' : '') + '. Nothing here can show it, so the ball goes where Newton says.';
    }
  }
  function setScene(name) {
    Q.scene = name; Q.ground = false;
    Array.prototype.forEach.call($('q-scene').children, function (b) { b.setAttribute('aria-pressed', b.getAttribute('data-scene') === name ? 'true' : 'false'); });
    $('q-speed-field').hidden = name === 'drop';
    $('q-height-field').hidden = name !== 'fence';
    $('q-thick-field').hidden = name !== 'fence';
    $('q-drop-field').hidden = name !== 'drop';
    $('q-ground').hidden = name !== 'drop';
    $('q-many').hidden = name !== 'doors';
    $('q-phase').parentNode.hidden = name === 'drop';
    $('q-restart').textContent = name === 'drop' ? 'Drop again' : 'Kick again';
    $('q-title').textContent = TEXT[name].title;
    $('q-premise').textContent = TEXT[name].premise;
    $('q-explain').innerHTML = TEXT[name].explain;
    reset();
  }

  var roClock = 0;
  function frame(now) {
    raf = requestAnimationFrame(frame);
    var dt = Math.min(0.05, (now - last) / 1000 || 0); last = now;
    advance(dt); draw();
    roClock += dt; if (roClock > 0.25) { roClock = 0; labels(); }
  }

  function bind() {
    Array.prototype.forEach.call($('q-scene').children, function (b) { b.addEventListener('click', function () { setScene(b.getAttribute('data-scene')); }); });
    $('q-speed').addEventListener('input', function (e) { Q.speed = Number(e.target.value); reset(); });
    $('q-height').addEventListener('input', function (e) { Q.eta = Number(e.target.value); reset(); });
    $('q-thick').addEventListener('input', function (e) { Q.thick = Number(e.target.value); reset(); });
    $('q-drop').addEventListener('input', function (e) { Q.drop = Number(e.target.value); Q.ground = false; reset(); });
    $('q-phase').addEventListener('change', function (e) { Q.phase = e.target.checked; });
    $('q-restart').addEventListener('click', function () { if (Q.scene === 'drop' && Q.ground) { Q.ground = false; reset(); } else launch(); });
    $('q-ground').addEventListener('click', function () { Q.ground = true; reset(); });
    $('q-many').addEventListener('click', function () { if (!Q.runs) finish(); sampleHits(100); labels(); });
    $('q-pause').addEventListener('click', function (e) { Q.paused = !Q.paused; e.target.setAttribute('aria-pressed', Q.paused ? 'true' : 'false'); e.target.textContent = Q.paused ? 'Resume' : 'Pause'; });
    $('q-look').addEventListener('click', function () {
      if (!Q.sim || Q.pr.classical || Q.done) return;
      if (Q.scene === 'drop') {
        var re = Q.sim.re, im = Q.sim.im, tot = 0, i, acc = 0;
        for (i = 0; i < N1; i++) tot += re[i] * re[i] + im[i] * im[i];
        var target = Math.random() * tot, hit = N1 - 1;
        for (i = 0; i < N1; i++) { acc += re[i] * re[i] + im[i] * im[i]; if (acc >= target) { hit = i; break; } }
        var z0 = Q.sim.z(hit);
        for (i = 0; i < N1; i++) { var dz = Q.sim.z(i) - z0, wgt = Math.exp(-dz * dz / (4 * 0.15 * 0.15)); re[i] *= wgt; im[i] *= wgt; }
        Q.sim.normalise();
        var sx = P.spread(Q.latS, Q.latT, M, Q.pr.hbar), u = Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(6.2832 * Math.random());
        Q.latX = Math.max(0.4, Math.min(L1 - 0.4, Q.latX + sx * u)); Q.latS = 0.25; Q.latT = 0;
        Q.marker = { x: Q.latX, y: z0, life: 1 };
      } else {
        var spot = Q.sim.measure(0.25, Math.random);
        Q.marker = { x: spot.x, y: spot.y, life: 1 };
      }
      Q.ref = 0;
    });
    $('q-nudge-btn').addEventListener('click', function () { A.set({ hbar: 0.1 }); });
  }

  CGH.quantum = {
    init: function () {
      canvas = $('q-canvas'); ctx = canvas.getContext('2d');
      bind(); setScene('doors');
      A.registerTab('quantum', {
        show: function () { last = performance.now(); cancelAnimationFrame(raf); raf = requestAnimationFrame(frame); },
        hide: function () { cancelAnimationFrame(raf); }
      });
      A.onChange(function () { reset(); });
    },
    state: Q
  };
})();
