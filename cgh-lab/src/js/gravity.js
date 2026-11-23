/* cGħ Lab — Gravity: two scenes about clocks.
 *   Time tower         uniform gravity (an accelerating frame): rate(h) = 1 + g h / c², exact.
 *   Black-hole mission circular equatorial orbits around a Kerr black hole (the Interstellar set-up).
 */
(function () {
  'use strict';
  var P = CGH.physics, A = CGH.app, $ = A.$, K = P.K;
  var canvas, ctx, raf = 0, last = 0, scene = 'tower';

  /* ================= tower ================= */
  var LO = -4, HI = 10, STOREY = P.REF.floor;
  var T = { floor: 0, lapse: 0, tau: [], me: 0 };
  function resetTower() { T.tau = []; for (var i = LO; i <= HI; i++) T.tau.push(0); T.me = 0; }
  function field() {
    var g = P.earthGravity(A.state.G), c = A.state.c;
    return { g: g, c: c, per: P.rindlerExcess(g, STOREY, c), horizon: P.rindlerHorizon(g, c) };
  }
  function rate(f, i) { return 1 + f.per * i; }                 // relative to the street clock
  function lowestFloor(f) { for (var i = LO; i <= 0; i++) if (rate(f, i) > 0.02) return i; return 0; }

  function stepTower(dt) {
    var f = field();
    var lo = lowestFloor(f);
    if (T.floor < lo) { T.floor = lo; $('floor').value = lo; }
    var dme = dt * Math.pow(10, T.lapse);                        // your own seconds: your watch always feels normal
    var dstreet = dme / rate(f, T.floor);
    for (var i = LO; i <= HI; i++) { var r = rate(f, i); if (r > 0) T.tau[i - LO] += r * dstreet; }
    T.me += dme;
    return f;
  }

  // Colour of light from a clock that runs D times faster than yours: blue above you, red below.
  function tint(D) {
    var t = Math.max(-1, Math.min(1, Math.log2(D) / 1.6));
    var w = [236, 232, 214], r = [214, 62, 44], b = [72, 118, 235], e = t < 0 ? r : b, a = Math.abs(t);
    return 'rgb(' + Math.round(w[0] + (e[0] - w[0]) * a) + ',' + Math.round(w[1] + (e[1] - w[1]) * a) + ',' + Math.round(w[2] + (e[2] - w[2]) * a) + ')';
  }
  function relText(f, i, j) {                                    // rate of floor i as judged from floor j
    var d = f.per * (i - j) / rate(f, j);
    if (i === j) return '×1';
    if (Math.abs(d) < 1e-3) return '×1 ' + (d > 0 ? '+ ' : '− ') + A.num(Math.abs(d), 2);
    return '×' + A.num(1 + d, 3);
  }

  function drawTower(f) {
    var s = A.fit(canvas, 2), w = s.w, h = s.h;
    ctx.setTransform(s.dpr, 0, 0, s.dpr, 0, 0);
    var ink = A.css('--ink'), muted = A.css('--muted'), line = A.css('--line'), surf = A.css('--surface'), warn = A.css('--warn');
    ctx.fillStyle = A.css('--sunk'); ctx.fillRect(0, 0, w, h);

    var narrow = w < 520;
    var n = HI - LO + 1, top = !narrow && w > 700 ? 26 : 12, fh = (h - top - 10) / n;
    var colLabel = narrow ? 30 : 64, cellW = narrow ? 64 : 120;
    var x0 = narrow ? 8 : Math.max(16, w * 0.08);
    var xCell = x0 + colLabel, xDial = xCell + cellW + (narrow ? 20 : 34), xTime = xDial + (narrow ? 20 : 30), xRate = narrow ? w - 8 : xTime + 150;
    function yTop(i) { return top + (HI - i) * fh; }
    var yStreet = yTop(0) + fh;

    // ground
    ctx.fillStyle = line; ctx.globalAlpha = 0.55; ctx.fillRect(0, yStreet, w, h - yStreet); ctx.globalAlpha = 1;
    ctx.strokeStyle = muted; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(0, yStreet); ctx.lineTo(w, yStreet); ctx.stroke();

    ctx.textBaseline = 'middle';
    for (var i = HI; i >= LO; i--) {
      var y = yTop(i), r = rate(f, i), alive = r > 0, mid = y + fh / 2;
      ctx.fillStyle = surf; ctx.strokeStyle = muted; ctx.lineWidth = 1;
      ctx.fillRect(xCell, y, cellW, fh); ctx.strokeRect(xCell + 0.5, y + 0.5, cellW - 1, fh - 1);
      // window, lit in the colour its light has when it reaches your floor
      ctx.fillStyle = alive ? tint(r / rate(f, T.floor)) : '#333';
      ctx.fillRect(xCell + cellW * 0.58, y + fh * 0.22, cellW * 0.3, fh * 0.5);
      ctx.strokeStyle = muted; ctx.strokeRect(xCell + cellW * 0.58 + 0.5, y + fh * 0.22 + 0.5, cellW * 0.3 - 1, fh * 0.5 - 1);

      ctx.fillStyle = muted; ctx.font = '12px ' + A.css('--font-data'); ctx.textAlign = 'right';
      ctx.fillText(narrow ? String(i) : (i === 0 ? 'street' : (i > 0 ? 'floor ' + i : 'level ' + i)), xCell - 8, mid);

      if (i === 0 || i === T.floor) {                            // the twin stays at street level; you are the filled one
        var you = i === T.floor, px = xCell + cellW * (you ? 0.28 : 0.14), py = y + fh * 0.5, sz = Math.min(fh * 0.34, 11);
        ctx.strokeStyle = ink; ctx.fillStyle = you ? ink : surf; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(px, py - sz * 0.75, sz * 0.42, 0, 6.2832); ctx.fill(); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(px - sz * 0.55, py + sz); ctx.lineTo(px, py - sz * 0.25); ctx.lineTo(px + sz * 0.55, py + sz); ctx.closePath(); ctx.fill(); ctx.stroke();
      }
      if (!alive) {
        ctx.fillStyle = warn; ctx.textAlign = 'left'; ctx.font = '12px ' + A.css('--font-text');
        ctx.fillText('past the horizon', xDial - 8, mid);
        continue;
      }
      var rad = Math.min(fh * 0.38, 13), ang = 6.2832 * ((T.tau[i - LO] / 10) % 1);
      ctx.beginPath(); ctx.arc(xDial, mid, rad, 0, 6.2832); ctx.fillStyle = surf; ctx.fill(); ctx.strokeStyle = ink; ctx.lineWidth = 1.25; ctx.stroke();
      ctx.beginPath(); ctx.moveTo(xDial, mid); ctx.lineTo(xDial + Math.sin(ang) * rad * 0.82, mid - Math.cos(ang) * rad * 0.82);
      ctx.strokeStyle = warn; ctx.lineWidth = 1.75; ctx.stroke();

      ctx.fillStyle = ink; ctx.font = (i === T.floor ? '500 ' : '') + '13px ' + A.css('--font-data'); ctx.textAlign = 'left';
      ctx.fillText(A.clock(T.tau[i - LO]), xTime, mid);
      ctx.fillStyle = i === T.floor ? ink : muted; ctx.textAlign = narrow ? 'right' : 'left';
      ctx.fillText(i === T.floor ? 'you' : relText(f, i, T.floor), xRate, mid);
    }
    // the horizon, if it is within the drawing
    var yH = yStreet + (f.horizon / STOREY) * fh;
    if (yH < h - 2) {
      ctx.strokeStyle = warn; ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(0, yH); ctx.lineTo(w, yH); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = warn; ctx.font = '12px ' + A.css('--font-text'); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
      ctx.fillText('horizon, ' + A.length(f.horizon, 2) + ' down', w - 8, yH - 3);
    }
    if (!narrow && w > 700) {
      ctx.fillStyle = muted; ctx.font = '12px ' + A.css('--font-text'); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('clock rate seen from your floor', xRate, 6);
    }
  }

  function towerReadouts(f) {
    var n = A.num, d = T.me - T.tau[0 - LO];
    var rows = [
      ['Gravity here, g', n(f.g) + ' m/s²'],
      ['Clock-rate change per storey, g·h/c²', n(f.per, 2)],
      ['Horizon below the street, c²/g', A.length(f.horizon, 3)],
      ['Your watch', A.clock(T.me)],
      ['Street twin’s watch', A.clock(T.tau[0 - LO])],
      [d >= 0 ? 'You are older by' : 'You are younger by', A.duration(Math.abs(d), 3)]
    ];
    $('grav-readouts').innerHTML = rows.map(function (r) { return '<div class="readout"><dt>' + r[0] + '</dt><dd>' + r[1] + '</dd></div>'; }).join('');
  }

  /* ================= black hole ================= */
  var H = { massExp: 38.3, spinS: 0, orbitS: 0, stay: 1, phiP: 0, phiS: 0, tFar: 0 };
  function hole() {
    var k = A.state, M = Math.pow(10, H.massExp);
    var delta = Math.pow(10, -14 * H.spinS);
    if (H.film) delta = 1.3e-14;
    var kk = P.kerr(delta);
    var dr = Math.pow(10, -9 + 11 * H.orbitS) - 1e-9;
    var rP = kk.rIsco + dr, rS = Math.max(10, 2 * rP);
    var fP = P.kerrDilation(rP, kk.a), fS = P.kerrDilation(rS, kk.a);
    return {
      M: M, kk: kk, rP: rP, rS: rS, fP: fP, fS: fS,
      rg: k.G * M / (k.c * k.c), tg: k.G * M / Math.pow(k.c, 3),
      omP: P.kerrOmega(rP, kk.a), omS: P.kerrOmega(rS, kk.a),
      hawking: P.hawking(M, delta, k)
    };
  }
  function massName(M) {
    var list = [[1, 'a bag of sugar'], [70, 'a person'], [1500, 'a car'], [1e12, 'a mountain'], [7.3e22, 'the Moon'], [K.Mearth, 'Earth'], [K.Msun, 'the Sun'], [8.5e36, 'the Milky Way’s black hole'], [1.99e38, 'Gargantua']];
    var best = list[0], bd = 1e9;
    list.forEach(function (e) { var d = Math.abs(Math.log10(M / e[0])); if (d < bd) { bd = d; best = e; } });
    return bd < 0.12 ? best[1] : (M > 1e29 ? A.num(M / K.Msun, 2) + ' Suns' : A.mass(M, 2));
  }
  function spinText(kk) {
    if (kk.delta >= 1) return 'none';
    if (kk.delta > 1e-3) return 'a = ' + kk.a.toFixed(3);
    return 'a = 1 − ' + A.num(kk.delta, 2);
  }

  function stepHole(dt) {
    var b = hole();
    var dCoord = dt * (2 * Math.PI / b.omP) / 6;                 // one planet orbit every six seconds of yours
    H.phiP += b.omP * dCoord; H.phiS += b.omS * dCoord; H.tFar += dCoord * b.tg;
    return b;
  }

  function drawHole(b) {
    var s = A.fit(canvas, 2), w = s.w, h = s.h;
    ctx.setTransform(s.dpr, 0, 0, s.dpr, 0, 0);
    var ink = A.css('--ink'), muted = A.css('--muted'), surf = A.css('--surface'), focus = A.css('--focus'), line = A.css('--line');
    ctx.fillStyle = A.css('--sunk'); ctx.fillRect(0, 0, w, h);
    var side = w > h * 1.25;
    var dw = side ? w * 0.56 : w, dh = side ? h : h * 0.58;
    var cx = dw / 2, cy = dh / 2, Rmax = Math.min(dw, dh) / 2 - 14, Rh = Rmax * 0.2;
    var rH = b.kk.rHorizon, span = Math.log(b.rS * 1.15 / rH);
    function R(r) { return Rh + (Rmax - Rh) * Math.log(Math.max(r, rH) / rH) / span; }
    function ring(r, style, dash, lw) {
      ctx.beginPath(); ctx.arc(cx, cy, R(r), 0, 6.2832); ctx.setLineDash(dash || []); ctx.strokeStyle = style; ctx.lineWidth = lw || 1; ctx.stroke(); ctx.setLineDash([]);
    }
    ring(b.rS, muted, [], 1);
    ring(b.kk.rIsco, muted, [2, 4], 1);
    ring(b.rP, focus, [], 1.75);
    ctx.beginPath(); ctx.arc(cx, cy, R(rH), 0, 6.2832); ctx.fillStyle = ink; ctx.fill();
    if (b.kk.a > 0.01) {                                          // spin arrow on the hole
      ctx.strokeStyle = surf; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(cx, cy, Rh * 0.55, -2.4, 0.9); ctx.stroke();
      var ax = cx + Rh * 0.55 * Math.cos(-2.4), ay = cy + Rh * 0.55 * Math.sin(-2.4);
      ctx.beginPath(); ctx.moveTo(ax - 5, ay + 3); ctx.lineTo(ax, ay); ctx.lineTo(ax + 6, ay + 4); ctx.stroke();
    }
    // bodies (angles run anticlockwise, the way the hole spins)
    var px = cx + R(b.rP) * Math.cos(-H.phiP), py = cy + R(b.rP) * Math.sin(-H.phiP);
    ctx.beginPath(); ctx.arc(px, py, 6, 0, 6.2832); ctx.fillStyle = focus; ctx.fill(); ctx.strokeStyle = surf; ctx.lineWidth = 1.5; ctx.stroke();
    var sx = cx + R(b.rS) * Math.cos(-H.phiS), sy = cy + R(b.rS) * Math.sin(-H.phiS);
    ctx.fillStyle = surf; ctx.strokeStyle = ink; ctx.lineWidth = 1.5; ctx.fillRect(sx - 5, sy - 5, 10, 10); ctx.strokeRect(sx - 5, sy - 5, 10, 10);

    ctx.font = '12px ' + A.css('--font-text'); ctx.fillStyle = muted; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('radius drawn on a log scale', 8, 6);
    // labels sit outside each orbit, on the far side from the hole
    function tag(text, x, y, color) {
      var d = Math.hypot(x - cx, y - cy) || 1, ux = (x - cx) / d, uy = (y - cy) / d;
      ctx.fillStyle = color; ctx.textAlign = ux > 0.3 ? 'left' : ux < -0.3 ? 'right' : 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(text, x + ux * 13, y + uy * 13);
    }
    tag('ship', sx, sy, muted);
    tag('planet', px, py, focus);

    // three clocks
    var bx = side ? dw + 6 : 14, by = side ? Math.max(18, h * 0.14) : dh + 4, bw = (side ? w - dw : w) - 22, rowH = side ? Math.min(92, (h - by - 8) / 3) : (h - dh - 8) / 3;
    var rows = [
      ['On the planet', H.tFar / b.fP, focus, '1 s here is ' + A.duration(b.fP, 3) + ' far away'],
      ['On the ship', H.tFar / b.fS, ink, '1 s here is ' + A.duration(b.fS, 3) + ' far away'],
      ['Far away', H.tFar, muted, 'the reference clock']
    ];
    rows.forEach(function (r, i) {
      var y = by + i * rowH;
      ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillStyle = r[2]; ctx.fillRect(bx, y + 3, 3, rowH - 12);
      ctx.fillStyle = muted; ctx.font = '12px ' + A.css('--font-text'); ctx.fillText(r[0], bx + 10, y);
      ctx.fillStyle = ink; ctx.font = '500 ' + (rowH > 60 ? 20 : 16) + 'px ' + A.css('--font-data'); ctx.fillText(A.clock(r[1]), bx + 10, y + 16);
      if (rowH > 52) { ctx.fillStyle = muted; ctx.font = '12px ' + A.css('--font-text'); ctx.fillText(r[3], bx + 10, y + (rowH > 60 ? 42 : 36), bw); }
    });
    void line;
  }

  function holeReadouts(b) {
    var n = A.num, stay = H.stay * 3600;
    var rows = [
      [n(H.stay) + ' h on the planet is, on the ship', A.duration(stay * b.fP / b.fS, 3)],
      ['and far away', A.duration(stay * b.fP, 3)],
      ['Time runs slower on the planet by', '×' + n(b.fP, 4)],
      ['Horizon radius', A.length(b.kk.rHorizon * b.rg, 3)],
      ['Planet orbit', n(b.rP, 4) + ' GM/c² = ' + A.length(b.rP * b.rg, 3)],
      ['One orbit, far-away time', A.duration(2 * Math.PI / b.omP * b.tg, 3)],
      ['Hawking temperature', n(b.hawking.temperature, 2) + ' K'],
      ['Evaporates in about', A.duration(b.hawking.lifetime, 2)]
    ];
    $('grav-readouts').innerHTML = rows.map(function (r) { return '<div class="readout"><dt>' + r[0] + '</dt><dd>' + r[1] + '</dd></div>'; }).join('');
    $('mass-out').textContent = massName(b.M);
    $('spin-out').textContent = spinText(b.kk);
    $('orbit-out').textContent = b.rP - b.kk.rIsco < 1e-6 ? 'innermost stable' : n(b.rP, 3) + ' GM/c²';
    $('stay-out').textContent = n(H.stay) + ' h';
  }

  /* ================= shared ================= */
  var TEXT = {
    tower: {
      title: 'Time tower',
      premise: 'A tower standing in uniform gravity, with a clock on every floor. All of them were set together at street level and carried to their floors. Choose a floor and wait there.',
      explain: '<div><h3>What to try</h3><ul>' +
        '<li>Slow light to 10 m/s, go to the tenth floor and let time pass. Your street-level twin falls behind. Go back down and the gap stays: you really are older.</li>' +
        '<li>Go below the street. Clocks there crawl, and a few metres down there is a level where they stop altogether, as judged from above.</li>' +
        '<li>Raise <span class="sym sym-g">G</span>. Stronger gravity does the same thing as slower light: only <span class="formula"><i>gh</i>/<i>c</i>²</span> matters.</li></ul></div>' +
        '<div><h3>The arithmetic</h3><p class="formula">rate(<i>h</i>) = 1 + <i>gh</i>/<i>c</i>² &nbsp; horizon at <i>h</i> = −<i>c</i>²/<i>g</i></p>' +
        '<p class="note">This is exact for a rigid tower held at constant acceleration, which by the equivalence principle is what standing in a uniform field means. It uses Earth’s pull with your <span class="sym">G</span>. With Earth’s real mass and <span class="sym">c</span> = 10 m/s the planet would sit far inside its own horizon, so the scene keeps the local pull and drops the planet. The real effect, 1.1×10⁻¹⁶ per metre, has been measured across a single millimetre with optical clocks.</p></div>'
    },
    hole: {
      title: 'Black-hole mission',
      premise: 'A planet circles a spinning black hole while the ship waits in a wide orbit. The crew go down for a few hours. The picture runs on far-away time: one planet orbit every six seconds.',
      explain: '<div><h3>What to try</h3><ul>' +
        '<li>With no spin, move the planet to the innermost stable orbit. Time there runs only 1.41 times slower. A film needs more.</li>' +
        '<li>Press “Miller’s planet”. The hole must spin within 1.3 parts in 10¹⁴ of the maximum before the innermost orbit gives one hour for seven years.</li>' +
        '<li>Change the mass or the dials. The ratio does not move: it depends only on spin and on the orbit measured in units of <span class="formula"><i>GM</i>/<i>c</i>²</span>. The constants set how big and how slow everything is.</li>' +
        '<li>Raise <span class="sym sym-h">ħ</span> and watch the last two readouts. Hawking’s temperature is the one widely trusted formula that needs all three constants. Nobody has measured it.</li></ul></div>' +
        '<div><h3>The arithmetic</h3><p class="formula">d<i>t</i>/d<i>τ</i> = (1 + <i>a</i>/<i>r</i><sup>3/2</sup>) / √(1 − 3/<i>r</i> + 2<i>a</i>/<i>r</i><sup>3/2</sup>)<br><i>T</i><sub>H</sub> = <i>ħc</i>³/8π<i>GMk</i><sub>B</sub> (no spin)</p>' +
        '<p class="note">Radii are in units of <i>GM</i>/<i>c</i>², orbits are circular, prograde and equatorial. The lifetime is the standard estimate for a non-spinning hole radiating photons only.</p></div>'
    }
  };

  function setScene(name) {
    scene = name;
    Array.prototype.forEach.call($('grav-scene').children, function (b) { b.setAttribute('aria-pressed', b.getAttribute('data-scene') === name ? 'true' : 'false'); });
    $('tower-controls').hidden = name !== 'tower';
    $('hole-controls').hidden = name !== 'hole';
    $('grav-title').textContent = TEXT[name].title;
    $('grav-premise').textContent = TEXT[name].premise;
    $('grav-explain').innerHTML = TEXT[name].explain;
    nudge();
  }
  function nudge() {
    var f = field(), el = $('grav-nudge');
    el.hidden = scene !== 'tower' || f.per > 1e-3;
    if (!el.hidden) $('grav-nudge-text').textContent = 'With these dials one storey changes a clock’s rate by ' + A.num(f.per, 2) + '. The effect is real and far too small to watch.';
  }

  var roClock = 0;
  function frame(now) {
    raf = requestAnimationFrame(frame);
    var dt = Math.min(0.05, (now - last) / 1000 || 0); last = now;
    roClock += dt;
    if (scene === 'tower') {
      var f = stepTower(dt); drawTower(f);
      if (roClock > 0.12) { roClock = 0; towerReadouts(f); $('floor-out').textContent = T.floor === 0 ? 'street' : String(T.floor); }
    } else {
      var b = stepHole(dt); drawHole(b);
      if (roClock > 0.12) { roClock = 0; holeReadouts(b); }
    }
  }

  function bind() {
    Array.prototype.forEach.call($('grav-scene').children, function (b) {
      b.addEventListener('click', function () { setScene(b.getAttribute('data-scene')); });
    });
    $('floor').addEventListener('input', function (e) { T.floor = Number(e.target.value); });
    $('tower-speed').addEventListener('input', function (e) { T.lapse = Number(e.target.value); $('tower-speed-out').textContent = '×' + A.num(Math.pow(10, T.lapse), 2); });
    $('tower-speed-out').textContent = '×1';
    $('tower-sync').addEventListener('click', resetTower);
    $('grav-nudge-btn').addEventListener('click', function () { A.set({ c: 10 }); });
    $('mass').addEventListener('input', function (e) { H.massExp = Number(e.target.value); });
    $('spin').addEventListener('input', function (e) { H.spinS = Number(e.target.value); H.film = false; });
    $('orbit').addEventListener('input', function (e) { H.orbitS = Number(e.target.value); });
    $('stay').addEventListener('input', function (e) { H.stay = Number(e.target.value); });
    $('hole-film').addEventListener('click', function () {
      H.massExp = 38.3; H.spinS = 1; H.film = true; H.orbitS = 0; H.stay = 1; H.tFar = 0;
      $('mass').value = 38.3; $('spin').value = 1; $('orbit').value = 0; $('stay').value = 1;
    });
    $('hole-plain').addEventListener('click', function () {
      H.spinS = 0; H.film = false; H.orbitS = 0; H.tFar = 0; $('spin').value = 0; $('orbit').value = 0;
    });
  }

  CGH.gravity = {
    init: function () {
      canvas = $('grav-canvas'); ctx = canvas.getContext('2d');
      resetTower(); bind(); setScene('tower');
      A.registerTab('gravity', {
        show: function () { nudge(); last = performance.now(); cancelAnimationFrame(raf); raf = requestAnimationFrame(frame); },
        hide: function () { cancelAnimationFrame(raf); }
      });
      A.onChange(function () { if (A.currentTab() === 'gravity') nudge(); });
    },
    tower: T, hole: H
  };
})();
