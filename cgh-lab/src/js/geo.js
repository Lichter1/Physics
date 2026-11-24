/* cGħ World — shared geometry and the JavaScript twin of the shader's light-ray integrator. No DOM.
 *
 * The world is a small planet with one street running round its equator.
 * Street coordinates: a = metres along the street, l = metres across it (+ is your right when
 * facing +a), h = height above the ground. The street repeats every 120 m, 20 times round.
 *
 * Light rays follow null geodesics of the Schwarzschild metric outside the planet. Written in
 * Cartesian form they obey  x'' = -(3/2) rs L² x / r⁵  with L = |x × x'| conserved, where the
 * parameter is affine and c·dt = dλ / (1 - rs/r). This is exact, not a weak-field approximation.
 */
(function (root) {
  'use strict';

  var NT = 20, LT = 120, R = NT * LT / (2 * Math.PI), EYE = 1.7, HMAX = 52;
  var START = { a: -10, l: 0 };
  var DOORS = { a: -6, l: 0, heading: Math.PI };       // where the wave demo starts: facing the two doors
  var TOWER = { a: 40, half: 1.5, height: 50, clockTop: 46, clockBase: 3 };
  var WALL = { a: -20, half: 0.2, span: 10, height: 3, door: 1.5, doorHalf: 0.6 };

  function mod(x, m) { return x - m * Math.floor(x / m); }
  function wrapA(a) { return mod(a + Math.PI * R, 2 * Math.PI * R) - Math.PI * R; }

  /* ---------- street coordinates <-> planet frame ---------- */
  // At a = l = 0 the up direction is +y, +a is +z and +l (your right) is -x.
  function toWorld(a, l, h) {
    var th = l / R, ph = a / R, r = R + h, ct = Math.cos(th);
    return [-r * Math.sin(th), r * ct * Math.cos(ph), r * ct * Math.sin(ph)];
  }
  function fromWorld(p) {
    var r = Math.hypot(p[0], p[1], p[2]);
    return { a: R * Math.atan2(p[2], p[1]), l: -R * Math.asin(Math.max(-1, Math.min(1, p[0] / r))), h: r - R };
  }
  // Unit vectors at a street point: up, along (+a), across (+l).
  function frame(a, l) {
    var th = l / R, ph = a / R, ct = Math.cos(th), st = Math.sin(th), cp = Math.cos(ph), sp = Math.sin(ph);
    return { up: [-st, ct * cp, ct * sp], along: [0, -sp, cp], across: [-ct, -st * cp, -st * sp] };
  }

  // The yard behind the two doors (block 0) is kept empty so the stripes are clean.
  function yard(ti, al) { return ti === 0 && al < -19.5 && al > -38.5; }

  /* ---------- what stands on the ground (for walking into things, and for the wave's walls) ---------- */
  function box2(x, y, hx, hy) {
    var dx = Math.abs(x) - hx, dy = Math.abs(y) - hy;
    return Math.hypot(Math.max(dx, 0), Math.max(dy, 0)) + Math.min(Math.max(dx, dy), 0);
  }
  // Signed distance (m) from a ground point to the nearest footprint. Negative inside.
  function footprint(a, l) {
    a = wrapA(a);
    var ti = Math.floor((a + LT / 2) / LT), al = a - ti * LT, ab = Math.abs(l);
    var d = box2(ab - 16, mod(al, 20) - 10, 6, 8);                      // houses
    if (!yard(ti, al)) {
      d = Math.min(d, box2(ab - 4.6, mod(al, 12) - 6, 0.08, 0.08));      // lamp posts
      d = Math.min(d, box2(ab - 6.3, mod(al, 6) - 3, 0.5, 0.5));         // crates
    }
    d = Math.min(d, box2(ab - 8.7, mod(al + 12, 24) - 12, 0.175, 0.175)); // tree trunks
    d = Math.min(d, box2(ab - 5.2, al, 0.15, 0.15));                     // clock gantry posts
    if (ti === 0) {
      d = Math.min(d, box2(l, al - TOWER.a, TOWER.half, TOWER.half));
      var wall = box2(l, al - WALL.a, WALL.span, WALL.half);
      var doors = box2(ab - WALL.door, al - WALL.a, WALL.doorHalf, 1);
      d = Math.min(d, Math.max(wall, -doors));
    }
    return d;
  }

  /* ---------- light rays ---------- */
  function len(v) { return Math.hypot(v[0], v[1], v[2]); }
  function dot(u, v) { return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]; }
  function cross(u, v) { return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]; }

  // A direction seen by a static observer at x becomes the ray's starting coordinate velocity.
  function launch(x, n, rs) {
    var r = len(x), u = [x[0] / r, x[1] / r, x[2] / r], nr = dot(n, u), f = 1 / Math.sqrt(1 - rs / r);
    return [nr * u[0] + (n[0] - nr * u[0]) * f, nr * u[1] + (n[1] - nr * u[1]) * f, nr * u[2] + (n[2] - nr * u[2]) * f];
  }

  /* Velocity-Verlet along the ray. opts: ground (radius, or 0 for none), escape (radius),
   * maxStep (fraction of r), steps (cap). Returns where it ended, the bend and c × coordinate time. */
  function trace(x, v, rs, opts) {
    opts = opts || {};
    var ground = opts.ground || 0, esc = opts.escape || 1e6, frac = opts.maxStep || 0.02, cap = opts.steps || 200000;
    x = x.slice(); v = v.slice();
    var Lv = cross(x, v), L2 = dot(Lv, Lv), k = -1.5 * rs * L2, ct = 0, n = 0, r = len(x), end = 'max';
    function acc(p, rr) { var s = k / Math.pow(rr, 5); return [s * p[0], s * p[1], s * p[2]]; }
    var a0 = acc(x, r);
    for (; n < cap; n++) {
      var h = frac * r / len(v);
      var xn = [x[0] + v[0] * h + 0.5 * a0[0] * h * h, x[1] + v[1] * h + 0.5 * a0[1] * h * h, x[2] + v[2] * h + 0.5 * a0[2] * h * h];
      var rn = len(xn), a1 = acc(xn, rn);
      v = [v[0] + 0.5 * (a0[0] + a1[0]) * h, v[1] + 0.5 * (a0[1] + a1[1]) * h, v[2] + 0.5 * (a0[2] + a1[2]) * h];
      ct += 0.5 * h * (1 / (1 - rs / r) + 1 / (1 - rs / rn));
      x = xn; r = rn; a0 = a1;
      if (ground && r < ground) { end = 'ground'; break; }
      if (r > esc && dot(x, v) > 0) { end = 'sky'; break; }
    }
    return { x: x, v: v, ct: ct, steps: n, end: end };
  }

  /* Where the sky stops, as an elevation angle above your horizontal (negative = below eye level).
   * Outside the photon sphere (R >= 1.5 rs) the edge is the ray that grazes the ground.
   * Inside it, rays escape only if their impact parameter is below 3√3 rs / 2: the sky is a cone overhead. */
  function horizon(rs, Rp, re) {
    if (rs <= 0) return -Math.acos(Rp / re);
    var b = Rp >= 1.5 * rs ? Rp / Math.sqrt(1 - rs / Rp) : 1.5 * Math.sqrt(3) * rs;
    var s = Math.min(1, b * Math.sqrt(1 - rs / re) / re);
    return Rp >= 1.5 * rs ? -Math.acos(s) : Math.PI / 2 - Math.asin(s);
  }

  // Rate of a static clock at height h compared with one at your eye.
  function clockRate(rs, h) { return Math.sqrt((1 - rs / (R + h)) / (1 - rs / (R + EYE))); }

  /* ---------- the full scene, mirrored from the shader (for the probe and for walking into things) ---------- */
  // Building heights use exact integer arithmetic so the GPU and JavaScript agree.
  function houseHeight(cell, side) { return 9 + 3 * mod(cell * 7 + side * 3 + 5, 6); }
  function sdBox(x, y, z, bx, by, bz) {
    var qx = Math.abs(x) - bx, qy = Math.abs(y) - by, qz = Math.abs(z) - bz;
    return Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0)) + Math.min(Math.max(qx, Math.max(qy, qz)), 0);
  }
  var MATS = ['the street', 'a house', '', 'a post', 'a lamp', 'a crate', '', 'a tree trunk', 'a tree', 'the street clock',
              'the clock tower', 'the wall with two doors', 'the bell over the left door'];
  // Distance to the nearest surface from a planet-frame point, and its material (same numbering as the shader).
  function scene(x) {
    var r = len(x), a = R * Math.atan2(x[2], x[1]), ti = Math.floor((a + 60) / 120);
    var l = -R * Math.asin(Math.max(-1, Math.min(1, x[0] / r))), h = r - R, al = l < 0 ? -l : l;
    a -= 120 * ti;
    if (h > HMAX) return { d: h - HMAX + 0.5, mat: 0 };
    var d = h, mat = 0, e, k;
    function take(v, m) { if (v < d) { d = v; mat = m; } }
    var side = l > 0 ? 1 : 0, c0 = Math.floor(a / 20), f = a / 20 - c0, cn = c0 + (f < 0.5 ? -1 : 1);
    for (k = 0; k < 2; k++) {
      var c = k === 0 ? c0 : cn, H = houseHeight(ti * 6 + c, side);
      take(sdBox(al - 16, h - 0.5 * H, a - 20 * c - 10, 6, 0.5 * H, 8), 1);
    }
    if (!yard(ti, a)) {
      var qz = mod(a, 12) - 6;
      take(sdBox(al - 4.6, h - 2.2, qz, 0.08, 2.2, 0.08), 3);
      take(sdBox(al - 4.6, h - 4.5, qz, 0.25, 0.14, 0.25), 4);
      take(sdBox(al - 6.3, h - 0.5, mod(a, 6) - 3, 0.5, 0.5, 0.5), 5);
    }
    var tz = mod(a + 12, 24) - 12;
    take(sdBox(al - 8.7, h - 1.3, tz, 0.175, 1.3, 0.175), 7);
    take(sdBox(al - 8.7, h - 4.0, tz, 1.3, 1.3, 1.3), 8);
    take(Math.min(sdBox(al - 5.2, h - 3.5, a, 0.15, 3.5, 0.15), sdBox(l, h - 6.85, a, 5.35, 0.15, 0.15)), 3);
    take(Math.max(Math.hypot(l, h - 5.4) - 1.25, Math.abs(a) - 0.06), 9);
    if (ti === 0) {
      take(sdBox(l, h - 25, a - 40, 1.5, 25, 1.5), 10);
      take(Math.max(sdBox(l, h - 1.5, a + 20, 10, 1.5, 0.2), -sdBox(al - 1.5, h - 1.5, a + 20, 0.6, 2, 1)), 11);
      take(sdBox(l - 1.5, h - 3.25, a + 20, 0.3, 0.2, 0.3), 12);
    }
    return { d: d, mat: mat };
  }

  // From a direction in the walker's frame to the frame of someone standing still there (inverse aberration).
  function deaberrate(np, beta) {
    var b2 = dot(beta, beta);
    if (b2 === 0) return { n: np.slice(), D: 1 };
    var g = 1 / Math.sqrt(1 - b2), nb = dot(np, beta), b = Math.sqrt(b2), k = ((g - 1) * nb / b - g * b) / b, den = g * (1 - nb);
    return { n: [(np[0] + k * beta[0]) / den, (np[1] + k * beta[1]) / den, (np[2] + k * beta[2]) / den], D: 1 / den };
  }

  /* The shader's light-ray march, step for step, keeping the path. Returns what the ray hit (kind 1),
   * or the sky (kind 2), with c × travel time ct and the path points. */
  function march(eye, n, rs, keep) {
    var re = len(eye), u = [eye[0] / re, eye[1] / re, eye[2] / re], x = eye.slice(), r = re, ct = 0, s = 0, mat = 0, kind = 0;
    var nu = dot(n, u), f = 1 / Math.sqrt(1 - rs / re), v = [0, 1, 2].map(function (i) { return nu * u[i] + (n[i] - nu * u[i]) * f; });
    var Lv = cross(x, v), k = -1.5 * rs * dot(Lv, Lv), acc = x.map(function (c) { return k * c / Math.pow(r, 5); });
    var path = keep ? [x.slice()] : null;
    for (var i = 0; i < 260; i++) {
      var h = r - R, d;
      if (h < HMAX) { var sc = scene(x); d = sc.d; mat = sc.mat; if (d < 0.004 + 0.0015 * s) { kind = 1; break; } }
      else d = h - HMAX + 0.5;
      if (r > 30 * R) { kind = 2; break; }
      var dl = Math.max(Math.min(0.8 * d, (h > 2 * HMAX ? 0.15 : 0.05) * r), 0.003 * s + 0.01), hs = dl / len(v);
      var xn = [0, 1, 2].map(function (j) { return x[j] + v[j] * hs + 0.5 * acc[j] * hs * hs; });
      var rn = len(xn), accn = xn.map(function (c) { return k * c / Math.pow(rn, 5); });
      for (var j = 0; j < 3; j++) v[j] += 0.5 * (acc[j] + accn[j]) * hs;
      ct += 0.5 * hs * (1 / (1 - rs / r) + 1 / (1 - rs / rn));
      if (rn < R) {
        var fr = (r - R) / Math.max(r - rn, 1e-6);
        ct -= (1 - fr) * hs / (1 - rs / R);
        x = [0, 1, 2].map(function (j) { return x[j] + (xn[j] - x[j]) * fr; }); s += fr * dl; mat = 0; kind = 1;
        if (keep) path.push(x.slice());
        break;
      }
      x = xn; r = rn; acc = accn; s += dl;
      if (keep) path.push(x.slice());
    }
    return { kind: kind, mat: mat, x: x, v: v, ct: ct, s: s, path: path, n0: n };
  }

  var api = {
    R: R, NT: NT, LT: LT, EYE: EYE, HMAX: HMAX, START: START, DOORS: DOORS, TOWER: TOWER, WALL: WALL,
    mod: mod, wrapA: wrapA, toWorld: toWorld, fromWorld: fromWorld, frame: frame, footprint: footprint,
    launch: launch, trace: trace, horizon: horizon, clockRate: clockRate, dot: dot, cross: cross, len: len,
    houseHeight: houseHeight, scene: scene, MATS: MATS, deaberrate: deaberrate, march: march
  };
  root.CGH = root.CGH || {};
  root.CGH.geo = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
