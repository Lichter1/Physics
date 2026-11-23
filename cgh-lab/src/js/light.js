/* cGħ Lab — Light: a first-person ride down a street with a tunable speed of light.
 *
 * The vertex shader draws every vertex at the event where its light left it (the intersection of
 * its worldline with the viewer's past light cone), Lorentz-transformed into the viewer's frame.
 * CGH.physics.apparent() is the same arithmetic in JavaScript and is what the tests check.
 */
(function () {
  'use strict';
  var P = CGH.physics, A = CGH.app, $ = A.$;
  var TILE = 120, EYE = 1.7, SPAN = 4, LAMP_PERIOD = 2, CLOCK_PERIOD = 20, TRAM_EFFORT = 7;

  /* ---------- geometry ---------- */
  function Mesh() { this.v = []; this.i = []; this.n = 0; }
  // A subdivided parallelogram. Straight edges turn into curves at speed, so nothing may be coarse.
  Mesh.prototype.quad = function (o, eu, ev, nu, nv, col, mat, uv0, uv1) {
    var nx = eu[1] * ev[2] - eu[2] * ev[1], ny = eu[2] * ev[0] - eu[0] * ev[2], nz = eu[0] * ev[1] - eu[1] * ev[0];
    var nl = Math.hypot(nx, ny, nz); nx /= nl; ny /= nl; nz /= nl;
    var lu = Math.hypot(eu[0], eu[1], eu[2]), lv = Math.hypot(ev[0], ev[1], ev[2]);
    uv0 = uv0 || [0, 0]; uv1 = uv1 || [lu, lv];
    var base = this.n, a, b;
    for (b = 0; b <= nv; b++) for (a = 0; a <= nu; a++) {
      var s = a / nu, t = b / nv;
      this.v.push(o[0] + s * eu[0] + t * ev[0], o[1] + s * eu[1] + t * ev[1], o[2] + s * eu[2] + t * ev[2],
        nx, ny, nz, col[0], col[1], col[2], uv0[0] + s * (uv1[0] - uv0[0]), uv0[1] + t * (uv1[1] - uv0[1]), mat);
      this.n++;
    }
    for (b = 0; b < nv; b++) for (a = 0; a < nu; a++) {
      var i0 = base + b * (nu + 1) + a, i1 = i0 + 1, i2 = i0 + nu + 1, i3 = i2 + 1;
      this.i.push(i0, i1, i2, i1, i3, i2);
    }
  };
  Mesh.prototype.box = function (c, s, col, mat, maxEdge, opts) {
    opts = opts || {};
    var hx = s[0] / 2, hy = s[1] / 2, hz = s[2] / 2;
    var kx = Math.max(1, Math.ceil(s[0] / maxEdge)), ky = Math.max(1, Math.ceil(s[1] / maxEdge)), kz = Math.max(1, Math.ceil(s[2] / maxEdge));
    this.quad([c[0] + hx, c[1] - hy, c[2] + hz], [0, 0, -s[2]], [0, s[1], 0], kz, ky, col, mat);
    this.quad([c[0] - hx, c[1] - hy, c[2] - hz], [0, 0, s[2]], [0, s[1], 0], kz, ky, col, mat);
    this.quad([c[0] - hx, c[1] - hy, c[2] + hz], [s[0], 0, 0], [0, s[1], 0], kx, ky, col, mat);
    this.quad([c[0] + hx, c[1] - hy, c[2] - hz], [-s[0], 0, 0], [0, s[1], 0], kx, ky, col, mat);
    this.quad([c[0] - hx, c[1] + hy, c[2] + hz], [s[0], 0, 0], [0, 0, -s[2]], kx, kz, opts.top || col, opts.topMat === undefined ? mat : opts.topMat);
    if (opts.bottom) this.quad([c[0] - hx, c[1] - hy, c[2] - hz], [s[0], 0, 0], [0, 0, s[2]], kx, kz, col, opts.topMat === undefined ? mat : opts.topMat);
  };

  var M_GROUND = 0, M_WALL = 1, M_PLAIN = 2, M_CLOCK = 3, M_LAMP = 4, M_SKY = 5, M_TRAM = 6, M_LEAF = 7, M_TRAMCLOCK = 8;

  function buildTile() {
    var m = new Mesh(), k, side;
    // ground: 1 m cells by the road, wider cells far out (still 1 m along the street so edges meet)
    m.quad([-24, 0, -60], [0, 0, 120], [48, 0, 0], 120, 48, [1, 1, 1], M_GROUND);
    m.quad([-152, 0, -60], [0, 0, 120], [128, 0, 0], 120, 16, [1, 1, 1], M_GROUND);
    m.quad([24, 0, -60], [0, 0, 120], [128, 0, 0], 120, 16, [1, 1, 1], M_GROUND);

    var z0 = [-58, -42, -24, -6, 12, 30, 46], len = [14, 16, 16, 16, 16, 14, 12];
    var hL = [15, 21, 12, 18, 9, 24, 15], hR = [12, 18, 24, 9, 15, 21, 12];
    var walls = [[0.62, 0.36, 0.28], [0.80, 0.72, 0.56], [0.55, 0.60, 0.66], [0.86, 0.82, 0.74], [0.70, 0.46, 0.34], [0.60, 0.64, 0.58], [0.78, 0.66, 0.60]];
    for (k = 0; k < 7; k++) {
      m.box([-16, hL[k] / 2, z0[k] + len[k] / 2], [12, hL[k], len[k]], walls[k], M_WALL, 2, { top: [0.32, 0.32, 0.34], topMat: M_PLAIN });
      m.box([16, hR[k] / 2, z0[k] + len[k] / 2], [12, hR[k], len[k]], walls[(k + 3) % 7], M_WALL, 2, { top: [0.32, 0.32, 0.34], topMat: M_PLAIN });
    }
    for (side = -1; side <= 1; side += 2) {
      for (k = 0; k < 10; k++) {                                   // lamp posts, every 12 m
        var lz = -54 + 12 * k;
        m.box([side * 4.6, 2.2, lz], [0.16, 4.4, 0.16], [0.2, 0.21, 0.23], M_PLAIN, 1.1);
        m.box([side * 4.6, 4.5, lz], [0.5, 0.28, 0.5], [1, 0.86, 0.5], M_LAMP, 0.5, { bottom: true });
      }
      for (k = 0; k < 20; k++) {                                   // one-metre crates: rulers you can ride past
        m.box([side * 6.3, 0.5, -57 + 6 * k], [1, 1, 1], k % 2 ? [0.78, 0.22, 0.2] : [0.2, 0.4, 0.75], M_PLAIN, 0.5);
      }
      for (k = 0; k < 5; k++) {                                    // trees
        var tz = -48 + 24 * k;
        m.box([side * 8.7, 1.3, tz], [0.35, 2.6, 0.35], [0.36, 0.26, 0.18], M_PLAIN, 1.3);
        m.box([side * 8.7, 4.0, tz], [2.8, 2.8, 2.8], [0.22, 0.48, 0.2], M_LEAF, 0.7, { bottom: true });
      }
      m.box([side * 5.2, 3.5, 0], [0.3, 7, 0.3], [0.2, 0.21, 0.23], M_PLAIN, 1);   // clock gantry posts
    }
    m.box([0, 6.85, 0], [10.7, 0.3, 0.3], [0.2, 0.21, 0.23], M_PLAIN, 1, { bottom: true });
    // clock faces, one looking each way down the street; u runs to the viewer's right
    m.quad([1.25, 4.15, -0.2], [-2.5, 0, 0], [0, 2.5, 0], 6, 6, [1, 1, 1], M_CLOCK, [-1, -1], [1, 1]);
    m.quad([-1.25, 4.15, 0.2], [2.5, 0, 0], [0, 2.5, 0], 6, 6, [1, 1, 1], M_CLOCK, [-1, -1], [1, 1]);
    return m;
  }
  function buildTram() {
    var m = new Mesh();
    m.box([0, 1.75, 0], [2.4, 2.7, 14], [0.93, 0.72, 0.12], M_TRAM, 0.5, { bottom: true });
    m.quad([1.21, 0.55, 0.5], [0, 0, -1], [0, 1, 0], 3, 3, [1, 1, 1], M_TRAMCLOCK, [-1, -1], [1, 1]);
    m.quad([-1.21, 0.55, -0.5], [0, 0, 1], [0, 1, 0], 3, 3, [1, 1, 1], M_TRAMCLOCK, [-1, -1], [1, 1]);
    return m;
  }
  function buildSky() {
    var m = new Mesh(), st = 24, sl = 48, i, j;
    for (i = 0; i <= st; i++) for (j = 0; j <= sl; j++) {
      var th = Math.PI * i / st, ph = 2 * Math.PI * j / sl;
      m.v.push(Math.sin(th) * Math.cos(ph), Math.cos(th), Math.sin(th) * Math.sin(ph), 0, 1, 0, 1, 1, 1, 0, 0, M_SKY);
      m.n++;
    }
    for (i = 0; i < st; i++) for (j = 0; j < sl; j++) {
      var a = i * (sl + 1) + j, b = a + sl + 1;
      m.i.push(a, a + 1, b, a + 1, b + 1, b);
    }
    return m;
  }

  /* ---------- shaders ---------- */
  var VS = [
    '#version 300 es',
    'precision highp float;',
    'layout(location=0) in vec3 aPos;',
    'layout(location=1) in vec3 aNormal;',
    'layout(location=2) in vec3 aColor;',
    'layout(location=3) in vec3 aUVM;',
    'uniform vec3 uRel;     // object origin (at world "now") minus the eye position',
    'uniform vec3 uV;       // viewer velocity / c, world frame',
    'uniform vec3 uU;       // object velocity / c, world frame',
    'uniform float uC;',
    'uniform float uT;      // world time now',
    'uniform int uMode;     // 0 seen, 1 simultaneous in the viewer frame, 2 Newton',
    'uniform int uSky;',
    'uniform mat4 uVP;',
    'out vec3 vN; out vec3 vCol; out vec3 vUVM; out vec3 vRest; out vec3 vDir;',
    'out float vD; out float vTe; out float vDist;',
    'void main() {',
    '  float bu2 = dot(uU, uU), bv2 = dot(uV, uV);',
    '  float gu = inversesqrt(1.0 - bu2), gv = inversesqrt(1.0 - bv2);',
    '  vec3 p = aPos;',
    '  if (bu2 > 0.0) { vec3 uh = uU * inversesqrt(bu2); p -= (1.0 - 1.0 / gu) * dot(p, uh) * uh; }   // a moving object is contracted in the world frame',
    '  vec3 d0 = uSky == 1 ? aPos * 3000.0 : p + uRel;',
    '  vec3 r; float cdt;',
    '  if (uMode == 0) {',
    '    float a = 1.0 - bu2, b = dot(d0, uU);',
    '    float cs = (-b + sqrt(b * b + a * dot(d0, d0))) / a;   // c × light travel time',
    '    r = d0 - uU * cs; cdt = -cs;',
    '  } else if (uMode == 1) {',
    '    cdt = dot(uV, d0) / (1.0 - dot(uV, uU));',
    '    r = d0 + uU * cdt;',
    '  } else { r = d0; cdt = 0.0; }',
    '  vec3 rp = r;',
    '  if (uMode != 2 && bv2 > 0.0) {',
    '    vec3 vh = uV * inversesqrt(bv2);',
    '    rp = r + (gv - 1.0) * dot(r, vh) * vh - gv * uV * cdt;',
    '  }',
    '  float rl = max(length(r), 1e-6);',
    '  vec3 rh = r / rl;',
    '  vD = uMode == 0 ? gv * (1.0 + dot(rh, uV)) / (gu * (1.0 + dot(rh, uU))) : 1.0;',
    '  vTe = uT + cdt / uC;',
    '  vDist = rl; vDir = rh;',
    '  vN = aNormal; vCol = aColor; vUVM = aUVM; vRest = aPos;',
    '  gl_Position = uVP * vec4(uSky == 1 ? normalize(rp) * 50.0 : rp, 1.0);',
    '}'
  ].join('\n');

  var FS = [
    '#version 300 es',
    'precision highp float;',
    'in vec3 vN; in vec3 vCol; in vec3 vUVM; in vec3 vRest; in vec3 vDir;',
    'in float vD; in float vTe; in float vDist;',
    'uniform int uDoppler; uniform int uBeam; uniform float uExposure; uniform float uTauScale;',
    'out vec4 outColor;',
    'const vec3 SUN = vec3(0.3665, 0.8378, 0.4050);',
    'const vec3 HAZE = vec3(0.76, 0.84, 0.92);',
    'float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }',
    '// A crude spectrum for a sunlit surface: its reflectance is known at the three colour bands,',
    '// guessed in the infrared and ultraviolet, and joined smoothly. Outside the visible range the',
    '// light available falls off the way sunlight does (a 5800 K black body, per unit frequency).',
    'float sun(float l) { float x = min(2480.7 / l, 60.0); return x * x * x / (exp(x) - 1.0); }',
    'float spec(float l, vec3 rgb, float ir, float uv) {',
    '  if (l < 465.0) return mix(uv, rgb.b, smoothstep(330.0, 465.0, l)) * sun(l) * 1.3598 * smoothstep(290.0, 330.0, l);',
    '  if (l < 545.0) return mix(rgb.b, rgb.g, (l - 465.0) / 80.0);',
    '  if (l < 610.0) return mix(rgb.g, rgb.r, (l - 545.0) / 65.0);',
    '  return mix(rgb.r, ir, smoothstep(610.0, 780.0, l)) * sun(l) * 0.8528;',
    '}',
    '// Each receptor reads the emitted wavelength that the Doppler factor D maps onto it.',
    'vec3 shifted(vec3 rgb, float ir, float uv, float D) {',
    '  vec3 lam = vec3(610.0, 545.0, 465.0) * D;',
    '  return vec3(spec(lam.r, rgb, ir, uv), spec(lam.g, rgb, ir, uv), spec(lam.b, rgb, ir, uv));',
    '}',
    'vec3 dial(vec2 uv, float t, float period, float ticks) {',
    '  float r = length(uv);',
    '  if (r > 1.0) discard;',
    '  vec3 col = vec3(0.96, 0.95, 0.90);',
    '  if (r > 0.92) col = vec3(0.10);',
    '  float ang = atan(uv.x, uv.y) / 6.2831853;',
    '  float tk = abs(fract(ang * ticks + 0.5) - 0.5), q = abs(fract(ang * 4.0 + 0.5) - 0.5);',
    '  if (r > 0.80 && r < 0.90 && tk < 0.09) col = vec3(0.10);',
    '  if (r > 0.66 && r < 0.90 && q < 0.012) col = vec3(0.10);',
    '  float ha = 6.2831853 * fract(t / period);',
    '  vec2 hd = vec2(sin(ha), cos(ha));',
    '  float along = dot(uv, hd), perp = abs(uv.x * hd.y - uv.y * hd.x);',
    '  if (along > -0.12 && along < 0.76 && perp < 0.055) col = vec3(0.78, 0.10, 0.08);',
    '  if (r < 0.08) col = vec3(0.10);',
    '  return col;',
    '}',
    'void main() {',
    '  int m = int(vUVM.z + 0.5);',
    '  vec3 col = vCol; float ir = -1.0, uv = -1.0, lit = 1.0, fog = 1.0;',
    '  float detail = 1.0 - smoothstep(35.0, 110.0, vDist);',
    '  if (m == 5) {',
    '    vec3 d = normalize(vDir);',
    '    float up = clamp(d.y, 0.0, 1.0);',
    '    col = mix(HAZE, vec3(0.26, 0.47, 0.86), pow(up, 0.55));',
    '    if (d.y < 0.0) col = mix(HAZE, vec3(0.50, 0.56, 0.52), clamp(-d.y * 6.0, 0.0, 1.0));',
    '    float s = dot(d, SUN);',
    '    col += vec3(1.0, 0.92, 0.75) * (smoothstep(0.9994, 0.9997, s) * 3.0 + pow(max(s, 0.0), 90.0) * 0.5);',
    '    ir = 0.12 + smoothstep(0.999, 0.9997, s) * 3.0; uv = 1.2; lit = 0.0; fog = 0.0;',
    '  } else if (m == 0) {',
    '    vec3 p = vRest; float ax = abs(p.x);',
    '    if (ax < 4.0) {',
    '      col = vec3(0.23, 0.24, 0.26) * (1.0 + detail * 0.16 * (hash(floor(p.xz * 4.0)) - 0.5));',
    '      if (min(abs(ax - 1.28), abs(ax - 2.72)) < 0.05) col = vec3(0.56, 0.57, 0.60);',
    '      if (ax < 0.09 && fract(p.z / 6.0) < 0.5) col = vec3(0.86);',
    '    } else if (ax < 4.3) {',
    '      col = mod(floor(p.z), 2.0) < 0.5 ? vec3(0.84, 0.20, 0.17) : vec3(0.93);',
    '    } else if (ax < 7.7) {',
    '      vec2 f = fract(p.xz);',
    '      col = vec3(0.70, 0.69, 0.66) * (1.0 + detail * 0.10 * (hash(floor(p.xz)) - 0.5));',
    '      if (f.x < 0.04 || f.y < 0.04) col *= 1.0 - 0.25 * detail;',
    '    } else {',
    '      col = vec3(0.30, 0.52, 0.24) * (1.0 + detail * 0.30 * (hash(floor(p.xz * 2.0)) - 0.5));',
    '      ir = 0.9; uv = 0.03;',
    '    }',
    '  } else if (m == 1) {',
    '    vec2 cell = fract(vUVM.xy / 3.0);',
    '    if (cell.x > 0.22 && cell.x < 0.78 && cell.y > 0.28 && cell.y < 0.80)',
    '      col = mix(vec3(0.15, 0.21, 0.29), vec3(0.42, 0.56, 0.70), 0.6 * hash(floor(vUVM.xy / 3.0) + floor(vCol.xy * 50.0 + 0.5)));',
    '    else if (vUVM.y < 0.5) col *= 0.6;',
    '  } else if (m == 3) {',
    '    col = dial(vUVM.xy, vTe, ' + CLOCK_PERIOD.toFixed(1) + ', 20.0); lit = 0.0;',
    '  } else if (m == 8) {',
    '    col = dial(vUVM.xy, vTe * uTauScale, ' + CLOCK_PERIOD.toFixed(1) + ', 20.0); lit = 0.0;',
    '  } else if (m == 4) {',
    '    bool on = fract(vTe / ' + LAMP_PERIOD.toFixed(1) + ') < 0.14;',
    '    col = on ? vec3(1.9, 1.7, 1.0) : vec3(0.30, 0.28, 0.24); lit = on ? 0.0 : 1.0;',
    '  } else if (m == 6) {',
    '    vec3 p = vRest;',
    '    if (abs(vN.y) < 0.5 && p.y > 1.75 && p.y < 2.65 && fract(p.z / 2.0 + 0.5) > 0.12 && fract(p.z / 2.0 + 0.5) < 0.88 && abs(p.z) < 6.6) col = vec3(0.16, 0.22, 0.30);',
    '    else if (p.y < 0.75) col = vec3(0.25, 0.26, 0.28);',
    '    else if (p.y > 1.45 && p.y < 1.6) col = vec3(0.75, 0.12, 0.10);',
    '  } else if (m == 7) {',
    '    col *= 1.0 + detail * 0.35 * (hash(floor(vRest.xy * 3.0) + floor(vRest.z * 3.0)) - 0.5);',
    '    ir = 0.95; uv = 0.03;',
    '  }',
    '  float lum = dot(col, vec3(0.3, 0.55, 0.15));',
    '  if (ir < 0.0) { ir = lum; uv = lum * 0.7; }',
    '  if (lit > 0.0) { float l = 0.42 + 0.70 * clamp(dot(normalize(vN), SUN), 0.0, 1.0); col *= l; ir *= l; uv *= l; }',
    '  if (fog > 0.0) {',
    '    float f = smoothstep(150.0, 430.0, vDist);',
    '    col = mix(col, HAZE, f); ir = mix(ir, 0.2, f); uv = mix(uv, 1.0, f);',
    '  }',
    '  vec3 c = col;',
    '  if (uDoppler == 1) c = shifted(col, ir, uv, vD);',
    '  if (uBeam == 1) c *= vD * vD * vD * uExposure;',
    '  vec3 hi = 0.6 + 0.4 * (1.0 - exp(-(max(c, vec3(0.6)) - 0.6) / 0.4));',
    '  c = mix(c, hi, step(vec3(0.6), c));',
    '  outColor = vec4(c, 1.0);',
    '}'
  ].join('\n');

  /* ---------- small matrix helpers (column-major) ---------- */
  function perspective(fovy, aspect, near, far) {
    var f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0];
  }
  function mul(a, b) {
    var o = new Float32Array(16);
    for (var c = 0; c < 4; c++) for (var r = 0; r < 4; r++) {
      o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
    return o;
  }

  /* ---------- state ---------- */
  var canvas, gl, prog, U = {}, meshes = {}, raf = 0, last = 0, ok = false;
  var S = {
    x: 2.2, z: -42, heading: 0, lookYaw: 0, lookPitch: 0.04, effort: 0,
    T: 60, tau: 60, mode: 0, doppler: true, beam: true, exposure: 1,
    trams: [{ x: -2, z: 70, dir: -1 }, { x: 2, z: -260, dir: 1 }],
    keys: {}, hint: 8
  };

  function compile(type, src) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  function upload(mesh) {
    var vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    var vb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vb);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(mesh.v), gl.STATIC_DRAW);
    for (var a = 0; a < 4; a++) { gl.enableVertexAttribArray(a); gl.vertexAttribPointer(a, 3, gl.FLOAT, false, 48, a * 12); }
    var ib = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(mesh.i), gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    return { vao: vao, count: mesh.i.length };
  }
  function setup() {
    canvas = $('light-canvas');
    gl = canvas.getContext('webgl2', { antialias: true, alpha: false });
    if (!gl) return false;
    prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    ['uRel', 'uV', 'uU', 'uC', 'uT', 'uMode', 'uSky', 'uVP', 'uDoppler', 'uBeam', 'uExposure', 'uTauScale'].forEach(function (n) { U[n] = gl.getUniformLocation(prog, n); });
    meshes.tile = upload(buildTile());
    meshes.tram = upload(buildTram());
    meshes.sky = upload(buildSky());
    gl.enable(gl.DEPTH_TEST);
    return true;
  }

  /* ---------- per-frame ---------- */
  function kinematics() {
    var c = A.state.c, v = P.speedFromEffort(S.effort, c), beta = v / c;
    return { c: c, v: v, beta: beta, gamma: P.gamma(beta), dir: [Math.sin(S.heading), 0, Math.cos(S.heading)] };
  }
  function wrap(d, L) { return d - L * Math.round(d / L); }

  function step(dtau) {
    var k = S.keys;
    if (k.faster) S.effort = Math.min(40, S.effort + 7 * dtau);
    if (k.slower) S.effort = Math.max(0, S.effort - 9 * dtau);
    if (k.left) S.heading += 1.3 * dtau;
    if (k.right) S.heading -= 1.3 * dtau;
    if (k.faster || k.slower) { $('effort').value = S.effort; $('effort-out').textContent = S.effort.toFixed(1) + ' m/s'; }
    var q = kinematics();
    var dT = q.gamma * dtau;                 // the town's clocks advance γ times faster than your watch
    S.tau += dtau; S.T += dT;
    S.x += q.dir[0] * q.v * dT; S.z += q.dir[2] * q.v * dT;
    if (S.x > 7.3) S.x = 7.3; if (S.x < -7.3) S.x = -7.3;
    var u = P.speedFromEffort(TRAM_EFFORT, q.c);
    S.trams.forEach(function (t) { t.z += t.dir * u * dT; });
    if (S.hint > 0) S.hint -= dtau;
    return q;
  }

  function render(q) {
    var f = A.fit(canvas, 1.5);
    gl.viewport(0, 0, canvas.width, canvas.height);
    var yaw = S.heading + S.lookYaw, pit = S.lookPitch;
    var fw = [Math.sin(yaw) * Math.cos(pit), Math.sin(pit), Math.cos(yaw) * Math.cos(pit)];
    var rt = [-Math.cos(yaw), 0, Math.sin(yaw)];
    var up = [rt[1] * fw[2] - rt[2] * fw[1], rt[2] * fw[0] - rt[0] * fw[2], rt[0] * fw[1] - rt[1] * fw[0]];
    var view = [rt[0], up[0], -fw[0], 0, rt[1], up[1], -fw[1], 0, rt[2], up[2], -fw[2], 0, 0, 0, 0, 1];
    var vp = mul(perspective(1.2, f.w / f.h, 0.12, 40000), view);

    var bv = [q.dir[0] * q.beta, 0, q.dir[2] * q.beta];
    // eye adaptation: beaming makes the view ahead brighter by D³, so expose for where you look
    var cosL = fw[0] * q.dir[0] + fw[2] * q.dir[2];
    var Dc = 1 / (q.gamma * (1 - q.beta * cosL));
    var target = Math.max(0.02, Math.min(60, Math.pow(Dc, -2.7)));
    S.exposure += (target - S.exposure) * 0.15;
    S.Dc = Dc;

    gl.clearColor(0.76, 0.84, 0.92, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(prog);
    gl.uniformMatrix4fv(U.uVP, false, vp);
    gl.uniform3fv(U.uV, bv);
    gl.uniform1f(U.uC, q.c);
    gl.uniform1f(U.uT, S.T % 3600);
    gl.uniform1i(U.uMode, S.mode);
    gl.uniform1i(U.uDoppler, S.doppler && S.mode === 0 ? 1 : 0);
    gl.uniform1i(U.uBeam, S.beam && S.mode === 0 ? 1 : 0);
    gl.uniform1f(U.uExposure, S.exposure);
    gl.uniform1f(U.uTauScale, 1);

    gl.depthMask(false);                                   // the sky is at infinity: aberration only
    gl.uniform1i(U.uSky, 1); gl.uniform3f(U.uU, 0, 0, 0); gl.uniform3f(U.uRel, 0, 0, 0);
    gl.bindVertexArray(meshes.sky.vao);
    gl.drawElements(gl.TRIANGLES, meshes.sky.count, gl.UNSIGNED_INT, 0);
    gl.depthMask(true);

    gl.uniform1i(U.uSky, 0);
    gl.bindVertexArray(meshes.tile.vao);
    var k0 = Math.round(S.z / TILE);
    for (var k = -SPAN; k <= SPAN; k++) {                  // the street repeats every 120 m, for ever
      gl.uniform3f(U.uRel, -S.x, -EYE, (k0 + k) * TILE - S.z);
      gl.drawElements(gl.TRIANGLES, meshes.tile.count, gl.UNSIGNED_INT, 0);
    }
    var bu = P.speedFromEffort(TRAM_EFFORT, q.c) / q.c;
    gl.bindVertexArray(meshes.tram.vao);
    gl.uniform1f(U.uTauScale, Math.sqrt(1 - bu * bu));     // the tram's own clock runs slow by 1/γ
    S.trams.forEach(function (t) {
      gl.uniform3f(U.uU, 0, 0, t.dir * bu);
      gl.uniform3f(U.uRel, t.x - S.x, -EYE, wrap(t.z - S.z, 2 * SPAN * TILE));
      gl.drawElements(gl.TRIANGLES, meshes.tram.count, gl.UNSIGNED_INT, 0);
    });
    gl.bindVertexArray(null);
  }

  function hud(q) {
    var n = A.num;
    $('light-hud-l').innerHTML = 'speed <b>' + n(q.v) + ' m/s</b> = ' + (q.beta < 1e-3 ? n(q.beta, 2) : q.beta.toFixed(3)) + ' c<br>γ = ' + (q.beta === 0 ? '1' : q.gamma < 1.001 ? '1 + ' + n(P.clockLoss(q.beta) * q.gamma, 2) : q.gamma.toFixed(3));
    $('light-hud-r').innerHTML = 'your watch <b>' + A.clock(S.tau) + '</b><br>town clock <b>' + A.clock(S.T) + '</b>';
    var D = S.Dc || 1, msg;
    if (S.mode === 1) msg = 'Your frame, this instant: lengths along your motion ×' + (1 / q.gamma).toFixed(3);
    else if (S.mode === 2) msg = 'Newton’s street: no light delay, no contraction';
    else if (Math.abs(D - 1) < 0.01) msg = 'Where you look: colours and clock rates unchanged';
    else msg = 'Where you look: light ×' + D.toFixed(2) + (D > 1 ? ' bluer' : ' redder') + ', clocks seem ×' + D.toFixed(2) + (D > 1 ? ' fast' : ' slow');
    if (S.hint > 0 && canvas.clientWidth > 640) msg += '<br>W S effort · A D steer · drag to look';
    $('light-hud-b').innerHTML = msg;

    // the next clock gantry ahead of you along the street
    var ahead = Math.cos(S.heading) >= 0 ? 1 : -1;
    var gz = (ahead > 0 ? Math.ceil(S.z / TILE) : Math.floor(S.z / TILE)) * TILE;
    var dist = Math.hypot(gz - S.z, S.x, 5.4 - EYE);
    var w = q.gamma * q.v;
    var rows = [
      ['Street covered per second on your watch, γv', n(w) + ' m'],
      ['Overhead clock ahead, ' + n(dist, 2) + ' m away, shows', A.clock(S.T - dist / q.c)],
      ['Its light is late by', A.duration(dist / q.c, 2)],
      ['A one-metre crate, in your frame', (1 / q.gamma).toFixed(3) + ' m'],
      ['Trams run at', n(P.speedFromEffort(TRAM_EFFORT, q.c)) + ' m/s']
    ];
    $('light-readouts').innerHTML = rows.map(function (r) { return '<div class="readout"><dt>' + r[0] + '</dt><dd>' + r[1] + '</dd></div>'; }).join('');
  }

  var hudClock = 0;
  function frame(now) {
    raf = requestAnimationFrame(frame);
    var dtau = Math.min(0.05, (now - last) / 1000 || 0); last = now;
    var q = step(dtau);
    render(q);
    hudClock += dtau;
    if (hudClock > 0.1) { hudClock = 0; hud(q); }
  }

  function nudge() {
    var c = A.state.c, best = P.speedFromEffort(40, c) / c, el = $('light-nudge');
    el.hidden = best > 0.2;
    if (!el.hidden) $('light-nudge-text').textContent = 'With c = ' + A.num(c) + ' m/s your hardest pedalling reaches ' + A.num(best, 2) + ' of light speed. The street will look perfectly ordinary.';
  }

  function bind() {
    $('effort').addEventListener('input', function (e) { S.effort = Number(e.target.value); $('effort-out').textContent = S.effort.toFixed(1) + ' m/s'; });
    $('effort-out').textContent = '0.0 m/s';
    Array.prototype.forEach.call($('light-mode').children, function (b) {
      b.addEventListener('click', function () {
        S.mode = Number(b.getAttribute('data-mode'));
        Array.prototype.forEach.call($('light-mode').children, function (o) { o.setAttribute('aria-pressed', o === b ? 'true' : 'false'); });
      });
    });
    $('opt-doppler').addEventListener('change', function (e) { S.doppler = e.target.checked; });
    $('opt-beam').addEventListener('change', function (e) { S.beam = e.target.checked; });
    $('light-stop').addEventListener('click', function () { S.effort = 0; $('effort').value = 0; $('effort-out').textContent = '0.0 m/s'; });
    $('light-turn').addEventListener('click', function () { S.heading += Math.PI; S.lookYaw = 0; });
    $('light-nudge-btn').addEventListener('click', function () { A.set({ c: 10 }); });

    var KEYMAP = { w: 'faster', ArrowUp: 'faster', s: 'slower', ArrowDown: 'slower', a: 'left', ArrowLeft: 'left', d: 'right', ArrowRight: 'right' };
    function key(ev, down) {
      if (A.currentTab() !== 'light' || !ev.key) return;
      var t = ev.target, onCanvas = t === canvas;
      if (!onCanvas && t && /^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(t.tagName) && ev.key.length > 1) return;   // leave arrow keys to sliders
      var act = KEYMAP[ev.key.length === 1 ? ev.key.toLowerCase() : ev.key];
      if (ev.key === ' ' && onCanvas && down) { S.effort = 0; $('effort').value = 0; $('effort-out').textContent = '0.0 m/s'; ev.preventDefault(); return; }
      if (!act) return;
      S.keys[act] = down;
      if (onCanvas) ev.preventDefault();
    }
    window.addEventListener('keydown', function (e) { key(e, true); });
    window.addEventListener('keyup', function (e) { key(e, false); });
    window.addEventListener('blur', function () { S.keys = {}; });

    var drag = null;
    canvas.addEventListener('pointerdown', function (e) { drag = { x: e.clientX, y: e.clientY }; try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* fine */ } canvas.focus({ preventScroll: true }); });
    canvas.addEventListener('pointermove', function (e) {
      if (!drag) return;
      S.lookYaw += (e.clientX - drag.x) * 0.005;
      S.lookPitch = Math.max(-1.2, Math.min(1.3, S.lookPitch + (e.clientY - drag.y) * 0.005));
      drag = { x: e.clientX, y: e.clientY };
    });
    function end() { drag = null; }
    canvas.addEventListener('pointerup', end); canvas.addEventListener('pointercancel', end);
    canvas.addEventListener('dblclick', function () { S.lookYaw = 0; S.lookPitch = 0.04; });

    Array.prototype.forEach.call(document.querySelectorAll('#light-view .pad button'), function (b) {
      var act = b.getAttribute('data-act');
      function on(e) { S.keys[act] = true; e.preventDefault(); }
      function off() { S.keys[act] = false; }
      b.addEventListener('pointerdown', on); b.addEventListener('pointerup', off);
      b.addEventListener('pointerleave', off); b.addEventListener('pointercancel', off);
    });
  }

  function show() {
    if (!ok) {
      try { ok = setup(); } catch (e) { ok = false; console.error(e); }
      if (!ok) { $('light-hud-b').textContent = 'This scene needs WebGL 2, which this browser did not provide.'; return; }
    }
    nudge();
    last = performance.now();
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(frame);
  }
  function hide() { cancelAnimationFrame(raf); S.keys = {}; }

  CGH.light = {
    init: function () {
      canvas = $('light-canvas');
      bind();
      A.registerTab('light', { show: show, hide: hide });
      A.onChange(function () { if (A.currentTab() === 'light') nudge(); });
    },
    state: S
  };
})();
