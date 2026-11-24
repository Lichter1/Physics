/* cGħ World — GPU programs.
 *   march  one light ray per pixel, traced backwards through curved spacetime to what it came from
 *   accum  blends this frame into a running average (the overlay of every place you might be)
 *   show   draws the average, one photon at a time when light is grainy
 */
(function () {
  'use strict';
  var g = CGH.geo, F = function (x) { return x.toFixed(6); };

  var VS = `#version 300 es
const vec2 P[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
out vec2 vUV;
void main() { vUV = P[gl_VertexID]; gl_Position = vec4(P[gl_VertexID], 0.0, 1.0); }`;

  var MARCH = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 outColor;
uniform vec3 uEye, uFwd, uRight, uUp, uBeta;
uniform vec2 uTan;
uniform float uRsRay, uRsShift, uRsClock, uC, uCdelay, uT, uExposure, uBell;
uniform int uDoppler, uBeam, uAberr;
const float R = ${F(g.R)};
const float HMAX = ${F(g.HMAX)};
const vec3 SUN = vec3(0.3015, 0.9045, 0.3015);

float hash1(float n) { return fract(sin(n * 12.9898) * 43758.5453); }
float hash2(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float sdBox(vec3 p, vec3 b) { vec3 q = abs(p) - b; return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0); }
float houseH(float cell, float side) { return 9.0 + 3.0 * mod(cell * 7.0 + side * 3.0 + 5.0, 6.0); }

// Street coordinates (across, height, along within the block) and the block index.
vec3 street(vec3 x, out float ti) {
  float r = length(x);
  float a = R * atan(x.z, x.y);
  ti = floor((a + 60.0) / 120.0);
  return vec3(-R * asin(clamp(x.x / r, -1.0, 1.0)), r - R, a - 120.0 * ti);
}

// Distance to the nearest surface, and which material it is.
float scene(vec3 x, out float mat) {
  float ti; vec3 p = street(x, ti);
  float l = p.x, h = p.y, a = p.z, al = abs(l);
  mat = 0.0;
  if (h > HMAX) return h - HMAX + 0.5;
  float d = h, e;
  float side = l > 0.0 ? 1.0 : 0.0, c0 = floor(a / 20.0), cn = c0 + (fract(a / 20.0) < 0.5 ? -1.0 : 1.0);
  for (int k = 0; k < 2; k++) {
    float c = k == 0 ? c0 : cn, H = houseH(ti * 6.0 + c, side);
    e = sdBox(vec3(al - 16.0, h - 0.5 * H, a - 20.0 * c - 10.0), vec3(6.0, 0.5 * H, 8.0));
    if (e < d) { d = e; mat = 1.0; }
  }
  if (!(ti == 0.0 && a < -19.5 && a > -38.5)) {          // the yard behind the two doors is kept empty
    vec3 q = vec3(al - 4.6, h, mod(a, 12.0) - 6.0);
    e = sdBox(q - vec3(0.0, 2.2, 0.0), vec3(0.08, 2.2, 0.08)); if (e < d) { d = e; mat = 3.0; }
    e = sdBox(q - vec3(0.0, 4.5, 0.0), vec3(0.25, 0.14, 0.25)); if (e < d) { d = e; mat = 4.0; }
    e = sdBox(vec3(al - 6.3, h - 0.5, mod(a, 6.0) - 3.0), vec3(0.5)); if (e < d) { d = e; mat = 5.0; }
  }
  vec3 t = vec3(al - 8.7, h, mod(a + 12.0, 24.0) - 12.0);
  e = sdBox(t - vec3(0.0, 1.3, 0.0), vec3(0.175, 1.3, 0.175)); if (e < d) { d = e; mat = 7.0; }
  e = sdBox(t - vec3(0.0, 4.0, 0.0), vec3(1.3)); if (e < d) { d = e; mat = 8.0; }
  e = min(sdBox(vec3(al - 5.2, h - 3.5, a), vec3(0.15, 3.5, 0.15)), sdBox(vec3(l, h - 6.85, a), vec3(5.35, 0.15, 0.15)));
  if (e < d) { d = e; mat = 3.0; }
  e = max(length(vec2(l, h - 5.4)) - 1.25, abs(a) - 0.06); if (e < d) { d = e; mat = 9.0; }
  if (ti == 0.0) {
    e = sdBox(vec3(l, h - 25.0, a - 40.0), vec3(1.5, 25.0, 1.5)); if (e < d) { d = e; mat = 10.0; }
    e = max(sdBox(vec3(l, h - 1.5, a + 20.0), vec3(10.0, 1.5, 0.2)), -sdBox(vec3(al - 1.5, h - 1.5, a + 20.0), vec3(0.6, 2.0, 1.0)));
    if (e < d) { d = e; mat = 11.0; }
    e = sdBox(vec3(l - 1.5, h - 3.25, a + 20.0), vec3(0.3, 0.2, 0.3)); if (e < d) { d = e; mat = 12.0; }
  }
  return d;
}
vec3 normalAt(vec3 x, float e) {
  float m; const vec2 k = vec2(1.0, -1.0);
  return normalize(k.xyy * scene(x + k.xyy * e, m) + k.yyx * scene(x + k.yyx * e, m) +
                   k.yxy * scene(x + k.yxy * e, m) + k.xxx * scene(x + k.xxx * e, m));
}

// Sunlit reflectance known at three colour bands, guessed in the infrared and ultraviolet; the light
// available outside the visible falls off like sunlight (a 5800 K black body).
float sunB(float lam) { float x = min(2480.7 / lam, 60.0); return x * x * x / (exp(x) - 1.0); }
float spec(float lam, vec3 rgb, float ir, float uv) {
  if (lam < 465.0) return mix(uv, rgb.b, smoothstep(330.0, 465.0, lam)) * sunB(lam) * 1.3598 * smoothstep(290.0, 330.0, lam);
  if (lam < 545.0) return mix(rgb.b, rgb.g, (lam - 465.0) / 80.0);
  if (lam < 610.0) return mix(rgb.g, rgb.r, (lam - 545.0) / 65.0);
  return mix(rgb.r, ir, smoothstep(610.0, 780.0, lam)) * sunB(lam) * 0.8528;
}
vec3 shifted(vec3 rgb, float ir, float uv, float D) {
  vec3 lam = vec3(610.0, 545.0, 465.0) * D;
  return vec3(spec(lam.r, rgb, ir, uv), spec(lam.g, rgb, ir, uv), spec(lam.b, rgb, ir, uv));
}
vec3 dial(vec2 uv, float t) {
  float r = length(uv);
  vec3 col = vec3(0.96, 0.95, 0.90);
  if (r > 0.92) return vec3(0.10);
  float ang = atan(uv.x, uv.y) / 6.2831853;
  if (r > 0.80 && abs(fract(ang * 20.0 + 0.5) - 0.5) < 0.09) col = vec3(0.10);
  if (r > 0.66 && abs(fract(ang * 4.0 + 0.5) - 0.5) < 0.012) col = vec3(0.10);
  float ha = 6.2831853 * fract(t / 20.0);
  vec2 hd = vec2(sin(ha), cos(ha));
  float along = dot(uv, hd), perp = abs(uv.x * hd.y - uv.y * hd.x);
  if (along > -0.12 && along < 0.76 && perp < 0.06) col = vec3(0.78, 0.10, 0.08);
  if (r < 0.08) col = vec3(0.10);
  return col;
}
vec3 sky(vec3 d, out float ir, out float uv) {
  vec3 col = vec3(0.015, 0.022, 0.045);
  vec3 cell = floor(d * 140.0);
  float h = hash2(cell.xy + 17.0 * cell.z);
  if (h > 0.994) col += vec3(0.8, 0.85, 1.0) * pow((h - 0.994) / 0.006, 3.0) * 2.5;
  vec3 mw = normalize(vec3(0.25, 0.35, -0.9));
  col += vec3(0.20, 0.19, 0.24) * exp(-pow(dot(d, mw) / 0.14, 2.0)) * (0.6 + 0.4 * hash2(floor(d.xy * 40.0)));
  float s = dot(d, SUN);
  col += vec3(1.0, 0.92, 0.78) * (smoothstep(0.9990, 0.9994, s) * 4.0 + pow(max(s, 0.0), 900.0) * 0.3);
  ir = dot(col, vec3(0.4)); uv = dot(col, vec3(0.5));
  return col;
}

void main() {
  // the pixel's direction in your frame, then in the frame of someone standing still where you are
  vec3 np = normalize(uFwd + vUV.x * uTan.x * uRight + vUV.y * uTan.y * uUp);
  float b2 = dot(uBeta, uBeta), Ddop = 1.0;
  vec3 n = np;
  if (b2 > 0.0) {
    float gm = inversesqrt(1.0 - b2), nb = dot(np, uBeta);
    vec3 bh = uBeta * inversesqrt(b2);
    if (uAberr == 1) n = (np + ((gm - 1.0) * dot(np, bh) - gm * sqrt(b2)) * bh) / (gm * (1.0 - nb));
    Ddop = 1.0 / (gm * (1.0 - nb));
  }
  // follow the light back along its curved path
  float re = length(uEye), r = re, ct = 0.0, s = 0.0, mat = 0.0;
  vec3 u = uEye / re, x = uEye;
  vec3 v = dot(n, u) * u + (n - dot(n, u) * u) * inversesqrt(1.0 - uRsRay / re);
  vec3 Lv = cross(x, v);
  float k = -1.5 * uRsRay * dot(Lv, Lv);
  vec3 acc = k * x / pow(r, 5.0);
  int kind = 0;
  for (int i = 0; i < 260; i++) {
    float h = r - R, d;
    if (h < HMAX) { d = scene(x, mat); if (d < 0.004 + 0.0015 * s) { kind = 1; break; } }
    else d = h - HMAX + 0.5;
    if (r > 30.0 * R) { kind = 2; break; }
    float dl = max(min(0.8 * d, (h > 2.0 * HMAX ? 0.15 : 0.05) * r), 0.003 * s + 0.01);
    float hs = dl / length(v);
    vec3 xn = x + v * hs + 0.5 * acc * hs * hs;
    float rn = length(xn);
    vec3 accn = k * xn / pow(rn, 5.0);
    v += 0.5 * (acc + accn) * hs;
    ct += 0.5 * hs * (1.0 / (1.0 - uRsRay / r) + 1.0 / (1.0 - uRsRay / rn));
    if (rn < R) {
      float f = (r - R) / max(r - rn, 1e-6);
      ct -= (1.0 - f) * hs / (1.0 - uRsRay / R);
      x = mix(x, xn, f); s += f * dl; r = R; mat = 0.0; kind = 1; break;
    }
    x = xn; r = rn; acc = accn; s += dl;
  }

  vec3 col; float ir = -1.0, uvl = -1.0, Dg;
  if (kind != 1) {
    col = kind == 2 ? sky(normalize(v), ir, uvl) : vec3(0.05, 0.06, 0.08);
    Dg = inversesqrt(1.0 - uRsShift / re);                    // starlight falling in is blueshifted
  } else {
    r = length(x);
    Dg = sqrt((1.0 - uRsShift / r) / (1.0 - uRsShift / re));
    float tau = sqrt(1.0 - uRsClock / r) * (uT - ct / uCdelay);    // the emitter's own clock when this light left it
    float ti; vec3 p = street(x, ti);
    float l = p.x, h = p.y, a = p.z, ag = a + 120.0 * ti;
    vec3 up = x / r, along = normalize(vec3(0.0, -x.z, x.y)), across = cross(along, up);
    vec3 nr = mat == 0.0 ? up : normalAt(x, 0.01 + 0.001 * s);
    float detail = 1.0 - smoothstep(40.0, 160.0, s), lit = 1.0;
    col = vec3(0.6);
    if (mat == 0.0) {
      float al = abs(l);
      if (al < 4.0) {
        col = vec3(0.23, 0.24, 0.26) * (1.0 + detail * 0.16 * (hash2(floor(vec2(l, a) * 4.0)) - 0.5));
        if (al < 0.09 && fract(a / 6.0) < 0.5) col = vec3(0.86);
        if (abs(al - 3.7) < 0.08) col = vec3(0.80);
      } else if (al < 4.3) col = mod(floor(ag), 2.0) < 0.5 ? vec3(0.84, 0.20, 0.17) : vec3(0.93);
      else if (al < 7.7) {
        vec2 f = fract(vec2(l, a));
        col = vec3(0.70, 0.69, 0.66) * (1.0 + detail * 0.1 * (hash2(floor(vec2(l, ag))) - 0.5));
        if (f.x < 0.04 || f.y < 0.04) col *= 1.0 - 0.25 * detail;
      } else {
        float fld = hash2(floor(vec2(l, ag) / 30.0));
        col = mix(vec3(0.30, 0.52, 0.24), vec3(0.62, 0.55, 0.28), step(0.7, fld) * step(24.0, al));
        col *= 1.0 + 0.3 * detail * (hash2(floor(vec2(l, ag) * 2.0)) - 0.5);
        ir = 0.9; uvl = 0.03;
      }
    } else if (mat == 1.0) {
      float c0 = floor(a / 20.0), hh = hash1(ti * 13.0 + c0 * 3.0 + (l > 0.0 ? 1.0 : 0.0));
      col = mix(vec3(0.62, 0.36, 0.28), vec3(0.80, 0.72, 0.56), hh);
      col = mix(col, vec3(0.55, 0.60, 0.66), step(0.66, fract(hh * 7.0)));
      if (dot(nr, up) > 0.7) col = vec3(0.32, 0.32, 0.34);
      else {
        vec2 w = fract(vec2(abs(dot(nr, across)) > 0.7 ? a : l, h) / 3.0);
        if (w.x > 0.22 && w.x < 0.78 && w.y > 0.28 && w.y < 0.80) col = vec3(0.17, 0.24, 0.32);
      }
    } else if (mat == 3.0) col = vec3(0.20, 0.21, 0.23);
    else if (mat == 4.0) {
      bool on = fract(tau / 2.0) < 0.14;
      col = on ? vec3(1.9, 1.7, 1.0) : vec3(0.30, 0.28, 0.24); lit = on ? 0.0 : 1.0;
    } else if (mat == 5.0) col = mod(floor(ag / 6.0), 2.0) < 0.5 ? vec3(0.78, 0.22, 0.20) : vec3(0.20, 0.40, 0.75);
    else if (mat == 7.0) col = vec3(0.36, 0.26, 0.18);
    else if (mat == 8.0) { col = vec3(0.22, 0.48, 0.20); ir = 0.95; uvl = 0.03; }
    else if (mat == 9.0) { col = dial(vec2(a < 0.0 ? l : -l, h - 5.4) / 1.25, tau); lit = 0.0; }
    else if (mat == 10.0) {
      col = vec3(0.74, 0.72, 0.68);
      float sa = dot(nr, along), sl = dot(nr, across);
      float fu = abs(sa) > 0.7 ? -sign(sa) * l : sign(sl) * (a - 40.0);
      for (int c = 0; c < 2; c++) {
        vec2 q = vec2(fu, h - (c == 0 ? 46.0 : 3.0)) / 1.3;
        if (abs(dot(nr, up)) < 0.5 && length(q) < 1.0) { col = dial(q, tau); lit = 0.0; }
      }
    } else if (mat == 12.0) { col = uBell > 0.5 ? vec3(0.5, 1.6, 1.9) : vec3(0.25, 0.27, 0.30); lit = uBell > 0.5 ? 0.0 : 1.0; }
    else if (mat == 11.0) col = abs(h - 2.2) < 0.15 ? vec3(0.15, 0.35, 0.75) : vec3(0.86, 0.85, 0.82);
    float lum = dot(col, vec3(0.3, 0.55, 0.15));
    if (ir < 0.0) { ir = lum; uvl = lum * 0.7; }
    if (lit > 0.0) { float sh = 0.30 + 0.75 * max(dot(nr, SUN), 0.0); col *= sh; ir *= sh; uvl *= sh; }
  }
  float D = Ddop * Dg;
  vec3 c = col;
  if (uDoppler == 1) c = shifted(col, ir, uvl, D);
  if (uBeam == 1) c *= D * D * D;
  c *= uExposure;
  outColor = vec4(c, 1.0);
}`;

  var ACCUM = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 outColor;
uniform sampler2D uNew, uOld;
uniform float uK;
void main() { vec2 t = vUV * 0.5 + 0.5; outColor = mix(texture(uOld, t), texture(uNew, t), uK); }`;

  var SHOW = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 outColor;
uniform sampler2D uImg;
uniform float uPhotons;
uniform uint uSeed;
uint pcg(uint v) { uint s = v * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
uint st;
float rnd() { st = pcg(st); return float(st) / 4294967296.0; }
float poisson(float m) {
  if (m > 30.0) { float a = max(rnd(), 1e-7), b = rnd(); return max(0.0, m + sqrt(m) * sqrt(-2.0 * log(a)) * cos(6.2831853 * b)); }
  float L = exp(-m), p = 1.0, k = 0.0;
  for (int i = 0; i < 80; i++) { p *= rnd(); if (p < L) break; k += 1.0; }
  return k;
}
void main() {
  vec3 c = max(texture(uImg, vUV * 0.5 + 0.5).rgb, 0.0);
  if (uPhotons < 1e5) {
    st = pcg(uint(gl_FragCoord.x) * 1973u + uint(gl_FragCoord.y) * 9277u + uSeed * 26699u);
    // each colour channel collects its own photons; a mid-grey pixel expects uPhotons of them
    c = vec3(poisson(c.r * uPhotons), poisson(c.g * uPhotons), poisson(c.b * uPhotons)) / uPhotons;
  }
  vec3 hi = 0.6 + 0.4 * (1.0 - exp(-(max(c, vec3(0.6)) - 0.6) / 0.4));
  outColor = vec4(mix(c, hi, step(vec3(0.6), c)), 1.0);
}`;

  CGH.shaders = { VS: VS, MARCH: MARCH, ACCUM: ACCUM, SHOW: SHOW };
})();
