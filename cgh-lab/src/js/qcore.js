/* cGħ Lab — quantum core.
 * A 2-D time-dependent Schrödinger solver (split-step Fourier) on an N×N periodic grid.
 * Pure arithmetic, no DOM: used by the Quantum scene and by the Node tests.
 *
 *   iħ ∂ψ/∂t = -(ħ²/2m) ∇²ψ + V ψ
 *
 * One step is  e^{-iV dt/2ħ} · F⁻¹ e^{-iħk² dt/2m} F · e^{-iV dt/2ħ},  which is unitary
 * for any dt, so probability is conserved to rounding error wherever there is no absorber.
 */
(function (root) {
  'use strict';

  function FFT(n) {
    this.n = n;
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    this.rev = new Uint32Array(n);
    var bits = Math.round(Math.log2(n));
    for (var i = 0; i < n / 2; i++) {
      this.cos[i] = Math.cos(2 * Math.PI * i / n);
      this.sin[i] = Math.sin(2 * Math.PI * i / n);
    }
    for (var j = 0; j < n; j++) {
      var r = 0;
      for (var b = 0; b < bits; b++) if (j & (1 << b)) r |= 1 << (bits - 1 - b);
      this.rev[j] = r;
    }
  }
  // In-place forward transform (e^{-2πi jk/n}), no scaling.
  FFT.prototype.forward = function (re, im) {
    var n = this.n, rev = this.rev, cos = this.cos, sin = this.sin, i, j, k, t;
    for (i = 0; i < n; i++) {
      j = rev[i];
      if (j > i) { t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (var size = 2; size <= n; size <<= 1) {
      var half = size >> 1, step = n / size;
      for (i = 0; i < n; i += size) {
        for (j = i, k = 0; j < i + half; j++, k += step) {
          var l = j + half;
          var tre = re[l] * cos[k] + im[l] * sin[k];
          var tim = -re[l] * sin[k] + im[l] * cos[k];
          re[l] = re[j] - tre; im[l] = im[j] - tim;
          re[j] += tre; im[j] += tim;
        }
      }
    }
  };
  // Inverse, including the 1/n.
  FFT.prototype.inverse = function (re, im) {
    this.forward(im, re);
    var n = this.n, s = 1 / n;
    for (var i = 0; i < n; i++) { re[i] *= s; im[i] *= s; }
  };

  /* opts: N (power of two), L (side length, m), m (mass, kg), hbar (J·s),
   * c (optional, m/s): use the relativistic energy sqrt((ħkc)² + (mc²)²) - mc² instead of (ħk)²/2m.
   * That is the free motion of a spinless particle with no pair creation; for c → ∞ it is Schrödinger. */
  function Sim(opts) {
    var N = this.N = opts.N;
    this.L = opts.L; this.dx = opts.L / N;
    this.m = opts.m; this.hbar = opts.hbar; this.c = opts.c || 0;
    this.re = new Float64Array(N * N);
    this.im = new Float64Array(N * N);
    this.V = new Float64Array(N * N);          // potential energy, J
    this.wall = new Uint8Array(N * N);         // 1 where a solid wall stands
    this.damp = new Float64Array(N * N);       // absorption rate, 1/s (0 = none)
    this.screen = new Float64Array(N);         // probability absorbed at the right edge, per row
    this.absorbed = 0;                         // total probability absorbed so far
    this.wallPhase = 1.0;                      // radians per step inside a wall (keep E·dt/ħ ≲ 0.1)
    this.time = 0;
    this._fft = new FFT(N);
    this._tr = new Float64Array(N); this._ti = new Float64Array(N);
    this._kc = new Float64Array(N * N); this._ks = new Float64Array(N * N);
    this._pc = new Float64Array(N * N); this._ps = new Float64Array(N * N);
    this._pd = new Float64Array(N * N);
    this._dt = -1;
    this.screenFrom = N;                       // columns >= this count as "the screen"
  }

  // Cell-centre coordinates: x to the right, y up (row 0 is the top of the picture).
  Sim.prototype.x = function (i) { return (i + 0.5) * this.dx; };
  Sim.prototype.y = function (j) { return this.L - (j + 0.5) * this.dx; };

  Sim.prototype.build = function (fn) {
    // fn(x, y) -> { V: joules, wall: bool, damp: 1/s }
    var N = this.N;
    for (var j = 0; j < N; j++) for (var i = 0; i < N; i++) {
      var c = fn(this.x(i), this.y(j)) || {};
      var q = j * N + i;
      this.V[q] = c.V || 0; this.wall[q] = c.wall ? 1 : 0; this.damp[q] = c.damp || 0;
    }
    this._dt = -1;
  };

  // A Gaussian packet: centre (x0,y0), widths (sx,sy) as standard deviations of |ψ|², velocity (vx,vy).
  Sim.prototype.gaussian = function (x0, y0, sx, sy, vx, vy) {
    var N = this.N, kx = this.m * vx / this.hbar, ky = this.m * vy / this.hbar;
    for (var j = 0; j < N; j++) for (var i = 0; i < N; i++) {
      var x = this.x(i) - x0, y = this.y(j) - y0;
      var a = Math.exp(-x * x / (4 * sx * sx) - y * y / (4 * sy * sy));
      var ph = kx * x + ky * y, q = j * N + i;
      this.re[q] = a * Math.cos(ph); this.im[q] = a * Math.sin(ph);
    }
    this.normalise();
    this.time = 0; this.absorbed = 0; this.screen.fill(0);
  };

  Sim.prototype.norm = function () {
    var s = 0, re = this.re, im = this.im;
    for (var q = 0; q < re.length; q++) s += re[q] * re[q] + im[q] * im[q];
    return s * this.dx * this.dx;
  };
  Sim.prototype.normalise = function () {
    var s = 1 / Math.sqrt(this.norm());
    for (var q = 0; q < this.re.length; q++) { this.re[q] *= s; this.im[q] *= s; }
  };

  // Mean position and spread of |ψ|², normalised by what is still on the grid.
  Sim.prototype.moments = function () {
    var N = this.N, re = this.re, im = this.im, s = 0, mx = 0, my = 0, xx = 0, yy = 0;
    for (var j = 0; j < N; j++) for (var i = 0; i < N; i++) {
      var q = j * N + i, p = re[q] * re[q] + im[q] * im[q], x = this.x(i), y = this.y(j);
      s += p; mx += p * x; my += p * y; xx += p * x * x; yy += p * y * y;
    }
    mx /= s; my /= s;
    return { norm: s * this.dx * this.dx, x: mx, y: my,
             sx: Math.sqrt(Math.max(0, xx / s - mx * mx)), sy: Math.sqrt(Math.max(0, yy / s - my * my)) };
  };

  // Probability in the half-plane x > x0 (plus whatever the screen has already swallowed).
  Sim.prototype.probRightOf = function (x0) {
    var N = this.N, re = this.re, im = this.im, s = 0;
    var i0 = Math.max(0, Math.ceil(x0 / this.dx - 0.5));
    for (var j = 0; j < N; j++) for (var i = i0; i < N; i++) { var q = j * N + i; s += re[q] * re[q] + im[q] * im[q]; }
    return s * this.dx * this.dx;
  };

  Sim.prototype._prepare = function (dt) {
    var N = this.N, hb = this.hbar, L = this.L, c = this.c, mc2 = this.m * c * c;
    var a = hb * dt / (2 * this.m);
    for (var j = 0; j < N; j++) {
      var fy = j < N / 2 ? j : j - N, ky = 2 * Math.PI * fy / L;
      for (var i = 0; i < N; i++) {
        var fx = i < N / 2 ? i : i - N, kx = 2 * Math.PI * fx / L, k2 = kx * kx + ky * ky;
        // kinetic energy × dt / ħ, written so that nothing cancels when c is large
        var ph = c ? hb * k2 * c * c * dt / (Math.sqrt(hb * hb * k2 * c * c + mc2 * mc2) + mc2) : a * k2, q = j * N + i;
        this._kc[q] = Math.cos(ph); this._ks[q] = -Math.sin(ph);
      }
    }
    var vw = this.wallPhase * hb / dt;          // wall height chosen so its phase per step is fixed
    this.wallHeight = vw;
    for (var q2 = 0; q2 < N * N; q2++) {
      var v = this.wall[q2] ? vw : this.V[q2];
      var p2 = v * dt / (2 * hb);
      this._pc[q2] = Math.cos(p2); this._ps[q2] = -Math.sin(p2);
      this._pd[q2] = this.damp[q2] > 0 ? Math.exp(-this.damp[q2] * dt / 2) : 1;
    }
    this._dt = dt;
  };

  Sim.prototype._halfPotential = function () {
    var N = this.N, re = this.re, im = this.im, pc = this._pc, ps = this._ps, pd = this._pd;
    var lost = 0, sf = this.screenFrom;
    for (var j = 0; j < N; j++) {
      var rowLost = 0;
      for (var i = 0; i < N; i++) {
        var q = j * N + i, a = re[q], b = im[q], d = pd[q];
        if (d !== 1) {
          var p = (a * a + b * b) * (1 - d * d);
          lost += p; if (i >= sf) rowLost += p;
          a *= d; b *= d;
        }
        re[q] = a * pc[q] - b * ps[q];
        im[q] = a * ps[q] + b * pc[q];
      }
      this.screen[j] += rowLost * this.dx * this.dx;
    }
    this.absorbed += lost * this.dx * this.dx;
  };

  Sim.prototype._fft2 = function (inverse) {
    var N = this.N, re = this.re, im = this.im, tr = this._tr, ti = this._ti, f = this._fft, i, j;
    for (j = 0; j < N; j++) {
      var o = j * N;
      for (i = 0; i < N; i++) { tr[i] = re[o + i]; ti[i] = im[o + i]; }
      if (inverse) f.inverse(tr, ti); else f.forward(tr, ti);
      for (i = 0; i < N; i++) { re[o + i] = tr[i]; im[o + i] = ti[i]; }
    }
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) { tr[j] = re[j * N + i]; ti[j] = im[j * N + i]; }
      if (inverse) f.inverse(tr, ti); else f.forward(tr, ti);
      for (j = 0; j < N; j++) { re[j * N + i] = tr[j]; im[j * N + i] = ti[j]; }
    }
  };

  Sim.prototype.step = function (dt) {
    if (dt !== this._dt) this._prepare(dt);
    var re = this.re, im = this.im, kc = this._kc, ks = this._ks, n = re.length;
    this._halfPotential();
    this._fft2(false);
    for (var q = 0; q < n; q++) {
      var a = re[q], b = im[q];
      re[q] = a * kc[q] - b * ks[q];
      im[q] = a * ks[q] + b * kc[q];
    }
    this._fft2(true);
    this._halfPotential();
    this.time += dt;
  };

  // A uniform force acting briefly: every momentum component shifts by ħ(dkx, dky).
  Sim.prototype.kick = function (dkx, dky) {
    var N = this.N, re = this.re, im = this.im;
    for (var j = 0; j < N; j++) for (var i = 0; i < N; i++) {
      var ph = dkx * this.x(i) + dky * this.y(j), c = Math.cos(ph), s = Math.sin(ph), q = j * N + i;
      var a = re[q], b = im[q];
      re[q] = a * c - b * s; im[q] = a * s + b * c;
    }
  };
  // Mean wavenumber <k>, from the momentum distribution |ψ(k)|². y grows upward, so row frequencies flip sign.
  Sim.prototype.meanK = function () {
    var N = this.N, L = this.L, sx = 0, sy = 0, s = 0, re0 = this.re, im0 = this.im;
    this.re = Float64Array.from(re0); this.im = Float64Array.from(im0);
    this._fft2(false);
    for (var j = 0; j < N; j++) {
      var ky = -2 * Math.PI * (j < N / 2 ? j : j - N) / L;
      for (var i = 0; i < N; i++) {
        var kx = 2 * Math.PI * (i < N / 2 ? i : i - N) / L, q = j * N + i;
        var p = this.re[q] * this.re[q] + this.im[q] * this.im[q];
        sx += p * kx; sy += p * ky; s += p;
      }
    }
    this.re = re0; this.im = im0;
    return [sx / s, sy / s];
  };
  // Move the whole wave by whole cells (di right, dj up). What slides off is dropped, new cells are empty.
  Sim.prototype.shift = function (di, dj) {
    var N = this.N, re = new Float64Array(N * N), im = new Float64Array(N * N);
    for (var j = 0; j < N; j++) {
      var js = j - dj;                                   // row 0 is the top: a window moved up reads rows above
      if (js < 0 || js >= N) continue;
      for (var i = 0; i < N; i++) {
        var is = i + di;
        if (is < 0 || is >= N) continue;
        re[j * N + i] = this.re[js * N + is]; im[j * N + i] = this.im[js * N + is];
      }
    }
    this.re = re; this.im = im;
  };

  /* A position measurement of finite sharpness: pick where the ball is found from |ψ|²,
   * then keep only the part of the wave within ~sigma of that spot (a Gaussian filter).
   * rand is a function returning uniform numbers in [0,1). Returns the spot found. */
  Sim.prototype.measure = function (sigma, rand) {
    var N = this.N, re = this.re, im = this.im, total = 0, q;
    for (q = 0; q < re.length; q++) total += re[q] * re[q] + im[q] * im[q];
    var target = rand() * total, acc = 0, hit = re.length - 1;
    for (q = 0; q < re.length; q++) { acc += re[q] * re[q] + im[q] * im[q]; if (acc >= target) { hit = q; break; } }
    var hi = hit % N, hj = (hit - hi) / N, x0 = this.x(hi), y0 = this.y(hj);
    for (var j = 0; j < N; j++) for (var i = 0; i < N; i++) {
      var x = this.x(i) - x0, y = this.y(j) - y0;
      var w = Math.exp(-(x * x + y * y) / (4 * sigma * sigma));
      q = j * N + i; re[q] *= w; im[q] *= w;
    }
    this.normalise();
    return { x: x0, y: y0 };
  };

  /* The same solver in one dimension, for problems that separate (a ball over a floor).
   * opts: N, L, m, hbar. Coordinate z runs from 0 (index 0) to L. */
  function Sim1D(opts) {
    var N = this.N = opts.N;
    this.L = opts.L; this.dz = opts.L / N; this.m = opts.m; this.hbar = opts.hbar;
    this.re = new Float64Array(N); this.im = new Float64Array(N);
    this.V = new Float64Array(N); this.wall = new Uint8Array(N); this.damp = new Float64Array(N);
    this.wallPhase = 1.0; this.time = 0; this._dt = -1;
    this._fft = new FFT(N);
    this._kc = new Float64Array(N); this._ks = new Float64Array(N);
    this._pc = new Float64Array(N); this._ps = new Float64Array(N); this._pd = new Float64Array(N);
  }
  Sim1D.prototype.z = function (i) { return (i + 0.5) * this.dz; };
  Sim1D.prototype.build = function (fn) {
    for (var i = 0; i < this.N; i++) {
      var c = fn(this.z(i)) || {};
      this.V[i] = c.V || 0; this.wall[i] = c.wall ? 1 : 0; this.damp[i] = c.damp || 0;
    }
    this._dt = -1;
  };
  Sim1D.prototype.set = function (fn) {       // fn(z) -> [re, im]
    for (var i = 0; i < this.N; i++) { var v = fn(this.z(i)); this.re[i] = v[0]; this.im[i] = v[1]; }
    this.normalise(); this.time = 0;
  };
  Sim1D.prototype.norm = function () {
    var s = 0; for (var i = 0; i < this.N; i++) s += this.re[i] * this.re[i] + this.im[i] * this.im[i];
    return s * this.dz;
  };
  Sim1D.prototype.normalise = function () {
    var s = 1 / Math.sqrt(this.norm());
    for (var i = 0; i < this.N; i++) { this.re[i] *= s; this.im[i] *= s; }
  };
  Sim1D.prototype.mean = function () {
    var s = 0, mz = 0;
    for (var i = 0; i < this.N; i++) { var p = this.re[i] * this.re[i] + this.im[i] * this.im[i]; s += p; mz += p * this.z(i); }
    return mz / s;
  };
  Sim1D.prototype._prepare = function (dt) {
    var N = this.N, hb = this.hbar, a = hb * dt / (2 * this.m), vw = this.wallPhase * hb / dt;
    for (var i = 0; i < N; i++) {
      var f = i < N / 2 ? i : i - N, k = 2 * Math.PI * f / this.L, ph = a * k * k;
      this._kc[i] = Math.cos(ph); this._ks[i] = -Math.sin(ph);
      var p2 = (this.wall[i] ? vw : this.V[i]) * dt / (2 * hb);
      this._pc[i] = Math.cos(p2); this._ps[i] = -Math.sin(p2);
      this._pd[i] = this.damp[i] > 0 ? Math.exp(-this.damp[i] * dt / 2) : 1;
    }
    this._dt = dt;
  };
  Sim1D.prototype._mul = function (c, s, d) {
    var re = this.re, im = this.im;
    for (var i = 0; i < this.N; i++) {
      var a = re[i], b = im[i];
      if (d) { a *= d[i]; b *= d[i]; }
      re[i] = a * c[i] - b * s[i]; im[i] = a * s[i] + b * c[i];
    }
  };
  Sim1D.prototype.step = function (dt) {
    if (dt !== this._dt) this._prepare(dt);
    this._mul(this._pc, this._ps, this._pd);
    this._fft.forward(this.re, this.im);
    this._mul(this._kc, this._ks, null);
    this._fft.inverse(this.re, this.im);
    this._mul(this._pc, this._ps, this._pd);
    this.time += dt;
  };

  /* Airy function Ai(x) on a grid, by integrating y'' = x y inwards from the decaying side.
   * Returns values on xs (ascending), up to an overall constant: enough to build
   * the stationary states of a ball on a floor. */
  function airyShape(xs) {
    var out = new Float64Array(xs.length);
    var x = Math.max(8, xs[xs.length - 1] + 1), h = 0.002;
    var y = 1e-12, dy = -Math.sqrt(x) * 1e-12;     // deep in the forbidden side, Ai'/Ai ≈ -√x
    var idx = xs.length - 1;
    while (idx >= 0) {
      // one RK4 step of (y, y') from x down to x - h
      var a1 = dy, b1 = x * y;
      var a2 = dy - 0.5 * h * b1, b2 = (x - 0.5 * h) * (y - 0.5 * h * a1);
      var a3 = dy - 0.5 * h * b2, b3 = (x - 0.5 * h) * (y - 0.5 * h * a2);
      var a4 = dy - h * b3, b4 = (x - h) * (y - h * a3);
      var yn = y - h * (a1 + 2 * a2 + 2 * a3 + a4) / 6;
      var dyn = dy - h * (b1 + 2 * b2 + 2 * b3 + b4) / 6;
      while (idx >= 0 && xs[idx] >= x - h) {
        var t = (x - xs[idx]) / h;
        out[idx] = y + t * (yn - y);
        idx--;
      }
      x -= h; y = yn; dy = dyn;
    }
    return out;
  }

  var api = { FFT: FFT, Sim: Sim, Sim1D: Sim1D, airyShape: airyShape };
  root.CGH = root.CGH || {};
  root.CGH.qcore = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
