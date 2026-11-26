/* Observatorium — astrophysics used by the instruments. SI units inside, astronomers' units at the edges.
 * Every relation here is one the course uses; the empirical calibrations say where they came from. */
(function (G) {
  'use strict';
  var A = G.OBS.astro = {};
  var D2R = Math.PI / 180;
  var C = A.C = {
    c: 2.99792458e8, G: 6.674e-11, h: 6.62607015e-34, k: 1.380649e-23, sigma: 5.670374e-8,
    Lsun: 3.828e26, Rsun: 6.957e8, Msun: 1.98847e30, Tsun: 5772, AU: 1.495978707e11, pc: 3.0856776e16,
    day: 86400, yr: 3.15576e7, MG_sun: 4.67, Mbol_sun: 4.74, MV_sun: 4.83, Rjup: 7.1492e7, Mjup: 1.89813e27,
    H0: 70, kms: 1e3
  };

  // ------------------------------------------------------------------ coordinates
  // ICRS (equatorial J2000) -> Galactic rotation matrix (Hipparcos definition, ESA 1997 vol. 1 §1.5.3).
  var RG = [[-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
    [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
    [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669]];
  function vec(lon, lat) {
    var cl = Math.cos(lat * D2R);
    return [cl * Math.cos(lon * D2R), cl * Math.sin(lon * D2R), Math.sin(lat * D2R)];
  }
  function ang(v) {
    var lon = Math.atan2(v[1], v[0]) / D2R;
    return [lon < 0 ? lon + 360 : lon, Math.asin(Math.max(-1, Math.min(1, v[2]))) / D2R];
  }
  A.vec = vec; A.ang = ang;
  A.equToGal = function (ra, dec) {
    var v = vec(ra, dec);
    return ang([RG[0][0] * v[0] + RG[0][1] * v[1] + RG[0][2] * v[2], RG[1][0] * v[0] + RG[1][1] * v[1] + RG[1][2] * v[2], RG[2][0] * v[0] + RG[2][1] * v[1] + RG[2][2] * v[2]]);
  };
  A.galToEqu = function (l, b) {
    var v = vec(l, b);
    return ang([RG[0][0] * v[0] + RG[1][0] * v[1] + RG[2][0] * v[2], RG[0][1] * v[0] + RG[1][1] * v[1] + RG[2][1] * v[2], RG[0][2] * v[0] + RG[1][2] * v[1] + RG[2][2] * v[2]]);
  };
  var EPS = 23.4392911 * D2R;
  A.equToEcl = function (ra, dec) {
    var v = vec(ra, dec), ce = Math.cos(EPS), se = Math.sin(EPS);
    return ang([v[0], ce * v[1] + se * v[2], -se * v[1] + ce * v[2]]);
  };
  /** Angle between two directions, degrees. */
  A.sep = function (ra1, de1, ra2, de2) {
    var a = vec(ra1, de1), b = vec(ra2, de2);
    var d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    return Math.acos(Math.max(-1, Math.min(1, d))) / D2R;
  };

  // ------------------------------------------------------------------ distances and magnitudes
  A.GAIA_ZP = 0.017; // mas, global parallax zero point to add (Lindegren et al. 2021, A&A 649, A4)
  A.distPc = function (plxMas) { return plxMas > 0 ? 1000 / plxMas : NaN; };
  A.distMod = function (dPc) { return 5 * Math.log10(dPc) - 5; };
  A.distFromMu = function (mu) { return Math.pow(10, mu / 5 + 1); };
  A.absMag = function (m, plxMas) { return plxMas > 0 ? m + 5 * Math.log10(plxMas / 100) : NaN; };
  A.fluxRatio = function (dm) { return Math.pow(10, -0.4 * dm); };

  // Empirical Gaia calibrations, fitted in pipeline/ to Gaia DR3's own temperatures (GSP-Phot) and FLAME
  // luminosities for 2816 stars within 100 pc (not white dwarfs). Valid for -0.1 < BP-RP < 3.6.
  var TPOLY = [0.00485747, -0.04839969, 0.19392372, -0.44831331, 4.01728836];   // log10 T(BP-RP), rms 0.014 dex
  var BCPOLY = [0.17382214, -0.98016866, 0.97813796, -0.21358207];              // BC_G(BP-RP), rms 0.12 mag
  function poly(c, x) { var y = 0; for (var i = 0; i < c.length; i++) y = y * x + c[i]; return y; }
  A.teffFromBpRp = function (c) { return isFinite(c) ? Math.pow(10, poly(TPOLY, Math.max(-0.1, Math.min(3.6, c)))) : NaN; };
  // Hydrogen-atmosphere white dwarfs: Gaia BP − RP against temperature (approximate, from DA model colours).
  var WDT = [[-0.56, 40000], [-0.42, 25000], [-0.30, 19000], [-0.18, 15000], [-0.07, 12000], [0.05, 10000], [0.25, 8000], [0.60, 6000], [0.85, 5000], [1.20, 4000]];
  A.teffWD = function (c) {
    if (!isFinite(c)) return NaN;
    if (c <= WDT[0][0]) return WDT[0][1];
    for (var k = 1; k < WDT.length; k++) if (c <= WDT[k][0]) { var u = (c - WDT[k - 1][0]) / (WDT[k][0] - WDT[k - 1][0]); return Math.pow(10, Math.log10(WDT[k - 1][1]) + u * (Math.log10(WDT[k][1]) - Math.log10(WDT[k - 1][1]))); }
    return WDT[WDT.length - 1][1];
  };
  A.bcG = function (c) { return isFinite(c) ? poly(BCPOLY, Math.max(-0.1, Math.min(3.6, c))) : NaN; };
  /** Bolometric luminosity in L_sun from Gaia absolute G magnitude and colour. */
  A.lumFromMG = function (MG, c) { return Math.pow(10, -0.4 * (MG + A.bcG(c) - C.Mbol_sun)); };
  /** B-V -> temperature (Ballesteros 2012, EPL 97, 34008), for the naked-eye sky. */
  A.teffFromBV = function (bv) { return 4600 * (1 / (0.92 * bv + 1.7) + 1 / (0.92 * bv + 0.62)); };
  /** Stefan–Boltzmann: radius in R_sun from L [L_sun] and T [K]. */
  A.radius = function (L, T) { return Math.sqrt(L) * Math.pow(C.Tsun / T, 2); };
  A.lumFromRT = function (R, T) { return R * R * Math.pow(T / C.Tsun, 4); };

  // ------------------------------------------------------------------ blackbodies
  /** Planck B_lambda (W m^-2 m^-1 sr^-1) at wavelength lam [m] and temperature T [K]. */
  A.planck = function (lam, T) {
    var x = C.h * C.c / (lam * C.k * T);
    return 2 * C.h * C.c * C.c / Math.pow(lam, 5) / (x > 700 ? Infinity : Math.expm1(x));
  };
  A.wienPeak = function (T) { return 2.897771955e-3 / T; };
  // Gaia passbands approximated as top hats (BP 330–680 nm, RP 630–1050 nm), photon-counting, with the zero
  // point chosen so that a 5772 K blackbody has the Sun's colour, BP-RP = 0.82 (Casagrande & VandenBerg 2018).
  A.BANDS = { BP: [330e-9, 680e-9], RP: [630e-9, 1050e-9] };
  function photons(T, band) {
    var n = 60, s = 0, a = band[0], b = band[1], dl = (b - a) / n;
    for (var i = 0; i < n; i++) { var l = a + (i + 0.5) * dl; s += A.planck(l, T) * l; }
    return s;
  }
  var ZP = null;
  A.bbBpRp = function (T) {
    var raw = -2.5 * Math.log10(photons(T, A.BANDS.BP) / photons(T, A.BANDS.RP));
    if (ZP === null) { ZP = 0; ZP = 0.82 - (-2.5 * Math.log10(photons(5772, A.BANDS.BP) / photons(5772, A.BANDS.RP))); }
    return raw + ZP;
  };
  /** Invert the blackbody colour by bisection: the colour temperature. */
  A.bbTeffFromBpRp = function (c) {
    var lo = 1500, hi = 60000;
    for (var i = 0; i < 60; i++) { var mid = Math.sqrt(lo * hi); if (A.bbBpRp(mid) > c) lo = mid; else hi = mid; }
    return Math.sqrt(lo * hi);
  };

  // ------------------------------------------------------------------ the parallax ellipse
  /** Apparent shift (east, north) in units of the parallax, for a star at (ra, dec), t in years from the
   *  March equinox. Uses the Sun's ecliptic longitude 2πt (circular orbit). */
  A.parallaxShift = function (ra, dec, t) {
    var ecl = A.equToEcl(ra, dec), lam = ecl[0] * D2R, bet = ecl[1] * D2R, ls = 2 * Math.PI * t;
    var dl = Math.sin(lam - ls), db = -Math.sin(bet) * Math.cos(lam - ls); // ecliptic east, north
    return { x: dl, y: db, beta: ecl[1] };
  };

  // ------------------------------------------------------------------ binary stars
  function solveKepler(M, e) {
    var E = e < 0.8 ? M : Math.PI;
    for (var i = 0; i < 30; i++) E -= (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
    return E;
  }
  /** Radial velocity (km/s) of a star in an orbit with period P, eccentricity e, argument of periastron
   *  w [deg], semi-amplitude K, systemic gamma, at orbital phase phi (0 = periastron). */
  A.rv = function (phi, e, w, K, gamma) {
    var E = solveKepler(2 * Math.PI * phi, e);
    var nu = 2 * Math.atan2(Math.sqrt(1 + e) * Math.sin(E / 2), Math.sqrt(1 - e) * Math.cos(E / 2));
    return gamma + K * (Math.cos(nu + w * D2R) + e * Math.cos(w * D2R));
  };
  /** Masses (M_sun) from a double-lined orbit: P days, K1, K2 km/s, eccentricity e, inclination i [deg]. */
  A.binaryMasses = function (P, K1, K2, e, i) {
    var s3 = Math.pow(Math.sin((i === undefined ? 90 : i) * D2R), 3);
    var tot = P * C.day * Math.pow((K1 + K2) * 1e3, 3) * Math.pow(1 - (e || 0) * (e || 0), 1.5) / (2 * Math.PI * C.G) / C.Msun / s3;
    var a = (K1 + K2) * 1e3 * P * C.day * Math.sqrt(1 - (e || 0) * (e || 0)) / (2 * Math.PI) / Math.sin((i === undefined ? 90 : i) * D2R);
    return { M1: tot * K2 / (K1 + K2), M2: tot * K1 / (K1 + K2), total: tot, aAU: a / C.AU, aRsun: a / C.Rsun };
  };
  /** Kepler's third law: period in years for a in AU and total mass in M_sun. */
  A.keplerP = function (aAU, M) { return Math.sqrt(Math.pow(aAU, 3) / (M || 1)); };

  // ------------------------------------------------------------------ main-sequence scalings
  A.ML_SLOPE = 3.86; // L ∝ M^a for the main-sequence stars of Torres et al. 2010 (fitted in pipeline/)
  A.massFromL = function (L, a) { return Math.pow(L, 1 / (a || A.ML_SLOPE)); };
  /** Main-sequence lifetime in years: t = 10 Gyr (M/M_sun)/(L/L_sun). */
  A.lifetime = function (M, a) { return 1e10 * Math.pow(M, 1 - (a || A.ML_SLOPE)); };

  // ------------------------------------------------------------------ microlensing and transits
  A.magnification = function (u) { return (u * u + 2) / (u * Math.sqrt(u * u + 4)); };
  /** Paczyński light curve in magnitudes with blending: fbl = source fraction of the baseline flux. */
  A.paczynski = function (t, t0, tE, u0, fbl, I0) {
    var tau = (t - t0) / tE, u = Math.sqrt(u0 * u0 + tau * tau);
    return I0 - 2.5 * Math.log10(fbl * A.magnification(u) + 1 - fbl);
  };
  /** Relative flux during a transit (trapezoid): phase in days from mid-transit, total duration T, ingress tau. */
  A.transit = function (dt, depth, T, tau) {
    var x = Math.abs(dt), half = T / 2;
    if (x >= half) return 1;
    if (x <= half - tau) return 1 - depth;
    return 1 - depth * (half - x) / tau;
  };

  // ------------------------------------------------------------------ galaxies
  /** Speed (km/s) the visible matter would give, SPARC convention (Lelli et al. 2016): gas can be negative. */
  A.vBaryon = function (vgas, vdisk, vbul, ydisk, ybul) {
    var v2 = vgas * Math.abs(vgas) + ydisk * vdisk * Math.abs(vdisk) + ybul * vbul * Math.abs(vbul);
    return Math.sign(v2) * Math.sqrt(Math.abs(v2));
  };
  /** Pseudo-isothermal dark halo: V(r) = Vinf sqrt(1 - (rc/r) atan(r/rc)). */
  A.vHalo = function (r, vinf, rc) { return r > 0 ? vinf * Math.sqrt(Math.max(0, 1 - rc / r * Math.atan(r / rc))) : 0; };
  /** MOND with the 'simple' interpolating function: g = gN/2 + sqrt(gN^2/4 + gN a0). a0 in m/s^2. */
  A.vMond = function (vN, rKpc, a0) {
    var r = rKpc * 1e3 * C.pc, gN = vN * Math.abs(vN) * 1e6 / r;
    if (gN <= 0) return 0;
    var g = gN / 2 + Math.sqrt(gN * gN / 4 + gN * a0);
    return Math.sqrt(g * r) / 1e3;
  };
  /** Mass (M_sun) enclosed within r [kpc] by a circular speed v [km/s]. */
  A.massEnclosed = function (v, rKpc) { return v * v * 1e6 * rKpc * 1e3 * C.pc / C.G / C.Msun; };
  /** Virial mass (M_sun) of a cluster: M = 3 sigma_los^2 R / G (sigma km/s, R Mpc). */
  A.virialMass = function (sigma, RMpc) { return 3 * sigma * sigma * 1e6 * RMpc * 1e6 * C.pc / C.G / C.Msun; };

  // ------------------------------------------------------------------ pulsars
  A.pulsarB = function (P, Pdot) { return 3.2e19 * Math.sqrt(P * Pdot); };     // gauss, magnetic dipole spin-down
  A.pulsarAge = function (P, Pdot) { return P / (2 * Pdot) / C.yr; };          // characteristic age, years

  // ------------------------------------------------------------------ cosmology (FLRW, matter + Lambda + curvature)
  /** Expansion rate in units of H0: E(z) = sqrt(Om (1+z)^3 + Ok (1+z)^2 + OL). Radiation ignored (z < 10). */
  A.E = function (z, Om, OL) { var a = 1 + z; return Math.sqrt(Math.max(1e-12, Om * a * a * a + (1 - Om - OL) * a * a + OL)); };
  /** Integral of f over [0, z] by Simpson's rule (n even). */
  function simpson(f, z, n) { n = n || 64; var h = z / n, s = f(0) + f(z); for (var i = 1; i < n; i++) s += (i % 2 ? 4 : 2) * f(i * h); return s * h / 3; }
  A.simpson = simpson;
  /** Transverse comoving distance in units of c/H0, with the curvature that 1 - Om - OL implies. */
  A.dM = function (z, Om, OL) {
    // integrate in u = ln(1+z), where the integrand (1+z)/E is smooth from z = 0 out to the CMB (z = 1100)
    var Ok = 1 - Om - OL, I = simpson(function (u) { var x = Math.expm1(u); return (1 + x) / A.E(x, Om, OL); }, Math.log1p(z), 64);
    if (Math.abs(Ok) < 1e-6) return I;
    var s = Math.sqrt(Math.abs(Ok));
    return Ok > 0 ? Math.sinh(s * I) / s : Math.sin(s * I) / s;
  };
  /** Luminosity distance in Mpc; angular-diameter distance is D_L / (1+z)^2. */
  A.DL = function (z, H0, Om, OL) { return (1 + z) * A.dM(z, Om, OL) * C.c / 1e3 / (H0 || C.H0); };
  A.DA = function (z, H0, Om, OL) { return A.dM(z, Om, OL) * C.c / 1e3 / (H0 || C.H0) / (1 + z); };
  /** Distance modulus m - M = 5 log10(D_L / 10 pc). */
  A.distModZ = function (z, H0, Om, OL) { return 5 * Math.log10(A.DL(z, H0, Om, OL) * 1e5); };
  /** An empty universe (Om = OL = 0, coasting): D_L = (c/H0) z (1 + z/2). */
  A.distModEmpty = function (z, H0) { return 5 * Math.log10(C.c / 1e3 / (H0 || C.H0) * z * (1 + z / 2) * 1e5); };
  /** Age of the universe at redshift z, in Gyr: integral over (1+z) of dz / ((1+z) H). */
  A.hubbleTimeGyr = function (H0) { return 977.8 / (H0 || C.H0); };
  A.age = function (z, H0, Om, OL) {
    // substitute a = 1/(1+z): t = (1/H0) * integral_0^a da / (a E(1/a - 1))
    var a1 = 1 / (1 + (z || 0)), f = function (a) { return a <= 0 ? 0 : 1 / (a * A.E(1 / a - 1, Om, OL)); };
    return A.hubbleTimeGyr(H0) * simpson(f, a1, 200);
  };
  A.lookback = function (z, H0, Om, OL) { return A.age(0, H0, Om, OL) - A.age(z, H0, Om, OL); };
  /** Deceleration parameter today: q0 = Om/2 - OL. Negative means the expansion is speeding up. */
  A.q0 = function (Om, OL) { return Om / 2 - OL; };

  // ------------------------------------------------------------------ the cosmic microwave background
  /** Blackbody intensity B_nu in MJy/sr at frequency f [GHz] (1 MJy/sr = 1e-20 W m^-2 Hz^-1 sr^-1). */
  A.planckNu = function (fGHz, T) {
    var nu = fGHz * 1e9, x = C.h * nu / (C.k * T);
    return 2 * C.h * nu * nu * nu / (C.c * C.c) / Math.expm1(x) * 1e20;
  };
  /** Peak of B_nu: nu_max = 58.79 GHz per kelvin. */
  A.wienNuGHz = function (T) { return 58.789 * T; };
  /** Number density of blackbody photons, per cm^3: n = 20.28 T^3. */
  A.photonDensity = function (T) { return 20.286 * T * T * T; };

  // ------------------------------------------------------------------ fast radio bursts
  /** Mean dispersion measure of the intergalactic medium (pc/cm^3) out to z (Macquart et al. 2020):
   *  DM = 3 c H0 Ob f_IGM chi_e / (8 pi G m_p) * integral (1+z)/E(z) dz, with f_IGM = 0.84 and chi_e = 0.875. */
  A.dmCosmic = function (z, Ob, H0, Om) {
    var h0 = (H0 || C.H0) * 1e3 / (1e6 * C.pc), mp = 1.67262192e-27;
    var pre = 3 * C.c * h0 / (8 * Math.PI * C.G * mp) / (C.pc * 1e6);   // per unit Ob f chi, in pc/cm^3
    var om = Om === undefined ? 0.31 : Om;
    return pre * Ob * 0.84 * 0.875 * simpson(function (x) { return (1 + x) / A.E(x, om, 1 - om); }, z, 24);
  };

  // ------------------------------------------------------------------ time and the horizon
  /** Julian Date from a JavaScript time (ms since 1970, UTC). */
  A.jd = function (ms) { return ms / 86400000 + 2440587.5; };
  /** Greenwich mean sidereal time, degrees (IAU 1982, good to a second over decades). */
  A.gmst = function (jd) { var d = jd - 2451545.0, T = d / 36525; return (((280.46061837 + 360.98564736629 * d + 0.000387933 * T * T) % 360) + 360) % 360; };
  /** Local sidereal time in degrees at east longitude lon. */
  A.lst = function (jd, lon) { return ((A.gmst(jd) + lon) % 360 + 360) % 360; };
  /** Equatorial to horizontal: [azimuth from north through east, altitude], degrees. */
  A.equToHor = function (ra, dec, lst, lat) {
    var H = (lst - ra) * D2R, d = dec * D2R, p = lat * D2R;
    var alt = Math.asin(Math.sin(p) * Math.sin(d) + Math.cos(p) * Math.cos(d) * Math.cos(H));
    var az = Math.atan2(-Math.cos(d) * Math.sin(H), Math.sin(d) * Math.cos(p) - Math.cos(d) * Math.sin(p) * Math.cos(H));
    return [((az / D2R) % 360 + 360) % 360, alt / D2R];
  };
  A.horToEqu = function (az, alt, lst, lat) {
    var a = az * D2R, h = alt * D2R, p = lat * D2R;
    var dec = Math.asin(Math.sin(p) * Math.sin(h) + Math.cos(p) * Math.cos(h) * Math.cos(a));
    var H = Math.atan2(-Math.sin(a) * Math.cos(h), Math.cos(p) * Math.sin(h) - Math.sin(p) * Math.cos(h) * Math.cos(a));
    return [((lst - H / D2R) % 360 + 360) % 360, dec / D2R];
  };
  /** The Sun's apparent position (low precision, ~0.01 deg; Astronomical Almanac). */
  A.sunPos = function (jd) {
    var n = jd - 2451545.0, L = (280.460 + 0.9856474 * n) % 360, g = (357.528 + 0.9856003 * n) % 360 * D2R;
    var lam = (L + 1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g)) * D2R, eps = (23.439 - 0.0000004 * n) * D2R;
    var ra = Math.atan2(Math.cos(eps) * Math.sin(lam), Math.cos(lam)) / D2R;
    return [(ra + 360) % 360, Math.asin(Math.sin(eps) * Math.sin(lam)) / D2R];
  };
  /** The Moon, to about a degree (main terms only): [ra, dec, illuminated fraction 0..1]. */
  A.moonPos = function (jd) {
    var d = jd - 2451545.0, L = 218.316 + 13.176396 * d, M = (134.963 + 13.064993 * d) * D2R, F = (93.272 + 13.229350 * d) * D2R;
    var lam = (L + 6.289 * Math.sin(M)) * D2R, bet = 5.128 * Math.sin(F) * D2R, eps = 23.439 * D2R;
    var x = Math.cos(bet) * Math.cos(lam), y = Math.cos(eps) * Math.cos(bet) * Math.sin(lam) - Math.sin(eps) * Math.sin(bet), z = Math.sin(eps) * Math.cos(bet) * Math.sin(lam) + Math.cos(eps) * Math.sin(bet);
    var ra = (Math.atan2(y, x) / D2R + 360) % 360, dec = Math.asin(z) / D2R, s = A.sunPos(jd), el = A.sep(ra, dec, s[0], s[1]);
    return [ra, dec, (1 - Math.cos(el * D2R)) / 2];
  };
  /** Airmass at altitude alt [deg] (Kasten & Young 1989; sec z above ~30 deg). */
  A.airmass = function (alt) { if (alt <= 0) return Infinity; var z = 90 - alt; return 1 / (Math.cos(z * D2R) + 0.50572 * Math.pow(96.07995 - z, -1.6364)); };
  /** The observatory the course's students are closest to: the Wise Observatory, Mitzpe Ramon. */
  A.WISE = { name: 'Wise Observatory, Mitzpe Ramon', lat: 30.5967, lon: 34.7633, tz: 2 };
})(typeof window !== 'undefined' ? window : globalThis);
