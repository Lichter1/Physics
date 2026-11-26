/* Observatorium — the catalogue: every dataset, every variable you can put on an axis, and which idea
 * (concept) you need before a derived variable unlocks. Observed quantities are always open; a derived one
 * opens when the mission that teaches its relation is done, the way the history went. */
(function (G) {
  'use strict';
  var OBS = G.OBS, A = OBS.astro;

  // ---------------------------------------------------------------------------------------------- concepts
  OBS.CONCEPTS = {
    parallax: { label: 'Parallax gives distance', rel: 'd [pc] = 1 / p [arcsec]', by: 'parallax' },
    invsq: { label: 'Inverse square gives luminosity', rel: 'M = m − 5 log₁₀(d / 10 pc),  L = 4πd²f', by: 'hrd' },
    thermo: { label: 'Colour gives temperature', rel: 'Wien: colour ratio of two bands → T', by: 'thermometer' },
    stefan: { label: 'Stefan–Boltzmann gives radius', rel: 'L = 4πR²σT⁴', by: 'radius' },
    kepler: { label: 'Orbits give mass', rel: 'M₁ + M₂ = P (K₁ + K₂)³ / (2πG sin³i)', by: 'binary' },
    masslum: { label: 'Mass sets luminosity', rel: 'L ∝ M^a on the main sequence;  t ≈ 10 Gyr · M/L', by: 'masslum' },
    leavitt: { label: 'Period gives luminosity', rel: 'M = a log₁₀ P + b for Cepheids', by: 'leavitt' },
    transit: { label: 'Transit depth gives size', rel: 'ΔF/F = (R_p / R★)²', by: 'transit' },
    cosmo: { label: 'Redshift gives distance and age', rel: 'D_L = (1+z) (c/H₀) ∫ dz/E(z),  E² = Ωm(1+z)³ + ΩΛ  (ΩΛ ≈ 0.7)', by: 'accel' }
  };
  var S = OBS.state = OBS.state || { concepts: {}, sandbox: false, lend: {} };
  // A mission lends you the ideas it teaches while it is open (S.lend), so its own exercise can use them.
  OBS.unlocked = function (needs) {
    if (!needs || !needs.length || S.sandbox) return true;
    return needs.every(function (k) { return S.concepts[k] || (S.lend && S.lend[k]); });
  };
  /** Whether the ideas a mission needs were learned in earlier missions (lent ones do not count). */
  OBS.learned = function (needs) {
    if (!needs || !needs.length || S.sandbox) return true;
    return needs.every(function (k) { return S.concepts[k]; });
  };

  // ---------------------------------------------------------------------------------------------- helpers
  function v(id, label, unit, f, o) {
    o = o || {}; o.id = id; o.label = label; o.unit = unit || ''; o.f = f; o.scale = o.scale || 'lin';
    return o;
  }
  function col(name) { return function (c, i) { return c[name][i]; }; }
  var plx = function (c, i) { return c.parallax[i] + A.GAIA_ZP; };
  var POS = [
    v('ra', 'Right ascension α', '°', col('ra'), { how: 'Position on the sky, measured against the reference stars.' }),
    v('dec', 'Declination δ', '°', col('dec'), { how: 'Position on the sky.' })
  ];
  // Gaia stars: observed and derived
  function gaiaStars(gcol, ccol) {
    var g = col(gcol), cc = col(ccol);
    function MG(c, i) { return A.absMag(c[gcol][i], plx(c, i)); }
    function L(c, i) { return A.lumFromMG(MG(c, i), c[ccol][i]); }
    // white dwarfs follow their own colour–temperature relation; the general one was fitted to ordinary stars
    function T(c, i) { return c.wd && c.wd[i] ? A.teffWD(c[ccol][i]) : A.teffFromBpRp(c[ccol][i]); }
    return POS.concat([
      v('G', 'Apparent brightness G', 'mag', g, { rev: true, how: 'Gaia G-band magnitude: m = −2.5 log₁₀(flux) + const. Bigger is fainter.' }),
      v('bp_rp', 'Colour BP − RP', 'mag', cc, { how: 'Blue magnitude minus red magnitude. Bigger is redder.' }),
      v('parallax', 'Parallax p', 'mas', plx, { needs: ['parallax'], scale: 'log', how: 'Half the yearly back-and-forth shift. Gaia value + 0.017 mas zero point.' }),
      v('pmra', 'Proper motion μ_α*', 'mas/yr', col('pmra'), { how: 'Drift across the sky, east–west.' }),
      v('pmdec', 'Proper motion μ_δ', 'mas/yr', col('pmdec'), { how: 'Drift across the sky, north–south.' }),
      v('dist', 'Distance d', 'pc', function (c, i) { return A.distPc(plx(c, i)); }, { needs: ['parallax'], scale: 'log', how: 'd [pc] = 1000 / p [mas]' }),
      v('vtan', 'Speed across the sky v_t', 'km/s', function (c, i) { return 4.74 * Math.hypot(c.pmra[i], c.pmdec[i]) / plx(c, i); }, { needs: ['parallax'], scale: 'log', how: 'v_t = 4.74 μ d  (μ in arcsec/yr, d in pc)' }),
      v('MG', 'Absolute magnitude M_G', 'mag', MG, { needs: ['parallax', 'invsq'], rev: true, how: 'M = m − 5 log₁₀(d / 10 pc): how bright it would look from 10 pc.' }),
      v('L', 'Luminosity L', 'L☉', L, { needs: ['parallax', 'invsq'], scale: 'log', how: 'From M_G plus a bolometric correction fitted to Gaia’s own luminosities: L = 10^(−0.4 (M_G + BC − 4.74)).' }),
      v('T', 'Temperature T', 'K', T, { needs: ['thermo'], scale: 'log', rev: true, how: 'From the colour, using a relation fitted to Gaia’s temperatures for 2,816 nearby stars (±3%).' }),
      v('R', 'Radius R', 'R☉', function (c, i) { return A.radius(L(c, i), T(c, i)); }, { needs: ['parallax', 'invsq', 'thermo', 'stefan'], scale: 'log', how: 'R = √L · (5772 K / T)²  in solar units, from L = 4πR²σT⁴.' }),
      v('Mms', 'Mass, if on the main sequence', 'M☉', function (c, i) { return A.massFromL(L(c, i)); }, { needs: ['parallax', 'invsq', 'masslum'], scale: 'log', how: 'M = L^(1/a) with your mass–luminosity slope a. Only meaningful for main-sequence stars.' }),
      v('tms', 'Main-sequence lifetime', 'yr', function (c, i) { var M = A.massFromL(L(c, i)); return A.lifetime(M); }, { needs: ['parallax', 'invsq', 'masslum'], scale: 'log', how: 't ≈ 10 Gyr · M / L' })
    ]);
  }
  var gaiaChecks = [
    v('T_gaia', 'Temperature (Gaia GSP-Phot)', 'K', col('teff_gspphot'), { needs: ['thermo'], scale: 'log', rev: true, how: 'Gaia’s own estimate from its low-resolution spectra. A check on yours.' }),
    v('R_gaia', 'Radius (Gaia FLAME)', 'R☉', col('radius_flame'), { needs: ['stefan'], scale: 'log', how: 'Gaia’s own radius estimate.' }),
    v('M_gaia', 'Mass (Gaia FLAME)', 'M☉', col('mass_flame'), { needs: ['masslum'], scale: 'log', how: 'Gaia’s estimate from stellar models.' })
  ];

  function galB(c, i) { return A.equToGal(c.ra[i], c.dec[i])[1]; }
  var GALB = [
    v('b', 'Galactic latitude b', '°', galB, { how: 'Angle above (+) or below (−) the plane of the Milky Way.' }),
    v('sinb', 'sin b', '', function (c, i) { return Math.sin(galB(c, i) * Math.PI / 180); }, { how: 'For objects spread evenly over the sky, sin b is spread evenly between −1 and +1.' })
  ];
  /** Lookback time in a flat universe with Ωm = 0.31, H0 = 70 (needs the 'cosmo' concept). */
  function tlb(zcol) { return v('tlb', 'Lookback time', 'Gyr', function (c, i) { return c[zcol][i] > 0 ? A.lookback(c[zcol][i], 70, 0.31, 0.69) : NaN; }, { needs: ['cosmo'], how: 'How long ago the light left: ∫ dz / ((1+z) H(z)) in the ΛCDM model you tested.' }); }
  var DS = OBS.DATASETS = {};
  // HYG gives Bayer names as 'Tau Phe' or 'Kap-1 Scl', and only the constellation ('Phe') for most stars
  var GREEK = { Alp: 'α', Bet: 'β', Gam: 'γ', Del: 'δ', Eps: 'ε', Zet: 'ζ', Eta: 'η', The: 'θ', Iot: 'ι', Kap: 'κ', Lam: 'λ', Mu: 'μ', Nu: 'ν', Xi: 'ξ', Omi: 'ο', Pi: 'π', Rho: 'ρ', Sig: 'σ', Tau: 'τ', Ups: 'υ', Phi: 'φ', Chi: 'χ', Psi: 'ψ', Ome: 'ω' };
  var SUP = { 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  function bayerName(b, hip) {
    var p = (b || '').split(' ');
    if (p.length === 2) { var g = p[0].split('-'); if (GREEK[g[0]]) return GREEK[g[0]] + (g[1] ? SUP[g[1]] || g[1] : '') + ' ' + p[1]; return b; }
    return (hip ? 'HIP ' + hip : 'a star') + (p[0] ? ' in ' + p[0] : '');
  }
  function def(d) { DS[d.id] = d; return d; }

  // ---------------------------------------------------------------------------------------------- the sky
  def({
    id: 'sky', file: 'sky', level: 'sky', title: 'The naked-eye sky', sub: 'Every star you can see without a telescope (V < 6.5): Hipparcos, via HYG',
    pos: true, mag: 'mag', bv: 'ci', x: 'ci', y: 'mag',
    name: function (c, i) { return c.name[i] || bayerName(c.bayer[i], c.hip[i]); },
    vars: POS.concat([
      v('mag', 'Apparent brightness V', 'mag', col('mag'), { rev: true, how: 'Visual magnitude (Hipparchus’ scale, made exact by Pogson in 1856): 5 magnitudes = ×100 in flux.' }),
      v('ci', 'Colour B − V', 'mag', col('ci'), { how: 'Blue magnitude minus visual magnitude. Bigger is redder.' }),
      v('dist', 'Distance d', 'pc', col('dist'), { needs: ['parallax'], scale: 'log', how: 'Hipparcos parallax: d = 1/p.' }),
      v('MV', 'Absolute magnitude M_V', 'mag', function (c, i) { return c.mag[i] - 5 * Math.log10(c.dist[i] / 10); }, { needs: ['parallax', 'invsq'], rev: true, how: 'M = m − 5 log₁₀(d / 10 pc)' }),
      v('LV', 'Visual luminosity L_V', 'L☉', function (c, i) { return Math.pow(10, -0.4 * (c.mag[i] - 5 * Math.log10(c.dist[i] / 10) - A.C.MV_sun)); }, { needs: ['parallax', 'invsq'], scale: 'log', how: 'Light in the V band only, compared with the Sun’s.' }),
      v('T', 'Temperature T', 'K', function (c, i) { return A.teffFromBV(c.ci[i]); }, { needs: ['thermo'], scale: 'log', rev: true, how: 'From B − V (Ballesteros 2012).' })
    ])
  });

  // ---------------------------------------------------------------------------------------------- stars
  def({
    id: 'nearby', file: 'nearby', level: 'stars', title: 'Stars within 100 pc', sub: '36,683 stars from Gaia DR3 (a sample of the 207,176 that pass the HR-diagram quality cuts)',
    pos: true, mag: 'G', gaiaColour: 'bp_rp', x: 'bp_rp', y: 'G', name: function (c, i) { return c.id[i] ? 'Gaia DR3 ' + c.id[i] : 'Gaia star at ' + OBS.fix(1000 / (c.parallax[i] + A.GAIA_ZP), 1) + ' pc'; },
    vars: gaiaStars('phot_g_mean_mag', 'bp_rp').concat(gaiaChecks, [
      v('wd', 'White dwarf (Gaia catalogue flag)', '', function (c, i) { return c.wd[i] ? 'white dwarf' : 'other'; }, { cat: true, needs: ['invsq'], how: 'GCNS white-dwarf probability > 0.5.' })
    ])
  });
  def({
    id: 'binaries', file: 'binaries', level: 'stars', title: 'Eclipsing double-lined binaries', sub: '195 binaries whose two velocity curves Gaia measured, and which also eclipse',
    pos: true, mag: 'G', gaiaColour: 'bp_rp', x: 'period', y: 'K1', name: function (c, i) { return 'Gaia DR3 ' + c.id[i]; },
    vars: POS.concat([
      v('G', 'Apparent brightness G', 'mag', col('phot_g_mean_mag'), { rev: true }),
      v('bp_rp', 'Colour BP − RP', 'mag', col('bp_rp')),
      v('period', 'Orbital period P', 'd', col('period'), { scale: 'log', how: 'Time for the velocity curves to repeat.' }),
      v('K1', 'Velocity amplitude, star 1', 'km/s', col('K1'), { how: 'Half the peak-to-peak swing of star 1’s Doppler velocity.' }),
      v('K2', 'Velocity amplitude, star 2', 'km/s', col('K2')),
      v('e', 'Eccentricity e', '', col('e')),
      v('M1', 'Mass of star 1', 'M☉', function (c, i) { return A.binaryMasses(c.period[i], c.K1[i], c.K2[i], c.e[i], 90).M1; }, { needs: ['kepler'], scale: 'log', how: 'Kepler’s third law with sin i = 1 (they eclipse).' }),
      v('M2', 'Mass of star 2', 'M☉', function (c, i) { return A.binaryMasses(c.period[i], c.K1[i], c.K2[i], c.e[i], 90).M2; }, { needs: ['kepler'], scale: 'log' })
    ])
  });
  def({
    id: 'torres', file: 'torres', level: 'stars', title: 'Stars weighed to 3%', sub: '190 stars in 95 eclipsing binaries (Torres, Andersen & Giménez 2010)',
    pos: false, x: 'mass', y: 'L', name: function (c, i) { return c.system[i] + ' ' + c.star[i]; },
    vars: [
      v('mass', 'Mass M', 'M☉', col('mass'), { needs: ['kepler'], scale: 'log', how: 'From the two velocity curves and the eclipses.' }),
      v('radius', 'Radius R', 'R☉', col('radius'), { needs: ['kepler'], scale: 'log', how: 'From the eclipse durations and the orbit size.' }),
      v('teff', 'Temperature T', 'K', col('teff'), { scale: 'log', rev: true, how: 'From the spectra and colours.' }),
      v('L', 'Luminosity L', 'L☉', function (c, i) { return Math.pow(10, c.logL[i]); }, { scale: 'log', how: 'L = 4πR²σT⁴.' }),
      v('logg', 'Surface gravity log g', 'cgs', col('logg')),
      v('period', 'Orbital period P', 'd', col('period'), { scale: 'log' }),
      v('stage', 'Main sequence or evolved', '', function (c, i) { return c.logg[i] > 3.9 ? 'main sequence' : 'evolved'; }, { cat: true, how: 'log g > 3.9 counted as main sequence.' })
    ]
  });
  def({
    id: 'pulsars', file: 'pulsars', level: 'galaxy', title: 'Pulsars', sub: 'Every pulsar in the ATNF catalogue with a measured period',
    pos: true, x: 'P', y: 'Pdot', name: function (c, i) { return 'PSR ' + c.name[i]; },
    vars: POS.concat(GALB, [
      v('dm', 'Dispersion measure DM', 'pc/cm³', col('dm'), { scale: 'log', how: 'Free electrons along the line of sight: low frequencies arrive later by Δt ∝ DM / ν².' }),
      v('P', 'Spin period P', 's', col('P'), { scale: 'log', how: 'Time between pulses.' }),
      v('Pdot', 'Spin-down rate Ṗ', 's/s', col('Pdot'), { scale: 'log', how: 'How fast the period grows.' }),
      v('B', 'Surface magnetic field', 'gauss', function (c, i) { return A.pulsarB(c.P[i], c.Pdot[i]); }, { scale: 'log', how: 'B ≈ 3.2×10¹⁹ √(P Ṗ) G, for a spinning magnet losing energy.' }),
      v('age', 'Characteristic age τ', 'yr', function (c, i) { return A.pulsarAge(c.P[i], c.Pdot[i]); }, { scale: 'log', how: 'τ = P / 2Ṗ' }),
      v('bin', 'In a binary?', '', function (c, i) { return c.binary[i] ? 'binary' : 'single'; }, { cat: true }),
      v('kind', 'Kind', '', function (c, i) { var t = c.type[i]; return /AXP/.test(t) ? 'magnetar' : c.P[i] < 0.03 ? 'millisecond' : 'ordinary'; }, { cat: true })
    ])
  });
  def({
    id: 'exoplanets', file: 'exoplanets', level: 'stars', title: 'Known exoplanets', sub: 'NASA Exoplanet Archive, one row per confirmed planet',
    pos: true, x: 'P', y: 'Mp', colourBy: 'method', name: function (c, i) { return c.pl_name[i]; },
    vars: POS.concat([
      v('P', 'Orbital period P', 'd', col('pl_orbper'), { scale: 'log' }),
      v('a', 'Orbit size a', 'AU', col('pl_orbsmax'), { scale: 'log' }),
      v('Mp', 'Planet mass', 'M_Jup', col('pl_bmassj'), { scale: 'log', how: 'Measured for some planets; for radial-velocity planets often M sin i, a lower limit; for most transiting planets an estimate from the radius (colour by “Where the mass comes from”).' }),
      v('massprov', 'Where the mass comes from', '', function (c, i) { return c.massprov ? c.massprov[i] || 'unknown' : 'unknown'; }, { cat: true, how: 'The archive’s pl_bmassprov: a measured mass, a minimum mass (M sin i), or an estimate from the radius with the Chen & Kipping (2017) relation.' }),
      v('Rp', 'Planet radius', 'R_Jup', col('pl_radj'), { scale: 'log', needs: ['transit'], how: 'From the transit depth, (R_p/R★)² = ΔF/F: only for planets that transit.' }),
      v('rho', 'Planet density', 'g/cm³', function (c, i) { return c.massprov && c.massprov[i] === 'estimated from radius' ? NaN : 1.326 * c.pl_bmassj[i] / Math.pow(c.pl_radj[i], 3); }, { scale: 'log', needs: ['transit'], how: 'ρ = M / (4/3 π R³) in g/cm³ (Jupiter 1.33, Earth 5.5, water 1). Only planets with a measured mass and a radius: an estimated mass would just return the relation it came from.' }),
      v('Teq', 'Equilibrium temperature', 'K', col('pl_eqt'), { scale: 'log' }),
      v('year', 'Year found', '', col('disc_year')),
      v('method', 'How it was found', '', col('method'), { cat: true }),
      v('dist', 'Distance to the system', 'pc', col('sy_dist'), { scale: 'log' }),
      v('st_teff', 'Host star temperature', 'K', col('st_teff'), { scale: 'log', rev: true })
    ])
  });
  def({
    id: 'kepler', file: 'kepler', level: 'stars', title: 'Kepler light curves', sub: 'Brightness of two stars every 30 minutes for 89 days (Kepler, 2009)',
    pos: false, series: true, x: 't', y: 'f', subset: { col: 'star', label: 'Star', def: 'HAT-P-7' },
    vars: [
      v('t', 'Time', 'd', col('t'), { how: 'Days since 1 Jan 2009 (Kepler time, BJD − 2454833).' }),
      v('f', 'Relative brightness', '', col('f'), { how: 'Flux divided by its 2-day running median.' })
    ]
  });

  // ---------------------------------------------------------------------------------------------- clusters
  OBS.CLUSTERS = {
    pleiades: { label: 'Pleiades (M45)', ra: 56.75, dec: 24.12, fov: 7, ebv: 0.045, pm: [19.99, -45.55] },
    hyades: { label: 'Hyades', ra: 66.75, dec: 15.87, fov: 30, ebv: 0 },
    praesepe: { label: 'Praesepe (M44)', ra: 130.1, dec: 19.67, fov: 6, ebv: 0.027 },
    m67: { label: 'M67', ra: 132.85, dec: 11.81, fov: 1.5, ebv: 0.041 },
    ngc6397: { label: 'NGC 6397 (globular)', ra: 265.175, dec: -53.674, fov: 1, ebv: 0.18 }
  };
  // A Pleiades field star counts as a member when it moves with the cluster: within 3 mas/yr of its mean motion
  // and between 6.4 and 8.4 mas in parallax. Everything in the other cluster files already passed such cuts.
  OBS.isMember = function (c, i) {
    if (c.cl[i] !== 'pleiades') return true;
    return Math.hypot(c.pmra[i] - 19.99, c.pmdec[i] + 45.55) < 3 && c.parallax[i] > 6.4 && c.parallax[i] < 8.4;
  };
  var clStars = gaiaStars('phot_g_mean_mag', 'bp_rp');
  def({
    id: 'clusters', file: 'clusters', level: 'clusters', title: 'Star clusters', sub: 'Gaia DR3 stars around five clusters',
    pos: true, mag: 'G', gaiaColour: 'bp_rp', x: 'bp_rp', y: 'G', name: function (c, i) { return OBS.CLUSTERS[c.cl[i]].label + ' star'; },
    subset: { col: 'cl', label: 'Cluster', def: 'pleiades', names: { pleiades: 'Pleiades field (3°)', hyades: 'Hyades', praesepe: 'Praesepe', m67: 'M67', ngc6397: 'NGC 6397', '*': 'All five' } },
    vars: clStars.concat([
      v('member', 'Moves with the cluster?', '', function (c, i) { return OBS.isMember(c, i) ? 'member' : 'field'; }, { cat: true, spoils: 'pleiades' }),     // would give mission 1.8's answer away
      v('cluster', 'Cluster', '', function (c, i) { return OBS.CLUSTERS[c.cl[i]].label; }, { cat: true }),
      v('bp_rp0', 'Colour, corrected for dust', 'mag', function (c, i) { return c.bp_rp[i] - 1.339 * OBS.CLUSTERS[c.cl[i]].ebv; }, { needs: ['parallax', 'invsq'], how: '(BP − RP)₀ = BP − RP − 1.339 E(B−V)' }),
      v('MGc', 'Absolute magnitude (cluster distance)', 'mag', function (c, i) {
        var k = c.cl[i], d = OBS.clusterDist(k);
        return c.phot_g_mean_mag[i] - A.distMod(d) - 2.74 * OBS.CLUSTERS[k].ebv;
      }, { needs: ['parallax', 'invsq'], rev: true, how: 'Every star given the cluster’s distance (median parallax of its members), corrected for dust: M = G − 5 log₁₀(d/10) − 2.74 E(B−V).' })
    ])
  });
  OBS.clusterDist = function (k) { var x = OBS.CLUSTERS[k]; return x.dist || NaN; };

  // ---------------------------------------------------------------------------------------------- Cepheids
  def({
    id: 'cep_ogle', file: 'cepheids_ogle', level: 'galaxies', title: 'Cepheids in the Magellanic Clouds', sub: 'OGLE-IV: 5,189 fundamental-mode Cepheids',
    pos: true, mag: 'I', x: 'P', y: 'I', name: function (c, i) { return c.id[i]; },
    subset: { col: 'gal', label: 'Galaxy', def: 'smc', names: { smc: 'Small Magellanic Cloud', lmc: 'Large Magellanic Cloud', '*': 'Both' } },
    vars: POS.concat([
      v('P', 'Pulsation period P', 'd', col('P'), { scale: 'log', how: 'Time between brightness peaks.' }),
      v('I', 'Mean brightness I', 'mag', col('I'), { rev: true, how: 'Average near-infrared (I-band) magnitude over a cycle.' }),
      v('V', 'Mean brightness V', 'mag', col('V'), { rev: true }),
      v('VI', 'Colour V − I', 'mag', function (c, i) { return c.V[i] - c.I[i]; }),
      v('amp', 'Amplitude (I)', 'mag', col('Iamp')),
      v('gal', 'Galaxy', '', function (c, i) { return c.gal[i] === 'smc' ? 'SMC' : 'LMC'; }, { cat: true })
    ])
  });
  def({
    id: 'cep_gaia', file: 'cepheids_gaia', level: 'galaxy', title: 'Cepheids measured by Gaia', sub: 'Milky Way Cepheids with parallaxes, and the Magellanic Clouds’',
    pos: true, mag: 'G', x: 'P', y: 'W', name: function (c, i) { return 'Gaia DR3 ' + c.id[i]; },
    subset: { col: 'region', label: 'Where', def: 'mw', names: { mw: 'Milky Way', lmc: 'Large Magellanic Cloud', smc: 'Small Magellanic Cloud', '*': 'All' } },
    vars: POS.concat([
      v('P', 'Pulsation period P', 'd', col('P'), { scale: 'log' }),
      v('G', 'Mean brightness G', 'mag', col('G'), { rev: true }),
      v('W', 'Dust-proof brightness W', 'mag', function (c, i) { return c.G[i] - 1.90 * (c.BP[i] - c.RP[i]); }, { rev: true, how: 'Wesenheit index W = G − 1.90 (BP − RP) (Ripepi et al. 2019): dust dims and reddens together, so this combination cancels it.' }),
      v('parallax', 'Parallax p', 'mas', plx, { needs: ['parallax'], scale: 'log' }),
      v('dist', 'Distance d', 'pc', function (c, i) { return A.distPc(plx(c, i)); }, { needs: ['parallax'], scale: 'log' }),
      v('MW', 'Absolute W', 'mag', function (c, i) { return c.G[i] - 1.90 * (c.BP[i] - c.RP[i]) + 5 * Math.log10(plx(c, i) / 100); }, { needs: ['parallax', 'invsq'], rev: true, how: 'W + 5 log₁₀(p / 100 mas)' }),
      v('region', 'Where', '', col('region'), { cat: true })
    ])
  });
  def({
    id: 'cep_lc', file: 'cepheid_lc', level: 'galaxies', title: 'Cepheid light curves', sub: 'OGLE-IV, I band, 2010–2015. Photometry © OGLE (Udalski et al. 2015; Soszyński et al.), shown for study.',
    pos: false, series: true, x: 't', y: 'mag', subset: { col: 'id', label: 'Star', def: 'OGLE-LMC-CEP-1109' },
    vars: [v('t', 'Time', 'HJD − 2450000', col('t')), v('mag', 'Brightness I', 'mag', col('mag'), { rev: true })]
  });

  // ---------------------------------------------------------------------------------------------- the Milky Way
  def({
    id: 'microlensing', file: 'microlensing', level: 'galaxy', title: 'Microlensing events', sub: 'OGLE-IV Early Warning System, 2019 (toward the Galactic bulge). Photometry © OGLE (Udalski et al. 2015), shown for study; OGLE asks to be contacted before it is used in publications.',
    pos: false, series: true, x: 't', y: 'mag', subset: { col: 'event', label: 'Event', def: '2019-BLG-0148' },
    vars: [v('t', 'Time', 'HJD − 2450000', col('t')), v('mag', 'Brightness I', 'mag', col('mag'), { rev: true })]
  });
  def({
    id: 'globulars', file: 'globulars', level: 'galaxy', title: 'Globular clusters', sub: 'All 157 in the Harris catalogue',
    pos: true, x: 'X', y: 'Y', name: function (c, i) { return c.name[i] ? c.id[i] + ' (' + c.name[i] + ')' : c.id[i]; },
    vars: POS.concat([
      v('l', 'Galactic longitude l', '°', col('l')), v('b', 'Galactic latitude b', '°', col('b')),
      v('R_sun', 'Distance from the Sun', 'kpc', col('R_sun'), { how: 'From the brightness of each cluster’s horizontal-branch (RR Lyrae) stars.' }),
      v('X', 'X (toward Sagittarius)', 'kpc', col('X'), { how: 'Sun at 0. X points toward Galactic longitude 0°.' }),
      v('Y', 'Y (direction of rotation)', 'kpc', col('Y')), v('Z', 'Z (out of the disk)', 'kpc', col('Z'))
    ])
  });

  // ---------------------------------------------------------------------------------------------- solar system
  def({
    id: 'smallbodies', file: 'smallbodies', level: 'solar', title: 'Asteroids and Kuiper-belt objects', sub: 'JPL Small-Body Database: 52,613 orbits',
    pos: false, x: 'a', y: 'e', colourBy: 'class',
    name: function (c, i) { return c.pdes[i] ? '(' + c.pdes[i] + ')' : 'main-belt asteroid'; },
    vars: [
      v('a', 'Orbit size a', 'AU', col('a'), { how: 'Semi-major axis.' }), v('e', 'Eccentricity e', '', col('e')), v('i', 'Inclination i', '°', col('i')),
      v('H', 'Absolute magnitude H', 'mag', col('H'), { rev: true }),
      v('P', 'Orbital period', 'yr', function (c, i) { return Math.pow(c.a[i], 1.5); }, { how: 'Kepler’s third law, P² = a³ (years, AU).' }),
      v('Pj', 'Period / Jupiter’s period', '', function (c, i) { return Math.pow(c.a[i] / 5.2026, 1.5); }, { how: 'Simple fractions (1/3, 2/5, 1/2) mean resonance.' }),
      v('class', 'Group', '', function (c, i) {
        return { MBA: 'main belt', IMB: 'main belt', OMB: 'outer belt', TJN: 'Jupiter Trojans', TNO: 'beyond Neptune', CEN: 'centaurs', MCA: 'Mars-crossers', APO: 'near-Earth', AMO: 'near-Earth', ATE: 'near-Earth', AST: 'other' }[c['class'][i]] || 'other';
      }, { cat: true })
    ]
  });

  // ---------------------------------------------------------------------------------------------- small datasets used in the lectures
  def({
    id: 'spectra', file: 'spectra', level: 'stars', title: 'Star spectra (SDSS)', sub: 'One real spectrum for each spectral class, B to M (SDSS DR17)',
    pos: false, x: 'lam', y: 'f', connect: true, colourBy: 'type', subset: { col: 'type', label: 'Class', def: '*', names: { '*': 'All six', B: 'B', A: 'A', F: 'F', G: 'G', K: 'K', M: 'M' } },
    catOrder: { type: ['B', 'A', 'F', 'G', 'K', 'M'] }, name: function (c, i) { return c.name[i]; },
    vars: [
      v('lam', 'Wavelength λ', 'nm', col('lam')),
      v('f', 'Brightness (scaled)', '', col('f'), { how: 'Flux per unit wavelength, scaled so the brightest part is 1.' }),
      v('type', 'Spectral class', '', col('type'), { cat: true })
    ]
  });
  def({
    id: 'sunspots', file: 'sunspots', level: 'solar', title: 'Sunspots since 1700', sub: 'Yearly mean sunspot number (SILSO, Royal Observatory of Belgium)',
    pos: false, x: 'year', y: 'sn', connect: true,
    vars: [v('year', 'Year', '', col('year'), { plain: true }), v('sn', 'Sunspot number', '', col('sn'), { how: 'Spots counted on the Sun, averaged over the year (version 2 of the international series).' })]
  });
  def({
    id: 'grbs', file: 'grbs', level: 'universe', title: 'Gamma-ray bursts', sub: 'Every burst in the Fermi GBM catalogue with a measured duration',
    pos: true, x: 't90', y: '#', name: function (c, i) { return c.name[i]; },
    vars: POS.concat(GALB, [
      v('t90', 'Duration T₉₀', 's', col('t90'), { scale: 'log', how: 'Time over which 90% of the burst’s gamma-ray counts arrive.' }),
      v('fluence', 'Fluence', 'erg/cm²', col('fluence'), { scale: 'log', how: 'Total energy received per unit area.' }),
      v('kind', 'Short or long', '', function (c, i) { return c.t90[i] < 2 ? 'short (< 2 s)' : 'long'; }, { cat: true })
    ])
  });

  // ---------------------------------------------------------------------------------------------- chapters 6-9: the Universe, light, galaxies
  var H73 = 73;
  def({
    id: 'supernovae', file: 'supernovae', level: 'universe', title: 'Type Ia supernovae', sub: 'Pantheon+: 1,466 supernovae with redshifts and standardised peak brightness (Scolnic et al. 2022)',
    pos: true, mag: 'mb', x: 'z', y: 'mu', err: { mu: 'emu', dmu: 'emu' }, name: function (c, i) { return 'SN ' + c.name[i]; },
    vars: POS.concat([
      v('z', 'Redshift z', '', col('z'), { scale: 'log', how: 'Shift of the host galaxy’s (or the supernova’s) lines: 1 + z = λ_observed / λ_emitted.' }),
      v('mb', 'Peak brightness m_B (standardised)', 'mag', col('mb'), { rev: true, how: 'Peak B magnitude corrected for the light curve’s width and colour, which makes type Ia supernovae standard candles.' }),
      v('mu', 'Distance modulus μ = m − M', 'mag', col('mu'), { how: 'm_B − M_B with M_B ≈ −19.25 from Cepheid-calibrated supernovae (SH0ES).' }),
      v('dL', 'Luminosity distance D_L', 'Mpc', function (c, i) { return Math.pow(10, c.mu[i] / 5 + 1) / 1e6; }, { scale: 'log', how: 'D_L = 10^(μ/5 + 1) pc: the distance at which the flux matches the luminosity.' }),
      v('dmu', 'Fainter than an empty universe by Δμ', 'mag', function (c, i) { return c.mu[i] - A.distModEmpty(c.z[i], H73); }, { how: 'μ minus the μ of a coasting universe with no matter and no Λ (H₀ = 73). Positive: farther, fainter.' }),
      v('v', 'Recession speed cz', 'km/s', function (c, i) { return c.z[i] * 299792.458; }, { scale: 'log', how: 'Valid only while z ≪ 1.' }),
      tlb('z')
    ])
  });
  def({
    id: 'cmbspec', file: 'cmb_spectrum', level: 'universe', title: 'The CMB spectrum', sub: 'COBE FIRAS: the brightness of the microwave sky at 43 frequencies (Fixsen et al. 1996)',
    pos: false, x: 'f', y: 'I', err: { I: 'err' },
    vars: [
      v('f', 'Frequency ν', 'GHz', col('f_ghz'), { how: 'ν = c / λ' }),
      v('lam', 'Wavelength λ', 'mm', function (c, i) { return 10 / c.nu_icm[i]; }, { how: 'λ = 1 / wavenumber' }),
      v('I', 'Brightness I_ν', 'MJy/sr', col('I'), { how: 'Power per unit area, frequency and solid angle; 1 MJy/sr = 10⁻²⁰ W m⁻² Hz⁻¹ sr⁻¹.' }),
      v('resid', 'Residual from a 2.725 K blackbody', 'kJy/sr', col('resid'), { how: 'Measured minus a perfect blackbody, a thousand times smaller than the signal.' })
    ]
  });
  def({
    id: 'cmbpower', file: 'cmb_power', level: 'universe', title: 'The CMB power spectrum', sub: 'Planck 2018: how strongly the temperature varies on each angular scale',
    pos: false, x: 'l', y: 'Dl', err: { Dl: 'err' },
    vars: [
      v('l', 'Multipole ℓ', '', col('l'), { how: 'Spots of angular size θ correspond to ℓ ≈ 180° / θ.' }),
      v('theta', 'Angular scale ≈ 180°/ℓ', '°', function (c, i) { return 180 / c.l[i]; }, { scale: 'log' }),
      v('Dl', 'Fluctuation power D_ℓ', 'μK²', col('Dl'), { how: 'D_ℓ = ℓ(ℓ+1) C_ℓ / 2π: the variance of the temperature on that scale.' })
    ]
  });
  def({
    id: 'cmbtheory', file: 'cmb_theory', level: 'universe', title: 'The Planck best-fit CMB spectrum', sub: 'ΛCDM model spectrum from the Planck 2018 fit (theory, not data)',
    pos: false, x: 'l', y: 'tt',
    vars: [v('l', 'Multipole ℓ', '', col('l')), v('tt', 'Model D_ℓ', 'μK²', col('tt'))]
  });
  def({
    id: 'zspec', file: 'zspec', level: 'galaxies', title: 'Spectra for measuring redshifts', sub: 'SDSS: a star-forming galaxy, an elliptical, a distant red galaxy and a quasar',
    pos: false, x: 'lam', y: 'f', connect: true, subset: { col: 'obj', label: 'Object', def: 'sf', names: { sf: 'Star-forming galaxy', ell: 'Elliptical galaxy', lrg: 'Distant red galaxy', qso: 'Quasar' } },
    vars: [v('lam', 'Observed wavelength λ', 'nm', col('lam')), v('f', 'Brightness (scaled)', '', col('f'), { how: 'Flux per unit wavelength, scaled so the brightest part is about 1.' })]
  });
  var BPT = { 1: 'star-forming', 2: 'star-forming', 3: 'composite', 4: 'AGN', 5: 'AGN', '-1': 'unclassified' };
  def({
    id: 'bpt', file: 'bpt', level: 'galaxies', title: 'Galaxies’ emission lines', sub: 'SDSS (MPA-JHU): 20,000 galaxies with Hα, Hβ, [N II] and [O III] all measured',
    pos: true, mag: null, x: 'n2ha', y: 'o3hb', name: function (c, i) { return 'SDSS galaxy at z = ' + c.z[i].toFixed(4); },
    vars: POS.concat([
      v('z', 'Redshift z', '', col('z')),
      v('n2ha', 'log [N II] 658.4 / Hα', '', function (c, i) { return Math.log10(c.nii[i] / c.ha[i]); }, { how: 'Ratio of two neighbouring lines: no dust or distance correction needed.' }),
      v('o3hb', 'log [O III] 500.7 / Hβ', '', function (c, i) { return Math.log10(c.oiii[i] / c.hb[i]); }, { how: 'High when the ionising light is hard, as from an accreting black hole.' }),
      v('logm', 'Stellar mass', 'log M☉', col('logm'), { how: 'From the galaxy’s colours and spectrum, fitted with stellar population models.' }),
      v('logsfr', 'Star formation rate', 'log M☉/yr', col('logsfr'), { how: 'Mostly from the Hα line, corrected for dust and the fibre aperture (Brinchmann et al. 2004).' }),
      v('ssfr', 'Specific star formation rate', 'log /yr', function (c, i) { return c.logsfr[i] - c.logm[i]; }, { how: 'SFR per unit stellar mass: how fast the galaxy is growing.' }),
      v('oh', 'Gas metallicity 12 + log(O/H)', '', col('oh'), { how: 'Oxygen abundance of the gas, from its emission lines (Tremonti et al. 2004). The Sun: 8.69.' }),
      v('lha', 'Hα luminosity (in the fibre)', 'erg/s', function (c, i) { var d = A.DL(c.z[i], 70, 0.3, 0.7) * 1e6 * A.C.pc * 100; return 4 * Math.PI * d * d * c.ha[i] * 1e-17; }, { scale: 'log', needs: ['cosmo'], how: 'L = 4π D_L² F, with F in the 3″ fibre (no dust correction).' }),
      v('cls', 'MPA-JHU classification', '', function (c, i) { return BPT[c.bptclass[i]] || 'unclassified'; }, { cat: true, how: 'From the lines, using the Kauffmann et al. (2003) and Kewley et al. (2001) lines.' }),
      tlb('z')
    ])
  });
  def({
    id: 'frbs', file: 'frbs', level: 'universe', title: 'Fast radio bursts', sub: '183 bursts with dispersion measures; 101 localized to a host galaxy with a redshift',
    pos: true, x: 'z', y: 'dm', name: function (c, i) { return c.name[i]; },
    vars: POS.concat(GALB, [
      v('dm', 'Dispersion measure DM', 'pc/cm³', col('dm'), { how: 'Column of free electrons: the delay of low frequencies, Δt = 4.15 ms · DM · (ν / 1 GHz)⁻².' }),
      v('z', 'Host redshift z', '', col('z'), { how: 'From the host galaxy’s spectrum, once the burst is localized.' }),
      v('rep', 'Repeats?', '', function (c, i) { return c.repeater[i] ? 'repeater' : 'one-off'; }, { cat: true }),
      v('tel', 'Telescope', '', col('telescope'), { cat: true }),
      tlb('z')
    ])
  });
  def({
    id: 'gw', file: 'gw', level: 'universe', title: 'Gravitational-wave mergers', sub: 'LIGO, Virgo and KAGRA (GWTC-1 to GWTC-5): 282 mergers, median source-frame masses',
    pos: false, x: 'mtot', y: 'mfinal', name: function (c, i) { return c.name[i]; },
    vars: [
      v('m1', 'Heavier object m₁', 'M☉', col('m1'), { scale: 'log', how: 'From the waveform: the chirp and the merger.' }),
      v('m2', 'Lighter object m₂', 'M☉', col('m2'), { scale: 'log' }),
      v('mtot', 'Total mass m₁ + m₂', 'M☉', function (c, i) { return c.m1[i] + c.m2[i]; }, { scale: 'log' }),
      v('mfinal', 'Final mass', 'M☉', col('mfinal'), { scale: 'log', how: 'From the ringdown of the merged black hole.' }),
      v('mrad', 'Mass radiated as waves', 'M☉', function (c, i) { return c.m1[i] + c.m2[i] - c.mfinal[i]; }, { how: 'm₁ + m₂ − M_final; times c² it is the energy carried off.' }),
      v('mchirp', 'Chirp mass', 'M☉', col('mchirp'), { scale: 'log', how: '(m₁m₂)^{3/5} / (m₁+m₂)^{1/5}: what sets how fast the frequency sweeps up.' }),
      v('dl', 'Luminosity distance', 'Mpc', col('dl'), { scale: 'log', how: 'From the amplitude of the waves: a standard siren, no ladder needed.' }),
      v('z', 'Redshift', '', col('z')),
      v('snr', 'Signal-to-noise ratio', '', col('snr'), { scale: 'log' }),
      v('kind', 'Kind', '', function (c, i) { return c.m1[i] < 3 ? 'two neutron stars' : c.m2[i] < 3 ? 'neutron star + black hole' : 'two black holes'; }, { cat: true, how: 'Objects under 3 M☉ counted as neutron stars.' }),
      v('year', 'Year', '', col('year'), { plain: true })
    ]
  });

  // ---------------------------------------------------------------------------------------------- galaxies
  def({
    id: 'rotation', file: 'rotation_curves', level: 'galaxies', title: 'Rotation curves', sub: 'SPARC: 175 disk galaxies (21-cm and Hα velocities)',
    pos: false, x: 'R', y: 'Vobs', err: 'eV', subset: { col: 'gal', label: 'Galaxy', def: 'NGC3198' },
    vars: [
      v('R', 'Distance from the centre R', 'kpc', col('R')),
      v('Vobs', 'Orbital speed V', 'km/s', col('Vobs'), { how: 'From the Doppler shift of gas, corrected for the tilt of the disk.' }),
      v('Vgas', 'Speed the gas alone would give', 'km/s', col('Vgas')),
      v('Vdisk', 'Speed the stars alone would give', 'km/s', col('Vdisk'), { how: 'For a mass-to-light ratio of 1 at 3.6 μm.' })
    ]
  });
  def({
    id: 'sparc', file: 'sparc', level: 'galaxies', title: 'Disk galaxies: spin and light', sub: 'SPARC galaxy table (Lelli, McGaugh & Schombert 2016)',
    pos: false, x: 'Vflat', y: 'L36', name: function (c, i) { return c.gal[i]; },
    vars: [
      v('Vflat', 'Flat rotation speed', 'km/s', col('Vflat'), { scale: 'log', how: 'The speed where the rotation curve levels off.' }),
      v('L36', 'Luminosity at 3.6 μm', '10⁹ L☉', col('L36'), { scale: 'log', how: 'Near-infrared light, a good tracer of stellar mass.' }),
      v('MHI', 'Hydrogen gas mass', '10⁹ M☉', col('MHI'), { scale: 'log' }),
      v('D', 'Distance', 'Mpc', col('D'), { scale: 'log' }),
      v('Reff', 'Half-light radius', 'kpc', col('Reff'), { scale: 'log' }),
      v('T', 'Hubble type (0 = S0 … 10 = irregular)', '', col('T'))
    ]
  });
  var H0 = A.C.H0, CK = A.C.c / 1e3;
  function Mr(c, i) { return c.r[i] - A.distMod(c.z[i] * CK / H0 * 1e6); }
  def({
    id: 'galaxies', file: 'galaxies', level: 'galaxies', title: 'Nearby galaxies', sub: 'SDSS DR18, 0.02 < z < 0.05, with Galaxy Zoo classifications',
    pos: true, mag: 'r', x: 'Mr', y: 'ur', name: function (c, i) { return 'SDSS galaxy at z = ' + c.z[i].toFixed(4); },
    vars: POS.concat([
      v('r', 'Apparent brightness r', 'mag', col('r'), { rev: true }),
      v('ur', 'Colour u − r', 'mag', function (c, i) { return c.u[i] - c.r[i]; }, { how: 'Ultraviolet minus red. Young hot stars make a galaxy blue.' }),
      v('gr', 'Colour g − r', 'mag', function (c, i) { return c.g[i] - c.r[i]; }),
      v('cz', 'Recession speed cz', 'km/s', function (c, i) { return c.z[i] * CK; }, { how: 'Doppler shift of the spectrum: v = cz.' }),
      v('d', 'Distance (Hubble’s law)', 'Mpc', function (c, i) { return c.z[i] * CK / H0; }, { how: 'd = cz / H₀ with H₀ = 70 km/s/Mpc.' }),
      v('Mr', 'Absolute magnitude M_r', 'mag', Mr, { rev: true, how: 'M = r − 5 log₁₀(d / 10 pc)' }),
      v('sigma', 'Star speeds inside it σ', 'km/s', col('sigma'), { scale: 'log', how: 'Width of the absorption lines in the spectrum.' }),
      v('morph', 'Shape (Galaxy Zoo volunteers)', '', function (c, i) { return c.p_el[i] > 0.8 ? 'elliptical' : c.p_sp[i] > 0.8 ? 'spiral' : 'unsure'; }, { cat: true, how: 'At least 80% of volunteer votes, debiased (Lintott et al. 2011).' })
    ])
  });
  def({
    id: 'coma', file: 'coma', level: 'galaxies', title: 'Toward the Coma cluster', sub: 'SDSS galaxies with redshifts in a 5° field',
    pos: true, mag: 'r', x: 'cz', y: '#', name: function (c, i) { return 'SDSS galaxy'; },
    vars: POS.concat([
      v('cz', 'Recession speed cz', 'km/s', function (c, i) { return c.z[i] * CK; }),
      v('r', 'Apparent brightness r', 'mag', col('r'), { rev: true }),
      v('gr', 'Colour g − r', 'mag', function (c, i) { return c.g[i] - c.r[i]; }),
      v('sepc', 'Distance from the cluster centre', '°', function (c, i) { return A.sep(c.ra[i], c.dec[i], 194.953, 27.981); })
    ])
  });

  // ---------------------------------------------------------------------------------------------- columns
  /** Load a dataset's table and attach a cache of computed columns. */
  OBS.open = function (id) {
    var ds = DS[id];
    return OBS.loadTable(ds.file).then(function (t) {
      if (!t.computed) t.computed = {};
      ds.table = t;
      if (id === 'clusters' && !OBS.CLUSTERS.pleiades.dist) clusterDistances(t);
      return ds;
    });
  };
  OBS.varOf = function (ds, id) {
    for (var i = 0; i < ds.vars.length; i++) if (ds.vars[i].id === id) return ds.vars[i];
    return null;
  };
  /** All values of a variable for a dataset (cached). Numbers in a Float64Array, categories in an Array. */
  OBS.column = function (ds, id) {
    var t = ds.table, key = ds.id + ':' + id;
    if (t.computed[key]) return t.computed[key];
    var vr = OBS.varOf(ds, id), n = t.n, out = vr.cat ? new Array(n) : new Float64Array(n), c = t.cols;
    for (var i = 0; i < n; i++) { var x = vr.f(c, i); out[i] = vr.cat ? x : (x === null || x === undefined || x === '' ? NaN : +x); }
    t.computed[key] = out;
    return out;
  };
  OBS.forget = function (ds) { if (ds.table) ds.table.computed = {}; };  // after a calibration changes (your M–L slope)
  function clusterDistances(t) {
    var c = t.cols;
    Object.keys(OBS.CLUSTERS).forEach(function (k) {
      var p = [];
      for (var i = 0; i < t.n; i++) if (c.cl[i] === k && OBS.isMember(c, i) && c.phot_g_mean_mag[i] < 17) p.push(c.parallax[i] + A.GAIA_ZP);
      p.sort(function (a, b) { return a - b; });
      OBS.CLUSTERS[k].plx = p[p.length >> 1];
      OBS.CLUSTERS[k].dist = 1000 / p[p.length >> 1];
      OBS.CLUSTERS[k].n = p.length;
    });
  }
  OBS.LEVELS = [
    { id: 'sky', label: 'The sky', scale: 'all directions', note: 'Positions, brightness and colour: what anyone can measure.' },
    { id: 'solar', label: 'Solar system', scale: '1–100 AU', note: 'The Sun and its family.' },
    { id: 'stars', label: 'Stars', scale: '1–100 pc', note: 'The solar neighbourhood: single stars, pairs, planets, corpses.' },
    { id: 'clusters', label: 'Star clusters', scale: '50 pc – 3 kpc', note: 'Stars born together.' },
    { id: 'galaxy', label: 'The Milky Way', scale: '1–30 kpc', note: 'Our galaxy and its halo.' },
    { id: 'galaxies', label: 'Galaxies', scale: '50 kpc – 300 Mpc', note: 'The Magellanic Clouds and beyond.' },
    { id: 'universe', label: 'The Universe', scale: '1–14 Gpc', note: 'Supernovae across the Universe, over the oldest light there is: the cosmic microwave background.' }
  ];
})(typeof window !== 'undefined' ? window : globalThis);
