/* Observatorium — missions. Each one retraces a step someone took with real data:
 *   brief    who, when, what they could measure; the variables, the relation, what it lets you deduce
 *   predict  a choice you lock in before looking
 *   observe  the lens and the sky are set up; a task the page checks for you
 *   explain  your numbers, what they mean, the real history, and where it is in the course
 * A mission can unlock concepts: new derived variables appear in every dataset's axis menus. */
(function (G) {
  'use strict';
  var OBS = G.OBS, A = OBS.astro;
  function fmt(v, s) { return '<span class="num">' + OBS.fmt(v, s || 3) + '</span>'; }
  function fix(v, d) { return '<span class="num">' + OBS.fix(v, d) + '</span>'; }
  function median(a) { a = a.filter(isFinite).sort(function (p, q) { return p - q; }); return a.length ? a[a.length >> 1] : NaN; }
  function pick(app, col) { var c = OBS.column(app.chart.ds, col); return app.chart.selected().map(function (i) { return c[i]; }); }
  function share(arr, fn) { return arr.length ? arr.filter(fn).length / arr.length : 0; }
  function pct(x) { return '<span class="num">' + Math.round(100 * x) + '%</span>'; }

  OBS.CHAPTERS = [
    { n: 1, title: 'Reading starlight', lecture: 'Lecture 1 · light, blackbodies, basic observations of stars, the HR diagram, clusters', hw: 'HW1, HW2 Q4' },
    { n: 2, title: 'Lives of stars', lecture: 'Lecture 2 · stellar evolution, remnants, pulsars, variable stars', hw: 'HW2, HW3 Q1, HW5' },
    { n: 3, title: 'Planets and lenses', lecture: 'Lecture 3 · exoplanets and how they are found, gravitational microlensing', hw: 'HW3 Q2, HW4 Q1–2' },
    { n: 4, title: 'The Sun’s family and the Galaxy', lecture: 'Lecture 4 · the solar system, the Milky Way, dark matter, MOND', hw: 'HW4 Q1, Q3 (MOND)' },
    { n: 5, title: 'Island universes', lecture: 'Lecture 5 · galaxies, their types, clusters and scaling relations', hw: 'HW5' },
    { n: 6, title: 'The expanding Universe', lecture: 'Lecture 7 · observational cosmology: redshift and supernovae, the CMB and its acoustic peaks, nucleosynthesis, quasars', hw: '' },
    { n: 7, title: 'The observer’s toolkit', lecture: 'Lectures 8–9 · telescopes, coordinates and time, the atmosphere, detectors, magnitude systems, data reduction', hw: '' },
    { n: 8, title: 'Every kind of light', lecture: 'Lectures 10–11 · radio, infrared, ultraviolet, X-ray and gamma-ray astronomy, gravitational waves', hw: '' },
    { n: 9, title: 'Reading galaxies', lecture: 'Lectures 12–13 · galaxy properties, redshifts, star formation and active nuclei, numerical tools', hw: '' }
  ];

  var M = OBS.MISSIONS = [
    // =================================================================================== chapter 1
    {
      id: 'sky', ch: 1, title: 'A sky without order', era: '≈ 130 BC – 1856', who: 'Hipparchus; Norman Pogson',
      course: 'Lecture 1 §1.1, §1.4', hw: 'HW1 Q1–2',
      brief: 'For two thousand years, every star catalogue held the same three numbers: where the star is, how bright it looks, and its colour. Hipparchus sorted stars into six magnitudes; in 1856 Pogson made the scale exact: five magnitudes are a factor of 100 in light, and a bigger magnitude is fainter.',
      vars: [['α, δ', 'Position on the sky', 'degrees', 'Measured against other stars'], ['V', 'Apparent brightness', 'mag', 'How much light reaches us'], ['B − V', 'Colour', 'mag', 'Blue magnitude minus visual; bigger is redder']],
      relation: 'f = L / 4πd²: the light we receive mixes how much a star gives out (L) with how far it is (d).',
      deduce: 'From the sky alone: nothing about what a star is. That is the problem this chapter solves.',
      question: 'Sirius and Rigel look almost equally bright (V = −1.46 and 0.13). Do they give out similar amounts of light?',
      choices: [['yes', 'Yes: similar brightness means similar power'], ['unknown', 'Not necessarily: we do not know how far away each one is'], ['rigel', 'No: Rigel is bluer, so it must be weaker']],
      answer: 'unknown',
      setup: function (app) {
        // the whole sky, and a chart that shows the very brightest stars (Sirius is V = −1.46)
        return app.open('sky', { x: 'ci', y: 'mag' }).then(function () { app.chart.setView({ x0: -0.4, x1: 2.2, y0: -1.8, y1: 6.7 }); app.sky.goTo(88, 0, 200, { mode: 'allsky', frame: 'equ' }); });
      },
      task: 'Each dot in the lens is one naked-eye star. Drag a box around the brightest stars, the ones above V = 1.0 at the top of the chart. Watch where they are on the sky.',
      progress: function (app) {
        var m = pick(app, 'mag'), n = m.length, ok = n >= 8 && n <= 30 && m.every(function (x) { return x < 1.6; });
        return { done: ok, text: n ? n + ' stars selected' + (m.some(function (x) { return x >= 1.6; }) ? ', some fainter than V = 1.6: draw a smaller box' : n < 8 ? ': box a few more of the brightest' : '') : 'Nothing selected yet' };
      },
      result: function (app) {
        var c = app.chart.ds.table.cols, sel = app.chart.selected();
        var blue = sel.filter(function (i) { return c.ci[i] < 0.3; }).length, red = sel.filter(function (i) { return c.ci[i] > 1.0; }).length;
        var names = sel.map(function (i) { return c.name[i]; }).filter(Boolean).slice(0, 12).join(', ');
        return 'You selected ' + fmt(sel.length, 2) + ' stars: ' + names + '. ' + fmt(blue, 2) + ' are blue (B − V < 0.3) and ' + fmt(red, 2) + ' are red (B − V > 1). The brightest stars come in every colour and lie all over the sky.';
      },
      explain: 'A star looks bright if it is powerful or if it is close, and from here the two are the same thing. Rigel is in fact about 5,000 times more powerful than Sirius but some 100 times farther away. Without distances, a chart of colour against brightness is a cloud with no physics in it. The next missions add distance, then temperature, size and mass, one relation at a time.',
      real: 'Hipparchus’ catalogue (around 130 BC) reached us through Ptolemy’s Almagest. Pogson defined the modern scale in 1856. The Hipparcos satellite (1989–93) measured the stars in this chart to a thousandth of a magnitude.'
    },
    {
      id: 'parallax', ch: 1, title: 'A year-long wobble', era: '1838', who: 'Friedrich Bessel', course: 'Lecture 1 §1.4 (distance)', hw: 'HW1 Q2 (the 2p trap)', unlocks: ['parallax'],
      brief: 'As the Earth goes round the Sun, a nearby star seems to trace a small ellipse against much more distant stars. Bessel chose 61 Cygni because it moves fast across the sky, a hint that it is close, and measured it with a heliometer for a year.',
      vars: [['p', 'Parallax', 'arcsec', 'Half the full width of the yearly ellipse'], ['d', 'Distance', 'parsec', '1 pc is where p = 1″ (3.26 light-years)']],
      relation: 'd [pc] = 1 / p [arcsec]. The ellipse is the Earth’s orbit seen from the star.',
      deduce: 'A distance for every star with a measurable parallax. Then flux plus distance gives luminosity.',
      question: 'Six months apart, 61 Cygni sits 0.57 arcseconds from where it was. What is its parallax?',
      choices: [['057', '0.57″'], ['029', '0.29″, half of it'], ['114', '1.14″, twice it']],
      answer: '029',
      setup: function (app, st) {
        app.sky.goTo(316.725, 38.749, 12, { mode: 'dome', frame: 'equ' });
        app.markers([{ ra: 316.725, dec: 38.749, label: '61 Cygni' }]);
        var I = app.instrument('Telescope on 61 Cygni A (offsets in milliarcseconds; proper motion removed)');
        st.scope = OBS.instruments.parallax(I.chart, I.tools, { name: '61 Cyg A', ra: 316.725, dec: 38.749, plx: 286.0 });
        st.input = OBS.ui.number(I.tools, { label: 'Your parallax', unit: 'arcsec', placeholder: 'e.g. 0.5' });
      },
      task: 'Press “Play a year” to watch the loop. The dotted line is the loop’s long axis. Mark the star at one end of it, press “Jump ahead 6 months” and mark it again: the readout says what fraction of the long axis your pair spans. Then type the parallax you deduce.',
      progress: function (app, st) {
        var s = st.scope.state, sep = s.sep, p = st.input.get(), frac = sep / s.longAxis, wide = frac > 0.95;
        var good = isFinite(p) && Math.abs(p - 0.286) < 0.015;
        return { done: wide && good, text: !isFinite(sep) ? 'Mark two positions' :
          !wide ? 'These marks are ' + OBS.fmt(sep / 1000, 3) + '″ apart, ' + Math.round(100 * frac) + '% of the long axis: across the ellipse it is narrower. Start at one end of the dotted long axis and jump 6 months.' :
          !isFinite(p) ? 'Full width found: ' + OBS.fmt(sep / 1000, 3) + '″. Now type the parallax.' : good ? 'Right' : 'Not quite: how is the full width of the ellipse related to p?' };
      },
      measured: function (app, st) { var p = st.input.get(); return Math.abs(p - 0.57) < 0.05 ? '057' : Math.abs(p - 1.14) < 0.1 ? '114' : '029'; },
      result: function (app, st) {
        var p = st.input.get(), d = 1 / p;
        return 'Your parallax: ' + fix(p, 3) + '″, so d = 1/p = ' + fix(d, 2) + ' pc = ' + fix(d * 3.2616, 1) + ' light-years. The light you see from 61 Cygni left it about ' + fmt(d * 3.2616, 2) + ' years ago.';
      },
      explain: 'Over half a year the Earth moves from one side of its orbit to the other, 2 AU, so the star crosses the full width of its ellipse: twice the parallax. That is the trap in HW1 Q2: a 0.2″ shift in half a year means p = 0.1″. The ellipse is flatter for stars near the ecliptic (its short axis is p sin β); 61 Cygni is at ecliptic latitude 52°.',
      real: 'Bessel announced p = 0.314″ in 1838; the modern value is 0.286″. Henderson (α Centauri) and Struve (Vega) measured parallaxes at almost the same time. Hipparcos measured 118,000 parallaxes; Gaia has measured 1.5 billion, and its parallax for 61 Cyg A is 286.0 ± 0.1 mas.',
      unlockText: 'Parallax and distance now appear in the axis menus.'
    },
    {
      id: 'hrd', ch: 1, title: 'Correct for distance', era: '1905–1913', who: 'Ejnar Hertzsprung; Henry Norris Russell', course: 'Lecture 1 §1.5', hw: 'HW1 Q2c', needs: ['parallax'], unlocks: ['invsq'],
      brief: 'Once parallaxes existed, Hertzsprung and Russell did the obvious thing: put every star at the same distance, on paper. The apparent brightness becomes the absolute magnitude, how bright the star would look from 10 parsecs.',
      vars: [['m', 'Apparent magnitude', 'mag', 'Measured'], ['d', 'Distance', 'pc', 'From parallax (unlocked)'], ['M', 'Absolute magnitude', 'mag', 'm − 5 log₁₀(d/10 pc)']],
      relation: 'M = m − 5 log₁₀(d / 10 pc), and L ∝ 10^(−0.4 M). A difference of 5 magnitudes is a factor 100 in power.',
      deduce: 'The luminosity of each star, and whether stars come in all combinations of colour and power.',
      question: 'These are the 8,900 stars you can see with the naked eye. In colour against apparent brightness they form a smear. If you replace apparent brightness by absolute magnitude, what happens?',
      choices: [['smear', 'Still a smear: stars come in every mix of colour and power'], ['bands', 'They collapse into a few narrow bands'], ['line', 'They fall on one straight line']],
      answer: 'bands',
      axes: true,     // the task asks the student to change the vertical axis
      setup: function (app, st) {
        st.stage = 0;
        return app.open('sky', { x: 'ci', y: 'mag' }).then(function () { app.sky.setMode('allsky'); app.chart.setView({ x0: -0.4, x1: 2.2 }); });
      },
      task: 'Stage 1: the naked-eye stars, colour against apparent brightness, a smear. Switch the vertical axis (Up) to “Absolute magnitude M_V”: every star is moved to 10 pc. Watch what happens.',
      progress: function (app, st) {
        if (st.stage === 0) {
          if (app.chart.yid !== 'MV') return { done: false, text: 'Switch Up to Absolute magnitude M_V' };
          if (!st.t0) st.t0 = Date.now();
          if (Date.now() - st.t0 < 2500) return { done: false, text: 'Bands: the main sequence running from bright blue to faint red, and the giants above it on the right.' };
          st.stage = 1;
          return app.open('nearby', { x: 'bp_rp', y: 'MG' }).then(function () {
            app.note('Stage 2: Gaia’s 36,000 stars within 100 pc, already in absolute magnitude. A volume like this holds the faint stars the naked eye misses.');
            return { done: false, text: 'Stage 2: box the faint sequence below the main one', task: 'Stage 2: Gaia’s stars within 100 pc, in absolute magnitude. Box the thin sequence of faint, white stars in the lower left, about 10 magnitudes below the Sun.' };
          });
        }
        var wd = pick(app, 'wd'), mg = pick(app, 'MG'), f = share(wd, function (x) { return x === 'white dwarf'; });
        return { done: wd.length >= 80 && f > 0.7, text: wd.length ? wd.length + ' selected, ' + Math.round(100 * f) + '% of them in the faint white sequence' + (f <= 0.7 ? ': tighten the box' : wd.length < 80 ? ': box more of it' : '') : 'Box the faint sequence below the main one' };
      },
      measured: function () { return 'bands'; },
      result: function (app) {
        var mg = pick(app, 'MG'), c = pick(app, 'bp_rp'), m = median(mg);
        var all = OBS.column(app.chart.ds, 'MG'), col = OBS.column(app.chart.ds, 'bp_rp'), wd = OBS.column(app.chart.ds, 'wd'), n = all.length, g = 0, w = 0;
        for (var i = 0; i < n; i++) { if (wd[i] === 'white dwarf') w++; else if (all[i] < 3.5 && col[i] > 0.9) g++; }
        return 'Your stars have median M_G = ' + fix(m, 1) + ' and colour ' + fix(median(c), 2) + ': as white as the Sun or whiter, yet ' + fmt(Math.pow(10, 0.4 * (m - 4.67)), 2) + ' times fainter. In the whole sample (which keeps every giant and a quarter of the white dwarfs) there are ' + fmt(g, 3) + ' red giants and ' + fmt(n - g - w, 3) + ' main-sequence stars.';
      },
      explain: 'Stars are not random. Most lie on the main sequence, a band where hotter means more powerful; these are stars burning hydrogen in their cores, and their position is set by their mass. Above it to the right are giants, cool yet bright, so they must be huge. Below to the left are white dwarfs, hot yet faint, so they must be tiny. You can only see this once distance is taken out.',
      real: 'Hertzsprung noticed giants and dwarfs in 1905–07; Russell drew the diagram for stars with parallaxes in 1913. Gaia’s 2018 version, made from 4 million stars with exactly these cuts, is the sharpest ever made (Gaia Collaboration, Babusiaux et al. 2018).',
      unlockText: 'Absolute magnitude and luminosity now appear in the axis menus.'
    },
    {
      id: 'thermometer', ch: 1, title: 'Colour is a thermometer', era: '1893–1900', who: 'Wilhelm Wien; Max Planck', course: 'Lecture 1 §1.2–1.4', hw: 'HW1 Q2–3', unlocks: ['thermo'],
      brief: 'A hot dense body glows with a spectrum that depends only on its temperature: Planck’s blackbody curve. Stars are close to blackbodies, so the ratio of their light in two filters tells their surface temperature, whatever their size or distance.',
      vars: [['F_BP / F_RP', 'Colour: blue light over red light', 'BP − RP in mag', 'Gaia’s two filters'], ['T', 'Surface temperature', 'K', 'The blackbody with the same colour']],
      relation: 'B_λ(T) = 2hc²/λ⁵ · 1/(e^{hc/λkT} − 1). In a ratio of two bands, R and d cancel; only T is left. Peak: λ_max = 2.898 mm·K / T.',
      deduce: 'A temperature for every star from photometry alone.',
      question: 'A star is much redder than the Sun (BP − RP = 2.5 against the Sun’s 0.82). Its surface is…',
      choices: [['hotter', 'hotter than the Sun'], ['cooler', 'cooler than the Sun'], ['cant', 'impossible to tell from colour alone']],
      answer: 'cooler',
      setup: function (app, st) {
        return OBS.open('nearby').then(function (ds) {
          var c = ds.table.cols;
          function find(target) {
            var best = -1, bd = 1;
            for (var i = 0; i < ds.table.n; i++) if (isFinite(c.teff_gspphot[i]) && !c.wd[i] && c.id[i]) { var d = Math.abs(c.bp_rp[i] - target); if (d < bd) { bd = d; best = i; } }
            return { label: (target < 0.3 ? 'Hot star' : target < 1.2 ? 'Sun-like star' : 'Red dwarf'), bprp: c.bp_rp[best], teff: c.teff_gspphot[best], id: c.id[best] };
          }
          st.stars = [find(0.15), find(0.82), find(2.5)];
          var I = app.instrument('Spectrograph: a blackbody seen through Gaia’s blue (BP) and red (RP) filters');
          st.matched = {};
          st.spec = OBS.instruments.spectrograph(I.chart, I.tools, st.stars, function (s) {
            var star = st.stars[s.star];
            if (Math.abs(A.bbBpRp(s.T) - star.bprp) < 0.03) st.matched[s.star] = s.T;
          });
        });
      },
      task: 'For each of the three real Gaia stars, slide the blackbody’s temperature until its colour matches the star’s measured colour. Watch the peak move.',
      progress: function (app, st) {
        var n = Object.keys(st.matched).length;
        return { done: n === 3, text: n + ' of 3 stars matched' };
      },
      result: function (app, st) {
        var rows = st.stars.map(function (s, k) {
          var T = st.matched[k];
          return '<tr><td>' + s.label + '</td><td>' + fix(s.bprp, 2) + '</td><td>' + fmt(T, 3) + ' K</td><td>' + fmt(s.teff, 3) + ' K</td></tr>';
        }).join('');
        return '<table class="mini"><thead><tr><th>Star</th><th>BP − RP</th><th>Your blackbody</th><th>Gaia’s temperature</th></tr></thead><tbody>' + rows + '</tbody></table>';
      },
      explain: 'Redder means cooler: Wien’s law moves the peak to longer wavelengths as T falls. Your blackbody temperatures agree with Gaia’s for the hot and Sun-like stars. For the red dwarf the blackbody reads too cold, because molecules in its cool atmosphere (titanium oxide, water) absorb blue light and make it redder than a blackbody of its temperature. From here on the page uses a colour–temperature relation fitted to Gaia’s own temperatures for 2,816 nearby stars. Tick “Show the limits”: on the long-wavelength side the curve follows Rayleigh–Jeans, B ∝ T/λ⁴, and a colour there is the same for every temperature. That is HW1 Q2b: you can only read T where the spectrum turns over.',
      real: 'Wien’s displacement law (1893) and Planck’s formula (1900). Gaia’s BP and RP photometers measure the colours of 1.5 billion stars; its GSP-Phot pipeline fits a full spectrum, not just a colour.',
      unlockText: 'Temperature now appears in the axis menus.'
    },
    {
      id: 'radius', ch: 1, title: 'How big is a white dwarf?', era: '1915–1926', who: 'Walter Adams; Arthur Eddington', course: 'Lecture 1 §1.5; Lecture 2 §2.4', hw: 'HW2 Q2', needs: ['parallax', 'invsq', 'thermo'], unlocks: ['stefan'],
      brief: 'Luminosity and temperature together give size: a star radiates σT⁴ from every square metre, so its power tells its area. In 1915 Adams found that faint Sirius B is hotter than Sirius A. Eddington worked out what that means for its size.',
      vars: [['L', 'Luminosity', 'L☉', 'Unlocked'], ['T', 'Temperature', 'K', 'Unlocked'], ['R', 'Radius', 'R☉', '√L · (5772 K / T)²']],
      relation: 'L = 4πR²σT⁴. On a chart of log L against log T, lines of constant radius are straight.',
      deduce: 'The size of every star, and where giants and dwarfs sit.',
      question: 'White dwarfs are as hot as the Sun or hotter, yet thousands of times fainter. Compared with the Sun they must be…',
      choices: [['same', 'about the same size'], ['small', 'about 100 times smaller, the size of the Earth'], ['large', 'larger but much less dense']],
      answer: 'small',
      setup: function (app) {
        return app.open('nearby', { x: 'T', y: 'L' }).then(function () {
          var C = OBS.colours();
          [0.01, 0.1, 1, 10, 100].forEach(function (R) {
            app.chart.add({ type: 'curve', id: 'R' + R, f: function (T) { return A.lumFromRT(R, T); }, colour: C['ink-3'], dash: [4, 4], width: 1, label: 'R = ' + R + ' R☉', labelAt: 3200 });
          });
          app.chart.add({ type: 'marker', id: 'sun', x: 5772, y: 1, label: 'Sun', drag: false });
        });
      },
      task: 'Dashed lines mark constant radius. Box the white dwarfs (hot but faint, bottom middle) and read off their size.',
      progress: function (app) {
        var wd = pick(app, 'wd'), f = share(wd, function (x) { return x === 'white dwarf'; });
        return { done: wd.length >= 50 && f > 0.7, text: wd.length ? wd.length + ' selected, ' + Math.round(100 * f) + '% white dwarfs' : 'Box the white dwarfs' };
      },
      result: function (app) {
        var R = median(pick(app, 'R')), all = OBS.column(app.chart.ds, 'R'), mx = 0;
        for (var i = 0; i < all.length; i++) if (isFinite(all[i]) && all[i] > mx) mx = all[i];
        return 'Median radius of your white dwarfs: ' + fmt(R, 2) + ' R☉ = ' + fmt(R / 0.009168, 2) + ' Earth radii. The largest star in the sample has R ≈ ' + fmt(mx, 2) + ' R☉: the radii span a factor of ' + fmt(mx / R, 2) + '.';
      },
      explain: 'A white dwarf packs about 0.6 solar masses into a ball the size of the Earth: a teaspoon would weigh tonnes. Ordinary pressure cannot hold that up; electron degeneracy pressure does, and only up to the Chandrasekhar mass, about 1.4 M☉ (HW2 Q2). Note the trap in HW2: heavier white dwarfs are smaller.',
      real: 'Adams measured Sirius B’s spectrum in 1915; Eddington called the implied density “absurd” in 1924; Fowler explained it with quantum degeneracy in 1926, and Chandrasekhar found the mass limit in 1930.',
      unlockText: 'Radius now appears in the axis menus.'
    },
    {
      id: 'binary', ch: 1, title: 'Weighing a star', era: '1889 →', who: 'Edward Pickering, Antonia Maury, Hermann Vogel', course: 'Lecture 1 §1.4 (mass)', hw: 'HW2 Q4', unlocks: ['kepler'],
      brief: 'The only direct way to weigh a star is to watch something orbit it. In a spectroscopic binary the two stars’ spectral lines shift back and forth as they swing round each other. If the pair also eclipses, we see the orbit edge-on.',
      vars: [['P', 'Orbital period', 'days', 'Time for the curves to repeat'], ['K₁, K₂', 'Velocity amplitudes', 'km/s', 'Half the swing of each Doppler curve'], ['i', 'Inclination', '°', '≈ 90° because it eclipses']],
      relation: 'M₁ + M₂ = P (K₁ + K₂)³ (1 − e²)^{3/2} / (2πG sin³i), and M₁/M₂ = K₂/K₁.',
      deduce: 'Both masses, and the size of the orbit.',
      question: 'Two stars orbit each other. Which one swings back and forth faster?',
      choices: [['heavy', 'The heavier one'], ['light', 'The lighter one'], ['same', 'Both the same']],
      answer: 'light',
      setup: function (app, st) {
        return OBS.open('binaries').then(function (ds) {
          var c = ds.table.cols, i = c.id.indexOf('3684484600212307712'), b = {};
          ['period', 'e', 'w', 'K1', 'K2', 'gamma', 'ra', 'dec', 'phot_g_mean_mag', 'parallax'].forEach(function (k) { b[k] = c[k][i]; });
          st.b = b;
          app.sky.goTo(b.ra, b.dec, 8, { mode: 'dome', frame: 'equ' });
          app.markers([{ ra: b.ra, dec: b.dec, label: 'Gaia DR3 3684484600212307712' }]);
          var I = app.instrument('Velocity curves of Gaia DR3 3684484600212307712 (G = ' + b.phot_g_mean_mag.toFixed(1) + ', ' + (1000 / b.parallax).toFixed(0) + ' pc), from Gaia’s orbit fit');
          st.bin = OBS.instruments.binary(I.chart, I.tools, b);
        });
      },
      task: 'Drag the four dashed rules to the top and bottom of each star’s curve. The readout turns them into amplitudes.',
      progress: function (app, st) {
        var s = st.bin.state, e1 = Math.abs(s.K1 / st.b.K1 - 1), e2 = Math.abs(s.K2 / st.b.K2 - 1);
        return { done: e1 < 0.04 && e2 < 0.04, text: 'K₁ ' + (e1 < 0.04 ? 'read well' : 'not yet') + ' · K₂ ' + (e2 < 0.04 ? 'read well' : 'not yet') };
      },
      measured: function (app, st) { return st.b.K2 > st.b.K1 ? 'light' : 'heavy'; },
      result: function (app, st) {
        var s = st.bin.state, m = A.binaryMasses(st.b.period, s.K1, s.K2, st.b.e, 90), g = A.binaryMasses(st.b.period, st.b.K1, st.b.K2, st.b.e, 90);
        return 'From your readings: M₁ = ' + fmt(m.M1, 3) + ' M☉ and M₂ = ' + fmt(m.M2, 3) + ' M☉, ' + fmt(m.aRsun, 3) + ' solar radii apart. Gaia’s own fit gives ' + fmt(g.M1, 3) + ' and ' + fmt(g.M2, 3) + ' M☉. The fainter star swings faster (K₂ = ' + fix(st.b.K2, 0) + ' km/s against ' + fix(st.b.K1, 0) + ') and is the lighter one.';
      },
      explain: 'Both stars circle their common centre of mass, so M₁v₁ = M₂v₂: the light star has the big orbit and the high speed (the trap in HW2 Q4). The period and the summed speeds give the orbit size, and Kepler’s third law turns size and period into total mass. Without the eclipse you only get M sin³i.',
      real: 'Mizar was the first spectroscopic binary (Pickering and Maury, 1889). Gaia DR3 (2022) published fitted orbits for 443,205 binaries; this pair is one of 407 whose two velocity curves were both measured and which also eclipse.',
      unlockText: 'Masses from orbits now appear in the axis menus.'
    },
    {
      id: 'masslum', ch: 1, title: 'Mass sets the light', era: '1924', who: 'Arthur Eddington', course: 'Lecture 1 §1.5; Lecture 2 §2.1', hw: 'HW2 Q3', needs: ['kepler'], unlocks: ['masslum'],
      brief: 'With masses from binaries, Eddington found that the luminosity of a main-sequence star depends almost only on its mass. These are 190 stars weighed to 3% from eclipsing binaries.',
      vars: [['M', 'Mass', 'M☉', 'From the binary orbit'], ['L', 'Luminosity', 'L☉', '4πR²σT⁴']],
      relation: 'L ∝ M^a: a straight line on a log–log plot, with slope a. Fuel ∝ M, burn rate ∝ L, so lifetime t ∝ M / L ∝ M^(1−a).',
      deduce: 'A mass for any main-sequence star from its luminosity, and how long it will live.',
      question: 'A main-sequence star ten times the Sun’s mass gives out how much more light?',
      choices: [['10', 'About 10 times'], ['100', 'About 100 times'], ['5000', 'Several thousand times']],
      answer: '5000',
      setup: function (app) {
        return app.open('torres', { x: 'mass', y: 'L', colourBy: 'stage' }).then(function () {
          app.chart.add({ type: 'line', id: 'fit', x1: 0.8, y1: 0.8, x2: 8, y2: 8, colour: OBS.colours().accent, label: 'your line' });
        });
      },
      task: 'Drag the two round handles until the line runs through the main-sequence stars. The slope is the power a in L ∝ M^a.',
      progress: function (app) {
        var L = app.chart.lineFit('fit'), rms = app.chart.lineRms('fit');
        return { done: L.slope > 3.1 && L.slope < 4.6 && rms < 0.4, text: 'Slope a = ' + L.slope.toFixed(2) + ', scatter ' + rms.toFixed(2) + ' dex' };
      },
      result: function (app) {
        var a = app.chart.lineFit('fit').slope;
        A.ML_SLOPE = a; OBS.store.set('mlslope', a);
        Object.keys(OBS.DATASETS).forEach(function (k) { OBS.forget(OBS.DATASETS[k]); });
        return 'Your slope: a = ' + fix(a, 2) + '. A 10 M☉ star is ' + fmt(Math.pow(10, a), 2) + ' times as luminous as the Sun and lives about ' + fmt(A.lifetime(10, a) / 1e6, 2) + ' million years; a 0.5 M☉ star lives ' + fmt(A.lifetime(0.5, a) / 1e9, 2) + ' billion years. The page now uses your slope for masses and lifetimes.';
      },
      explain: 'Heavier stars squeeze their cores harder, run hotter and burn far faster, so they shine much more and die young. The lecture quotes L ∝ M^3.5; these data give about 3.9, steeper below 2 M☉ and shallower above. That one exponent is behind HW2 Q3 and the cluster ages in the next chapter.',
      real: 'Eddington’s mass–luminosity relation (1924), from a handful of binaries. The data here are the 95 systems of Torres, Andersen and Giménez (2010), whose masses and radii are good to 3%.',
      unlockText: 'Main-sequence masses and lifetimes now appear in the axis menus, computed with your slope.'
    },
    {
      id: 'pleiades', ch: 1, title: 'A cluster hiding in the field', era: '1930; 1997–2018', who: 'Robert Trumpler; the Hipparcos and Gaia teams', course: 'Lecture 1 §1.6', hw: 'HW1 Q4', needs: ['parallax', 'invsq'],
      brief: 'Stars in a cluster were born together, so they share a distance and a motion. That makes a cluster a calibrated sample: its main sequence can be compared with the nearby one to give its distance, without a parallax.',
      vars: [['μ_α*, μ_δ', 'Proper motion', 'mas/yr', 'Drift across the sky'], ['G, BP − RP', 'Brightness and colour', 'mag', 'Measured'], ['m − M', 'Distance modulus', 'mag', 'The shift between the cluster’s sequence and the nearby one']],
      relation: 'm − M = 5 log₁₀(d / 10 pc) + A (A = dimming by dust).',
      deduce: 'The cluster’s distance, from photometry alone.',
      question: 'In a chart of the stars’ motion across the sky, the Pleiades will show up as…',
      choices: [['clump', 'a tight clump, apart from the other stars'], ['spread', 'spread out like everything else'], ['line', 'a line']],
      answer: 'clump',
      setup: function (app, st) {
        st.stage = 0;
        app.sky.goTo(56.75, 24.12, 9, { mode: 'dome', frame: 'equ' });
        return app.open('clusters', { sub: 'pleiades', x: 'pmra', y: 'pmdec', colourBy: 'none' }).then(function () {
          app.chart.setView({ x0: -40, x1: 60, y0: -90, y1: 30 });
        });
      },
      task: 'Stage 1: these are all 8,547 Gaia stars within 3° of the Pleiades with parallax > 2 mas. Find the clump in proper motion and box it. Its stars light up on the sky.',
      progress: function (app, st) {
        var ch = app.chart;
        if (st.stage === 0) {
          var sel = ch.selected(), c = ch.ds.table.cols, mem = sel.filter(function (i) { return OBS.isMember(c, i); }).length;
          var ok = sel.length >= 300 && mem / sel.length > 0.8;
          if (ok) {
            st.stage = 1; st.members = sel.slice(); st.cols = c;     // the cluster table, kept for the result
            var keep = new Uint8Array(ch.ds.table.n); sel.forEach(function (i) { keep[i] = 1; });
            return OBS.open('nearby').then(function (nb) {
              var xs = OBS.column(nb, 'bp_rp'), ys = OBS.column(nb, 'MG');
              ch.setFilter(function (cc, i) { return keep[i] === 1; });
              ch.setAxes('bp_rp', 'G'); ch.clearSel();
              app.note('Stage 2: your ' + sel.length + ' stars, colour against apparent brightness. The green points are the 100 pc stars at their absolute magnitudes.');
              st.slide = OBS.instruments.slideRef(ch, app.tools, { xs: xs, ys: ys, value: 2, min: 0, max: 10, label: 'Shift the nearby stars down by (distance modulus)', onchange: function (mu, out) { st.mu = mu; out.innerHTML = 'Shift ' + fix(mu, 2) + ' mag → distance ' + fmt(A.distFromMu(mu - 2.74 * 0.045), 3) + ' pc (after 0.12 mag of dust)'; } });
            }).then(function () { return { done: false, text: 'Stage 2: slide the nearby main sequence down onto the cluster', task: 'Stage 2: your cluster stars, colour against apparent brightness, with the 100 pc stars in green at their absolute magnitudes. Slide the green main sequence down until it lies on the cluster’s: the shift is the cluster’s distance modulus.' }; });
          }
          return { done: false, text: sel.length ? sel.length + ' selected, ' + Math.round(100 * mem / Math.max(1, sel.length)) + '% moving with the cluster' : 'Box the clump' };
        }
        var mu = st.mu, good = mu > 5.55 && mu < 6.05;
        return { done: good, text: 'Stage 2: shift ' + mu.toFixed(2) + ' mag' + (good ? ': matched' : '') };
      },
      result: function (app, st) {
        var d = A.distFromMu(st.mu - 2.74 * 0.045), c = st.cols || app.chart.ds.table.cols;
        var p = median(st.members.map(function (i) { return c.parallax[i] + A.GAIA_ZP; })), dp = 1000 / p, agree = Math.abs(d - dp) / dp < 0.1;
        return 'Main-sequence fitting: ' + fmt(d, 3) + ' pc. Gaia’s parallaxes of the same ' + st.members.length + ' stars: median ' + fix(p, 2) + ' mas, so ' + fmt(dp, 3) + ' pc. ' + (agree ? 'Two independent methods agree.' : 'The two differ by ' + pct(Math.abs(d - dp) / dp) + ': look again at where the shifted sequence sits.');
      },
      explain: 'Members share a motion, so they stand out in proper motion even though on the sky they are mixed with thousands of field stars. Members share a distance, so in colour against apparent magnitude they draw a clean main sequence, shifted from the nearby one by the distance modulus. This is how distances to farther clusters, too far for parallax, were found: clusters are the second rung of the distance ladder (HW1 Q4 uses the same idea).',
      real: 'In 1997 Hipparcos put the Pleiades at 118 pc, while main-sequence fitting said 133. The “Pleiades problem” lasted 17 years, until radio interferometry (Melis et al. 2014, 136 pc) and then Gaia (135 pc) sided with the stellar physics.'
    },
    // =================================================================================== chapter 2
    {
      id: 'turnoff', ch: 2, title: 'Clocks in clusters', era: '1952 →', who: 'Allan Sandage; Martin Schwarzschild', course: 'Lecture 2 §2.1–2.3', hw: 'HW2 Q3, HW3 Q1b', needs: ['parallax', 'invsq', 'masslum'],
      brief: 'All stars in a cluster were born at once, but they age at different speeds. Massive stars use up their hydrogen first and leave the main sequence. The point where a cluster’s main sequence ends, the turn-off, falls as the cluster ages.',
      vars: [['M_G', 'Absolute magnitude (cluster distance)', 'mag', 'From each cluster’s Gaia parallax, corrected for dust'], ['L_TO', 'Turn-off luminosity', 'L☉', 'Where the main sequence ends'], ['t', 'Age', 'yr', 'Lifetime of a turn-off star']],
      relation: 'M_TO = L_TO^(1/a) (your mass–luminosity slope), and t ≈ 10 Gyr · (M_TO / M☉)^(1 − a).',
      deduce: 'The age of each cluster, and so a lower limit on the age of the Galaxy.',
      question: 'In an old cluster the main sequence…',
      choices: [['top', 'stops lower down: the bright blue stars are gone'], ['full', 'still reaches up to the bright blue stars'], ['shift', 'moves to the right as a whole']],
      answer: 'top',
      setup: function (app, st) {
        return app.open('clusters', { sub: '*', x: 'bp_rp0', y: 'MGc', colourBy: 'cluster', filter: function (c, i) { return c.cl[i] !== 'hyades' && OBS.isMember(c, i); } }).then(function () {
          var C = OBS.colours(), pal = [C.c1, C.c2, C.c3, C.c4];
          app.chart.setView({ x0: -0.4, x1: 2.2, y0: -3, y1: 8 });
          st.lines = [['pleiades', 'Pleiades', -0.5], ['praesepe', 'Praesepe', 1.5], ['m67', 'M67', 4.5], ['ngc6397', 'NGC 6397', 6.0]];
          var cats = app.chart.categories().cats;
          st.lines.forEach(function (l) { app.chart.add({ type: 'hline', id: l[0], value: l[2], label: l[1] + ' turn-off', colour: pal[cats.indexOf(OBS.CLUSTERS[l[0]].label)] || C.accent, dash: [6, 4] }); });
        });
      },
      task: 'Four clusters, each star placed at its cluster’s distance. Drag each coloured rule to the brightness where that cluster’s main sequence ends and turns off to the right.',
      progress: function (app, st) {
        var R = { pleiades: [-3, 0.3], praesepe: [-0.6, 1.4], m67: [2.0, 3.5], ngc6397: [2.9, 4.3] }, ok = 0, msg = [];
        st.lines.forEach(function (l) { var v = app.chart.get(l[0]).value, r = R[l[0]]; if (v >= r[0] && v <= r[1]) ok++; else msg.push(l[1]); });
        return { done: ok === 4, text: ok + ' of 4 turn-offs placed' + (msg.length ? '. Still off: ' + msg.join(', ') : '') };
      },
      result: function (app, st) {
        var lit = { pleiades: '112 Myr', praesepe: '650 Myr', m67: '4 Gyr', ngc6397: '13 Gyr' }, a = A.ML_SLOPE;
        var rows = st.lines.map(function (l) {
          var MG = app.chart.get(l[0]).value, col = { pleiades: -0.1, praesepe: 0.2, m67: 0.65, ngc6397: 0.58 }[l[0]];
          var L = A.lumFromMG(MG, col), M = A.massFromL(L, a), t = A.lifetime(M, a);
          return '<tr><td>' + l[1] + '</td><td>' + fix(MG, 1) + '</td><td>' + fmt(L, 2) + '</td><td>' + fmt(M, 2) + '</td><td>' + (t > 1e9 ? fmt(t / 1e9, 2) + ' Gyr' : fmt(t / 1e6, 2) + ' Myr') + '</td><td>' + lit[l[0]] + '</td></tr>';
        }).join('');
        return '<table class="mini"><thead><tr><th>Cluster</th><th>M_G at turn-off</th><th>L [L☉]</th><th>M [M☉]</th><th>Your age</th><th>Models</th></tr></thead><tbody>' + rows + '</tbody></table>';
      },
      explain: 'The turn-off falls with age because the stars that remain on the main sequence are lighter and longer-lived. Your simple clock works well for the young clusters. It reads too young for the old ones, by about 2× for M67 and 3× for NGC 6397, for two reasons: turn-off stars have already brightened as helium built up in their cores, and the metal-poor stars of a globular cluster are brighter for their mass. Stellar-evolution models (isochrones) correct both. Look also at M67’s blue stragglers above its turn-off (stars that gained mass from a companion), its red giants, and NGC 6397’s horizontal branch.',
      real: 'Sandage and Schwarzschild (1952) explained the turn-off with stellar evolution; Sandage dated globular clusters through the 1950s–70s. The oldest globulars, about 13 billion years old, set a minimum age for the Universe that cosmology must respect.'
    },
    {
      id: 'leavitt', ch: 2, title: 'Leavitt’s law', era: '1908–1912', who: 'Henrietta Swan Leavitt', course: 'Lecture 2 (variable stars); Lecture 4 (the distance ladder)', hw: 'HW5 (distance ladder)', unlocks: ['leavitt'],
      brief: 'At Harvard, Leavitt measured thousands of variable stars on photographic plates of the Magellanic Clouds. The Clouds are far away compared with their own size, so all their stars are at nearly the same distance: any pattern in apparent brightness is a pattern in true brightness.',
      vars: [['P', 'Pulsation period', 'days', 'From the light curve'], ['⟨m⟩', 'Mean apparent magnitude', 'mag', 'Averaged over a cycle']],
      relation: '⟨m⟩ = a log₁₀ P + b for stars at one distance. Pulsating stars are a standard candle if a and b are universal.',
      deduce: 'A star’s luminosity from its period, and so its distance from its flux.',
      question: 'Leavitt plotted period against brightness for Cepheids in the Small Magellanic Cloud. She found that…',
      choices: [['slow', 'the brighter ones pulse more slowly'], ['fast', 'the brighter ones pulse faster'], ['none', 'there is no relation']],
      answer: 'slow',
      setup: function (app, st) {
        st.stage = 0;
        app.sky.goTo(80.9, -69.75, 14, { mode: 'dome', frame: 'equ' });
        app.markers([{ ra: 80.9, dec: -69.75, label: 'Large Magellanic Cloud' }, { ra: 13.19, dec: -72.83, label: 'Small Magellanic Cloud' }]);
        return app.open('cep_lc', { sub: 'OGLE-LMC-CEP-1109' }).then(function () {
          st.fold = OBS.instruments.fold(app.chart, app.tools, { P0: 5, min: 1, max: 40, fine: 0.25, start: false, shift: true });
        });
      },
      task: 'Stage 1: five years of OGLE measurements of one Cepheid in the LMC. Tick “Fold”, then find the trial period at which the scattered points line up into one clean pulse: slide it, or click a dip in the sharpness curve below the sliders and fine-tune. (Dips at half or double the true period are aliases: look at the folded curve.)',
      progress: function (app, st) {
        var ch = app.chart;
        if (st.stage === 0) {
          var x = OBS.column(ch.ds, 't'), y = OBS.column(ch.ds, 'mag'), t = [], m = [];
          for (var i = 0; i < x.length; i++) if (ch.mask[i]) { t.push(x[i]); m.push(y[i]); }
          var s = OBS.instruments.foldScatter(t, m, st.fold.P), best = OBS.instruments.foldScatter(t, m, 11.8606);
          if (ch.fold && s < 1.6 * best) {
            st.stage = 1; st.P = st.fold.P;
            app.tools.innerHTML = ''; ch.fold = null;     // the fold controls belong to stage 1 only
            return app.open('cep_ogle', { sub: 'smc', x: 'P', y: 'I' }).then(function () {
              app.sky.goTo(13.19, -72.83, 8);
              ch = app.chart;
              ch.add({ type: 'line', id: 'pl', x1: 2, y1: 16, x2: 20, y2: 16, colour: OBS.colours().accent, label: 'your line' });
              app.note('Stage 2: 2,750 Cepheids in the Small Magellanic Cloud, mean I-band brightness against period (log scale).');
              return { done: false, text: 'Stage 2: fit a line through the Cepheids' };
            });
          }
          return { done: false, text: ch.fold ? 'Folding at ' + st.fold.P.toFixed(4) + ' d: scatter ' + (s / best).toFixed(1) + '× the sharpest' : 'Tick “Fold” first' };
        }
        var L = ch.lineFit('pl'), rms = ch.lineRms('pl');
        return { done: L && L.slope < -2.5 && L.slope > -3.6 && rms < 0.4, text: 'Stage 2: slope ' + (L ? L.slope.toFixed(2) : '–') + ' mag per decade of period, scatter ' + rms.toFixed(2) + ' mag' };
      },
      result: function (app, st) {
        var L = app.chart.lineFit('pl');
        st.slope = L.slope; OBS.store.set('plslope', L.slope);
        return 'Your period for OGLE-LMC-CEP-1109: ' + fix(st.P, 3) + ' days (OGLE: 11.861). Your law: I = ' + fix(L.slope, 2) + ' log₁₀ P + ' + fix(L.intercept, 2) + '. A Cepheid with a 30-day period is ' + fix(-L.slope * Math.log10(30), 1) + ' magnitudes, about ' + fmt(Math.pow(10, -0.4 * L.slope * Math.log10(30)), 2) + ' times, brighter than a 1-day one.';
      },
      explain: 'Cepheids are giants that swell and shrink because a layer of helium in their envelope traps heat when compressed and releases it when expanded. Bigger, more luminous Cepheids pulse more slowly, much as a big bell rings lower. Since all these stars are at one distance, the slope is real. What Leavitt could not get was the zero point: how bright a 10-day Cepheid really is. That needs Cepheids with known distances: the next mission.',
      real: 'Leavitt published 1,777 variables in 1908 and the period–luminosity relation for 25 SMC Cepheids in 1912. Hubble used her law in 1924 to show that the Andromeda nebula is a separate galaxy. OGLE has since found about 10,000 Cepheids in the Magellanic Clouds.',
      unlockText: 'You have the slope of the period–luminosity law. Calibrate it next.'
    },
    {
      id: 'yardstick', ch: 2, title: 'A yardstick for the cosmos', era: '1913; 2019', who: 'Ejnar Hertzsprung; the Araucaria project', course: 'Lecture 4 (distance ladder); Lecture 5', hw: 'HW5 (distance ladder)', needs: ['parallax', 'invsq', 'leavitt'],
      brief: 'Gaia measured parallaxes for hundreds of Milky Way Cepheids. Their absolute magnitudes calibrate Leavitt’s law; comparing them with the Large Magellanic Cloud’s Cepheids at the same period gives the LMC’s distance. To beat the dust, use the Wesenheit brightness W = G − 1.90 (BP − RP), which dust leaves unchanged.',
      vars: [['W', 'Dust-proof brightness', 'mag', 'G − 1.90 (BP − RP)'], ['M_W', 'Absolute W', 'mag', 'From Gaia parallax'], ['μ', 'Distance modulus', 'mag', 'W − M_W at the same period']],
      relation: 'μ = m − M = 5 log₁₀(d / 10 pc). Five magnitudes is a factor 10 in distance.',
      deduce: 'The distance to the LMC: the anchor for every extragalactic distance.',
      question: 'Two Cepheids pulse with the same period. One looks 5 magnitudes fainter. How much farther away is it?',
      choices: [['5', '5 times'], ['10', '10 times'], ['100', '100 times']],
      answer: '10',
      setup: function (app, st) {
        return OBS.open('cep_gaia').then(function (ds) {
          var reg = OBS.column(ds, 'region'), P = OBS.column(ds, 'P'), MW = OBS.column(ds, 'MW'), c = ds.table.cols, xs = [], ys = [];
          for (var i = 0; i < ds.table.n; i++) if (reg[i] === 'mw' && (c.parallax[i] + A.GAIA_ZP) / c.parallax_error[i] > 10 && isFinite(MW[i])) { xs.push(P[i]); ys.push(MW[i]); }
          st.nref = xs.length;
          app.sky.goTo(80.9, -69.75, 16, { mode: 'dome', frame: 'equ' });
          return app.open('cep_gaia', { sub: 'lmc', x: 'P', y: 'W' }).then(function () {
            app.chart.setView({ x0: 0.8, x1: 60, y0: 9, y1: 19 });
            app.note('Coloured: Gaia Cepheids in the LMC (apparent W). Grey: ' + xs.length + ' Milky Way Cepheids at their absolute W (parallax better than 10%).');
            st.slide = OBS.instruments.slideRef(app.chart, app.tools, { xs: xs, ys: ys, value: 15.5, min: 5, max: 25, under: false, size: 2.6, alpha: 0.85, label: 'Shift the Milky Way Cepheids by μ', onchange: function (mu, out) { st.mu = mu; out.innerHTML = 'μ = ' + fix(mu, 2) + ' → LMC at ' + fmt(A.distFromMu(mu) / 1000, 3) + ' kpc'; } });
          });
        });
      },
      task: 'The green points are Milky Way Cepheids, placed by their Gaia parallaxes; the gold ones are in the LMC. Slide the green ones down until they lie on the LMC’s: the shift is the LMC’s distance modulus.',
      progress: function (app, st) { var ok = st.mu > 18.25 && st.mu < 18.95; return { done: ok, text: 'μ = ' + st.mu.toFixed(2) + (ok ? ': on top' : '') }; },
      result: function (app, st) {
        var d = A.distFromMu(st.mu) / 1000;
        return 'Your distance to the Large Magellanic Cloud: ' + fmt(d, 3) + ' kpc, about ' + fmt(d * 3262, 2) + ' light-years. The best modern value, from eclipsing binaries, is 49.6 ± 0.6 kpc (Pietrzyński et al. 2019).';
      },
      explain: 'Matching the two sequences is the same move as main-sequence fitting for the Pleiades, one rung further out. Gaia’s Cepheids put the LMC a few percent beyond the eclipsing-binary distance: LMC Cepheids have fewer heavy elements, which changes their brightness at a given period by about 0.1–0.2 mag. Calibrating that metallicity term is part of today’s debate over the Hubble constant.',
      real: 'Hertzsprung made the first calibration in 1913 from 13 Galactic Cepheids’ motions, and got the SMC far too close. Baade found in 1952 that there are two kinds of Cepheids, which doubled the size of the Universe overnight. The SH0ES team (Riess et al.) uses Cepheids calibrated like this to measure H₀.'
    },
    {
      id: 'pulsars', ch: 2, title: 'Lighthouses', era: '1967; 1982', who: 'Jocelyn Bell Burnell and Antony Hewish; Don Backer', course: 'Lecture 2 §2.4 (remnants, P–Ṗ diagram)', hw: 'HW2 Q1',
      brief: 'In 1967 Jocelyn Bell found a radio source ticking every 1.337 seconds: a spinning neutron star sweeping a beam past the Earth. A pulsar slows down as its magnetic field radiates away its spin energy. Two numbers per pulsar, P and Ṗ, tell its field and its age.',
      vars: [['P', 'Spin period', 's', 'Time between pulses'], ['Ṗ', 'Spin-down rate', 's/s', 'Change of period per second'], ['B, τ', 'Field and age', 'G, yr', 'B ≈ 3.2×10¹⁹ √(PṖ),  τ = P / 2Ṗ']],
      relation: 'Magnetic-dipole braking: lines of constant B and constant age are straight on a log P – log Ṗ chart.',
      deduce: 'Which pulsars are young and strongly magnetised, which are old, and where the fastest ones come from.',
      question: 'Millisecond pulsars spin fastest of all. Are they the youngest?',
      choices: [['young', 'Yes: they have not slowed down yet'], ['old', 'No: they are old pulsars spun up again by a companion'], ['same', 'They are the same age as the rest']],
      answer: 'old',
      setup: function (app) {
        return app.open('pulsars', { x: 'P', y: 'Pdot', colourBy: 'bin' }).then(function () {
          var C = OBS.colours();
          [1e8, 1e10, 1e12, 1e14].forEach(function (B) { app.chart.add({ type: 'curve', id: 'B' + B, f: function (P) { return Math.pow(B / 3.2e19, 2) / P; }, colour: C['ink-3'], dash: [4, 4], width: 1, label: 'B = 10^' + Math.log10(B) + ' G', labelAt: B > 1e13 ? 3 : B > 1e11 ? 0.3 : 0.004 }); });
          [1e3, 1e6, 1e9].forEach(function (T) { app.chart.add({ type: 'curve', id: 'T' + T, f: function (P) { return P / (2 * T * A.C.yr); }, colour: C.c5, dash: [2, 3], width: 1, label: 'τ = 10^' + Math.log10(T) + ' yr', labelAt: T > 1e8 ? 0.002 : 1 }); });
          app.chart.add({ type: 'marker', id: 'crab', x: 0.0334, y: 4.21e-13, label: 'Crab pulsar', drag: false });
        });
      },
      task: 'Box the millisecond pulsars, the separate group at the bottom left. The readout below the chart counts how many are in binaries.',
      progress: function (app) { var P = pick(app, 'P'), m = median(P); return { done: P.length >= 200 && m < 0.02, text: P.length + ' selected, median period ' + OBS.fmt(m * 1000, 2) + ' ms' }; },
      result: function (app) {
        var b = pick(app, 'bin'), all = OBS.column(app.chart.ds, 'bin'), P = OBS.column(app.chart.ds, 'P'), ob = 0, on = 0, ages = pick(app, 'age');
        for (var i = 0; i < all.length; i++) if (P[i] > 0.1) { on++; if (all[i] === 'binary') ob++; }
        return pct(share(b, function (x) { return x === 'binary'; })) + ' of your millisecond pulsars are in binaries, against ' + pct(ob / on) + ' of ordinary pulsars. Their median characteristic age is ' + fmt(median(ages) / 1e9, 2) + ' billion years.';
      },
      explain: 'A newborn pulsar like the Crab starts at the top left: young, fast-ageing, with a field of 10¹² G. It drifts down to the right as it slows, until the beam switches off. Millisecond pulsars are old neutron stars that were spun up by gas falling onto them from a companion star in a binary: hence the binaries, the weak fields (10⁸ G) and the great ages.',
      real: 'Bell and Hewish found the first pulsar in 1967 (Hewish shared the 1974 Nobel Prize; Bell did not). Backer et al. found the first millisecond pulsar in 1982, and Alpar et al. proposed recycling the same year. The Hulse–Taylor binary pulsar (1974) was the first evidence for gravitational waves.'
    },
    // =================================================================================== chapter 3
    {
      id: 'transit', ch: 3, title: 'A shadow crosses a star', era: '2000; 2009', who: 'David Charbonneau, Greg Henry; the Kepler mission', course: 'Lecture 3 (exoplanet detection)', hw: 'HW4 Q1 (occultations)', unlocks: ['transit'],
      brief: 'If a planet’s orbit is edge-on to us, the planet blocks a little of its star once every orbit. Kepler stared at 150,000 stars for four years. Here are 89 days of HAT-P-7.',
      vars: [['P', 'Orbital period', 'days', 'Time between dips'], ['ΔF/F', 'Transit depth', '', 'Fraction of the starlight blocked'], ['R★', 'Star’s radius', 'R☉', 'From its luminosity and temperature']],
      relation: 'ΔF / F = (R_p / R★)². Kepler’s third law turns P into the orbit size.',
      deduce: 'The planet’s size, its orbit, and with a radial-velocity mass, its density.',
      question: 'A planet twice as wide as Jupiter crosses the same star. Compared with Jupiter’s, its transit dip is…',
      choices: [['2', 'twice as deep'], ['4', 'four times as deep'], ['same', 'just as deep, but longer']],
      answer: '4',
      setup: function (app, st) {
        app.sky.goTo(292.25, 47.97, 18, { mode: 'dome', frame: 'equ' });
        app.markers([{ ra: 292.2474, dec: 47.9695, label: 'HAT-P-7' }]);
        return app.open('kepler', { sub: 'HAT-P-7' }).then(function () {
          st.fold = OBS.instruments.fold(app.chart, app.tools, { P0: 1.3, min: 0.5, max: 10, fine: 0.03, start: false, shift: true });
          app.chart.add({ type: 'hline', id: 'depth', value: 0.9985, label: 'bottom of the dip', colour: OBS.colours().c2, dash: [5, 4] });
        });
      },
      task: 'Find the dips, tick “Fold” and tune the trial period until all the dips stack on top of each other. Then drag the rule to the bottom of the dip.',
      progress: function (app, st) {
        var P = st.fold.P, okP = app.chart.fold && app.chart.sub === 'HAT-P-7' && Math.abs(P / 2.204737 - 1) < 0.002, d = 1 - app.chart.get('depth').value, okD = Math.abs(d / 0.0067 - 1) < 0.15;
        return { done: okP && okD, text: (app.chart.fold ? 'Period ' + P.toFixed(4) + ' d' + (okP ? ' (dips stacked)' : '') : 'Tick “Fold”') + ' · depth ' + (d * 100).toFixed(2) + '%' + (okD ? '' : '') };
      },
      result: function (app, st) {
        var d = 1 - app.chart.get('depth').value, k = Math.sqrt(d), Rs = 2.0, Rp = k * Rs * A.C.Rsun / A.C.Rjup, a = Math.pow(1.47 * Math.pow(st.fold.P / 365.25, 2), 1 / 3);
        return 'Depth ' + fix(d * 100, 2) + '% → R_p / R★ = ' + fix(k, 3) + '. HAT-P-7 has R★ = 2.0 R☉, so the planet’s radius is ' + fix(Rp, 2) + ' times Jupiter’s. Its year is ' + fix(st.fold.P, 4) + ' days; with the star’s 1.47 M☉, Kepler’s third law puts it ' + fix(a, 3) + ' AU from the star, ' + fix(0.387 / a, 0) + ' times closer than Mercury is to the Sun.';
      },
      explain: 'Area blocked over area of the star: double the planet’s width and you block four times the light. HAT-P-7 b is a hot Jupiter, about one and a half times Jupiter’s radius and heated above 2,000 K. Look at the bottom of the folded dip: it is rounded, because the star’s edge is darker than its centre (limb darkening). For a harder case, switch the star to HAT-P-11: a Neptune-sized planet with a dip of 0.4%, and starspots that the planet crosses.',
      real: 'Charbonneau et al. and Henry et al. (1999–2000) saw the first transit, of HD 209458 b, with a 10 cm telescope. Kepler (2009–2018) found more than 2,600 planets this way; TESS continues across the whole sky.',
      unlockText: 'Transit depth now gives planet sizes.'
    },
    {
      id: 'census', ch: 3, title: 'Who finds which planets', era: '1995 →', who: 'Michel Mayor and Didier Queloz', course: 'Lecture 3 (detection methods, hot Jupiters, statistics)', hw: '',
      brief: 'In 1995 Mayor and Queloz found a planet half as massive as Jupiter orbiting 51 Pegasi every 4.2 days, far closer than anyone expected. Today about 6,500 planets are known. Every detection method sees some planets more easily than others.',
      vars: [['P, a', 'Orbital period and size', 'days, AU', 'Transits and Doppler wobble'], ['M_p', 'Planet mass', 'M_Jup', 'Doppler (M sin i) or timing'], ['method', 'How it was found', '', 'Transit, radial velocity, microlensing, imaging, …']],
      relation: 'A transit is seen only if the orbit is edge-on: probability ≈ R★ / a. A Doppler wobble scales as M_p / √a.',
      deduce: 'Which planets are common, and which only look common because they are easy to find.',
      question: 'Most known planets were found by transits. Does that mean most planets orbit close to their stars?',
      choices: [['yes', 'Yes: the census says so'], ['bias', 'No: transits are far easier to catch for close-in planets'], ['cant', 'It is impossible to say anything']],
      answer: 'bias',
      setup: function (app) { return app.open('exoplanets', { x: 'P', y: 'Mp', colourBy: 'method' }).then(function () { app.sky.setMode('allsky'); }); },
      task: 'Box the hot Jupiters: planets heavier than about 0.3 Jupiter masses with periods shorter than about 10 days (the clump at the top left).',
      progress: function (app) { var P = pick(app, 'P'), Mp = pick(app, 'Mp'); return { done: P.length >= 150 && median(P) < 6 && median(Mp) > 0.4, text: P.length + ' selected, median period ' + OBS.fmt(median(P), 2) + ' d, median mass ' + OBS.fmt(median(Mp), 2) + ' M_Jup' }; },
      result: function (app) {
        var m = pick(app, 'method'), f = share(m, function (x) { return x === 'Transit'; });
        return pct(f) + ' of your hot Jupiters were found by transits. A planet on a 3-day orbit round a Sun-like star transits for about one observer in ' + fmt(1 / (A.C.Rsun / (Math.pow(3 / 365.25, 2 / 3) * A.C.AU)), 2) + '; on Jupiter’s 12-year orbit, about one in ' + fmt(1 / (A.C.Rsun / (5.2 * A.C.AU)), 2) + '.';
      },
      explain: 'Each method leaves its own footprint in this chart. Transits favour big planets on short orbits; radial velocities favour heavy ones; microlensing finds planets a few AU out, thousands of parsecs away; imaging finds young giants far from their stars. Careful surveys correct for these biases and find that hot Jupiters are rare, around 1% of Sun-like stars, while smaller planets are common.',
      real: 'Mayor and Queloz shared the 2019 Nobel Prize for 51 Pegasi b. The data are from the NASA Exoplanet Archive’s table of confirmed planets.'
    },
    {
      id: 'microlens', ch: 3, title: 'A star brightens for a month', era: '1936; 1986; 1993', who: 'Albert Einstein; Bohdan Paczyński; the OGLE team', course: 'Lecture 3 (history of lensing); Lecture 4 (MACHOs)', hw: 'HW3 Q2, HW4 Q2',
      brief: 'Mass bends light. When a star passes almost exactly in front of a more distant one, it acts as a lens: the background star’s images are too close together to separate, but their total brightness rises and falls. Paczyński proposed in 1986 to watch millions of stars toward the Galactic bulge to catch this.',
      vars: [['t₀', 'Time of closest approach', 'days', 'Peak of the light curve'], ['t_E', 'Einstein time', 'days', 'Time to cross one Einstein radius: R_E / v'], ['u₀', 'Closest approach', 'Einstein radii', 'Sets the peak magnification'], ['f_s', 'Source fraction', '', 'Share of the baseline light that is lensed (the rest is blended neighbours)']],
      relation: 'A(u) = (u² + 2) / (u √(u² + 4)),  u(t) = √(u₀² + ((t − t₀)/t_E)²).',
      deduce: 'How long the lens took to pass, and so a combination of its mass, distance and speed.',
      question: 'A foreground star drifts in front of a background star. The background star…',
      choices: [['dim', 'dims, because the foreground star blocks it'], ['bright', 'brightens and fades again, symmetrically in time'], ['colour', 'changes colour']],
      answer: 'bright',
      setup: function (app, st) {
        app.sky.goTo(266.4, -29.0, 30, { mode: 'dome', frame: 'gal' });
        app.markers([{ ra: 269.8946, dec: -29.0638, label: 'OGLE-2019-BLG-0148' }]);
        return Promise.all([app.open('microlensing', { sub: '2019-BLG-0148' }), OBS.loadJSON('manifest')]).then(function (r) {
          var p = r[1].microlensing.events['2019-BLG-0148'], ds = app.chart.ds, err = OBS.column(ds, 'mag') && ds.table.cols.err;
          st.ews = p;
          var T0 = p.tmax - 2450000;
          function f(t, q) { return A.paczynski(t, q.t0, q.tE, q.u0, q.fbl, q.I0); }
          var errf = function (i) { return Math.sqrt(err[i] * err[i] + 0.008 * 0.008); };
          app.chart.setView({ x0: T0 - 150, x1: T0 + 150 });
          st.model = OBS.instruments.model(app.chart, app.tools, {
            label: 'Paczyński curve', f: f, err: errf,
            params: [
              { id: 't0', label: 'Time of peak t₀', min: T0 - 60, max: T0 + 60, value: T0 - 35, fmt: function (x) { return (x).toFixed(1); }, unit: 'HJD − 2450000' },
              { id: 'tE', label: 'Einstein time t_E', min: 5, max: 200, value: 20, log: true, unit: 'd' },
              { id: 'u0', label: 'Closest approach u₀', min: 0.01, max: 1.5, value: 0.8, log: true, unit: 'R_E' },
              { id: 'fbl', label: 'Source fraction f_s', min: 0.2, max: 1, value: 1, fmt: function (x) { return x.toFixed(2); } },
              { id: 'I0', label: 'Baseline brightness', min: p.I0 - 0.3, max: p.I0 + 0.3, value: p.I0 + 0.1, fmt: function (x) { return x.toFixed(3); }, unit: 'mag' }
            ],
            describe: function (q) { return 'Peak magnification A(u₀) = ' + fix(A.magnification(q.u0), 2) + '.'; }
          });
          var q = { t0: T0, tE: p.tE, u0: p.u0, fbl: p.fbl, I0: p.I0 }, xs = OBS.column(ds, 't'), ys = OBS.column(ds, 'mag'), s = 0, n = 0;
          for (var i = 0; i < xs.length; i++) if (app.chart.mask[i]) { var dd = (ys[i] - f(xs[i], q)) / errf(i); s += dd * dd; n++; }
          st.target = 1.5 * s / (n - 5);
        });
      },
      task: 'Adjust the five sliders until the model runs through the points. Aim for χ²/N below the target shown.',
      progress: function (app, st) { var r = st.model.p._fit.red; return { done: r < st.target, text: 'χ²/N = ' + OBS.fmt(r, 3) + ' (target below ' + OBS.fmt(st.target, 2) + ')' }; },
      result: function (app, st) {
        var p = st.model.p, e = st.ews;
        return 'Your fit: t_E = ' + fix(p.tE, 1) + ' d, u₀ = ' + fix(p.u0, 3) + ', peak magnification ' + fix(A.magnification(p.u0), 2) + ', source fraction ' + fix(p.fbl, 2) + '. OGLE’s automatic fit: t_E = ' + fix(e.tE, 1) + ' d, u₀ = ' + fix(e.u0, 3) + '.';
      },
      explain: 'The curve is symmetric because the lens moves in a straight line, and it has the same shape in every colour because gravity bends all wavelengths equally: two signatures that tell lensing from a variable star. t_E is the only physical number in a single light curve, and it mixes the lens mass, the distances and the speed (the trap in HW3 Q2): a 0.3 M☉ star halfway to the bulge, moving at 150 km/s, gives about 25 days; this event’s 58 days means a heavier or slower lens. A planet round the lens adds a short spike when one of the images passes near it (HW3 Q2b).',
      real: 'Einstein worked out stellar microlensing in 1936 and thought it would never be seen. Paczyński’s 1986 proposal led to the OGLE, MACHO and EROS surveys; the first events were seen in 1993. Microlensing showed that compact objects (MACHOs) cannot be most of the Milky Way’s dark matter (Lecture 4).'
    },
    // =================================================================================== chapter 4
    {
      id: 'kirkwood', ch: 4, title: 'Gaps in the belt', era: '1866', who: 'Daniel Kirkwood', course: 'Lecture 4 (asteroids, Kuiper belt, TNOs)', hw: '',
      brief: 'By 1866 about 90 asteroids were known, and Kirkwood noticed that none orbited at certain distances from the Sun. Here are 50,000 orbits from JPL’s Small-Body Database.',
      vars: [['a', 'Orbit size (semi-major axis)', 'AU', 'From the orbit fit'], ['P', 'Orbital period', 'yr', 'P² = a³'], ['P / P_J', 'Period over Jupiter’s', '', '(a / 5.203)^{3/2}']],
      relation: 'Kepler’s third law, P² = a³. An asteroid whose period is a simple fraction of Jupiter’s gets a kick at the same place every few orbits.',
      deduce: 'Where Jupiter clears the belt, and where it gathers asteroids instead.',
      question: 'Jupiter orbits at 5.2 AU. Where would you expect few asteroids?',
      choices: [['even', 'Nowhere special: they are spread evenly'], ['res', 'Where an asteroid’s year is a simple fraction of Jupiter’s'], ['near', 'Only close to Jupiter']],
      answer: 'res',
      setup: function (app, st) {
        return app.open('smallbodies', { x: 'a', y: '#' }).then(function () {
          app.chart.bins = 220; app.chart.setView({ x0: 1.9, x1: 3.6 });
          var C = OBS.colours();
          st.rules = ['g1', 'g2', 'g3'];
          [2.2, 2.65, 3.1].forEach(function (x, k) { app.chart.add({ type: 'vline', id: st.rules[k], value: x, label: 'gap ' + (k + 1), colour: [C.c1, C.c2, C.c4][k], dash: [5, 4] }); });
        });
      },
      task: 'This histogram counts asteroids by orbit size. Drag the three rules into the three deepest gaps between 2.2 and 3.4 AU.',
      progress: function (app, st) {
        var G = [2.065, 2.502, 2.825, 2.958, 3.279], used = {}, ok = 0;
        st.rules.forEach(function (r) { var x = app.chart.get(r).value; G.forEach(function (g, k) { if (Math.abs(x - g) < 0.03 && !used[k]) { used[k] = 1; ok++; } }); });
        return { done: ok === 3, text: ok + ' of 3 rules in a gap' };
      },
      result: function (app, st) {
        var FR = [[1, 4], [1, 3], [2, 5], [3, 7], [1, 2], [3, 5], [2, 3], [1, 1]];
        return '<table class="mini"><thead><tr><th>Gap</th><th>a [AU]</th><th>P / P_Jupiter</th><th>Nearest fraction</th></tr></thead><tbody>' + st.rules.map(function (r, k) {
          var a = app.chart.get(r).value, q = Math.pow(a / 5.2026, 1.5), best = FR.reduce(function (b, f) { return Math.abs(f[0] / f[1] - q) < Math.abs(b[0] / b[1] - q) ? f : b; });
          return '<tr><td>' + (k + 1) + '</td><td>' + fix(a, 3) + '</td><td>' + fix(q, 3) + '</td><td>' + best[0] + '/' + best[1] + '</td></tr>';
        }).join('') + '</tbody></table>';
      },
      explain: 'Each gap sits where an asteroid would go round the Sun 3, 5/2, 7/3 or 2 times for each orbit of Jupiter. Repeated kicks at the same point of the orbit pump up the eccentricity until the asteroid crosses Mars’ orbit and is removed. Resonances can also protect: switch the vertical axis to eccentricity and widen the range to 6 AU to see the Hildas piled up at 3:2 (4.0 AU) and the Trojans sharing Jupiter’s orbit (1:1). Beyond Neptune, Pluto and the “plutinos” sit at Neptune’s 3:2 resonance, 39.4 AU.',
      real: 'Kirkwood (1866–67) explained the gaps with Jupiter’s resonances. The Kuiper belt’s first member after Pluto, 1992 QB₁, was found by Jewitt and Luu in 1992.'
    },
    {
      id: 'shapley', ch: 4, title: 'Where is the centre?', era: '1918', who: 'Harlow Shapley', course: 'Lecture 4 (the Milky Way, the Great Debate)', hw: '',
      brief: 'Globular clusters are bright and visible from far away. Shapley measured distances to them from their variable stars and found them crowded into one part of the sky, around Sagittarius. Here are all 157 in the Harris catalogue.',
      vars: [['l, b', 'Galactic position', '°', 'Along and across the Milky Way'], ['R☉', 'Distance', 'kpc', 'From RR Lyrae stars'], ['X, Y, Z', 'Position in space', 'kpc', 'Sun at the origin']],
      relation: 'The clusters orbit the centre of the Galaxy, so their swarm is centred on it.',
      deduce: 'Our distance from the centre of the Milky Way.',
      question: 'The globular clusters crowd into one part of the sky. What does that mean?',
      choices: [['local', 'Nothing in particular: a local clump'], ['off', 'The Sun is far from the centre of the Galaxy, and the clusters surround that centre'], ['dust', 'Dust hides the clusters in other directions']],
      answer: 'off',
      setup: function (app, st) {
        return app.open('globulars', { x: 'X', y: 'Y' }).then(function () {
          app.sky.goTo(266.4, -28.9, 200, { mode: 'allsky', frame: 'gal' });
          app.chart.equal = true; app.chart.setView({ x0: -25, x1: 35, y0: -25, y1: 25 });
          app.chart.add({ type: 'marker', id: 'sun', x: 0, y: 0, label: 'Sun', drag: false });
          app.chart.add({ type: 'point', id: 'centre', x: 0, y: 12, label: 'centre?', colour: OBS.colours().c2 });
        });
      },
      task: 'The sky shows where the clusters are; the lens shows them from above the Galactic plane. Drag the cross to the middle of the swarm.',
      progress: function (app) { var p = app.chart.get('centre'), d = Math.hypot(p.x - 7.0, p.y - 0.1); return { done: d < 1.5, text: 'Cross at X = ' + p.x.toFixed(1) + ', Y = ' + p.y.toFixed(1) + ' kpc' + (d < 1.5 ? ': in the middle' : '') }; },
      result: function (app) {
        var p = app.chart.get('centre'), R = Math.hypot(p.x, p.y);
        return 'You put the centre of the Galaxy ' + fmt(R, 2) + ' kpc from the Sun, about ' + fmt(R * 3262, 2) + ' light-years, toward Sagittarius.';
      },
      explain: 'The clusters form a round swarm around the Galaxy’s centre, and we are far out in the disk. The median of the clusters is a little short of the true distance because clusters on the far side, behind the dust of the bulge, are harder to find. The modern distance, from the orbit of the star S2 round the black hole Sgr A*, is 8.2 kpc (GRAVITY Collaboration 2019).',
      real: 'Shapley got about 15 kpc in 1918, too large because nobody yet knew that dust dims distant objects (Trumpler showed it in 1930). In the 1920 Great Debate he argued, wrongly, that the spiral nebulae were inside the Milky Way; Curtis argued they were other galaxies.'
    },
    {
      id: 'rotation', ch: 4, title: 'Too fast at the edge', era: '1970–1985', who: 'Vera Rubin and Kent Ford; Albert Bosma', course: 'Lecture 4 (dark matter, MOND)', hw: 'HW4 Q3 (MOND)',
      brief: 'Gas far out in a spiral galaxy orbits its centre. Its speed tells how much mass lies inside its orbit. NGC 3198’s hydrogen extends to 44 kpc, far beyond its stars. SPARC gives the measured speeds, and the speeds the stars and gas alone would produce.',
      vars: [['V(R)', 'Orbital speed', 'km/s', '21-cm and Hα Doppler shifts'], ['V_bar', 'Speed from visible matter', 'km/s', 'Stars (times a mass-to-light ratio) and gas'], ['Υ', 'Mass-to-light ratio', 'M☉ / L☉', 'Mass of stars per unit light']],
      relation: 'V² = G M(&lt;R) / R. Visible matter alone: V_bar² = V_gas² + Υ V_disk². A dark halo adds V_h²; MOND changes the force law instead.',
      deduce: 'How much unseen mass the galaxy needs, and where it is.',
      question: 'Gas far beyond the stars of NGC 3198 orbits…',
      choices: [['fall', 'more slowly the farther out it is'], ['flat', 'at the same speed, however far out'], ['rise', 'faster the farther out it is']],
      answer: 'flat',
      setup: function (app, st) {
        return app.open('rotation', { sub: 'NGC3198' }).then(function () {
          var ds = app.chart.ds, c = ds.table.cols, R = c.R, map = {}, C = OBS.colours();
          for (var i = 0; i < ds.table.n; i++) if (app.chart.mask[i]) map[R[i]] = i;
          function idx(r) { return map[r]; }
          function bar(r, p) { var i = idx(r); return i === undefined ? NaN : A.vBaryon(c.Vgas[i], c.Vdisk[i], c.Vbul[i], p.Y, 0.7); }
          app.chart.setView({ x0: 0, x1: 47, y0: 0, y1: 200 });
          st.mond = false;
          var box = document.createElement('label'); box.className = 'check'; box.innerHTML = '<input type="checkbox" id="rc-mond"> Use modified gravity (MOND, a₀ = 1.2×10⁻¹⁰ m/s²) instead of a dark halo';
          app.tools.appendChild(box);
          box.querySelector('input').addEventListener('change', function (e) {
            st.mond = e.target.checked;
            ['vh', 'rc'].forEach(function (k) { var inp = st.model.sliders[k].el.querySelector('input'); inp.disabled = st.mond; inp.title = st.mond ? 'MOND has no dark halo: untick the box to use these' : ''; });
            st.model.update();
          });
          function model(r, p) {
            var vb = bar(r, p);
            if (!isFinite(vb)) return NaN;
            if (st.mond) return A.vMond(vb, r, 1.2e-10);
            var vh = A.vHalo(r, p.vh, p.rc);
            return Math.sqrt(vb * Math.abs(vb) + vh * vh);
          }
          // curves are drawn through the measured radii only
          function curve(fn) { var rs = Object.keys(map).map(Number).sort(function (a, b) { return a - b; }); return function (x) { var j = 0; while (j < rs.length - 1 && rs[j + 1] < x) j++; if (x < rs[0] || x > rs[rs.length - 1]) return NaN; var r0 = rs[j], r1 = rs[Math.min(rs.length - 1, j + 1)], f0 = fn(r0), f1 = fn(r1); return r1 === r0 ? f0 : f0 + (f1 - f0) * (x - r0) / (r1 - r0); }; }
          st.model = OBS.instruments.model(app.chart, app.tools, {
            label: 'model', err: function (i) { return Math.max(2, c.eV[i]); },
            f: function (x, p) { return idx(x) !== undefined ? model(x, p) : curve(function (r) { return model(r, p); })(x); },
            extra: [{ id: 'vbar', label: 'stars + gas', colour: C.c4, f: function (x, p) { return curve(function (r) { return bar(r, p); })(x); } }],
            params: [
              { id: 'Y', label: 'Stellar mass-to-light ratio Υ', min: 0.1, max: 2.5, value: 0.5, fmt: function (x) { return x.toFixed(2); }, unit: 'M☉/L☉' },
              { id: 'vh', label: 'Dark halo: speed far out', min: 0, max: 250, value: 0, fmt: function (x) { return Math.round(x); }, unit: 'km/s' },
              { id: 'rc', label: 'Dark halo: core radius', min: 0.3, max: 30, value: 5, log: true, unit: 'kpc' }
            ],
            // a fit 'without dark matter' counts once the student has actually tried Υ (not the starting value)
            onchange: function (p, q) { if (p.vh < 3 && !st.mond && Math.abs(p.Y - 0.5) > 0.01) st.bestNoDM = Math.min(st.bestNoDM || Infinity, q.red); if (st.mond) st.bestMond = Math.min(st.bestMond || Infinity, q.red); }
          });
        });
      },
      task: 'First keep the halo speed at 0 and try to fit the data with stars and gas alone by changing Υ. Then add a dark halo until χ²/N is below 2. Try the MOND box too.',
      progress: function (app, st) {
        var q = st.model.p._fit, tried = isFinite(st.bestNoDM);
        return { done: tried && !st.mond && st.model.p.vh > 20 && q.red < 2, text: (tried ? 'Best without dark matter: χ²/N = ' + OBS.fmt(st.bestNoDM, 3) + '. ' : 'Try without a halo first: change Υ with the halo speed at 0. ') + 'Now: χ²/N = ' + OBS.fmt(q.red, 3) + (st.mond ? ' (MOND is on: untick it to finish with a dark halo)' : '') };
      },
      result: function (app, st) {
        var c = app.chart.ds.table.cols, sel = app.chart.shown(), last = sel[sel.length - 1], R = c.R[last], V = c.Vobs[last], p = st.model.p;
        var Mdyn = A.massEnclosed(V, R), vb = A.vBaryon(c.Vgas[last], c.Vdisk[last], c.Vbul[last], p.Y, 0.7), Mbar = A.massEnclosed(vb, R);
        return 'Within ' + fix(R, 1) + ' kpc the measured speed of ' + fix(V, 0) + ' km/s needs ' + fmt(Mdyn, 2) + ' M☉. Your stars and gas (Υ = ' + fix(p.Y, 2) + ') supply ' + fmt(Mbar, 2) + ' M☉: ' + fix(Mdyn / Mbar, 1) + ' times less. Best fit with no dark matter: χ²/N = ' + fmt(st.bestNoDM, 2) + '; with a halo: ' + fmt(p._fit.red, 2) + (isFinite(st.bestMond) ? '; with MOND: ' + fmt(st.bestMond, 2) : '') + '.';
      },
      explain: 'If the mass were where the light is, speeds would fall as 1/√R beyond the stars, like the planets round the Sun. They stay flat, so the enclosed mass keeps growing in proportion to R: there is a halo of matter that does not shine. MOND, with one universal constant and no halo (HW4 Q3), brings χ²/N from about 200 down to about 10 for this galaxy: far closer than Newton with the visible matter alone, though not as close as a halo with two free parameters (a slightly larger distance to NGC 3198 helps it). That MOND does this well across hundreds of galaxies, and badly for galaxy clusters, is why both ideas are still argued about.',
      real: 'Rubin and Ford measured flat rotation in Andromeda from 1970; Bosma (1978) extended curves with 21-cm radio data; van Albada et al. (1985) made NGC 3198 the textbook case for a dark halo. Milgrom proposed MOND in 1983.'
    },
    // =================================================================================== chapter 5
    {
      id: 'redblue', ch: 5, title: 'Red and blue galaxies', era: '1926; 2001; 2007', who: 'Edwin Hubble; Iskra Strateva; Galaxy Zoo volunteers', course: 'Lecture 5 (galaxy types)', hw: 'HW5 (the Hubble sequence)',
      brief: 'Hubble sorted galaxies by shape: ellipticals, spirals, irregulars. The Sloan Digital Sky Survey measured colours for a million; Galaxy Zoo asked the public to classify their shapes. Here are 14,000 nearby SDSS galaxies.',
      vars: [['u − r', 'Colour', 'mag', 'Ultraviolet minus red light'], ['M_r', 'Absolute magnitude', 'mag', 'Distance from redshift, d = cz / H₀'], ['shape', 'Galaxy Zoo classification', '', 'Vote fractions from volunteers']],
      relation: 'Blue light comes from hot, young, short-lived stars. A galaxy that stops forming stars turns red within a billion years.',
      deduce: 'Which galaxies are still forming stars, and whether that goes with shape.',
      question: 'Compared with spiral galaxies, elliptical galaxies are…',
      choices: [['redder', 'redder'], ['bluer', 'bluer'], ['same', 'the same colour']],
      answer: 'redder',
      setup: function (app) { return app.open('galaxies', { x: 'Mr', y: 'ur', colourBy: 'none' }).then(function () { app.chart.setView({ x0: -23.5, x1: -16, y0: 0.6, y1: 3.4 }); app.sky.goTo(180, 30, 200, { mode: 'allsky', frame: 'equ', photo: false }); }); },
      task: 'Box the red sequence: the tight band of red galaxies across the top of the chart.',
      progress: function (app) {
        var u = pick(app, 'ur'), red = u.filter(function (x) { return x > 2.2; }).length / Math.max(1, u.length), ok = u.length >= 800 && median(u) > 2.35 && red > 0.85;
        return { done: ok, text: u.length + ' selected, median u − r = ' + OBS.fix(median(u), 2) + (u.length && !ok && red <= 0.85 ? ': ' + Math.round(100 * (1 - red)) + '% of your box is blue cloud, tighten it' : '') };
      },
      result: function (app) {
        var m = pick(app, 'morph'), all = OBS.column(app.chart.ds, 'morph'), sel = app.chart.sel, rest = [];
        for (var i = 0; i < all.length; i++) if (!sel[i] && app.chart.mask[i]) rest.push(all[i]);
        return 'Of your red galaxies, ' + pct(share(m, function (x) { return x === 'elliptical'; })) + ' were called elliptical by Galaxy Zoo and ' + pct(share(m, function (x) { return x === 'spiral'; })) + ' spiral. Of all the others: ' + pct(share(rest, function (x) { return x === 'elliptical'; })) + ' elliptical and ' + pct(share(rest, function (x) { return x === 'spiral'; })) + ' spiral.';
      },
      explain: 'Galaxies come in two colour families: a blue cloud of star-forming disks and a red sequence that has stopped making stars. Shape and colour go together, but not perfectly: some red galaxies are spirals that ran out of gas, and a few blue ones are ellipticals that recently gained some. Switch “Colour by” to the Galaxy Zoo shape to see it point by point.',
      real: 'Hubble’s classification (1926), the tuning fork. Strateva et al. (2001) found the colour bimodality in early SDSS data. Galaxy Zoo (Lintott et al. 2008) gathered tens of millions of classifications from volunteers.'
    },
    {
      id: 'zwicky', ch: 5, title: 'Weighing a cluster of galaxies', era: '1933', who: 'Fritz Zwicky', course: 'Lecture 5 (galaxy clusters, dark matter)', hw: 'HW5',
      brief: 'Zwicky measured the speeds of eight galaxies in the Coma cluster and found them moving far too fast for the visible mass to hold them together. Here are 1,532 galaxies with SDSS redshifts in a 5° field around Coma.',
      vars: [['cz', 'Recession speed', 'km/s', 'Doppler shift of the spectrum'], ['σ', 'Spread of speeds', 'km/s', 'Standard deviation of cz in the cluster'], ['R', 'Cluster size', 'Mpc', 'From the angle and the distance cz / H₀']],
      relation: 'Virial theorem for a bound cluster: M ≈ 3σ²R / G.',
      deduce: 'The mass of the cluster, and how it compares with the mass in its stars.',
      question: 'Galaxies in the Coma cluster move relative to each other at about…',
      choices: [['100', '100 km/s'], ['1000', '1,000 km/s'], ['10000', '10,000 km/s']],
      answer: '1000',
      setup: function (app) {
        app.sky.goTo(194.95, 27.98, 9, { mode: 'dome', frame: 'equ', photo: false });
        app.markers([{ ra: 194.953, dec: 27.981, label: 'Coma cluster' }]);
        return app.open('coma', { x: 'cz', y: '#' }).then(function () { app.chart.bins = 70; app.chart.setView({ x0: 0, x1: 24000 }); });
      },
      task: 'Most galaxies in this direction belong to Coma, which shows as a spike in the speed histogram. Box the spike (drag across it).',
      progress: function (app) {
        var v = pick(app, 'cz'), n = v.length, m = v.reduce(function (a, x) { return a + x; }, 0) / Math.max(1, n);
        var s = Math.sqrt(v.reduce(function (a, x) { return a + (x - m) * (x - m); }, 0) / Math.max(1, n - 1));
        return { done: n >= 400 && m > 6400 && m < 7600 && s > 650 && s < 1250, text: n + ' galaxies, mean cz ' + Math.round(m) + ' km/s, spread σ = ' + Math.round(s) + ' km/s' };
      },
      result: function (app) {
        var c = app.chart.ds.table.cols, sel = app.chart.selected(), v = sel.map(function (i) { return c.z[i] * 299792.458; });
        var m = v.reduce(function (a, x) { return a + x; }, 0) / v.length, s = Math.sqrt(v.reduce(function (a, x) { return a + (x - m) * (x - m); }, 0) / (v.length - 1));
        var d = m / A.C.H0, r2 = 0, Lr = 0;
        sel.forEach(function (i) { var sep = A.sep(c.ra[i], c.dec[i], 194.953, 27.981) * Math.PI / 180 * d; r2 += sep * sep; Lr += Math.pow(10, -0.4 * (c.r[i] - A.distMod(d * 1e6) - 4.65)); });
        var R = Math.sqrt(r2 / sel.length), Mv = A.virialMass(s, R), Ms = 5 * Lr;
        return 'σ = ' + fmt(s, 3) + ' km/s; distance cz/H₀ = ' + fmt(d, 3) + ' Mpc; rms radius of your galaxies ' + fix(R, 2) + ' Mpc. Virial mass M ≈ 3σ²R/G = ' + fmt(Mv, 2) + ' M☉. The galaxies’ red light adds up to ' + fmt(Lr, 2) + ' L☉; at 5 M☉ per L☉ that is ' + fmt(Ms, 2) + ' M☉ in stars, ' + fmt(Mv / Ms, 2) + ' times less than the cluster needs.';
      },
      explain: 'Galaxies in Coma move at about 1,000 km/s relative to each other. With a size of a few megaparsecs, they would fly apart in a billion years unless tens of times more mass than the stars holds them in. Part of the missing mass is hot X-ray gas, several times the mass of the stars, but even with it most of the mass is dark. Lensing of background galaxies by clusters gives the same answer.',
      real: 'Zwicky (1933) called the missing mass “dunkle Materie”. With the Hubble constant of the time, 558 km/s/Mpc, he overestimated the mass-to-light ratio, but the conclusion has survived every test since.'
    },
    {
      id: 'tully', ch: 5, title: 'Spin tells power', era: '1977', who: 'Brent Tully and Richard Fisher', course: 'Lecture 5 (scaling relations: Tully–Fisher, Faber–Jackson)', hw: 'HW5 (MOND, distance ladder)',
      brief: 'The faster a disk galaxy spins, the more mass it has, and so the more light. The spin can be measured from the width of a single radio line without knowing the distance, which makes the relation a distance indicator.',
      vars: [['V_flat', 'Flat rotation speed', 'km/s', 'Width of the 21-cm line or the rotation curve'], ['L', 'Luminosity', '10⁹ L☉', 'At 3.6 μm, where the light traces the stars’ mass']],
      relation: 'L ∝ V^a: a straight line on log–log axes. MOND predicts M_bar = V⁴ / (G a₀), slope 4 exactly.',
      deduce: 'A galaxy’s luminosity from its spin, and its distance from its flux.',
      question: 'A disk galaxy spins twice as fast as another. It is about how much brighter?',
      choices: [['2', '2 times'], ['4', '4 times'], ['16', 'about 16 times']],
      answer: '16',
      setup: function (app) {
        return app.open('sparc', { x: 'Vflat', y: 'L36', filter: function (c, i) { return c.Vflat[i] > 0 && c.Q[i] < 3; } }).then(function () {
          app.chart.add({ type: 'line', id: 'tf', x1: 60, y1: 0.5, x2: 250, y2: 3, colour: OBS.colours().accent, label: 'your line' });
        });
      },
      task: 'Drag the line through the galaxies. Its slope is the power a in L ∝ V^a.',
      progress: function (app) {
        // the window comes from fits to these galaxies: least squares 4.4, bisector 4.6, orthogonal 4.8
        var L = app.chart.lineFit('tf'), r = app.chart.lineRms('tf'), ok = L.slope > 3.6 && L.slope < 5.4 && r < 0.40;
        var hint = ok ? '' : r >= 0.40 ? ': move the line into the middle of the cloud' : L.slope <= 3.6 ? ': too shallow, tilt it up' : ': too steep, tilt it down';
        return { done: ok, text: 'Slope a = ' + L.slope.toFixed(2) + ', scatter ' + r.toFixed(2) + ' dex' + hint };
      },
      result: function (app) {
        var L = app.chart.lineFit('tf');
        return 'Your relation: L ∝ V^' + fix(L.slope, 2) + '. A galaxy spinning at 200 km/s should give out about ' + fmt(Math.pow(10, L.at(Math.log10(200))), 2) + '×10⁹ L☉ at 3.6 μm. Measure its flux and you have its distance.';
      },
      explain: 'Twice the speed means about 2⁴ = 16 times the light. The relation is tight in the infrared, where the light follows the stellar mass, and tighter still when the gas mass is added (the baryonic Tully–Fisher relation), with a slope close to 4. For elliptical galaxies the same idea with the spread of star speeds σ is the Faber–Jackson relation, L ∝ σ⁴.',
      real: 'Tully and Fisher published the relation in 1977 and used it to measure distances to the Virgo cluster. Lelli, McGaugh and Schombert’s SPARC sample (2016) measures it with Spitzer 3.6 μm photometry.'
    },
    // =================================================================================== chapter 6 (lecture 7)
    {
      id: 'accel', ch: 6, title: 'Fainter than expected', era: '1998', who: 'Saul Perlmutter, Brian Schmidt and Adam Riess', course: 'Lecture 7 (cosmological redshift, the Hubble diagram, luminosity distance)', unlocks: ['cosmo'],
      brief: 'Type Ia supernovae all reach nearly the same peak brightness once corrected for how fast they fade: standard candles visible across the Universe. In 1998 two teams measured them out to z ≈ 0.8, expecting to see how strongly gravity slows the expansion. Here are 1,466 of them from the Pantheon+ compilation.',
      vars: [['z', 'Redshift', '', '1 + z = how much the Universe has grown since the light left'], ['μ', 'Distance modulus', 'mag', 'Peak magnitude minus the calibrated absolute magnitude'], ['Ωm, ΩΛ', 'Matter and dark energy', 'fractions of the critical density', 'What you fit'], ['H₀', 'Hubble constant', 'km/s/Mpc', 'Sets the overall scale']],
      relation: 'μ = 5 log₁₀(D_L / 10 pc), with D_L(z) = (1+z) (c/H₀) ∫₀ᶻ dz′/E(z′) and E(z) = √(Ωm(1+z)³ + Ωk(1+z)² + ΩΛ). Different universes bend the Hubble diagram differently at high z.',
      deduce: 'What the Universe is made of, whether its expansion is slowing down or speeding up, and how old it is.',
      question: 'Compared with a universe of matter only, whose expansion gravity slows down, distant supernovae turned out to look…',
      choices: [['brighter', 'brighter: closer than expected'], ['same', 'just as expected'], ['fainter', 'fainter: farther away than expected']],
      answer: 'fainter',
      setup: function (app, st) {
        return app.open('supernovae', { x: 'z', y: 'dmu' }).then(function () {
          var ch = app.chart, ds = ch.ds, c = ds.table.cols, C = OBS.colours();
          ch.setView({ x0: 0.01, x1: 2.4, y0: -0.75, y1: 0.75 });
          app.sky.goTo(0, 0, 200, { mode: 'allsky', frame: 'equ', photo: false });
          // binned averages, to guide the eye
          var edges = [0.01, 0.02, 0.035, 0.06, 0.1, 0.16, 0.25, 0.4, 0.6, 0.9, 1.4, 2.4];
          for (var b = 0; b < edges.length - 1; b++) {
            var sw = 0, sy = 0, sz = 0;
            for (var i = 0; i < ds.table.n; i++) if (c.z[i] >= edges[b] && c.z[i] < edges[b + 1]) { var w = 1 / (c.emu[i] * c.emu[i]); sw += w; sy += w * (c.mu[i] - A.distModEmpty(c.z[i], 73)); sz += w * c.z[i]; }
            if (sw) ch.add({ type: 'marker', id: 'bin' + b, x: sz / sw, y: sy / sw, colour: C.c2, drag: false });
          }
          var chi = OBS.snChi2(c); st.chiMin = Infinity;
          for (var om = 0.05; om <= 1.0; om += 0.05) for (var ol = 0; ol <= 1.4; ol += 0.05) st.chiMin = Math.min(st.chiMin, chi(om, ol).chi2);
          st.chiMatter = chi(1, 0).chi2;
          st.model = OBS.instruments.model(ch, app.tools, {
            label: 'your universe', err: function (i) { return c.emu[i]; },
            f: function (z, p) { return A.distModZ(z, p.H0, p.Om, p.OL) - A.distModEmpty(z, 73); },
            extra: [{ id: 'matter', label: 'matter only (Ωm = 1)', colour: C.c4, f: function (z, p) { return A.distModZ(z, p.H0, 1, 0) - A.distModEmpty(z, 73); } }],
            params: [
              { id: 'Om', label: 'Ωm: matter', min: 0, max: 1.5, value: 1, fmt: function (x) { return x.toFixed(2); } },
              { id: 'OL', label: 'ΩΛ: dark energy', min: -0.5, max: 1.5, value: 0, fmt: function (x) { return x.toFixed(2); } },
              { id: 'H0', label: 'H₀', min: 60, max: 80, value: 73, step: 0.1, fmt: function (x) { return x.toFixed(1); }, unit: 'km/s/Mpc' }
            ],
            describe: function (p) {
              var q0 = A.q0(p.Om, p.OL), ok = p.Om * 1.0 + p.OL < 2.2;
              return 'Ωm + ΩΛ = ' + fix(p.Om + p.OL, 2) + (Math.abs(p.Om + p.OL - 1) < 0.03 ? ' (flat)' : p.Om + p.OL > 1 ? ' (closed)' : ' (open)') + '; q₀ = ' + fix(q0, 2) + ': the expansion is ' + (q0 < 0 ? '<b>speeding up</b>' : '<b>slowing down</b>') + (ok ? '; age ' + fix(A.age(0, p.H0, p.Om, p.OL), 1) + ' Gyr.' : '.');
            }
          });
        });
      },
      task: 'Each dot is one supernova, shown as how much fainter it looks than in an empty, coasting universe; the orange circles are averages. The model starts as a universe of matter only. Change Ωm and ΩΛ (and H₀) until your curve runs through the averages.',
      progress: function (app, st) {
        var p = st.model.p, q = p._fit, d = q.chi2 - st.chiMin;
        return { done: d < 8 && A.q0(p.Om, p.OL) < -0.15 && p.Om > 0.1 && p.Om < 0.65, text: 'χ² is ' + Math.round(d) + ' above the best possible fit; the expansion is ' + (A.q0(p.Om, p.OL) < 0 ? 'speeding up' : 'slowing down') + ' in your universe' };
      },
      result: function (app, st) {
        var p = st.model.p, d = p._fit.chi2 - st.chiMin;
        return 'Your universe: Ωm = ' + fix(p.Om, 2) + ', ΩΛ = ' + fix(p.OL, 2) + ', H₀ = ' + fix(p.H0, 1) + ' km/s/Mpc; q₀ = ' + fix(A.q0(p.Om, p.OL), 2) + ', age ' + fix(A.age(0, p.H0, p.Om, p.OL), 1) + ' Gyr. Its χ² is within ' + fix(d, 0) + ' of the best fit; a universe of matter only is worse by ' + fmt(st.chiMatter - st.chiMin, 3) + '. At z = 0.5 your supernovae are ' + fix(A.distModZ(0.5, p.H0, p.Om, p.OL) - A.distModZ(0.5, p.H0, 1, 0), 2) + ' mag fainter than in a matter-only universe.';
      },
      explain: 'Distant supernovae are about 0.2–0.3 magnitudes, some 25%, fainter than in a universe that is only slowing down: they are farther away, so the expansion has been speeding up for the last 5 billion years or so. Something with negative pressure, called dark energy, makes up about 70% of the Universe’s energy; Einstein’s cosmological constant Λ is the simplest form. Combined with the CMB (mission 6.3), which says space is flat, the supernovae give Ωm ≈ 0.3 and ΩΛ ≈ 0.7. From now on, a redshift gives a distance and an age: lookback time is unlocked for the supernovae, the fast radio bursts and the SDSS galaxies’ emission lines.',
      real: 'The Supernova Cosmology Project (Perlmutter) and the High-z Supernova Search Team (Schmidt, Riess) announced the acceleration in 1998; they shared the 2011 Nobel Prize. Pantheon+ (2022) combines 1,550 supernovae: Ωm = 0.334 ± 0.018 for a flat universe. Its Cepheid calibration gives H₀ = 73, the “Hubble tension” with the CMB’s 67.4.'
    },
    {
      id: 'cmb', ch: 6, title: 'The afterglow', era: '1964–1990', who: 'Arno Penzias and Robert Wilson; John Mather and the COBE team', course: 'Lecture 7 (the cosmic microwave background)',
      brief: 'Penzias and Wilson found an unexplained hiss in their horn antenna in 1964, the same from every direction, with the temperature of a body at about 3 K. In 1990 COBE’s FIRAS instrument measured its spectrum across the peak, at 43 frequencies. Here are those measurements.',
      vars: [['ν', 'Frequency', 'GHz', 'FIRAS scanned 60–600 GHz'], ['I_ν', 'Brightness', 'MJy/sr', 'Power per area, frequency and solid angle'], ['T', 'Temperature', 'K', 'What you fit']],
      relation: 'I_ν = B_ν(T) = (2hν³/c²) / (e^{hν/kT} − 1): a blackbody. In an expanding universe, T falls as 1/(1+z) and the shape is kept.',
      deduce: 'The temperature of the Universe today, how many photons fill it, and when it became transparent.',
      question: 'If the young Universe was hot, dense and in thermal equilibrium, today’s microwave sky should have the spectrum of…',
      choices: [['bb', 'a blackbody, cooled by the expansion'], ['pl', 'a power law, like radio galaxies'], ['lines', 'bright emission lines of hydrogen']],
      answer: 'bb',
      setup: function (app, st) {
        return app.open('cmbspec', { x: 'f', y: 'I' }).then(function () {
          app.chart.setView({ x0: 0, x1: 700, y0: 0, y1: 900 });      // room for the 3.5 K starting curve's peak
          app.sky.goTo(266.4, -28.9, 200, { mode: 'allsky', frame: 'gal', band: 'cmb' });
          var c = app.chart.ds.table.cols;
          st.model = OBS.instruments.model(app.chart, app.tools, {
            label: 'blackbody', err: function (i) { return c.err[i]; },
            f: function (f, p) { return A.planckNu(f, p.T + p.dT); },
            params: [
              { id: 'T', label: 'Temperature', min: 2, max: 4, value: 3.5, fmt: function (x) { return x.toFixed(3); }, unit: 'K' },
              { id: 'dT', label: 'Fine tune', min: -0.01, max: 0.01, value: 0, fmt: function (x) { return (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(4); }, unit: 'K' }
            ],
            describe: function (p) { var T = p.T + p.dT; return 'T = ' + fix(T, 4) + ' K: peak at ' + fix(A.wienNuGHz(T), 1) + ' GHz (λ = ' + fix(299.792458 / A.wienNuGHz(T), 2) + ' mm), ' + fix(A.photonDensity(T), 0) + ' photons in every cm³.'; }
          });
        });
      },
      task: 'The curve starts at 3.5 K, about what Penzias and Wilson first estimated. Slide the temperature until the blackbody runs through FIRAS’s points; use the fine tune: the error bars are hundreds of times smaller than the dots.',
      progress: function (app, st) { var q = st.model.p._fit; return { done: q.red < 3, text: 'χ²/N = ' + OBS.fmt(q.red, 3) + ' at T = ' + OBS.fix(st.model.p.T + st.model.p.dT, 4) + ' K' }; },
      result: function (app, st) {
        var T = st.model.p.T + st.model.p.dT;
        return 'T = ' + fix(T, 4) + ' K, with χ²/N = ' + fmt(st.model.p._fit.red, 3) + ': a blackbody to better than 1 part in 10,000. It peaks at ' + fix(A.wienNuGHz(T), 1) + ' GHz and fills space with ' + fix(A.photonDensity(T), 0) + ' photons per cm³, about a billion for every proton. Hydrogen became neutral when the radiation cooled to about 3,000 K, at 1 + z = 3000 / ' + fix(T, 3) + ' = ' + fix(3000 / T, 0) + '.';
      },
      explain: 'Only a body in thermal equilibrium emits a perfect blackbody, and nothing in today’s Universe is dense enough to have made this one. It is the light of the whole Universe at the moment it became transparent, 380,000 years after the Big Bang, when it was about 3,000 K: since then every wavelength has been stretched 1,100 times. That is why the Big Bang model, rather than a steady-state universe, became the standard one.',
      real: 'Alpher and Herman predicted a 5 K background in 1948; Penzias and Wilson (Nobel 1978) found it by accident in 1965. FIRAS’s spectrum, shown by Mather in 1990 to a standing ovation, earned him and George Smoot (who led the measurement of its tiny variations) the 2006 Nobel Prize. The best value is 2.72548 ± 0.00057 K (Fixsen 2009).'
    },
    {
      id: 'peaks', ch: 6, title: 'The shape of space', era: '2000–2018', who: 'BOOMERanG, WMAP and Planck', course: 'Lecture 7 (anisotropies, acoustic peaks, the geometry of the Universe)',
      brief: 'The CMB’s temperature varies by about 1 part in 100,000 from place to place. Sound waves in the young plasma make spots of a preferred physical size: the distance sound could travel before recombination, which physics fixes. How big that ruler looks on the sky depends on the geometry of space. Here is Planck’s measurement of how strong the variations are on each angular scale.',
      vars: [['ℓ', 'Multipole', '', 'ℓ ≈ 180° / angular size'], ['D_ℓ', 'Fluctuation power', 'μK²', 'Variance of the temperature on that scale'], ['Ω_total', 'Total density', 'critical density', '1 = flat; more = closed; less = open']],
      relation: 'θ ≈ r_s / D_A(z = 1100): the sound horizon r_s seen at the angular-diameter distance of the CMB. Curvature changes D_A, and so slides every peak in ℓ.',
      deduce: 'Whether space is flat, closed like a sphere or open like a saddle.',
      question: 'What will the CMB say about the geometry of space?',
      choices: [['closed', 'curved like a sphere (Ω > 1)'], ['flat', 'flat (Ω = 1)'], ['open', 'curved like a saddle (Ω < 1)']],
      answer: 'flat',
      setup: function (app, st) {
        return Promise.all([app.open('cmbpower', { x: 'l', y: 'Dl' }), OBS.open('cmbtheory')]).then(function (r) {
          var th = r[1].table.cols, TT = new Float64Array(2700), c = app.chart.ds.table.cols;
          for (var i = 0; i < r[1].table.n; i++) TT[th.l[i]] = th.tt[i];
          function tt(l) { if (l < 2 || l > 2598) return NaN; var k = Math.floor(l), f = l - k; return TT[k] * (1 - f) + TT[k + 1] * f; }
          var DA0 = A.DA(1090, 67.4, 0.315, 0.685);
          st.ratio = function (Ot) { return A.DA(1090, 67.4, 0.315, Ot - 0.315) / DA0; };
          app.chart.setView({ x0: 0, x1: 2550, y0: -200, y1: 6600 });
          app.sky.goTo(266.4, -28.9, 200, { mode: 'allsky', frame: 'gal', band: 'cmb' });
          st.model = OBS.instruments.model(app.chart, app.tools, {
            label: 'your universe', err: function (i) { return c.err[i]; },
            f: function (l, p) { return tt(l / st.ratio(p.Ot + p.dO)); },
            params: [
              { id: 'Ot', label: 'Ω_total = Ωm + ΩΛ', min: 0.7, max: 1.3, value: 0.8, fmt: function (x) { return x.toFixed(3); } },
              { id: 'dO', label: 'Fine tune', min: -0.02, max: 0.02, value: 0, fmt: function (x) { return (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(4); } }
            ],
            describe: function (p) { var O = p.Ot + p.dO, l1 = 220 * st.ratio(O); return 'Ω_total = ' + fix(O, 3) + ' (' + (Math.abs(O - 1) < 0.01 ? 'flat' : O > 1 ? 'closed' : 'open') + '): the first peak sits at ℓ ≈ ' + fix(l1, 0) + ', spots about ' + fix(180 / l1, 2) + '° across.'; }
          });
        });
      },
      task: 'The blue curve (“your universe”) is the spectrum the physics of the early plasma predicts, drawn for the universe on the slider. It starts open (Ω = 0.8), where the same ruler looks smaller and every peak moves right. Slide Ω_total until the peaks line up with Planck’s points.',
      progress: function (app, st) { var p = st.model.p, q = p._fit; return { done: q.red < 3, text: 'Ω_total = ' + OBS.fix(p.Ot + p.dO, 3) + ', χ²/N = ' + OBS.fmt(q.red, 3) }; },
      result: function (app, st) {
        var O = st.model.p.Ot + st.model.p.dO, l1 = 220 * st.ratio(O);
        return 'Ω_total = ' + fix(O, 3) + ' (χ²/N = ' + fmt(st.model.p._fit.red, 3) + '). The first peak is at ℓ ≈ ' + fix(l1, 0) + ': the biggest hot and cold spots are about ' + fix(180 / l1, 2) + '° across, about twice the size of the full Moon. A 2% change in Ω would already move the peaks by a whole bin.';
      },
      explain: 'The peaks are sound waves caught at maximum compression or rarefaction at recombination. Their physical size is known, so their angle measures the geometry: in a closed universe the ruler would look bigger (peaks to the left), in an open one smaller. Planck’s peaks sit where a flat universe puts them, to within half a percent. Flat space needs Ω_total = 1, yet matter is only 0.3: the rest is the dark energy the supernovae found. The height of the second peak relative to the first also weighs the ordinary matter: Ω_b ≈ 0.049.',
      real: 'BOOMERanG and MAXIMA located the first peak in 2000; WMAP (2003–2012) and Planck (2013–2018) measured the spectrum to ℓ ≈ 2500. Planck 2018: Ω_k = 0.001 ± 0.002 (with BAO), Ωm = 0.315, H₀ = 67.4 km/s/Mpc, age 13.80 Gyr.'
    },
    {
      id: 'forest', ch: 6, title: 'A forest of clouds', era: '1963–1971', who: 'Maarten Schmidt; Roger Lynds', course: 'Lecture 7 (quasars as cosmological probes, the Lyman-α forest)',
      brief: 'Quasars are bright enough to see across most of the Universe, and their light passes through everything in between. Here is an SDSS spectrum of a quasar. First find its redshift from its broad emission lines; then look at what happened to its light on the way.',
      vars: [['λ', 'Observed wavelength', 'nm', 'SDSS spectrograph'], ['z', 'Redshift', '', 'λ_observed = (1 + z) λ_rest'], ['Lyα', 'Lyman-α', '121.6 nm', 'Hydrogen’s strongest line, n = 2 → 1']],
      relation: '1 + z = λ_observed / λ_rest for every line. Each cloud of hydrogen between us and the quasar absorbs Lyα at its own redshift, so at λ = 121.6 nm × (1 + z_cloud).',
      deduce: 'How far the quasar is, and the hydrogen spread through intergalactic space along the way.',
      question: 'Blueward of the quasar’s own Lyman-α line, its spectrum will look…',
      choices: [['smooth', 'smooth, like the red side'], ['forest', 'cut by many narrow absorption lines'], ['dark', 'completely dark']],
      answer: 'forest',
      setup: function (app, st) {
        return app.open('zspec', { sub: 'qso' }).then(function () {
          app.chart.setView({ x0: 380, x1: 920, y0: 0, y1: 1.15 });
          app.sky.goTo(266.4, -28.9, 200, { mode: 'allsky', frame: 'gal', band: 'visible', photo: true });
          return OBS.loadJSON('zspec_meta').then(function (m) {
            st.meta = m.filter(function (x) { return x.obj === 'qso'; })[0];
            app.sky.markers = [{ ra: st.meta.ra, dec: st.meta.dec, label: 'the quasar' }];
            st.fit = OBS.instruments.zfit(app.chart, app.tools, { lines: OBS.LINES.quasar, z: 1.0, min: 0, max: 4, forest: true });
          });
        });
      },
      task: 'The dashed lines are a quasar’s emission lines (Lyα, C IV, C III], …) at the trial redshift. Slide z until they sit on the broad bumps; the shaded band then marks where intervening hydrogen would absorb.',
      progress: function (app, st) { var d = Math.abs(st.fit.z - st.meta.z); return { done: d < 0.01, text: 'z = ' + st.fit.z.toFixed(3) + (d < 0.01 ? '' : d < 0.05 ? ': close, use the fine tune' : '') }; },
      result: function (app, st) {
        var z = st.fit.z, lya = 121.6 * (1 + z), zlo = 380 / 121.6 - 1;
        return 'Your redshift: z = ' + fix(z, 3) + ' (SDSS: ' + fix(st.meta.z, 3) + '). Its Lyman-α line is at ' + fix(lya, 0) + ' nm, in blue-green light. Every dip between 380 and ' + fix(lya, 0) + ' nm is Lyα absorbed by a hydrogen cloud at a redshift between ' + fix(zlo, 2) + ' and ' + fix(z, 2) + '. The light left the quasar when the Universe was ' + fix(A.age(z, 67.4, 0.315, 0.685), 1) + ' billion years old.';
      },
      explain: 'Each cloud along the line of sight absorbs at its own redshift, so the spectrum blueward of Lyα becomes a forest of lines, a one-dimensional map of hydrogen through billions of light-years. Only about 1 atom in 10,000 of that gas is neutral: the intergalactic medium was ionised by the first stars and quasars (reionization, finished by z ≈ 6). Metal lines in the clouds show they are nearly pristine, and their deuterium checks Big Bang nucleosynthesis.',
      real: 'Schmidt identified 3C 273’s lines at z = 0.158 in 1963. Lynds described the “forest” in 1971. Gunn and Peterson (1965) predicted that fully neutral gas would black out everything blueward of Lyα; the first such trough was seen in a z = 6.28 SDSS quasar in 2001, marking the end of reionization.'
    },
    // =================================================================================== chapter 7 (lectures 8–9)
    {
      id: 'night', ch: 7, title: 'Planning a night', era: 'every clear night', who: 'observers at the Wise Observatory, Mitzpe Ramon', course: 'Lecture 8 (coordinate systems, sidereal time, visibility, airmass)',
      brief: 'A telescope can only look through the night sky, and best near the zenith, where the light crosses the least air. When a target is up depends on its right ascension, the date and the hour; how high it gets depends on its declination and your latitude. The sky now shows the horizon at the Wise Observatory (30.6° N).',
      vars: [['α, δ', 'Right ascension, declination', 'h, °', 'The target’s place on the sky'], ['LST', 'Local sidereal time', 'h', 'The right ascension on the meridian now'], ['h', 'Altitude', '°', 'Above the horizon'], ['X', 'Airmass', '', '≈ 1 / sin h: how much air the light crosses']],
      relation: 'A star crosses the meridian when LST = α, at altitude h_max = 90° − |φ − δ|. The sidereal day is 23 h 56 min, so each night stars rise 3 min 56 s earlier.',
      deduce: 'When to observe a target, and how much the atmosphere will dim and blur it.',
      question: 'In which month is the Orion Nebula highest in the sky at midnight?',
      choices: [['jun', 'June'], ['sep', 'September'], ['dec', 'December']],
      answer: 'dec',
      setup: function (app, st) {
        app.sky.layers.photo = false; app.sky.band = 'visible';
        app.instrument('Planning a night · the sky over Mitzpe Ramon');
        st.planner = OBS.instruments.night(app, app.tools, {
          targets: [{ name: 'Orion Nebula (M42)', ra: 83.82, dec: -5.39 }, { name: 'Globular cluster M13', ra: 250.42, dec: 36.46 }, { name: 'Andromeda galaxy (M31)', ra: 10.68, dec: 41.27 }, { name: 'Galactic centre (Sgr A*)', ra: 266.42, dec: -29.01 }, { name: 'Pleiades', ra: 56.75, dec: 24.12 }],
          target: 0, day: 30, hour: 21
        });
        return Promise.resolve();
      },
      task: 'Choose a date and a local time when the Orion Nebula is within 1° of the highest it can get, while the sky is fully dark (the Sun more than 18° below the horizon).',
      progress: function (app, st) {
        var p = st.planner, ok = +p.pick.value === 0 && p.max - p.alt < 1 && p.sunAlt < -18;
        return { done: ok, text: +p.pick.value !== 0 ? 'Choose the Orion Nebula as the target' : 'Altitude ' + p.alt.toFixed(1) + '° of a possible ' + p.max.toFixed(1) + '°; Sun at ' + Math.round(p.sunAlt) + '°' };
      },
      result: function (app, st) {
        var p = st.planner, d = new Date(p.t + 2 * 3600000);
        return 'On ' + d.toUTCString().slice(5, 16) + ' at ' + d.toUTCString().slice(17, 22) + ' (UTC+2) the Orion Nebula stands ' + fix(p.alt, 1) + '° high at azimuth ' + fix(p.az, 0) + '° (' + (Math.abs(p.az - 180) < 10 ? 'due south' : p.az < 180 ? 'south-east' : 'south-west') + '), at airmass ' + fix(p.X, 2) + ': the atmosphere dims it by only ' + fix(0.21 * p.X, 2) + ' mag. The Sun is ' + fix(-p.sunAlt, 0) + '° below the horizon.';
      },
      explain: 'A target is best seen at night when it is opposite the Sun: Orion (α = 5.6 h) is opposite the Sun (α ≈ 17.6 h) in mid-December, so it crosses the meridian at midnight then; in June it is up only in daylight. Because the sidereal day is 4 minutes shorter than the solar day, the same star transits 2 hours earlier every month. Its highest altitude never changes: 90° − |φ − δ| = 90 − (30.6 + 5.4) = 54°. Low targets pay twice: more air dims them (extinction) and blurs them (seeing).',
      real: 'The Wise Observatory, run by Tel Aviv University since 1971, sits at 875 m on the rim of the Ramon crater. Professional observatories rank sites by clear nights, seeing and darkness: Mauna Kea, the Atacama desert and La Palma lead.'
    },
    {
      id: 'photometry', ch: 7, title: 'Counting photons', era: '1970s →', who: 'CCD astronomers; the DESI Legacy Surveys', course: 'Lecture 9 (magnitude systems, zero points, photometry and noise)',
      brief: 'Every star in a modern image is a few hundred pixels of counts on top of the sky. Measuring its brightness means adding up the light inside an aperture and subtracting the sky measured in a ring around it. This is a real r-band image from the Legacy Surveys, already calibrated: each pixel is in nanomaggies, so m = 22.5 − 2.5 log₁₀(flux) on the AB system.',
      vars: [['F', 'Flux in the aperture', 'nanomaggies', 'Sum of the pixels minus the sky'], ['σ', 'Noise', 'nanomaggies', 'From the sky’s scatter, growing with the aperture’s area'], ['m', 'AB magnitude', 'mag', 'm = 22.5 − 2.5 log₁₀ F'], ['FWHM', 'Star width', '″', 'The seeing']],
      relation: 'S/N = F / σ, with σ ≈ σ_sky √N_pix. Too small an aperture loses starlight; too big adds noise. The flux “converges” (the curve of growth) at a few times the seeing.',
      deduce: 'Calibrated magnitudes and their errors, and how big an aperture to use.',
      question: 'Making the aperture bigger and bigger around a faint star…',
      choices: [['better', 'always improves the measurement: more light'], ['peak', 'helps at first, then adds more sky noise than starlight'], ['none', 'makes no difference']],
      answer: 'peak',
      setup: function (app, st) {
        app.instrument('Aperture photometry · Legacy Surveys DR9, r band');
        app.sky.goTo(248.0, 15.0, 2, { mode: 'dome', frame: 'equ', photo: true });
        app.sky.markers = [{ ra: 248.0, dec: 15.0, label: 'the field' }];
        return OBS.loadJSON('photfield').then(function (field) { st.field = field; st.phot = OBS.instruments.photometry(app, app.tools, field); });
      },
      task: 'Click star A. Change the aperture radius and watch the curve of growth (blue) and the signal-to-noise (orange). Then measure three of the marked stars with an aperture big enough that your magnitudes agree with SDSS’s within 0.15.',
      progress: function (app, st) {
        var d = st.phot.done, n = 0; Object.keys(d).forEach(function (k) { if (Math.abs(d[k].m - st.field.stars[k].r) < 0.15) n++; });
        return { done: n >= 3, text: n + ' of 3 stars within 0.15 mag of SDSS' };
      },
      result: function (app, st) {
        var d = st.phot.done, ks = Object.keys(d), dm = ks.map(function (k) { return d[k].m - st.field.stars[k].r; }), m = dm.reduce(function (a, x) { return a + x; }, 0) / dm.length;
        return 'You measured ' + ks.length + ' stars; on average your magnitudes differ from SDSS’s by ' + fix(m, 2) + ' mag. Stars here are ' + fix(median(ks.map(function (k) { return d[k].fwhm; }).filter(isFinite)), 2) + '″ wide (the seeing at the time).' + (function () {
          var f = ks.slice().sort(function (a, b) { return st.field.stars[b].r - st.field.stars[a].r; })[0];
          return f === undefined ? '' : ' The faintest you measured, at r = ' + fix(st.field.stars[f].r, 1) + ', gave S/N = ' + fix(d[f].snr, 0) + ' with your aperture of ' + fix(d[f].r, 1) + ' px: ' + (d[f].snr < 10 ? 'the sky behind it is as bright as the star, so its noise sets the error.' : d[f].snr < 60 ? 'the sky’s noise inside that big aperture is already much of the error; a radius near the seeing would give a better S/N.' : 'bright enough that the sky’s noise matters little.');
        })();
      },
      explain: 'The aperture has to hold most of a star’s light, but every extra pixel adds the sky’s noise. For a faint star the best signal-to-noise comes from a radius about the size of the seeing; the total flux needs two to three times that (or a small aperture plus an “aperture correction” measured on bright stars). Magnitudes only mean something with a zero point: here the images are calibrated so that 1 nanomaggie is 3.631 μJy, m_AB = 22.5. SDSS and the Legacy Surveys use slightly different r filters and fit profiles rather than using apertures, so a few hundredths of a magnitude difference is normal.',
      real: 'CCDs replaced photographic plates in the 1980s: they record 90% of photons instead of 1%, and are linear. The AB system (Oke & Gunn 1983) defines m = −2.5 log₁₀ f_ν − 48.60 in cgs units; Vega magnitudes instead set Vega to zero in every band.'
    },
    // =================================================================================== chapter 8 (lectures 10–11)
    {
      id: 'bursts', ch: 8, title: 'Where do the bursts come from?', era: '1991–2000', who: 'BATSE on the Compton Gamma Ray Observatory', course: 'Lecture 11 (gamma-ray astronomy, GRBs)',
      brief: 'Gamma-ray bursts were found by satellites watching for nuclear tests; for 25 years nobody knew how far away they were. If they came from the Milky Way, like pulsars, they would crowd toward its plane or its centre. Here are 4,430 bursts seen by Fermi’s GBM, on the sky in Galactic coordinates.',
      vars: [['l, b', 'Galactic longitude and latitude', '°', 'b = 0 is the plane of the Milky Way'], ['sin b', 'Sine of the latitude', '', 'Uniform between −1 and +1 for an even sky']],
      relation: 'For objects spread evenly over the sky, sin b is spread evenly: 17% lie within 10° of the plane. A disk population crowds to b = 0.',
      deduce: 'Whether the bursts belong to the Milky Way or lie far beyond it.',
      question: 'Compared with pulsars, which crowd into the Milky Way’s plane, the bursts turned out to…',
      choices: [['plane', 'crowd into the plane too'], ['centre', 'crowd toward the Galactic centre'], ['iso', 'come equally from every direction']],
      answer: 'iso',
      setup: function (app, st) {
        return app.open('grbs', { x: 'sinb', y: '#' }).then(function () {
          app.chart.bins = 20; app.chart.setView({ x0: -1, x1: 1 });
          app.sky.goTo(266.4, -28.9, 200, { mode: 'allsky', frame: 'gal', photo: false });
          var out = OBS.ui.readout(app.tools), row = OBS.ui.row(app.tools);
          function show() { var s = OBS.isotropy(app.chart.ds, app.chart.mask); out.innerHTML = '<b>' + app.chart.ds.title + '</b>: ' + s.n + ' objects; within 10° of the plane: <span class="num">' + Math.round(100 * s.plane) + '%</span> (an even sky: 17%). Toward the Galactic centre, ⟨cos θ⟩ = <span class="num">' + s.cosgc.toFixed(3) + '</span>; flattening ⟨sin² b − ⅓⟩ = <span class="num">' + s.sinb2.toFixed(3) + '</span> (both 0 ± ' + s.err.toFixed(3) + ' if even).'; }
          function go(id) { return app.open(id, { x: 'sinb', y: '#' }).then(function () { app.chart.bins = 20; app.chart.setView({ x0: -1, x1: 1 }); if (id === 'pulsars') st.sawPulsars = true; show(); }); }
          OBS.ui.button(row, 'Compare with the pulsars', function () { go('pulsars'); });
          OBS.ui.button(row, 'Back to the bursts', function () { go('grbs'); });
          app.tools.appendChild(out); show();
          st.go = go;
        });
      },
      task: 'The histogram counts bursts by sin b. Compare it with the pulsars (button), then come back and drag a box over the bursts within 10° of the plane (|sin b| < 0.17).',
      progress: function (app, st) {
        var ds = app.chart.ds, sel = app.chart.selected(), sb = ds.id === 'grbs' ? OBS.column(ds, 'sinb') : null;
        var ok = sb && sel.length > 300 && sel.every(function (i) { return Math.abs(sb[i]) < 0.22; });
        return { done: !!(ok && st.sawPulsars), text: (st.sawPulsars ? '' : 'Look at the pulsars first. ') + (ds.id === 'grbs' ? sel.length + ' bursts selected' : 'Now go back to the bursts') };
      },
      result: function (app, st) {
        var g = OBS.isotropy(app.chart.ds, app.chart.mask), n = app.chart.selected().length;
        return 'You selected ' + n + ' of ' + g.n + ' bursts, ' + pct(n / g.n) + ', near the plane; an even sky puts 17% there. Their mean ⟨cos θ⟩ toward the Galactic centre is ' + fix(g.cosgc, 3) + ' and ⟨sin² b − ⅓⟩ = ' + fix(g.sinb2, 3) + ': zero within ±' + fix(g.err, 3) + '. Pulsars, by contrast, have most of their number within 10° of the plane.';
      },
      explain: 'An even sky leaves three possibilities. The bursts could be very near, within a few hundred parsecs, closer than the thickness of the Milky Way’s disk; but then the faint ones would be as evenly spread and as numerous as an even, nearby population predicts, and they are too few. They could fill a huge halo around the Milky Way, far bigger than its disk, so that our off-centre position barely shows; that was the Galactic side of the 1995 debate. Or they are cosmological, far beyond all Galactic structure. BATSE’s 2,700 bursts were already isotropic in 1991–2000, and their faint ones were too few for an even, nearby population. In 1997 BeppoSAX found X-ray afterglows, and GRB 970508’s spectrum gave z = 0.835: billions of light-years. To be seen so far, a burst must release as much energy in seconds as the Sun in its whole life, almost certainly beamed into jets.',
      real: 'Meegan et al. (1992) showed BATSE’s isotropy; the Great Debate of 1995 (Lamb vs Paczyński) argued Galactic halo vs cosmological. Swift (2004) and Fermi (2008) followed; GRB 170817A, 1.7 s after GW170817, tied short bursts to neutron-star mergers.'
    },
    {
      id: 'frb', ch: 8, title: 'Weighing the missing baryons', era: '2007–2020', who: 'Duncan Lorimer; Jean-Pierre Macquart and the ASKAP team', course: 'Lecture 10 (radio astronomy, fast radio bursts, the missing baryons)',
      brief: 'A fast radio burst lasts a millisecond, and its low frequencies arrive later than its high ones: the radio waves were slowed by free electrons on the way. The delay measures the electrons along the path, the dispersion measure DM. For 101 bursts localized to a host galaxy with a redshift, DM can be compared with distance. The CMB and Big Bang nucleosynthesis say ordinary matter is Ω_b = 0.049; most of it had never been seen.',
      vars: [['DM', 'Dispersion measure', 'pc/cm³', 'Delay Δt ∝ DM / ν² across the band'], ['z', 'Host redshift', '', 'From the host galaxy’s spectrum'], ['Ω_b', 'Ordinary matter', 'critical density', 'What you fit']],
      relation: 'DM = DM_Milky Way + DM_host + DM_cosmic(z), with DM_cosmic ≈ 3cH₀Ω_b f_IGM / (8πGm_p) × ∫(1+z)/E(z) dz ≈ 935 z pc/cm³ for Ω_b = 0.049.',
      deduce: 'How much ionised gas lies between galaxies, the baryons nobody could count.',
      question: 'If the missing baryons fill the space between galaxies as hot, thin gas, the dispersion measure should…',
      choices: [['grow', 'grow with distance, by about 1,000 pc/cm³ per unit of redshift'], ['flat', 'hardly grow: space between galaxies is empty'], ['fall', 'fall with distance']],
      answer: 'grow',
      setup: function (app, st) {
        return app.open('frbs', { x: 'z', y: 'dm', filter: function (c, i) { return c.z[i] > 0; } }).then(function () {
          app.chart.setView({ x0: 0, x1: 1.1, y0: 0, y1: 2000 });
          app.sky.goTo(266.4, -28.9, 200, { mode: 'allsky', frame: 'gal', band: 'radio' });
          st.model = OBS.instruments.model(app.chart, app.tools, {
            label: 'Milky Way + host + intergalactic gas', err: function () { return 150; },
            f: function (z, p) { return p.a + A.dmCosmic(z, p.Ob); },
            params: [
              { id: 'Ob', label: 'Ω_b, all ordinary matter (the model puts 84% of it between galaxies)', min: 0, max: 0.12, value: 0, fmt: function (x) { return x.toFixed(3); } },
              { id: 'a', label: 'DM from the Milky Way and the host', min: 0, max: 500, value: 150, fmt: function (x) { return Math.round(x); }, unit: 'pc/cm³' }
            ],
            describe: function (p) { return 'Ω_b = ' + fix(p.Ob, 3) + ': a burst at z = 0.5 crosses ' + fix(A.dmCosmic(0.5, p.Ob), 0) + ' pc/cm³ of intergalactic electrons.'; }
          });
        });
      },
      task: 'The model starts with no intergalactic gas (Ω_b = 0). Raise Ω_b, and set the offset from our Galaxy and the hosts, until the curve runs through the middle of the bursts. (The scatter is real: hosts and the cosmic web differ from burst to burst.)',
      progress: function (app, st) { var p = st.model.p; return { done: p.Ob > 0.035 && p.Ob < 0.065 && p._fit.red < 2, text: 'Ω_b = ' + p.Ob.toFixed(3) + ', χ²/N = ' + OBS.fmt(p._fit.red, 3) }; },
      result: function (app, st) {
        var p = st.model.p;
        return 'Your fit: Ω_b = ' + fix(p.Ob, 3) + ' (CMB and nucleosynthesis: 0.049), with ' + fix(p.a, 0) + ' pc/cm³ from our Galaxy and the hosts. A burst at z = 1 crosses about ' + fix(A.dmCosmic(1, p.Ob), 0) + ' pc/cm³: ' + fmt(A.dmCosmic(1, p.Ob) * 3.0857e18, 2) + ' free electrons in every square centimetre of its path.';
      },
      explain: 'The dispersion measure grows with distance at just the rate the baryons predicted by the CMB would give if they are spread between galaxies as warm-hot ionised gas. Stars, cold gas and the hot gas in clusters add up to less than half of Ω_b; the rest is the warm-hot intergalactic medium (WHIM), too thin and too warm to see in emission. Fast radio bursts count every electron along the way, wherever it is.',
      real: 'Lorimer and Narkevic found the first burst in 2001 Parkes data (published 2007). Most are extragalactic: their DMs far exceed what the Milky Way gives at their latitude. Macquart et al. (2020) used five ASKAP-localized bursts to find the missing baryons. In April 2020 FRB 200428 came from the magnetar SGR 1935+2154 in our own Galaxy.'
    },
    {
      id: 'gw', ch: 8, title: 'Hearing black holes merge', era: '2015 →', who: 'LIGO, Virgo and KAGRA', course: 'Lecture 11 (gravitational waves, multi-messenger astronomy)',
      brief: 'On 14 September 2015 both LIGO detectors recorded a chirp lasting 0.2 seconds: two black holes of about 35 and 30 M☉ merging 1.3 billion light-years away. From the waveform come the masses before the merger and the mass of the black hole left behind. Here are 282 mergers from the catalogues to 2025.',
      vars: [['m₁, m₂', 'Masses before the merger', 'M☉', 'From the inspiral chirp'], ['M_f', 'Final mass', 'M☉', 'From the ringdown'], ['ΔM', 'Mass radiated', 'M☉', 'm₁ + m₂ − M_f'], ['ε', 'Fraction radiated', '', 'What you fit']],
      relation: 'M_f = (1 − ε)(m₁ + m₂); the energy carried away by gravitational waves is E = ε (m₁ + m₂) c².',
      deduce: 'How much mass a merger turns into gravitational waves, and how many black holes of each mass exist.',
      question: 'When two black holes merge, the black hole left behind weighs…',
      choices: [['sum', 'exactly the sum of the two'], ['less', 'a few percent less: the rest leaves as gravitational waves'], ['more', 'more than the sum']],
      answer: 'less',
      setup: function (app, st) {
        // the mass that disappears, against the total: a 4% effect is a clear slope here, invisible on final-against-total
        return app.open('gw', { x: 'mtot', y: 'mrad', colourBy: 'kind' }).then(function () {
          app.chart.setView({ y0: -1, y1: 14 });
          app.sky.goTo(266.4, -28.9, 200, { mode: 'allsky', frame: 'gal', band: 'visible', photo: true });
          var c = app.chart.ds.table.cols;
          st.model = OBS.instruments.model(app.chart, app.tools, {
            label: 'radiated = ε × total', err: function (i) { return 0.03 * c.mfinal[i]; },
            f: function (m, p) { return p.eps * m; },
            extra: [{ id: 'sum', label: 'no loss', colour: OBS.colours()['ink-3'], f: function () { return 0; } }],
            params: [{ id: 'eps', label: 'Fraction radiated ε', min: 0, max: 0.15, value: 0, step: 0.001, fmt: function (x) { return (100 * x).toFixed(1) + '%'; } }],
            describe: function (p) { return 'ε = ' + fix(100 * p.eps, 1) + '%: GW150914 (34.6 + 30.0 M☉) would radiate ' + fix(p.eps * 64.6, 1) + ' M☉, ' + fmt(p.eps * 64.6 * A.C.Msun * A.C.c * A.C.c, 2) + ' J.'; }
          });
        });
      },
      task: 'Each dot is one merger: the mass that disappeared (m₁ + m₂ − M_final) against the total mass before. The line starts at “no loss”, zero. Raise ε until the line runs through the dots. Heavier mergers lose more solar masses, but the same fraction.',
      progress: function (app, st) { var p = st.model.p, ok = p.eps > 0.03 && p.eps < 0.055 && p._fit.red < 1; return { done: ok, text: 'ε = ' + (100 * p.eps).toFixed(1) + '%, χ²/N = ' + OBS.fmt(p._fit.red, 3) + (ok ? '' : p.eps <= 0.03 ? ': the line runs below most dots, raise ε' : p.eps >= 0.055 ? ': the line runs above most dots, lower ε' : '') }; },
      result: function (app, st) {
        var e = st.model.p.eps, E = e * 64.6 * A.C.Msun * A.C.c * A.C.c;
        return 'About ' + fix(100 * e, 1) + '% of the mass is radiated. For GW150914 that is ' + fix(e * 64.6, 1) + ' M☉, or ' + fmt(E, 2) + ' J, emitted in about 0.2 s: a peak power of some 4×10⁴⁹ W, more than all the stars in the observable Universe shine.';
      },
      explain: 'Gravitational waves carry energy, so the merged black hole is lighter than its parts by 3–5%, the most efficient process known (fusion releases 0.7%). The catalogue also maps the black-hole population: masses from a few to over 100 M☉, many heavier than any black hole seen in X-ray binaries, and few objects between about 2.5 and 5 M☉, the “mass gap” between neutron stars and black holes. GW170817, two neutron stars at 40 Mpc, was also seen as a short gamma-ray burst and a kilonova that forged gold and platinum.',
      real: 'Hulse and Taylor’s binary pulsar (1974) showed orbits shrinking exactly as gravitational waves predict (Nobel 1993). Weiss, Barish and Thorne won the 2017 Nobel Prize for LIGO. GW170817’s distance from the waves and redshift from its host gave H₀ = 70 (+12, −8) km/s/Mpc: a standard siren.'
    },
    // =================================================================================== chapter 9 (lectures 12–13)
    {
      id: 'redshift', ch: 9, title: 'How far is that galaxy?', era: '1912 → today', who: 'Vesto Slipher; the Sloan Digital Sky Survey', course: 'Lectures 9 and 12 (spectroscopy, galaxy spectra, redshifts)',
      brief: 'A galaxy’s spectrum carries the lines of its gas and its stars, all shifted by the same factor 1 + z. Star-forming galaxies shine in emission lines of hydrogen, oxygen and nitrogen; old galaxies show absorption lines of calcium, magnesium and sodium, and a sharp step at 400 nm where metals absorb the blue light: the 4000 Å break. Here are three SDSS galaxies.',
      vars: [['λ', 'Observed wavelength', 'nm', 'SDSS spectrograph'], ['z', 'Redshift', '', '(λ_obs − λ_rest) / λ_rest'], ['cz', 'Recession speed', 'km/s', 'Only for z ≪ 1']],
      relation: 'λ_observed = (1 + z) λ_rest for every line; for small z, d ≈ cz / H₀.',
      deduce: 'Distances to galaxies, and from many of them, the structure of the Universe in three dimensions.',
      question: 'A galaxy of old stars at z = 0.46 will show its 4000 Å break at about…',
      choices: [['400', '400 nm'], ['585', '585 nm'], ['750', '750 nm']],
      answer: '585',
      setup: function (app, st) {
        st.z = {}; st.cur = 'sf';
        return OBS.loadJSON('zspec_meta').then(function (meta) {
          st.meta = {}; meta.forEach(function (m) { st.meta[m.obj] = m; });
          var row;
          function go(k) {
            st.cur = k;
            return app.open('zspec', { sub: k }).then(function () {
              app.chart.setView({ x0: 380, x1: 920, y0: 0, y1: k === 'sf' ? 2.6 : 1.2 });
              app.tools.innerHTML = ''; row = OBS.ui.row(app.tools);
              var names = { sf: 'Star-forming galaxy', ell: 'Elliptical', lrg: 'Distant red galaxy' }, btn = {};
              Object.keys(names).forEach(function (j) { btn[j] = OBS.ui.button(row, names[j] + (st.z[j] !== undefined ? ' ✓' : ''), function () { go(j); }, j === k ? 'small primary' : 'small ghost'); });
              var m = st.meta[k]; app.sky.markers = [{ ra: m.ra, dec: m.dec, label: m.name.split(' (')[0] }];
              st.fit = OBS.instruments.zfit(app.chart, app.tools, { lines: k === 'sf' ? OBS.LINES.emission : OBS.LINES.absorption, main: k === 'sf' ? 3 : 0, z: 0, min: 0, max: 0.6 });
              // a measurement is what you record when the lines sit on the features, not a value the slider swept past
              var rec = OBS.ui.row(app.tools), msg = OBS.ui.readout(app.tools);
              OBS.ui.button(rec, 'Record this redshift', function () {
                var tol = k === 'lrg' ? 0.004 : 0.002, z = st.fit.z;
                if (Math.abs(z - m.z) < tol) {
                  st.z[k] = z; btn[k].innerHTML = names[k] + ' ✓';
                  msg.innerHTML = 'Recorded <span class="num">z = ' + z.toFixed(4) + '</span> for the ' + names[k].toLowerCase() + '.' + (Object.keys(st.z).length < 3 ? ' Now the next galaxy.' : '');
                } else {
                  delete st.z[k]; btn[k].innerHTML = names[k];
                  msg.innerHTML = 'Not yet: the ' + (k === 'sf' ? 'dashed lines are not on the emission peaks. Find the tallest peak (Hα with [N II] beside it) and bring the Hα line onto it; the others should then fall into place.' : 'dotted lines are not on the dips. Find the 4000 Å step first, where the spectrum drops to the blue, and put Ca H and K just blueward of it; Mg and Na should then sit in dips too.');
                }
              }, 'small primary');
            });
          }
          st.go = go;
          app.sky.goTo(180, 30, 200, { mode: 'allsky', frame: 'equ', photo: false });
          return go('sf');
        });
      },
      task: 'For each galaxy, slide the redshift until the lines match, then press “Record this redshift”: emission lines (dashed red) for the star-forming galaxy; absorption lines (dotted orange: Ca K and H, Mg, Na) for the elliptical and the distant red galaxy. Switch galaxies with the buttons.',
      progress: function (app, st) { var n = Object.keys(st.z).length; return { done: n >= 3, text: n + ' of 3 galaxies measured' + (st.fit ? '; now z = ' + st.fit.z.toFixed(4) : '') }; },
      result: function (app, st) {
        return ['sf', 'ell', 'lrg'].map(function (k) { var z = st.z[k], m = st.meta[k]; return m.name.split(' (')[0] + ': z = ' + fix(z, 4) + ' (SDSS ' + fix(m.z, 4) + ')' + (z < 0.15 ? ', d ≈ cz/H₀ = ' + fix(z * 299792.458 / 70, 0) + ' Mpc' : ', light ' + fix(A.lookback(z, 70, 0.31, 0.69), 1) + ' Gyr old'); }).join('; ') + '.';
      },
      explain: 'Emission lines are easy: Hα and [N II] stand out even in a noisy spectrum. Absorption lines need a good signal, but the 4000 Å break is so strong that it can be found even in broad-band colours: that is how photometric redshifts work for millions of galaxies without spectra, together with the Lyman break at 91.2–121.6 nm for the most distant ones. At z = 0.46 the break has moved to 585 nm, from the blue to the yellow.',
      real: 'Slipher measured the first galaxy redshifts at Lowell Observatory from 1912; Hubble combined them with distances in 1929. SDSS has measured spectra of over 4 million galaxies and quasars; DESI is now taking 40 million.'
    },
    {
      id: 'bpt', ch: 9, title: 'Young stars or a black hole?', era: '1981–2003', who: 'Baldwin, Phillips and Terlevich; Kauffmann and Kewley', course: 'Lectures 12–13 (galaxy properties: emission lines, the BPT diagram, star formation)',
      brief: 'The gas in a galaxy glows in emission lines when something ionises it: the ultraviolet of young massive stars, or the much harder radiation of an accreting black hole. The ratios of neighbouring lines tell which, without any correction for dust or distance. Here are 20,000 SDSS galaxies with all four lines measured.',
      vars: [['[N II]/Hα', 'Nitrogen to hydrogen', 'log', '658.4 nm / 656.3 nm'], ['[O III]/Hβ', 'Oxygen to hydrogen', 'log', '500.7 nm / 486.1 nm'], ['SFR', 'Star formation rate', 'M☉/yr', 'From Hα: SFR = 7.9×10⁻⁴² L(Hα) [erg/s] (Kennicutt 1998)']],
      relation: 'Harder ionising light heats the gas more and reaches further, boosting the forbidden lines [N II] and [O III] relative to hydrogen’s.',
      deduce: 'Which galaxies host an active nucleus, and the star formation rate of the rest.',
      question: 'Gas lit by an accreting black hole, rather than by young stars, shows…',
      choices: [['strong', 'stronger [N II] and [O III] relative to the hydrogen lines'], ['weak', 'weaker [N II] and [O III]'], ['none', 'no emission lines at all']],
      answer: 'strong',
      setup: function (app, st) {
        return app.open('bpt', { x: 'n2ha', y: 'o3hb' }).then(function () {
          app.chart.setView({ x0: -1.6, x1: 0.6, y0: -1.3, y1: 1.4 });
          app.sky.goTo(180, 30, 200, { mode: 'allsky', frame: 'equ', photo: false });
          app.chart.add({ type: 'line', id: 'cut', x1: -1.4, y1: 1.1, x2: 0.4, y2: 1.1, colour: OBS.colours().accent, label: 'your dividing line' });
          var out = OBS.ui.readout(app.tools), ds = app.chart.ds, cls = OBS.column(ds, 'cls'), x = OBS.column(ds, 'n2ha'), y = OBS.column(ds, 'o3hb');
          function score() {
            var L = app.chart.get('cut'), ok = 0, n = 0, agn = 0;
            if (!L) return;
            function side(px, py) { return (L.x2 - L.x1) * (py - L.y1) - (L.y2 - L.y1) * (px - L.x1) > 0; }
            var agnSide = side(0.3, 0.8);     // the AGN side is the one that holds the upper right of the diagram, whichever way the line was drawn
            for (var i = 0; i < x.length; i++) {
              if (cls[i] !== 'star-forming' && cls[i] !== 'AGN') continue;
              var isAgn = cls[i] === 'AGN';
              n++; if ((side(x[i], y[i]) === agnSide) === isAgn) ok++; if (isAgn) agn++;
            }
            st.agree = ok / n;
            out.innerHTML = 'Your line agrees with the MPA–JHU classification for <span class="num">' + Math.round(100 * st.agree) + '%</span> of the star-forming and AGN galaxies (the side of your line toward the upper right = AGN). Composites are left out.';
          }
          OBS.onMission('overlay', function (e) { if (e.chart === app.chart && app.chart.ds === ds) score(); });
          app.tools.appendChild(out); score(); st.score = score;
        });
      },
      task: 'The diagram is shaped like a seagull. Drag the ends of the line so that it separates the left wing (star-forming galaxies) from the right wing (galaxies with active nuclei). Colour by “MPA-JHU classification” to check.',
      progress: function (app, st) { return { done: st.agree > 0.95, text: 'Agreement ' + Math.round(100 * (st.agree || 0)) + '% (95% needed)' }; },
      result: function (app, st) {
        var ds = app.chart.ds, c = ds.table.cols, n = 0, sf = [], cls = OBS.column(ds, 'cls');
        for (var i = 0; i < ds.table.n; i++) if (cls[i] === 'star-forming') { sf.push(c.logsfr[i]); n++; }
        sf.sort(function (a, b) { return a - b; });
        return 'Your line agrees with the published classification for ' + pct(st.agree) + ' of the star-forming and AGN galaxies (composites left out). Of the ' + ds.table.n.toLocaleString('en-US') + ', ' + n.toLocaleString('en-US') + ' are powered by star formation; their median rate is ' + fmt(Math.pow(10, sf[sf.length >> 1]), 2) + ' M☉ per year, about the Milky Way’s.';
      },
      explain: 'Young O and B stars ionise hydrogen but few of their photons have the energy to excite the forbidden lines strongly; an accretion disc’s hard ultraviolet and X-rays do. So active galaxies lie up and to the right. Kauffmann et al. (2003) drew the empirical edge of the star-forming wing; Kewley et al. (2001) the theoretical maximum for star formation; between them lie composites. The Hα line of the star-forming galaxies measures their star formation rate; plotted against stellar mass it forms the “main sequence” of star-forming galaxies, and their oxygen abundance rises with mass (the mass–metallicity relation): try those axes in the lens.',
      real: 'Baldwin, Phillips and Terlevich proposed the diagram in 1981. With SDSS spectra of hundreds of thousands of galaxies, the MPA–JHU group (Brinchmann, Kauffmann, Tremonti and others) measured lines, masses, star formation rates and metallicities for all of them.'
    },
    {
      id: 'chain', ch: 9, title: 'Let the chain find it', era: '1953 → today', who: 'Metropolis and colleagues; cosmologists everywhere', course: 'Lecture 13 (fitting models: χ², grids, Markov chain Monte Carlo)', needs: ['cosmo'],
      brief: 'To find the best parameters of a model you compare it with the data for many trial values and keep the best. With two parameters a grid works; with ten it becomes impossible. A Markov chain walks through parameter space, spending more time where the fit is good, so that its steps sample the probability of each value. Here is the supernova χ² from mission 6.1 as a map of Ωm and ΩΛ.',
      vars: [['χ²', 'Misfit', '', 'Σ (data − model)² / error²'], ['Ωm, ΩΛ', 'Parameters', '', 'Matter and dark energy'], ['acceptance', 'Share of steps taken', '', 'Tuned by the step size']],
      relation: 'Metropolis rule: from θ propose θ′; accept it if χ² falls, otherwise with probability e^{−Δχ²/2}. The steps then sample P(θ | data) ∝ e^{−χ²/2}.',
      deduce: 'Best values and their uncertainties, without mapping the whole space.',
      question: 'A grid with 100 values per parameter costs 10⁴ χ² calculations for 2 parameters. For 10 parameters it costs…',
      choices: [['1e5', '10⁵'], ['1e20', '10²⁰'], ['1e6', '10⁶']],
      answer: '1e20',
      setup: function (app, st) {
        app.instrument('A Markov chain on the supernova χ² map');
        return OBS.open('supernovae').then(function (ds) {
          app.sky.goTo(266.4, -28.9, 200, { mode: 'allsky', frame: 'gal', band: 'cmb' });
          st.mc = OBS.instruments.mcmc(app, app.tools, ds.table.cols);
        });
      },
      task: 'Press “Run the chain” and watch where it goes: it starts far away, in a slowing universe. The step size it starts with is too small. Tune it until a quarter to a half of the steps are accepted (press “Start again” after a change), then set the burn-in so that it drops the walk in from the start (the grey part of the trace, up to the dotted line where the chain reaches the gold region), and let it run to a few thousand steps.',
      progress: function (app, st) {
        var m = st.mc, s = m.stats, acc = m.accRate(), hint = '';
        if (!m.tries) hint = 'Run the chain';
        else if (acc > 0.6) hint = 'steps too small: the chain creeps and its samples repeat each other';
        else if (acc < 0.15) hint = 'steps too large: almost every step is rejected';
        else if (m.inAt === null) hint = 'still walking in toward the gold region';
        else if (m.burn < m.inAt) hint = 'raise the burn-in to at least ' + m.inAt + ': before that the chain was still walking in';
        else if (!s || s.n < 3000) hint = 'let it run: ' + (s ? s.n : 0) + ' of 3000 samples after burn-in';
        var done = !!(s && s.n >= 3000 && acc > 0.15 && acc < 0.6 && m.inAt !== null && m.burn >= m.inAt && Math.abs(s.OL[0] - m.best.OL) < 0.12);
        return { done: done, text: 'Acceptance ' + Math.round(100 * acc) + '%' + (s ? ', ' + s.n + ' samples after burn-in' : '') + (done ? '' : '. ' + hint.charAt(0).toUpperCase() + hint.slice(1) + '.') };
      },
      result: function (app, st) {
        var s = st.mc.stats;
        return 'From ' + s.n + ' samples: Ωm = ' + fix(s.Om[0], 2) + ' ± ' + fix(s.Om[1], 2) + ', ΩΛ = ' + fix(s.OL[0], 2) + ' ± ' + fix(s.OL[1], 2) + ', and ' + pct(s.pacc) + ' of the samples describe an accelerating universe. The grid needed ' + st.mc.evals.toLocaleString('en-US') + ' χ² evaluations for two parameters; your chain used ' + st.mc.tries.toLocaleString('en-US') + '. With ten parameters the grid would need 10²⁰, while a chain needs roughly ten times more steps than here, not 10¹⁶ times.';
      },
      explain: 'A chain is a random walk with a bias toward good fits: a step uphill in χ² is sometimes accepted, so the chain never gets stuck and visits each region as often as its probability. Too small a step and it creeps (acceptance near 100%, strongly correlated samples); too large and it is rejected almost always. The supernovae alone constrain a long, thin valley, Ωm and ΩΛ rising together; the CMB’s flatness cuts across it. Uncertainties from a chain, a bootstrap (refitting resampled data) or a jackknife (leaving one point out at a time) all answer the same question: how much would the answer change with other data like these?',
      real: 'Metropolis, the Rosenbluths and the Tellers published the algorithm in 1953 for physics simulations at Los Alamos; Hastings generalised it in 1970. Since about 2000 almost every cosmological parameter, including Planck’s, comes from Markov chains.'

    }
  ];

  // ---------------------------------------------------------------- check yourself (before and after)
  OBS.QUIZ = [
    { q: 'A star shifts by 0.2″ against the background between January and July. Its distance is', c: ['5 pc', '10 pc', '20 pc'], a: 1, m: 'parallax', why: 'The shift between January and July is twice the parallax, so p = 0.1″ and d = 1/p = 10 pc.' },
    { q: 'Two stars have the same colour; one looks 100 times fainter than the other. From these numbers alone you can say', c: ['it is 10 times farther away', 'it gives out 100 times less light', 'nothing about its distance or power'], a: 2, m: 'hrd', why: 'Flux falls as L/d², so 100 times fainter could be 10 times farther, 100 times weaker, or any mix: you need the distance.' },
    { q: 'You measure the ratio of a 30,000 K star’s brightness at two radio wavelengths. You can find its temperature from it', c: ['yes, Wien’s law works at any wavelength', 'no, in the Rayleigh–Jeans tail the ratio is the same for every hot body', 'only if you know its distance'], a: 1, m: 'thermometer', why: 'At radio wavelengths every hot body is on the Rayleigh–Jeans tail, B ∝ T/λ², so the ratio of two radio fluxes does not depend on T.' },
    { q: 'A star is twice as hot as the Sun and has the same luminosity. Its radius is', c: ['¼ of the Sun’s', '½ of the Sun’s', 'the same as the Sun’s'], a: 0, m: 'radius', why: 'L = 4πR²σT⁴: with T doubled, T⁴ grows 16 times, so R² must shrink 16 times and R is ¼ of the Sun’s.' },
    { q: 'In a binary, star A’s spectral lines swing by ±40 km/s and star B’s by ±80 km/s. The heavier star is', c: ['A', 'B', 'they have equal mass'], a: 0, m: 'binary', why: 'Both stars circle the common centre of mass, M_A v_A = M_B v_B: the star that moves less is the heavier, twice as heavy here.' },
    { q: 'Cluster X’s main sequence ends at M_G = 3; cluster Y’s at M_G = 0. The older one is', c: ['X', 'Y', 'you cannot tell without spectra'], a: 0, m: 'turnoff', why: 'Bright, massive stars leave the main sequence first; a turn-off at a fainter M_G means only less massive stars remain, so the cluster is older.' },
    { q: 'A Cepheid in a distant galaxy has a 30-day period and looks 25 magnitudes fainter than its absolute magnitude. Its distance is', c: ['1 Mpc', '1 kpc', '1 Gpc'], a: 0, m: 'yardstick', why: 'm − M = 25 gives d = 10^(25/5 + 1) pc = 10⁶ pc = 1 Mpc.' },
    { q: 'A planet blocks 1% of its star’s light. Its radius is', c: ['1% of the star’s', '10% of the star’s', '0.01% of the star’s'], a: 1, m: 'transit', why: 'The depth is the ratio of areas, (R_p/R★)² = 0.01, so R_p/R★ = 0.1.' },
    { q: 'Far out in a disk galaxy the rotation speed stays constant with radius. The mass inside radius R grows like', c: ['R', '1/R', 'it stays constant'], a: 0, m: 'rotation', why: 'V² = GM(<R)/R with V constant gives M(<R) ∝ R: mass keeps growing where the light has run out.' },
    { q: 'A microlensing light curve differs from a variable star because it is', c: ['periodic', 'the same in every colour and symmetric in time', 'redder at the peak'], a: 1, m: 'microlens', why: 'Gravity bends light of every colour equally, and a lens passing by gives a symmetric rise and fall that happens once.' },
    { q: 'An asteroid at 2.5 AU goes round the Sun this many times per orbit of Jupiter (5.2 AU)', c: ['2', '3', '5'], a: 1, m: 'kirkwood', why: 'Kepler’s third law: P ∝ a^1.5, so P_Jupiter/P = (5.2/2.5)^1.5 ≈ 3: the 3:1 resonance, a Kirkwood gap.' },
    { q: 'Most red galaxies in the local Universe are', c: ['spirals full of dust', 'ellipticals that no longer form stars', 'very distant galaxies'], a: 1, m: 'redblue', why: 'Red means old stars and no new ones; in the local Universe these are mostly ellipticals (dusty edge-on spirals are a minority).' },
    { q: 'Distant type Ia supernovae look fainter than a universe of matter alone predicts. This means the expansion', c: ['is slowing down', 'is speeding up', 'has stopped'], a: 1, m: 'accel', why: 'Fainter at a given redshift means farther away than a decelerating universe allows: the expansion has been speeding up.' },
    { q: 'The CMB is a 2.725 K blackbody today. When it was emitted, at z ≈ 1100, its temperature was about', c: ['3 K', '300 K', '3,000 K'], a: 2, m: 'cmb', why: 'Expansion stretches every wavelength by 1 + z, so T = 2.725 × 1101 ≈ 3,000 K, when hydrogen became neutral.' },
    { q: 'A star stands 30° above the horizon. The air its light crosses, compared with straight up, is about', c: ['the same', 'twice as much', 'ten times as much'], a: 1, m: 'night', why: 'Airmass ≈ sec z = 1/sin(altitude) = 1/sin 30° = 2.' },
    { q: 'Two radio dishes 10 km apart observe at 21 cm. They resolve details of about', c: ['4 arcseconds', '4 arcminutes', '4 degrees'], a: 0, m: 'frb', why: 'θ ≈ λ/B = 0.21 m / 10⁴ m = 2.1×10⁻⁵ rad ≈ 4 arcseconds.' },
    { q: 'Gamma-ray bursts come equally from every direction, while pulsars crowd to the Milky Way’s plane. So the bursts are most likely', c: ['inside the Galactic disk', 'far beyond the Milky Way', 'in the Solar System'], a: 1, m: 'bursts', why: 'A disk population would crowd to the plane and a halo one toward the centre; an even sky means beyond all Galactic structure.' },
    { q: 'A galaxy has strong [O III]/Hβ and [N II]/Hα line ratios. Its gas is most likely lit by', c: ['young massive stars', 'an accreting black hole', 'old red giants'], a: 1, m: 'bpt', why: 'Hard ultraviolet and X-rays from an accretion disc boost the forbidden lines far more than young stars can.' }
  ];
})(typeof window !== 'undefined' ? window : globalThis);
