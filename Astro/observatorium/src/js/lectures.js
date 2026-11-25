/* Observatorium — the lectures. Before each mission, a short lecture in the observatory: a few steps of theory
 * and reasoning, each with its equation, an explainer to play with, and a cue that turns the sky (and switches
 * to photographs) to what is being discussed. After the mission, practice problems like the homework's.
 *   sky: {ra, dec, fov, mode, frame, photo, grid, markers:[{ra, dec, label}]}
 *   ex:  an explainer id (explainers.js) and its options
 *   practice: [{q, a, tol (relative), unit, sol}]
 * Equations are TeX, typeset by MathJax. */
(function (G) {
  'use strict';
  var OBS = G.OBS;
  var ORI = { ra: 83.8, dec: 2, fov: 42, mode: 'dome', frame: 'equ' };
  function mk(ra, dec, label) { return { ra: ra, dec: dec, label: label }; }
  var BET = mk(88.793, 7.407, 'Betelgeuse'), RIG = mk(78.634, -8.202, 'Rigel'), SIR = mk(101.287, -16.716, 'Sirius');

  var L = OBS.LECTURES = {
    // =================================================================================== chapter 1
    sky: {
      beats: [
        { h: 'Light is all we get', sky: Object.assign({ photo: false, markers: [BET, RIG] }, ORI),
          html: '<p>Almost everything in astronomy comes from light that happens to reach the Earth. A telescope is a bucket for it: the area of its mirror sets how many photons it collects per second. That is HW1 Q1: a 4 cm lens gathers 33 times the light of a 7 mm pupil.</p><p>From one picture of the sky we can measure three things for every star: <b>where</b> it is, <b>how much light</b> arrives, and its <b>colour</b>. This chapter is about how much can be deduced from those three numbers.</p>' },
        { h: 'Magnitudes', sky: { ra: 101.287, dec: -16.716, fov: 30, markers: [SIR] }, ex: { id: 'magnitude' },
          eq: 'm_1 - m_2 = -2.5\\,\\log_{10}\\frac{F_1}{F_2}',
          html: '<p>Hipparchus ranked stars from first to sixth magnitude. In 1856 Pogson noticed that a first-magnitude star is about 100 times brighter than a sixth and made it exact: 5 magnitudes are a factor of 100 in flux. Bigger numbers are fainter. Sirius is \\(m_V = -1.46\\); the naked-eye limit is about 6.5; Hubble reaches 31.</p>' },
        { h: 'Brightness is not power', sky: { ra: 89, dec: -4, fov: 55, markers: [SIR, RIG] },
          eq: 'F = \\frac{L}{4\\pi d^2}',
          html: '<p>The flux \\(F\\) we measure (watts per square metre at the telescope) depends on two things: the star’s luminosity \\(L\\), its total power, and its distance \\(d\\), because the light spreads over a sphere of area \\(4\\pi d^2\\).</p><p>Sirius and Rigel look almost equally bright. Without their distances we cannot tell a nearby dim star from a distant brilliant one. One equation, two unknowns.</p>' },
        { h: 'Colour', sky: Object.assign({ photo: true, markers: [BET, RIG] }, ORI, { fov: 28 }),
          eq: 'B - V = m_B - m_V = -2.5\\,\\log_{10}\\frac{F_B}{F_V} + \\text{const}', ex: { id: 'filters' },
          html: '<p>To measure colour, astronomers photograph a star through standard coloured filters. The classic set is the <b>UBV system</b> of Harold Johnson and William Morgan (1953): <b>U</b> (ultraviolet, around 365 nm), <b>B</b> (blue, around 440 nm) and <b>V</b> (around 550 nm, yellow-green). V stands for <b>visual</b>: its band was chosen to match the sensitivity of the human eye (the dashed curve below), so V magnitudes continue the brightnesses observers had judged by eye since Hipparchus.</p><p>The difference of the two magnitudes, \\(B - V\\), is the <b>colour index</b>; bigger means redder. Betelgeuse (\\(B-V = 1.85\\)) is orange-red and Rigel (\\(-0.03\\)) blue-white: compare them in the photograph. Colour is a <i>ratio</i> of two fluxes, so the distance cancels: it belongs to the star itself. Mission 1.4 turns it into a temperature, using Gaia’s own two broad bands, BP and RP, instead of B and V.</p>' },
        { h: 'Where things are', sky: { ra: 60, dec: 20, fov: 100, grid: true, photo: false },
          html: '<p>Positions use <b>right ascension</b> \\(\\alpha\\) (hours, measured eastward along the celestial equator) and <b>declination</b> \\(\\delta\\) (degrees north or south of it): the Earth’s longitude and latitude projected onto the sky. The grid is now on: each line of right ascension is 2 hours, each circle of declination 30°.</p><p>Catalogues of positions, magnitudes and colours are the raw material of everything that follows: Hipparchus’ 850 stars, Hipparcos’ 118,000, Gaia’s 1.8 billion.</p>' }
      ],
      practice: [
        { q: 'Star A is 3.0 magnitudes brighter than star B. How many times more light do we receive from A?', a: 15.85, tol: 0.03, unit: '×', sol: '\\(F_A/F_B = 10^{0.4 \\times 3} = 10^{1.2} \\approx 15.8\\).' },
        { q: 'The naked-eye limit is about magnitude 6.5 and Sirius is −1.5. How many times fainter is the faintest star you can see than Sirius?', a: 1585, tol: 0.03, unit: '×', sol: '\\(\\Delta m = 8\\), so \\(10^{0.4 \\times 8} = 10^{3.2} \\approx 1600\\).' }
      ]
    },
    parallax: {
      beats: [
        { h: 'The hardest measurement', sky: { ra: 316.725, dec: 38.749, fov: 22, mode: 'dome', frame: 'equ', photo: false, markers: [mk(316.725, 38.749, '61 Cygni')] },
          html: '<p>Until 1838 nobody had measured the distance to any star. Copernicus’ Sun-centred system predicted that stars should shift as the Earth moves round its orbit; no shift was seen, and this was used as an argument against him. The shift was real. It was just smaller than anyone could measure.</p>' },
        { h: 'The geometry', ex: { id: 'parallax' },
          eq: '\\tan p = \\frac{1\\ \\text{AU}}{d} \\;\\Rightarrow\\; d\\,[\\text{pc}] = \\frac{1}{p\\,[\\text{arcsec}]}',
          html: '<p>Look at the star from opposite sides of the Earth’s orbit. The half-angle it sweeps against much more distant stars is the <b>parallax</b> \\(p\\). Because \\(p\\) is tiny, \\(\\tan p \\approx p\\) in radians. A <b>parsec</b> is the distance at which \\(p = 1″\\): 206,265 AU, or 3.26 light-years.</p>' },
        { h: 'How small', sky: { ra: 316.725, dec: 38.749, fov: 0.312, photo: true, markers: [mk(316.725, 38.749, '61 Cyg A and B')] },
          html: '<p>The nearest star, Proxima Centauri, has \\(p = 0.77″\\): the width of a coin seen from 5 km. This photograph shows 61 Cygni, a pair of orange dwarfs (the bright double near the centre). Its 0.29″ took Bessel a year of nights with a heliometer, a telescope with a split lens built to measure small angles between stars.</p>' },
        { h: 'The half-year trap', eq: '\\text{shift over six months} = 2p',
          html: '<p>Over half a year the Earth moves 2 AU, from one side of its orbit to the other. The star therefore crosses the <i>whole width</i> of its ellipse: twice the parallax. This is the trap in HW1 Q2: a 0.2″ shift in half a year means \\(p = 0.1″\\) and \\(d = 10\\) pc.</p><p>The ellipse has a semi-major axis \\(p\\) along the ecliptic and \\(p\\sin\\beta\\) across it, where \\(\\beta\\) is the star’s ecliptic latitude. Only the long axis gives the full \\(2p\\).</p>' },
        { h: 'From Bessel to Gaia', ex: { id: 'ladder', opts: { hi: 1 } },
          html: '<p>Hipparcos (1989–93) measured parallaxes to about 1 milliarcsecond for 118,000 stars. Gaia measures them to 0.02 mas for bright stars: distances good to 1% out to hundreds of parsecs. Everything farther rests on methods calibrated with parallaxes: the <b>distance ladder</b>.</p>' }
      ],
      practice: [
        { q: 'A star shifts by 0.2″ between January and July. What is its distance in parsecs?', a: 10, tol: 0.02, unit: 'pc', sol: 'January and July are opposite sides of the orbit, so \\(2p = 0.2″\\), \\(p = 0.1″\\), \\(d = 1/p = 10\\) pc.' },
        { q: 'Proxima Centauri has p = 0.768″. How far is it in light-years?', a: 4.247, tol: 0.02, unit: 'ly', sol: '\\(d = 1/0.768 = 1.302\\) pc \\(= 1.302 \\times 3.2616 = 4.25\\) ly.' }
      ]
    },
    hrd: {
      beats: [
        { h: 'The inverse-square law', sky: { ra: 0, dec: 0, fov: 200, mode: 'allsky', frame: 'equ', photo: false }, ex: { id: 'inverse' },
          eq: 'F = \\frac{L}{4\\pi d^2}',
          html: '<p>Light leaving a star spreads over a sphere whose area grows as \\(d^2\\): twice as far, a quarter of the flux. With a parallax we know \\(d\\), so we can finally undo the dimming and ask how powerful each star really is.</p>' },
        { h: 'Absolute magnitude', eq: 'M = m - 5\\log_{10}\\frac{d}{10\\ \\text{pc}}, \\qquad m - M = 5\\log_{10} d - 5',
          html: '<p>The <b>absolute magnitude</b> \\(M\\) is the magnitude a star would have if it were 10 pc away. \\(m - M\\) is the <b>distance modulus</b>, a distance written as a magnitude. The Sun has \\(M_V = 4.83\\) and \\(M_G = 4.67\\); at 10 pc it would be a faint naked-eye star.</p>' },
        { h: 'From magnitude to power', eq: '\\frac{L}{L_\\odot} = 10^{-0.4\\,(M_\\text{bol} - 4.74)}, \\qquad M_\\text{bol} = M_G + BC',
          html: '<p>A filter sees only part of a spectrum. The <b>bolometric correction</b> \\(BC\\) adds the light outside the band. It is large for very hot stars (light in the ultraviolet) and very cool ones (light in the infrared). The page uses a correction fitted to Gaia’s own luminosities of nearby stars.</p>' },
        { h: 'Hertzsprung and Russell',
          html: '<p>Around 1905 Hertzsprung noticed that stars of the same colour come in two very different luminosities: <b>dwarfs</b> and <b>giants</b>. In 1913 Russell plotted absolute magnitude against spectral type (OBAFGKM, a temperature sequence) for stars with parallaxes. That diagram organised stellar astrophysics: the main sequence, giants, supergiants, white dwarfs.</p>' },
        { h: 'What the diagram means',
          html: '<p>A star’s place on the diagram is set mostly by its <b>mass</b> and its <b>age</b>. The main sequence is where stars fuse hydrogen in their cores. That stage takes 90% of a star’s life, so about 90% of stars are found there. Giants have used up their core hydrogen and swollen; white dwarfs are the cores left behind. You are about to see it with 36,000 stars measured by Gaia.</p>' }
      ],
      practice: [
        { q: 'A star of apparent magnitude 10 is 100 pc away. What is its absolute magnitude?', a: 5, tol: 0.01, unit: 'mag', sol: '\\(M = 10 - 5\\log_{10}(100/10) = 10 - 5 = 5\\).' },
        { q: 'Star X has M = −0.2 and the Sun M = 4.8 in the same band. How many times more luminous is X in that band?', a: 100, tol: 0.02, unit: '×', sol: '\\(\\Delta M = 5\\) is a factor of exactly 100.' }
      ]
    },
    thermometer: {
      beats: [
        { h: 'Blackbodies', sky: Object.assign({ photo: true, markers: [BET, RIG] }, ORI, { fov: 28 }), ex: { id: 'planck' },
          eq: 'B_\\lambda(T) = \\frac{2hc^2}{\\lambda^5}\\,\\frac{1}{e^{hc/\\lambda k T} - 1}',
          html: '<p>A dense, hot body in equilibrium glows with a spectrum fixed by its temperature alone. A star’s photosphere is close to this. Planck found the formula in 1900 by assuming that light is emitted in quanta of energy \\(E = h\\nu\\): the beginning of quantum mechanics (Lecture 1 §1.2–1.3).</p>' },
        { h: 'Wien and Stefan', eq: '\\lambda_\\text{max} = \\frac{2.898\\ \\text{mm K}}{T}, \\qquad F_\\text{surface} = \\sigma T^4',
          html: '<p>Hotter bodies peak at shorter wavelengths, and give out much more power per square metre. The Sun (5772 K) peaks at 502 nm and looks white. Betelgeuse (about 3,600 K) peaks in the infrared and looks orange; Rigel (about 12,000 K) peaks in the ultraviolet and looks blue. Compare them in the photograph.</p>' },
        { h: 'Colour cancels size and distance', eq: 'F_\\lambda = \\pi B_\\lambda(T)\\left(\\frac{R}{d}\\right)^2 \\;\\Rightarrow\\; \\frac{F_{\\lambda_1}}{F_{\\lambda_2}} = \\frac{B_{\\lambda_1}(T)}{B_{\\lambda_2}(T)}',
          html: '<p>The flux at every wavelength carries the same factor \\((R/d)^2\\). In a ratio of two wavelengths it cancels, so the colour depends on \\(T\\) only. That is why a single colour index is a thermometer, and why in HW1 Q2 the colour gave \\(T\\) before you knew anything else about the star.</p>' },
        { h: 'Which limit you are in', eq: '\\frac{hc}{\\lambda kT} \\gg 1:\\; B_\\lambda \\approx \\frac{2hc^2}{\\lambda^5}e^{-hc/\\lambda kT} \\qquad \\frac{hc}{\\lambda kT} \\ll 1:\\; B_\\lambda \\approx \\frac{2ckT}{\\lambda^4}',
          html: '<p>On the short-wavelength side of the peak (the Wien tail) the ratio of two bands changes exponentially with \\(T\\): an excellent thermometer. Far past the peak (Rayleigh–Jeans) every hot body has the same slope \\(\\lambda^{-4}\\), so the ratio is the same for every temperature (HW1 Q2b). Choose filters before or near the peak.</p>' },
        { h: 'Real stars are not perfect blackbodies',
          html: '<p>Atoms and molecules in a star’s atmosphere absorb at particular wavelengths, removing light mostly in the blue. Cool stars are worst: molecules such as titanium oxide carve deep bands. A colour temperature can therefore differ from the true surface (effective) temperature, most of all for red dwarfs. You will see this in the mission. Gaia avoids it by fitting a whole low-resolution spectrum.</p>' }
      ],
      practice: [
        { q: 'A star’s spectrum peaks at 290 nm. What is its temperature?', a: 9993, tol: 0.02, unit: 'K', sol: '\\(T = 2.898\\times10^{-3}/290\\times10^{-9} \\approx 10{,}000\\) K.' },
        { q: 'Two stars have the same radius. One is twice as hot. How many times more luminous is it?', a: 16, tol: 0.01, unit: '×', sol: '\\(L \\propto R^2T^4\\), so \\(2^4 = 16\\).' }
      ]
    },
    radius: {
      beats: [
        { h: 'Luminosity and temperature give size', sky: { ra: 101.287, dec: -16.716, fov: 18, mode: 'dome', frame: 'equ', photo: false, markers: [mk(101.287, -16.716, 'Sirius A and B')] }, ex: { id: 'stefan' },
          eq: 'L = 4\\pi R^2 \\sigma T^4 \\;\\Rightarrow\\; \\frac{R}{R_\\odot} = \\left(\\frac{L}{L_\\odot}\\right)^{1/2}\\left(\\frac{T_\\odot}{T}\\right)^2',
          html: '<p>Each square metre of a star’s surface radiates \\(\\sigma T^4\\). The total power is that times the area \\(4\\pi R^2\\). So once we have \\(L\\) (from distance and flux) and \\(T\\) (from colour), the radius follows.</p>' },
        { h: 'Lines of equal size', eq: '\\log L = 2\\log R + 4\\log T + \\text{const}',
          html: '<p>On a chart of \\(\\log L\\) against \\(\\log T\\), stars of the same radius lie on straight lines of slope 4. Giants sit above the main sequence because they are big; white dwarfs sit below because they are small. The diagram is also a size chart.</p>' },
        { h: 'Sirius B',
          html: '<p>In 1844 Bessel noticed that Sirius wobbles, as if pulled by an unseen companion. Alvan Clark saw it in 1862, 10,000 times fainter than Sirius A. In 1915 Adams took its spectrum: white, as hot as Sirius A. Hot and faint can only mean tiny. Sirius B has about the Sun’s mass in a ball the size of the Earth.</p>' },
        { h: 'What holds a white dwarf up', eq: 'P_\\text{deg} \\propto \\rho^{5/3}\\ \\text{(slow electrons)},\\qquad P_\\text{deg} \\propto \\rho^{4/3}\\ \\text{(relativistic)} \\;\\Rightarrow\\; M_\\text{Ch} \\approx 1.4\\,M_\\odot',
          html: '<p>Electrons packed this tightly cannot share quantum states (Pauli’s principle), so they resist further squeezing even when cold: <b>degeneracy pressure</b>. When they become relativistic the pressure grows more slowly with density, like gravity does, and above one mass gravity wins: the Chandrasekhar mass (HW2 Q2). One consequence: heavier white dwarfs are <i>smaller</i>.</p>' }
      ],
      practice: [
        { q: 'A star has L = 100 L☉ and T = 2,886 K (half the Sun’s). What is its radius?', a: 40, tol: 0.02, unit: 'R☉', sol: '\\(R = \\sqrt{100}\\times 2^2 = 40\\,R_\\odot\\): a red giant.' },
        { q: 'A white dwarf has L = 0.003 L☉ and T = 10,000 K. What is its radius?', a: 0.01825, tol: 0.04, unit: 'R☉', sol: '\\(R = \\sqrt{0.003}\\,(5772/10000)^2 = 0.0548 \\times 0.333 = 0.018\\,R_\\odot\\), about twice the Earth’s radius.' }
      ]
    },
    binary: {
      beats: [
        { h: 'Stars in pairs', sky: { ra: 200.981, dec: 54.925, fov: 6, mode: 'dome', frame: 'equ', photo: false, markers: [mk(200.981, 54.925, 'Mizar'), mk(47.042, 40.956, 'Algol')] },
          html: '<p>About half of Sun-like stars have a companion. In a <b>visual binary</b> we watch the orbit (Sirius A and B). In a <b>spectroscopic binary</b> the stars are too close to separate, but their spectral lines shift back and forth: Mizar, here, was the first (Pickering and Maury, 1889). In an <b>eclipsing binary</b> one star passes in front of the other: Algol dims every 2.87 days.</p>' },
        { h: 'Doppler shifts', eq: 'v_r = c\\,\\frac{\\Delta\\lambda}{\\lambda}',
          html: '<p>Motion toward us shortens wavelengths; motion away lengthens them. The H-alpha line at 656.3 nm shifted by 0.22 nm means 100 km/s. Only the component along our line of sight shows: the <b>radial velocity</b>.</p>' },
        { h: 'Around the centre of mass', ex: { id: 'orbit2' }, eq: 'M_1 a_1 = M_2 a_2 \\;\\Rightarrow\\; \\frac{M_1}{M_2} = \\frac{K_2}{K_1}',
          html: '<p>Both stars circle their common centre of mass, the lighter one on the bigger orbit. In the same period it must move faster, so the ratio of the two velocity amplitudes \\(K_1, K_2\\) is the inverse ratio of the masses. This is the trap in HW2 Q4: the fast star is the light one.</p>' },
        { h: 'Kepler’s third law', eq: 'P^2 = \\frac{4\\pi^2 a^3}{G(M_1+M_2)},\\quad a = \\frac{(K_1+K_2)P}{2\\pi\\sin i} \\;\\Rightarrow\\; M_1+M_2 = \\frac{P\\,(K_1+K_2)^3}{2\\pi G\\,\\sin^3 i}',
          html: '<p>The period and the summed speeds give the size of the orbit, and Kepler’s third law turns size and period into total mass. The catch is the inclination \\(i\\): from velocities alone we only get \\(M\\sin^3 i\\). If the stars eclipse, the orbit is nearly edge-on and \\(\\sin i \\approx 1\\).</p>' },
        { h: 'Your pair', sky: { ra: 197.1246, dec: -2.6789, fov: 0.234, photo: true, markers: [mk(197.1246, -2.6789, 'Gaia DR3 3684484600212307712')] },
          html: '<p>This 7.7-magnitude star in Virgo, 183 pc away, is two F-type stars 2.7 days apart. Gaia measured both stars’ velocity curves from its spectra, and its photometry shows the eclipses. You will read the curves yourself and weigh both stars.</p>' }
      ],
      practice: [
        { q: 'K₁ = 50 km/s, K₂ = 100 km/s, P = 10 days, circular and edge-on. What is the total mass?', a: 3.50, tol: 0.03, unit: 'M☉', sol: '\\(M = P(K_1+K_2)^3/2\\pi G = 864{,}000 \\times (1.5\\times10^5)^3 / (2\\pi \\times 6.674\\times10^{-11}) = 6.95\\times10^{30}\\) kg \\(= 3.5\\,M_\\odot\\).' },
        { q: 'For the same pair, what is the mass of the heavier star?', a: 2.33, tol: 0.03, unit: 'M☉', sol: '\\(M_1/M_2 = K_2/K_1 = 2\\), so \\(M_1 = \\tfrac{2}{3}\\times 3.5 = 2.33\\,M_\\odot\\). It is the slower star.' }
      ]
    },
    masslum: {
      beats: [
        { h: 'Why mass sets the light', sky: Object.assign({ photo: false, markers: [RIG, mk(217.429, -62.680, 'Proxima Centauri')] }, ORI, { fov: 70 }),
          eq: 'L \\propto M^{a},\\qquad a \\approx 3\\text{–}4',
          html: '<p>A star is a balance between gravity and pressure. More mass squeezes harder, so the core must be hotter and denser. Fusion rates rise steeply with temperature (the CNO cycle roughly as \\(T^{16}\\)), and energy leaks out faster through a hotter interior. The result is a steep power law of mass: Rigel, about 20 M☉, out-shines the Sun some 100,000 times.</p>' },
        { h: 'Reading power laws', ex: { id: 'powerlaw' }, eq: '\\log L = a\\,\\log M + c',
          html: '<p>On logarithmic axes a power law is a straight line whose slope is the exponent. That is how the relation is measured: plot \\(\\log L\\) against \\(\\log M\\) for stars weighed in binaries and fit the slope.</p>' },
        { h: 'Lifetimes', eq: 't_\\text{MS} \\approx 10\\ \\text{Gyr}\\;\\frac{M/M_\\odot}{L/L_\\odot} \\approx 10\\ \\text{Gyr}\\,\\left(\\frac{M}{M_\\odot}\\right)^{1-a}',
          html: '<p>The fuel grows like \\(M\\), the burn rate like \\(L\\). A 20 M☉ star (HW2 Q3) lives about 10 million years; a 0.2 M☉ red dwarf longer than the present age of the Universe. Every cluster age in the next chapter comes from this one line.</p>' },
        { h: 'Where it stops working',
          html: '<p>The relation holds only on the main sequence. Giants and white dwarfs of the same mass have very different luminosities. At the very top, above about 50 M☉, radiation pressure takes over and \\(L\\) rises only about in proportion to \\(M\\).</p>' }
      ],
      practice: [
        { q: 'With L ∝ M^3.5 and the Sun living 10 Gyr, how long does a 5 M☉ star live?', a: 179, tol: 0.03, unit: 'Myr', sol: '\\(t = 10\\ \\text{Gyr}\\times 5^{1-3.5} = 10 \\times 5^{-2.5} = 0.179\\) Gyr \\(= 179\\) Myr.' }
      ]
    },
    pleiades: {
      beats: [
        { h: 'Star clusters', sky: { ra: 56.75, dec: 24.12, fov: 2.028, mode: 'dome', frame: 'equ', photo: true },
          html: '<p><b>Open clusters</b> such as the Pleiades hold hundreds to thousands of young stars in the Galactic disk; the blue haze is dust reflecting their light. <b>Globular clusters</b> hold 10⁵–10⁶ old stars in the halo. A cluster’s stars formed together from one cloud: same age, same composition, same distance. That makes a cluster a controlled experiment.</p>' },
        { h: 'Moving together', eq: 'v_t = 4.74\\,\\mu\\,d \\qquad (v_t\\ \\text{in km/s},\\ \\mu\\ \\text{in arcsec/yr},\\ d\\ \\text{in pc})',
          html: '<p>Cluster stars share a velocity, so they share a <b>proper motion</b>, their slow drift across the sky. On the sky they are mixed with unrelated foreground and background stars; in a chart of proper motions they stand apart as a clump.</p>' },
        { h: 'Main-sequence fitting', ex: { id: 'cmd' }, eq: 'm - M = 5\\log_{10}\\frac{d}{10\\ \\text{pc}} + A',
          html: '<p>Plot the members’ colours against their apparent magnitudes. Because all of them are at the same distance, they draw a clean main sequence, shifted from the nearby one (in absolute magnitudes) by the distance modulus. Slide one onto the other and read off \\(d\\). \\(A\\) is the dimming by dust (0.12 mag for the Pleiades).</p>' },
        { h: 'The moving-cluster method', sky: { ra: 66.75, dec: 15.87, fov: 26, photo: false, markers: [mk(66.75, 15.87, 'Hyades')] }, eq: 'd = \\frac{v_r \\tan\\theta}{4.74\\,\\mu}',
          html: '<p>A nearby cluster like the Hyades is big on the sky, and perspective makes its proper motions converge toward one point. The angle \\(\\theta\\) to that point, with the radial velocity \\(v_r\\), gives the distance directly. HW1 Q4 is the same idea: a cluster receding at a known speed shrinks on the sky by \\(\\Delta\\theta/\\theta = \\Delta d/d\\).</p>' },
        { h: 'The Pleiades problem', sky: { ra: 56.75, dec: 24.12, fov: 2.028, photo: true },
          html: '<p>In 1997 Hipparcos put the Pleiades at 118 pc. Main-sequence fitting said 133 pc. Either the parallaxes had a systematic error, or our models of young stars were wrong. The argument lasted 17 years, until radio interferometry (136 pc, 2014) and Gaia (135 pc) settled it in favour of the stars. Measurements have errors their error bars don’t show.</p>' }
      ],
      practice: [
        { q: 'A cluster’s main sequence appears 10 magnitudes fainter than the nearby main sequence, with no dust. How far is it?', a: 1000, tol: 0.02, unit: 'pc', sol: '\\(m - M = 10\\) ⇒ \\(d = 10^{10/5 + 1} = 1000\\) pc.' },
        { q: 'A cluster star has proper motion 0.05″/yr at 100 pc. What is its speed across the sky?', a: 23.7, tol: 0.02, unit: 'km/s', sol: '\\(v_t = 4.74 \\times 0.05 \\times 100 = 23.7\\) km/s.' }
      ]
    },
    // =================================================================================== chapter 2
    turnoff: {
      beats: [
        { h: 'How a star ages', sky: { ra: 132.85, dec: 11.81, fov: 0.546, mode: 'dome', frame: 'equ', photo: true, markers: [mk(132.85, 11.81, 'M67')] },
          html: '<p>On the main sequence a star fuses hydrogen into helium in its core. When the core hydrogen runs out, the core contracts and heats, hydrogen burns in a shell around it, and the envelope swells and cools: a <b>red giant</b>. Stars above about 0.5 M☉ then ignite helium in the core. Low-mass stars end as white dwarfs; stars above about 8 M☉ explode as supernovae.</p>' },
        { h: 'Who leaves first', ex: { id: 'turnoff' }, eq: 't_\\text{MS} \\approx 10\\ \\text{Gyr}\\,(M/M_\\odot)^{1-a} \\;\\Rightarrow\\; M_\\text{TO} = \\left(\\frac{t}{10\\ \\text{Gyr}}\\right)^{1/(1-a)}',
          html: '<p>In a cluster born all at once, the most massive stars leave the main sequence first. At any age there is one mass that is just leaving: the <b>turn-off</b>. Measure the turn-off luminosity, convert it to a mass with the mass–luminosity relation, and its lifetime is the cluster’s age (HW3 Q1b).</p>' },
        { h: 'Two populations', sky: { ra: 265.175, dec: -53.674, fov: 0.468, photo: true, markers: [mk(265.175, -53.674, 'NGC 6397')] },
          html: '<p>Globular clusters such as NGC 6397 are poor in elements heavier than helium: they formed early, from gas that few stars had yet enriched. Baade called these old, metal-poor stars <b>Population II</b>, and the younger metal-rich disk stars, like the Sun and M67, <b>Population I</b>. Fewer metals make a star of a given mass hotter and brighter, which shifts its whole sequence.</p>' },
        { h: 'Oddities in the diagram',
          html: '<p><b>Blue stragglers</b> sit above the turn-off, younger-looking than the cluster: stars that gained mass from a companion or merged with one (Lecture 2). The <b>horizontal branch</b> holds old stars burning helium in their cores. Its RR Lyrae stars all have about the same luminosity, which made them Shapley’s distance markers (chapter 4).</p>' }
      ],
      practice: [
        { q: 'A cluster’s turn-off stars have 2 M☉. Using t = 10 Gyr · M^(−2.5), how old is it?', a: 1768, tol: 0.03, unit: 'Myr', sol: '\\(10\\ \\text{Gyr}\\times 2^{-2.5} = 1.77\\) Gyr.' }
      ]
    },
    leavitt: {
      beats: [
        { h: 'The Harvard computers', sky: { ra: 13.19, dec: -72.83, fov: 3.51, mode: 'dome', frame: 'equ', photo: true, markers: [mk(13.19, -72.83, 'Small Magellanic Cloud')] },
          html: '<p>Around 1900 Harvard College Observatory employed women as “computers” to measure its glass photographic plates: Annie Jump Cannon classified spectra, Antonia Maury found binaries, Henrietta Swan Leavitt searched for variable stars by comparing plates of the Magellanic Clouds taken on different nights. (Switch to <b>Plate view</b>, top right, to see the sky the way they did: dark stars on glass.)</p>' },
        { h: 'Why Cepheids pulse', sky: { ra: 337.293, dec: 58.415, fov: 12, photo: false, markers: [mk(337.293, 58.415, 'δ Cephei')] }, ex: { id: 'cepheid' }, eq: 'P \\propto \\frac{1}{\\sqrt{G\\bar\\rho}}',
          html: '<p>In a layer where helium is partly ionised, compression makes the gas more opaque, so it traps heat instead of releasing it. The layer is pushed out, overshoots, cools, becomes transparent and falls back: a heat engine (Eddington’s κ-mechanism). The period is roughly the time sound takes to cross the star, which depends on its mean density. Big, tenuous, luminous stars pulse slowly. δ Cephei, the prototype, has a period of 5.37 days.</p>' },
        { h: 'Finding a period', eq: '\\phi = \\frac{t - t_0}{P} \\bmod 1',
          html: '<p>A Cepheid is measured on scattered nights, never once per cycle. To see its light curve, <b>fold</b> the times on a trial period: replace each time by its phase. At the right period every cycle stacks onto one clean curve; a little off, and the cycles slide past each other and smear. With five years of data the period comes out to a few parts in 10,000.</p>' },
        { h: 'The law', eq: '\\langle m \\rangle = a\\log_{10}P + b \\;\\Rightarrow\\; M = a\\log_{10}P + b\'',
          html: '<p>All the Small Magellanic Cloud’s stars are at nearly the same distance, so differences in apparent magnitude are differences in luminosity. Leavitt plotted 25 Cepheids in 1912: a straight line in \\(\\log P\\). Measure a period anywhere, read off the luminosity, compare it with the flux, and you have a distance. Only the zero point \\(b\'\\) was missing.</p>' }
      ],
      practice: [
        { q: 'Using M_V = −2.43 (log P − 1) − 4.05, a 10-day Cepheid has V = 15.0, with no dust. How far is it?', a: 64.6, tol: 0.03, unit: 'kpc', sol: '\\(M_V = -4.05\\), \\(m - M = 19.05\\), \\(d = 10^{19.05/5+1} = 64.6\\) kpc.' }
      ]
    },
    yardstick: {
      beats: [
        { h: 'Calibrating the yardstick', sky: { ra: 80.89, dec: -69.76, fov: 14, mode: 'dome', frame: 'equ', photo: true, markers: [mk(80.89, -69.76, 'Large Magellanic Cloud')] }, eq: 'M = a\\log_{10}P + b,\\qquad b \\text{ from Cepheids with parallaxes}',
          html: '<p>The slope of Leavitt’s law comes from one galaxy. The zero point needs Cepheids whose distances are known independently: Gaia parallaxes of Milky Way Cepheids, or Cepheids in clusters with main-sequence distances. Here you will use several hundred Gaia parallaxes.</p>' },
        { h: 'Dust', eq: 'A_G \\approx 2.74\\,E(B-V),\\qquad W = G - 1.90\\,(BP - RP)',
          html: '<p>Dust in the Galactic disk dims starlight and reddens it, in proportion: the dimming \\(A\\) is a fixed multiple of the reddening. The <b>Wesenheit</b> combination \\(W\\) subtracts exactly the dimming that goes with the observed reddening, so a star has the same \\(W\\) with or without dust in front of it. Milky Way Cepheids in the dusty disk need this.</p>' },
        { h: 'The distance ladder', ex: { id: 'ladder', opts: { hi: 3 } },
          html: '<p>Parallaxes calibrate Cepheids; Cepheids in nearby galaxies calibrate type Ia supernovae and the Tully–Fisher relation; those reach the Hubble flow. Every rung inherits the errors of the rungs below it.</p>' },
        { h: 'Island universes', sky: { ra: 10.685, dec: 41.269, fov: 2.496, photo: true, markers: [mk(10.685, 41.269, 'Andromeda (M31)')] },
          html: '<p>In 1924 Hubble found Cepheids in the Andromeda nebula. Their periods and faintness put it far outside the Milky Way: Andromeda is a galaxy like ours. That settled the 1920 Great Debate. Hubble’s distance was too small by more than a factor of 2, because the Cepheid calibration of the time mixed two kinds of pulsating stars (corrected by Baade in 1952).</p>' },
        { h: 'Today', eq: 'H_0 = 73.0 \\pm 1.0\\ \\text{(Cepheids + SNe Ia)} \\quad\\text{vs}\\quad 67.4 \\pm 0.5\\ \\text{(CMB)}\\ \\ \\text{km s}^{-1}\\,\\text{Mpc}^{-1}',
          html: '<p>The Cepheid ladder gives a Hubble constant about 8% higher than the one inferred from the cosmic microwave background. Either some rung has a hidden error, the metallicity dependence of Leavitt’s law among them, or the cosmological model is missing something. You will meet it again in the cosmology lectures.</p>' }
      ],
      practice: [
        { q: 'The LMC’s distance modulus is 18.48. How far is it?', a: 49.7, tol: 0.02, unit: 'kpc', sol: '\\(d = 10^{18.48/5 + 1} = 49{,}659\\) pc \\(\\approx 49.7\\) kpc.' }
      ]
    },
    pulsars: {
      beats: [
        { h: 'What a supernova leaves', sky: { ra: 83.633, dec: 22.0145, fov: 0.195, mode: 'dome', frame: 'equ', photo: true, markers: [mk(83.633, 22.0145, 'Crab Nebula (SN 1054)')] }, eq: '\\rho \\approx \\frac{3M}{4\\pi R^3} \\approx \\frac{3 \\times 1.4 M_\\odot}{4\\pi\\,(12\\ \\text{km})^3} \\approx 4\\times10^{17}\\ \\text{kg m}^{-3}',
          html: '<p>When the iron core of a massive star collapses, electrons are pressed into protons and the core becomes a ball of neutrons, about 1.4 M☉ in 12 km: the density of an atomic nucleus. The outer star is blown off as a supernova. Chinese astronomers saw this one in 1054; the Crab Nebula is its debris, still expanding at 1,500 km/s.</p>' },
        { h: 'A lighthouse', ex: { id: 'pulsar' },
          html: '<p>Collapse spins the core up, as a skater pulls in her arms, and squeezes its magnetic field to \\(10^{8}\\)–\\(10^{15}\\) gauss. Charged particles stream along the magnetic poles and beam radio waves. If the magnetic axis is tilted from the spin axis, the beams sweep round like a lighthouse, and we see a pulse each time one crosses the Earth.</p>' },
        { h: 'Spinning down', eq: '\\dot E = 4\\pi^2 I\\,\\frac{\\dot P}{P^3},\\qquad B \\approx 3.2\\times10^{19}\\sqrt{P\\dot P}\\ \\text{G},\\qquad \\tau = \\frac{P}{2\\dot P}',
          html: '<p>A rotating magnet radiates, so the pulsar slows down. From \\(P\\) and \\(\\dot P\\) alone you get the energy it loses, its field and a characteristic age. The Crab pulsar has \\(P = 33\\) ms and \\(\\dot P = 4.2\\times10^{-13}\\): it loses \\(4.5\\times10^{31}\\) W, just what the nebula radiates, and \\(\\tau = 1{,}260\\) yr, close to its true age of 970 years.</p>' },
        { h: 'Recycled pulsars',
          html: '<p>Some pulsars spin hundreds of times a second yet slow down very little. They are old neutron stars that were <b>recycled</b>: gas falling on them from a companion star spun them up again (Alpar et al. 1982). If that is right, millisecond pulsars should be in binaries far more often than ordinary pulsars. You will check.</p>' }
      ],
      practice: [
        { q: 'A pulsar has P = 1 s and Ṗ = 10⁻¹⁵. What is its characteristic age?', a: 15.8, tol: 0.03, unit: 'Myr', sol: '\\(\\tau = P/2\\dot P = 5\\times10^{14}\\) s \\(= 15.8\\) Myr.' }
      ]
    },
    // =================================================================================== chapter 3
    transit: {
      beats: [
        { h: 'Kepler’s stare', sky: { ra: 290.67, dec: 44.5, fov: 24, mode: 'dome', frame: 'equ', photo: false, markers: [mk(290.67, 44.5, 'Kepler’s field'), mk(292.2474, 47.9695, 'HAT-P-7')] },
          html: '<p>From 2009 to 2013 the Kepler telescope stared at one patch of sky between Cygnus and Lyra, measuring the brightness of 150,000 stars every 30 minutes. It was looking for planets that pass in front of their stars.</p>' },
        { h: 'A shadow on the star', ex: { id: 'transit' }, eq: '\\frac{\\Delta F}{F} \\approx \\left(\\frac{R_p}{R_\\star}\\right)^2',
          html: '<p>If a planet’s orbit is nearly edge-on to us, it blocks a fraction of the star’s disc once per orbit: the ratio of the two areas. Jupiter in front of the Sun would block 1%, the Earth 0.008%. The depth gives the planet’s size relative to its star.</p>' },
        { h: 'Period and orbit', eq: 'a^3 = \\frac{G M_\\star P^2}{4\\pi^2},\\qquad T_\\text{dur} \\approx \\frac{P}{\\pi}\\,\\frac{R_\\star}{a}',
          html: '<p>The time between dips is the orbital period; Kepler’s third law with the star’s mass gives the orbit size. The duration of the dip gives another check on the star’s size and the impact parameter. A radial-velocity mass then gives the density: rock or gas.</p>' },
        { h: 'HAT-P-7 b', sky: { ra: 292.2474, dec: 47.9695, fov: 0.234, photo: true, markers: [mk(292.2474, 47.9695, 'HAT-P-7')] },
          html: '<p>A planet slightly larger than Jupiter, orbiting an F star every 2.2 days, so close that its day side is above 2,000 K. Kepler’s data are precise enough to show the planet’s glow vanish as it goes behind the star. The same geometry applies to the occultations in HW4 Q1, where a small Kuiper-belt object blocks a background star.</p>' }
      ],
      practice: [
        { q: 'A transit 1% deep crosses a star of 1 R☉. How big is the planet, in Jupiter radii (R☉ = 9.73 R_J)?', a: 0.973, tol: 0.03, unit: 'R_J', sol: '\\(R_p/R_\\star = \\sqrt{0.01} = 0.1\\), so \\(R_p = 0.1 \\times 9.73 = 0.97\\,R_J\\).' }
      ]
    },
    census: {
      beats: [
        { h: '51 Pegasi b', sky: { ra: 344.367, dec: 20.769, fov: 14, mode: 'dome', frame: 'equ', photo: false, markers: [mk(344.367, 20.769, '51 Pegasi')] },
          html: '<p>In 1995 Mayor and Queloz found a planet half as massive as Jupiter orbiting the Sun-like star 51 Pegasi every 4.2 days, ten times closer than Mercury is to the Sun. Nobody had expected giant planets there. They probably formed farther out, beyond the snow line where ice is available, and migrated inward.</p>' },
        { h: 'The Doppler wobble', eq: 'K = 28.4\\ \\text{m s}^{-1}\\;\\frac{M_p\\sin i}{M_J}\\left(\\frac{P}{1\\ \\text{yr}}\\right)^{-1/3}\\left(\\frac{M_\\star}{M_\\odot}\\right)^{-2/3}',
          html: '<p>A planet and its star orbit their common centre of mass, just like a binary star, so the star’s spectrum shifts back and forth. Jupiter moves the Sun at 12.5 m/s; the Earth at 9 cm/s. Heavy planets on short orbits give the biggest signal.</p>' },
        { h: 'Each method has its bias', ex: { id: 'selection' }, eq: 'P_\\text{transit} \\approx \\frac{R_\\star}{a}',
          html: '<p>A transit is seen only by observers near the orbital plane, a fraction \\(R_\\star/a\\) of them, and short periods give many transits to catch. So transit surveys favour big planets close in; Doppler surveys favour heavy planets close in. Microlensing finds planets a few AU out; direct imaging finds young giants far out.</p>' },
        { h: 'Where planets form',
          html: '<p>Planets grow in the disc of gas and dust around a young star. Inside the <b>snow line</b> only rock and metal condense; beyond it ice adds solid material, enough for cores to grow large and capture gas. The <b>habitable zone</b> is the range of distances where a planet could hold liquid water. Correcting for the biases, small planets turn out to be common and hot Jupiters rare: around 1% of Sun-like stars.</p>' }
      ],
      practice: [
        { q: 'What is the chance that a planet at 0.05 AU round a Sun-like star transits for a random observer (R☉ = 0.00465 AU)?', a: 0.093, tol: 0.03, unit: '', sol: '\\(R_\\star/a = 0.00465/0.05 = 0.093\\), about 9%.' },
        { q: 'How large is the Sun’s wobble caused by a 1-Jupiter-mass planet on a 4-day orbit?', a: 128, tol: 0.04, unit: 'm/s', sol: '\\(K = 28.4 \\times (4/365.25)^{-1/3} = 28.4 \\times 4.50 = 128\\) m/s.' }
      ]
    },
    microlens: {
      beats: [
        { h: 'Gravity bends light', sky: { ra: 266.4, dec: -28.9, fov: 45, mode: 'dome', frame: 'gal', photo: true, markers: [mk(269.8946, -29.0638, 'OGLE-2019-BLG-0148')] }, eq: '\\alpha = \\frac{4GM}{c^2 b}',
          html: '<p>General relativity predicts that light passing a mass \\(M\\) at distance \\(b\\) is deflected by \\(\\alpha\\): 1.75″ at the edge of the Sun, twice Newton’s value. Eddington’s eclipse expedition measured it in 1919. Toward the Galactic bulge, where you are looking, millions of stars are lined up behind one another: the best place to catch one star lensing another.</p>' },
        { h: 'The Einstein radius', eq: '\\theta_E = \\sqrt{\\frac{4GM}{c^2}\\,\\frac{D_{LS}}{D_L D_S}},\\qquad R_E = \\theta_E D_L',
          html: '<p>If a source sits exactly behind a lens, its light forms a ring of angular radius \\(\\theta_E\\). For a 0.3 M☉ star halfway to the bulge (\\(D_L = 4\\), \\(D_S = 8\\) kpc), \\(\\theta_E \\approx 0.55\\) mas and \\(R_E \\approx 2.2\\) AU. That is far too small to see the images separately: we only see the total brightness change. Hence <b>micro</b>lensing (HW4 Q2 works the same formula for a whole galaxy).</p>' },
        { h: 'Magnification', ex: { id: 'lens' }, eq: 'A(u) = \\frac{u^2+2}{u\\sqrt{u^2+4}},\\qquad u(t) = \\sqrt{u_0^2 + \\left(\\frac{t-t_0}{t_E}\\right)^2}',
          html: '<p>With the separation \\(u\\) in Einstein radii, the two images together are \\(A(u)\\) times brighter than the source. As the lens drifts past, \\(u\\) falls to its minimum \\(u_0\\) and rises again: a symmetric bump, the same in every colour, because gravity bends all wavelengths equally.</p>' },
        { h: 'What one light curve can tell', eq: 't_E = \\frac{R_E}{v_\\perp} \\approx \\frac{2.2\\ \\text{AU}}{150\\ \\text{km/s}} \\approx 25\\ \\text{d}',
          html: '<p>The only physical number in a simple light curve is \\(t_E\\), and it mixes the lens mass, the two distances and the relative speed (HW3 Q2). A planet round the lens adds a short spike when one of the images passes close to it. Neighbouring stars blended into the same pixel dilute the bump: the source fraction \\(f_s\\).</p>' },
        { h: 'What surveys found', sky: { ra: 269.8946, dec: -29.0638, fov: 0.312, frame: 'equ', photo: true, markers: [mk(269.8946, -29.0638, 'the event’s star')] },
          html: '<p>In the 1990s MACHO, EROS and OGLE monitored millions of stars to test whether the Milky Way’s dark matter is made of compact objects (MACHOs, Lecture 4). They found too few events for that. OGLE now finds about 2,000 events a year and has found dozens of planets this way. This photograph is the crowded field around the star you will fit.</p>' }
      ],
      practice: [
        { q: 'What is the peak magnification of an event with u₀ = 0.1?', a: 10.04, tol: 0.02, unit: '×', sol: '\\(A = (0.01 + 2)/(0.1\\sqrt{4.01}) = 2.01/0.2002 = 10.0\\).' }
      ]
    },
    // =================================================================================== chapter 4
    kirkwood: {
      beats: [
        { h: 'The Sun’s smaller family', sky: { ra: 180, dec: 0, fov: 120, mode: 'dome', frame: 'equ', photo: false, grid: true },
          html: '<p>Between Mars and Jupiter orbit more than a million asteroids; Ceres, the first, was found in 1801. Beyond Neptune lies the Kuiper belt (Pluto, Eris, Makemake, Haumea and thousands of smaller bodies), and much farther the Oort cloud, the source of long-period comets. All move close to the ecliptic, the plane of the planets (Lecture 4).</p>' },
        { h: 'Kepler’s third law', eq: 'P^2\\,[\\text{yr}] = a^3\\,[\\text{AU}]',
          html: '<p>For anything orbiting the Sun, the period in years squared equals the orbit size in AU cubed. An asteroid at 2.5 AU takes \\(2.5^{3/2} = 3.95\\) years: exactly a third of Jupiter’s 11.86.</p>' },
        { h: 'Resonance', ex: { id: 'resonance' }, eq: '\\frac{P}{P_J} = \\left(\\frac{a}{a_J}\\right)^{3/2} = \\frac{p}{q}',
          html: '<p>Jupiter tugs on every asteroid each time it passes. Usually the tugs come from all directions and average out. When the asteroid’s period is a simple fraction of Jupiter’s, it meets Jupiter again and again at the same points of its orbit, the tugs add up, and its orbit is stretched until it crosses Mars’ and is removed.</p>' },
        { h: 'Beyond Neptune',
          html: '<p>Resonances can also protect. Pluto crosses Neptune’s orbit but never meets Neptune, because it goes round twice for every three Neptune orbits (the 3:2 resonance, 39.4 AU). Hundreds of “plutinos” share it. Kuiper-belt objects are too small to see directly at 40 AU unless they pass in front of a star: the occultations of HW4 Q1.</p>' }
      ],
      practice: [
        { q: 'What is the orbit size of an asteroid whose period is exactly 1/3 of Jupiter’s (a_J = 5.20 AU)?', a: 2.50, tol: 0.01, unit: 'AU', sol: '\\(a = 5.20 \\times (1/3)^{2/3} = 5.20 \\times 0.481 = 2.50\\) AU.' }
      ]
    },
    shapley: {
      beats: [
        { h: 'Our galaxy, edge-on', sky: { ra: 266.4, dec: -28.9, fov: 200, mode: 'allsky', frame: 'gal', photo: true },
          html: '<p>This is the whole sky in one photograph, in Galactic coordinates: the Milky Way runs along the middle, with its centre in Sagittarius. We live inside a disk of stars, gas and dust about 30 kpc across, around a central bulge, inside a roughly spherical halo of old stars and globular clusters. The two smudges below the disk to the right are the Magellanic Clouds.</p>' },
        { h: 'Globular clusters', sky: { ra: 250.423, dec: 36.461, fov: 0.312, mode: 'dome', frame: 'equ', photo: true, markers: [mk(250.423, 36.461, 'M13')] }, eq: 'M_V(\\text{RR Lyrae}) \\approx +0.6',
          html: '<p>Globular clusters are among the oldest things in the Galaxy, and bright enough to see across it. Each contains RR Lyrae stars, horizontal-branch pulsators with almost the same luminosity. Shapley used them as standard candles to measure distances to the clusters.</p>' },
        { h: 'Shapley’s argument', ex: { id: 'shapley' },
          html: '<p>The globular clusters orbit the centre of the Galaxy, so their swarm is centred on it. If the Sun were at the centre, they would appear all round the sky. Instead they crowd toward Sagittarius. In 1918 Shapley concluded that the Sun is far out in the disk, and found the centre about 15 kpc away.</p>' },
        { h: 'Dust, and the true distance', sky: { ra: 266.405, dec: -28.936, fov: 22, frame: 'gal', photo: true, markers: [mk(266.405, -28.936, 'Sgr A*')] },
          html: '<p>Shapley’s distance was too large: interstellar dust dims distant clusters, and dimming made them look farther away. Trumpler proved the dust existed in 1930. Today the best distance to the centre is 8.2 kpc, from the orbit of the star S2 round the black hole Sgr A* (GRAVITY 2019). The dark lanes in this photograph are that dust.</p>' }
      ],
      practice: [
        { q: 'RR Lyrae stars (M_V = +0.6) in a cluster appear at V = 15.6, with no dust. How far is the cluster?', a: 10, tol: 0.02, unit: 'kpc', sol: '\\(m - M = 15\\) ⇒ \\(d = 10^{15/5+1} = 10^4\\) pc \\(= 10\\) kpc.' }
      ]
    },
    rotation: {
      beats: [
        { h: 'Speed tells mass', sky: { ra: 154.979, dec: 45.549, fov: 0.195, mode: 'dome', frame: 'equ', photo: true, markers: [mk(154.979, 45.549, 'NGC 3198')] }, eq: '\\frac{v^2}{r} = \\frac{G M(<r)}{r^2} \\;\\Rightarrow\\; M(<r) = \\frac{v^2 r}{G}',
          html: '<p>Gas on a circular orbit moves at the speed that balances the pull of all the mass inside its orbit. Measure the speed at each radius and you have weighed the galaxy, radius by radius. In the Solar System nearly all the mass is in the Sun, so speeds fall as \\(r^{-1/2}\\): Neptune moves at 5 km/s, the Earth at 30.</p>' },
        { h: 'Measuring rotation', sky: { ra: 10.685, dec: 41.269, fov: 2.496, photo: true, markers: [mk(10.685, 41.269, 'M31')] }, eq: 'v = \\frac{v_\\text{obs} - v_\\text{sys}}{\\sin i}',
          html: '<p>Neutral hydrogen emits at 21 cm (a hyperfine transition, Lecture 3) and extends far beyond a galaxy’s visible stars. Its Doppler shift across the disk, corrected for the tilt \\(i\\), gives the rotation speed. Vera Rubin and Kent Ford measured Andromeda’s rotation from 1970 using optical emission lines.</p>' },
        { h: 'Flat curves', ex: { id: 'rotation' },
          html: '<p>If the mass were where the light is, speeds would fall beyond the stars. Instead they stay flat. A flat curve means \\(M(<r) \\propto r\\): the enclosed mass keeps growing where there is almost no light. The usual explanation is a halo of <b>dark matter</b> around every galaxy.</p>' },
        { h: 'Or different gravity?', eq: 'g = \\sqrt{g_N a_0}\\ \\ (g_N \\ll a_0) \\;\\Rightarrow\\; v^4 = G M a_0,\\qquad a_0 \\approx 1.2\\times10^{-10}\\ \\text{m s}^{-2}',
          html: '<p>Milgrom’s MOND (1983) changes the force law instead: below a tiny acceleration \\(a_0\\) gravity weakens more slowly than \\(1/r^2\\). Then speeds stay flat, with \\(v^4 = GMa_0\\), and no halo is needed (HW5). You will test both on NGC 3198.</p>' }
      ],
      practice: [
        { q: 'A galaxy rotates at a flat 220 km/s out to 8 kpc. How much mass lies inside 8 kpc, in units of 10¹⁰ M☉?', a: 9.0, tol: 0.03, unit: '×10¹⁰ M☉', sol: '\\(M = v^2 r/G = (2.2\\times10^5)^2 \\times 2.47\\times10^{20} / 6.674\\times10^{-11} = 1.79\\times10^{41}\\) kg \\(= 9.0\\times10^{10}\\,M_\\odot\\).' }
      ]
    },
    // =================================================================================== chapter 5
    redblue: {
      beats: [
        { h: 'Hubble’s sequence', sky: { ra: 187.706, dec: 12.391, fov: 0.156, mode: 'dome', frame: 'equ', photo: true, markers: [mk(187.706, 12.391, 'M87')] },
          html: '<p>Hubble (1926) sorted galaxies by shape. <b>Ellipticals</b> (E0–E7, by flattening) are smooth and featureless; this is M87, a giant elliptical at the heart of the Virgo cluster. <b>Spirals</b> (Sa–Sd, by how tightly wound and how big the bulge) have disks and arms; barred spirals (SB) have a bar. Lenticulars (S0) are in between; irregulars have no order.</p>' },
        { h: 'A spiral', sky: { ra: 210.802, dec: 54.349, fov: 0.312, photo: true, markers: [mk(210.802, 54.349, 'M101')] },
          html: '<p>M101, the Pinwheel. The arms are blue and knotted with pink star-forming regions; the centre is yellower. A galaxy’s colour is the summed colour of its stars, so it tells us their ages.</p>' },
        { h: 'Colour is age', ex: { id: 'colourmix' },
          html: '<p>Massive blue stars live less than 100 million years. If a galaxy is blue, it has formed stars recently. Once star formation stops, the blue stars die within a few hundred million years, and the galaxy turns red and stays red. Ellipticals are red: their stars are old.</p>' },
        { h: 'Two families, a million galaxies',
          html: '<p>The Sloan Digital Sky Survey measured colours and redshifts for about a million galaxies. Galaxy Zoo (2007) asked volunteers to classify their shapes: more than 100,000 people made tens of millions of classifications. Together they show that galaxies fall into two colour families, a blue cloud and a red sequence, and that colour follows shape only roughly.</p>' }
      ],
      practice: [
        { q: 'Galaxy A has u − r = 2.5 and galaxy B has u − r = 1.5. By what factor is A’s ratio of ultraviolet to red flux smaller than B’s?', a: 2.512, tol: 0.02, unit: '×', sol: 'A difference of 1 magnitude in the colour is a factor of \\(10^{0.4} = 2.512\\) in the flux ratio.' }
      ]
    },
    zwicky: {
      beats: [
        { h: 'Clusters of galaxies', sky: { ra: 194.953, dec: 27.981, fov: 1.248, mode: 'dome', frame: 'equ', photo: true, markers: [mk(194.953, 27.981, 'Coma cluster')] },
          html: '<p>Galaxies gather in groups and clusters. The Coma cluster holds more than a thousand galaxies, most of them ellipticals and lenticulars, about 100 Mpc away. Almost every smudge in this photograph is a galaxy.</p>' },
        { h: 'Redshift and distance', eq: 'z = \\frac{\\Delta\\lambda}{\\lambda},\\qquad v = cz\\ \\ (v \\ll c),\\qquad d = \\frac{v}{H_0}',
          html: '<p>A galaxy’s spectrum is shifted to the red by its recession. Hubble’s law turns that into a distance. Within a cluster, though, the galaxies also move about inside the cluster’s gravity. That spread of speeds around the mean, \\(\\sigma\\), is what weighs the cluster.</p>' },
        { h: 'The virial theorem', ex: { id: 'virial' }, eq: '2K + W = 0 \\;\\Rightarrow\\; M \\approx \\frac{3\\sigma^2 R}{G}',
          html: '<p>For a cluster that has settled, the kinetic energy \\(K\\) and the gravitational energy \\(W\\) balance: twice the one equals minus the other. With the measured line-of-sight speed spread \\(\\sigma\\) and the cluster’s size \\(R\\), that gives the mass needed to hold the galaxies in.</p>' },
        { h: 'Where the mass is',
          html: '<p>Zwicky found in 1933 that Coma needs hundreds of times more mass than its galaxies’ light suggests. Since then, X-ray telescopes have found hot gas between the galaxies, several times the mass of the stars but still not enough. Gravitational lensing of background galaxies by clusters, the research field of your lecturer, measures the total mass directly and confirms it: most of a cluster’s mass is dark.</p>' }
      ],
      practice: [
        { q: 'A cluster has σ = 1000 km/s and R = 1.5 Mpc. What is its virial mass 3σ²R/G, in units of 10¹⁵ M☉?', a: 1.05, tol: 0.03, unit: '×10¹⁵ M☉', sol: '\\(3 \\times (10^6)^2 \\times 4.63\\times10^{22} / 6.674\\times10^{-11} = 2.08\\times10^{45}\\) kg \\(= 1.05\\times10^{15}\\,M_\\odot\\).' }
      ]
    },
    tully: {
      beats: [
        { h: 'Spin from one radio line', sky: { ra: 154.979, dec: 45.549, fov: 0.195, mode: 'dome', frame: 'equ', photo: true, markers: [mk(154.979, 45.549, 'NGC 3198')] }, eq: 'W_{21} \\approx 2\\,V_\\text{flat}\\sin i',
          html: '<p>The 21-cm line from a whole disk galaxy is broadened by its rotation: gas on one side approaches, gas on the other recedes. The width of that single line gives the rotation speed, without any image and without knowing the distance.</p>' },
        { h: 'Why L ∝ V⁴', ex: { id: 'tf' }, eq: 'M = \\frac{V^2 R}{G},\\quad L \\propto M,\\quad R^2 \\propto \\frac{L}{\\Sigma} \\;\\Rightarrow\\; L \\propto \\frac{V^4}{G^2\\Sigma}',
          html: '<p>A faster spin means more mass inside the orbit; more mass means more stars and more light. If disks have similar surface brightness \\(\\Sigma\\), the algebra gives \\(L \\propto V^4\\). Tully and Fisher found this relation empirically in 1977.</p>' },
        { h: 'A distance indicator', eq: 'V \\to L \\to M,\\qquad m - M = 5\\log_{10}d - 5',
          html: '<p>Measure \\(V\\) from the line width, read \\(L\\) (or the absolute magnitude) off the relation, compare with the apparent magnitude, and you have the distance: the same template as Cepheids (HW5). Its calibration comes from galaxies with Cepheid distances.</p>' },
        { h: 'Relatives',
          html: '<p>For ellipticals the equivalent is the <b>Faber–Jackson</b> relation, \\(L \\propto \\sigma^4\\), with the spread of star speeds \\(\\sigma\\) in place of rotation. Adding gas to the stars gives the <b>baryonic</b> Tully–Fisher relation, whose slope is very close to 4, exactly what MOND predicts from \\(V^4 = GMa_0\\).</p>' }
      ],
      practice: [
        { q: 'Galaxy A spins at 300 km/s, galaxy B at 150 km/s. With L ∝ V⁴, how many times more luminous is A?', a: 16, tol: 0.01, unit: '×', sol: '\\(2^4 = 16\\).' }
      ]
    }
  };

  // attach to the missions: the first beat's sky is where the dome turns the moment a mission opens
  OBS.MISSIONS.forEach(function (m) {
    var l = L[m.id]; if (!l) return;
    m.lecture = l.beats; m.practice = l.practice || [];
    m.target = l.beats[0].sky || null;
  });
})(typeof window !== 'undefined' ? window : globalThis);
