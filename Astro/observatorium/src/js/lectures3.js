/* Observatorium — lecture steps added after the November 2026 audit: topics the slides teach that the first
 * lectures skipped, with their key numbers. Each entry is inserted after the named step of a mission's lecture
 * (or at the end), and extra practice problems are appended. Loaded after lectures.js and lectures2.js. */
(function (G) {
  'use strict';
  var OBS = G.OBS;
  function mk(ra, dec, label) { return { ra: ra, dec: dec, label: label }; }

  var ADD = {
    // ============================================================================== chapter 1 (lecture 1)
    thermometer: {
      beats: [
        { after: 'Blackbodies', h: 'Light and the windows of the air',
          eq: 'c = \\lambda \\nu, \\qquad E = h\\nu',
          html: '<p>Light is an electromagnetic wave: oscillating electric and magnetic fields, at right angles to each other and to the direction of travel, moving at \\(c\\) in vacuum, and carrying energy in packets \\(E = h\\nu\\). The spectrum runs from gamma rays (picometres) through X-rays, ultraviolet, visible (400–700 nm), infrared and microwaves to radio waves metres long. The atmosphere lets through only two broad windows, visible light and radio (about 1 cm to 10 m), plus a few narrow infrared ones; everything else needs mountains, balloons or space. Our eyes see 400–700 nm because the Sun is brightest there and the air is clear there: chapter 8 opens the other windows.</p>' },
        { after: 'Which limit you are in', h: 'A blackbody fit does not give the distance',
          eq: '\\frac{F_\\nu}{\\pi B_\\nu(T)} = \\left(\\frac{R}{d}\\right)^2',
          html: '<p>Fitting a star’s spectrum with a blackbody gives its temperature from the shape. The height of the fit gives only the ratio \\(R/d\\), the star’s angular size: a star twice as big and twice as far looks identical. To get either one you need the other, from a parallax (chapter 1) or a measured radius (an eclipsing binary, interferometry). This is why the colour cancels size and distance, and why the next missions need parallaxes.</p>',
          example: '<p>The Sun: \\(R_\\odot / 1\\,\\text{AU} = 6.96\\times10^8 / 1.496\\times10^{11} = 4.65\\times10^{-3}\\), so \\((R/d)^2 = 2.2\\times10^{-5}\\): the flux we receive is \\(2.2\\times10^{-5}\\) of the flux leaving its surface, \\(\\pi B\\).</p>' },
        { after: 'The spectral sequence', h: 'What else a spectrum tells',
          html: '<p>Beyond temperature and composition: the <b>width</b> of a line measures how fast the star spins (one limb approaches, the other recedes) and how dense its atmosphere is (giants have narrower lines than dwarfs of the same temperature, which is how luminosity classes are read). The <b>shape</b> of a line shows gas flowing in or out (P Cygni profiles of stellar winds). A magnetic field splits lines into several components, the <b>Zeeman effect</b>, which measures sunspot fields of a few thousand gauss. And the shift of the whole pattern gives the velocity along the line of sight (the Doppler effect of mission 1.6).</p>' }
      ],
      practice: [
        { q: 'A star’s blackbody fit gives (R/d)² = 10⁻¹⁸. Its parallax puts it at 10 pc. What is its radius, in solar radii? (1 pc = 3.086×10¹⁶ m, R☉ = 6.96×10⁸ m)', a: 0.443, tol: 0.03, unit: 'R☉', sol: '\\(R/d = 10^{-9}\\), so \\(R = 10^{-9} \\times 3.086\\times10^{17}\\ \\text{m} = 3.09\\times10^{8}\\) m \\(= 0.44\\,R_\\odot\\): a red dwarf.' }
      ]
    },
    hrd: {
      beats: [
        { after: 'Hertzsprung and Russell', h: 'First drawn for clusters',
          html: '<p>The diagram was found twice. Hertzsprung (1911) plotted the stars of the Pleiades and the Hyades: every star in a cluster is at the same distance, so apparent and absolute magnitude differ by the same constant and the pattern appears <i>without any parallax</i> (mission 1.8 uses exactly this). Russell (1913) used the few stars with parallaxes. Both saw that most stars lie on one band, the main sequence, with giants above it.</p><p>Spectra separate the branches too: the Yerkes (MK) <b>luminosity classes</b> add a Roman numeral to the spectral type, from V (dwarfs, the main sequence) through IV (subgiants), III (giants), II (bright giants) to I (supergiants). The Sun is G2 V, Arcturus K1.5 III, Rigel B8 Ia. Cooler than M come the brown dwarfs, types L and T.</p>' }
      ],
      practice: [
        { q: 'In a cluster, star A looks 5 magnitudes brighter than star B. How many times more luminous is A?', a: 100, tol: 0.01, unit: '×', sol: 'All members are at the same distance, so the difference in apparent magnitude is the difference in absolute magnitude: 5 magnitudes is a factor \\(10^{0.4 \\times 5} = 100\\) in luminosity.' }
      ]
    },
    radius: {
      beats: [
        { after: 'Lines of equal size', h: 'Hot stars are big',
          eq: 'L \\propto T^{8}\\ \\text{(main sequence)} \\;\\Rightarrow\\; R^2 = \\frac{L}{4\\pi\\sigma T^4} \\propto T^4 \\;\\Rightarrow\\; R \\propto T^2',
          html: '<p>Along the main sequence luminosity rises roughly as \\(T^8\\), so the radius grows as \\(T^2\\): hotter main-sequence stars are also bigger. The smallest stars, at about half the Sun’s temperature, are about a quarter of its radius; the hottest, at five times, about 25 times. Off the main sequence the range is far wider: giants reach about 1 AU (200 R☉) and supergiants several AU, while a white dwarf is the size of the Earth, about 10⁴ km.</p>',
          example: '<p>A main-sequence star twice as hot as the Sun: \\(R \\approx 2^2 = 4\\,R_\\odot\\) and \\(L = R^2 T^4 = 16 \\times 16 = 256\\,L_\\odot\\), which is indeed \\(2^8\\).</p>' }
      ],
      practice: [
        { q: 'A main-sequence star is 3 times hotter than the Sun. With R ∝ T², how many times more luminous is it?', a: 6561, tol: 0.01, unit: 'L☉', sol: '\\(R = 9\\,R_\\odot\\), so \\(L = R^2 T^4 = 81 \\times 81 = 6561\\,L_\\odot = 3^8\\).' }
      ]
    },
    masslum: {
      beats: [
        { h: 'A sequence of masses',
          html: '<p>The main sequence is not, as was first thought, a cooling track that stars slide down: it is a sequence of <b>masses</b>, each star sitting where its mass puts it for most of its life. It runs from about 0.08 M☉ to about 100 M☉. Below 0.08 M☉ the core never gets hot enough to fuse hydrogen: a brown dwarf, glowing only from its contraction. Above about 100 M☉ the pressure of radiation, which grows as \\(T^4\\), overwhelms the gas pressure, which grows only as \\(T\\), and the star blows off its outer layers. The exponent in \\(L \\propto M^a\\) is not one number either: close to 4 near a solar mass, smaller for red dwarfs and for the most massive stars. Your fitted slope is an average over the stars you used.</p>' }
      ]
    },

    // ============================================================================== chapter 2 (lecture 2)
    turnoff: {
      beats: [
        { after: 'How a star ages', h: 'After the main sequence, step by step',
          html: '<p>When the core hydrogen runs out, a Sun-like star becomes a <b>subgiant</b>, then climbs the <b>red giant branch</b> while hydrogen burns in a shell (about 10% of its main-sequence life). At 10⁸ K and about 10³ g/cm³ the core ignites helium in a <b>helium flash</b>; the star settles on the <b>horizontal branch</b> (under 1% of its life), burning helium into carbon and oxygen. With two shells burning, it swells again on the <b>asymptotic giant branch</b>, pulses, and sheds its envelope as a planetary nebula, leaving a carbon–oxygen white dwarf that cools for ever (a “black dwarf” would take longer than the age of the Universe). Stars above about 8 M☉ go on to burn carbon, neon, oxygen and silicon, up to iron.</p>',
          example: '<p><b>The lecturer’s scaling.</b> With \\(L \\propto M^4\\), a 10 M☉ star is \\(10^4\\) times as luminous and holds 10 times the fuel, so it lives \\(10/10^4 = 1/1000\\) as long as the Sun: about 10 Myr. A 0.1 M☉ star lives about a thousand times longer than the Sun. So an old population, whose massive stars are long gone, looks red.</p>' },
        { after: 'What makes stars shine', h: 'Where the elements come from',
          html: '<p>Hydrogen and helium (and a trace of lithium) come from the Big Bang (chapter 6). Stars make the rest. Helium burns to carbon by the <b>triple-alpha</b> process, then oxygen, neon, magnesium, silicon, and fusion stops at <b>iron</b>, the most tightly bound nucleus. Heavier elements are built by adding neutrons: slowly in AGB stars (the <b>s-process</b>: barium, lead), and rapidly in explosive events (the <b>r-process</b>: gold, platinum, uranium), mostly in merging neutron stars (GW170817 in chapter 8) and some rare supernovae. The very first stars, <b>Population III</b>, formed from pure hydrogen and helium; none has been found.</p>' },
        { after: 'How stars end', h: 'How much energy a supernova releases',
          eq: 'E_K = \\tfrac{1}{2} M v^2 = \\tfrac{1}{2} (3 \\times 2\\times10^{30}\\ \\text{kg}) (10^7\\ \\text{m/s})^2 \\approx 3\\times10^{44}\\ \\text{J} = 3\\times10^{51}\\ \\text{erg}',
          html: '<p>A core-collapse supernova ejects a few solar masses at about 10,000 km/s. Only about 1% of that kinetic energy comes out as light, yet for weeks the star outshines its galaxy. Most of the collapse energy, about 100 times more again, leaves as neutrinos: Kamiokande and IMB caught 25 of them from SN 1987A. Type II supernovae (collapsing massive stars) show hydrogen and a plateau in their light curves; type Ia (exploding white dwarfs, chapter 6) show no hydrogen and a fast, standard decline.</p>' }
      ],
      practice: [
        { q: 'With L ∝ M⁴ and the Sun living 10 Gyr, how long does a 10 M☉ star live, in Myr?', a: 10, tol: 0.02, unit: 'Myr', sol: '\\(t \\propto M/L = M^{-3}\\): \\(10\\ \\text{Gyr} \\times 10^{-3} = 10\\) Myr.' },
        { q: 'A supernova ejects 3 M☉ at 10⁴ km/s. What is the kinetic energy, in units of 10⁴⁴ J? (M☉ = 1.989×10³⁰ kg)', a: 2.98, tol: 0.03, unit: '×10⁴⁴ J', sol: '\\(\\tfrac12 \\times 5.97\\times10^{30} \\times (10^7)^2 = 2.98\\times10^{44}\\) J \\(= 3\\times10^{51}\\) erg.' }
      ]
    },
    pulsars: {
      beats: [
        { after: 'A lighthouse', h: 'Too fast for a white dwarf',
          eq: '\\frac{GM}{R^2} \\ge \\Omega^2 R \\;\\Rightarrow\\; \\rho = \\frac{M}{\\tfrac43 \\pi R^3} \\ge \\frac{3\\pi}{G P^2}',
          html: '<p>A star spinning with period \\(P\\) holds together only if gravity at its equator beats the centrifugal pull. That sets a minimum mean density, whatever the mass. For the Crab pulsar, \\(P = 33\\) ms, it is \\(1.3\\times10^{14}\\) kg/m³. A white dwarf, at about \\(10^{9}\\) kg/m³, would fly apart; only a neutron star, as dense as an atomic nucleus (\\(2\\times10^{17}\\) kg/m³), survives. That argument, from a period alone, is how the 1968 pulsars were identified.</p>',
          example: '<p>The fastest known pulsar, PSR J1748−2446ad, spins in 1.40 ms: \\(\\rho \\ge 3\\pi/(6.67\\times10^{-11} \\times (1.40\\times10^{-3})^2) = 7.2\\times10^{16}\\) kg/m³, a third of nuclear density.</p>' },
        { after: 'Spinning down', h: 'A pulsar’s life on the P–Ṗ chart', ex: { id: 'pulsarlife' },
          html: '<p>A pulsar is born at the upper left of the P–Ṗ diagram, spinning fast and slowing fast (the Crab). As it ages it moves right, along lines of constant magnetic field, crossing the lines of characteristic age. When the voltage its spin can generate drops too low (roughly \\(B/P^2\\) below a threshold) it stops shining: the <b>death line</b>, beyond which lies the pulsar graveyard. <b>Magnetars</b>, at the top right, have fields of \\(10^{14}\\)–\\(10^{15}\\) gauss and shine from the decay of their fields, not their spin. A companion can later spin a dead pulsar up again to milliseconds: the recycled pulsars at the bottom left.</p>' },
        { after: 'Black holes', h: 'Clocks and tides near a black hole',
          eq: '1 + z = \\frac{1}{\\sqrt{1 - R_s/r}}, \\qquad \\Delta a \\approx \\frac{2GM}{r^3}\\,\\ell',
          html: '<p>A clock at distance \\(r\\) from a black hole ticks slower, by the factor above, than one far away, and its light arrives redshifted by the same factor. At \\(r = R_s\\) the factor is infinite: a falling object seems to freeze at the horizon and fade away exponentially. The difference of gravity across a body of length \\(\\ell\\), the <b>tide</b>, grows as \\(1/r^3\\): near a stellar-mass black hole it would stretch a person long before the horizon; near a supermassive one the horizon is passed without noticing.</p>',
          example: '<p>Light emitted at \\(r = 4R_s\\) arrives with \\(1 + z = 1/\\sqrt{1 - 1/4} = 1.155\\): a 120 MHz signal is received at 104 MHz.</p>' }
      ],
      practice: [
        { q: 'What is the minimum mean density of the Crab pulsar (P = 33.4 ms), in units of 10¹⁴ kg/m³?', a: 1.27, tol: 0.03, unit: '×10¹⁴ kg/m³', sol: '\\(3\\pi/(G P^2) = 9.425/(6.674\\times10^{-11} \\times 1.116\\times10^{-3}) = 1.27\\times10^{14}\\) kg/m³.' },
        { q: 'A signal emitted at 120 MHz from r = 4 R_s is received far away at what frequency, in MHz?', a: 103.9, tol: 0.01, unit: 'MHz', sol: '\\(120\\sqrt{1 - 1/4} = 120 \\times 0.866 = 103.9\\) MHz.' }
      ]
    },
    yardstick: {
      beats: [
        { after: 'The distance ladder', h: 'A ring of light around SN 1987A',
          eq: 'd = \\frac{R}{\\theta} = \\frac{c\\,t}{\\theta}',
          fig: { img: 'sn1987a', caption: 'SN 1987A in the LMC, with the ring of gas the flash lit up (Hubble).' },
          html: '<p>An independent rung to the LMC. The flash of SN 1987A lit up a ring of gas thrown off by the star thousands of years before. The ring brightened about 250 days after the explosion: that delay, times \\(c\\), is the ring’s physical radius. Hubble measured its angular radius, 0.85″. Their ratio is the distance, with no ladder at all. Detached eclipsing binaries, whose radii come from their light and velocity curves, give 49.6 kpc to 1%; Cepheids must agree with both.</p>',
          example: '<p>\\(R = c \\times 250\\ \\text{d} = 3.0\\times10^8 \\times 2.16\\times10^7\\ \\text{s} = 6.5\\times10^{15}\\) m \\(= 0.21\\) pc. \\(\\theta = 0.85″ = 4.1\\times10^{-6}\\) rad. \\(d = 0.21/4.1\\times10^{-6} = 51\\) kpc.</p>' }
      ],
      practice: [
        { q: 'A light echo ring around a supernova brightens 250 days after the explosion and has an angular radius of 0.85″. How far away is it, in kpc?', a: 50.9, tol: 0.03, unit: 'kpc', sol: '\\(R = c\\,t = 6.48\\times10^{15}\\) m \\(= 0.210\\) pc; \\(\\theta = 0.85/206265 = 4.12\\times10^{-6}\\) rad; \\(d = 0.210/4.12\\times10^{-6} = 5.09\\times10^4\\) pc.' }
      ]
    },

    // ============================================================================== chapter 3 (lecture 3)
    transit: {
      beats: [
        { after: 'Period and orbit', h: 'How long a transit lasts, and how hot the planet is',
          eq: ['T_\\text{dur} \\approx \\frac{P}{\\pi}\\,\\frac{R_\\star}{a}', 'T_\\text{eq} = T_\\star \\sqrt{\\frac{R_\\star}{2a}}\\,(1 - A)^{1/4}'],
          html: '<p>A planet crossing the middle of its star covers a path of length \\(2R_\\star\\) out of its orbit’s \\(2\\pi a\\), so the transit lasts the fraction \\(R_\\star/(\\pi a)\\) of the period (shorter if it crosses off-centre: the impact parameter). The same closeness heats it: a planet that absorbs a fraction \\(1 - A\\) of its star’s light and re-radiates it over its whole surface settles at the equilibrium temperature above. Hot Jupiters at 0.05 AU from a Sun-like star reach about 1,250 K; the Earth, at 1 AU, 278 K (255 K with its albedo of 0.3; the greenhouse effect adds the rest).</p>',
          example: '<p>A 3-day orbit round a Sun-like star: \\(a = 0.041\\) AU, \\(R_\\star/a = 0.114\\), so a central transit lasts \\(3\\ \\text{d} \\times 0.114/\\pi = 0.109\\) d \\(\\approx 2.6\\) h.</p>' }
      ],
      practice: [
        { q: 'A hot Jupiter orbits a Sun-like star (5772 K, 1 R☉) at 0.05 AU. What is its equilibrium temperature with no albedo, in K? (R☉ = 6.96×10⁸ m, 1 AU = 1.496×10¹¹ m)', a: 1245, tol: 0.02, unit: 'K', sol: '\\(5772\\sqrt{6.96\\times10^8/(2 \\times 7.48\\times10^9)} = 5772 \\times 0.216 = 1245\\) K.' }
      ]
    },
    census: {
      beats: [
        { after: 'The Doppler wobble', h: 'How small a wobble',
          eq: 'K = \\left(\\frac{2\\pi G}{P}\\right)^{1/3} \\frac{m_p \\sin i}{M_\\star^{2/3}}',
          html: '<p>A planet and its star circle their common centre of mass, so the star moves at a speed \\(K\\) along our line of sight. Jupiter makes the Sun wobble at 12.5 m/s with a 12-year period; the same planet at 1 AU would give 28 m/s; the Earth gives 9 cm/s. The best spectrographs (HARPS, ESPRESSO) now reach about 1 m/s and below, enough for super-Earths close to small stars. Above about 13 Jupiter masses an object burns deuterium: the conventional line between a planet and a <b>brown dwarf</b>.</p>' },
        { after: 'Where stars are born', h: 'Gas between the stars',
          eq: 'R_S = \\left(\\frac{3 Q}{4\\pi n^2 \\alpha_B}\\right)^{1/3}',
          html: '<p>A hot O star ionises the gas around it out to the <b>Strömgren radius</b>, where recombinations use up all its ionising photons: an <b>H II region</b> at about 10⁴ K, glowing in recombination lines (Hα gives the Orion Nebula its red). Its green comes from <b>forbidden lines</b> of doubly ionised oxygen, [O III] at 495.9 and 500.7 nm, once blamed on a new element, “nebulium”: in gas this thin, ions excited by collisions have time to decay by transitions too slow to compete in a laboratory. Elsewhere the interstellar medium is cold atomic hydrogen, seen at 21 cm; molecular clouds, traced by CO; and dust, which dims and reddens starlight (most strongly in the blue: interstellar reddening) and glows in the far infrared.</p>',
          example: '<p>An O star emitting \\(Q = 10^{49}\\) ionising photons per second into gas of \\(n = 10^4\\) cm⁻³ (\\(\\alpha_B = 2.6\\times10^{-13}\\) cm³/s): \\(R_S^3 = 3\\times10^{49}/(4\\pi \\times 10^8 \\times 2.6\\times10^{-13}) = 9.2\\times10^{52}\\) cm³, \\(R_S = 4.5\\times10^{17}\\) cm \\(= 0.15\\) pc.</p>' }
      ],
      practice: [
        { q: 'What radial-velocity amplitude does a Jupiter-mass planet (1.898×10²⁷ kg) on a 1-year circular orbit produce on a Sun-like star (1.989×10³⁰ kg), seen edge-on, in m/s?', a: 28.4, tol: 0.03, unit: 'm/s', sol: '\\((2\\pi G/P)^{1/3} = (2\\pi \\times 6.674\\times10^{-11}/3.156\\times10^7)^{1/3} = 2.37\\times10^{-6}\\); \\(M_\\star^{2/3} = 1.58\\times10^{20}\\); \\(K = 2.37\\times10^{-6} \\times 1.898\\times10^{27}/1.58\\times10^{20} = 28.4\\) m/s.' },
        { q: 'The same O star (Strömgren radius 0.15 pc at 10⁴ cm⁻³) sits in gas of 100 cm⁻³. What is its Strömgren radius, in pc?', a: 3.2, tol: 0.05, unit: 'pc', sol: '\\(R_S \\propto n^{-2/3}\\): \\(0.15 \\times (100)^{2/3} = 0.15 \\times 21.5 = 3.2\\) pc.' }
      ]
    },
    microlens: {
      beats: [
        { after: 'What surveys found', h: 'Could the halo be made of MACHOs?',
          eq: '\\tau \\sim \\frac{v^2}{c^2} \\sim 10^{-6}',
          html: '<p>If the Milky Way’s dark halo were made of compact objects (MACHOs: dead stars, brown dwarfs, planets), at any moment a fraction \\(\\tau\\) of distant stars would be lensed. For a halo in balance, \\(\\tau\\) is about \\((v/c)^2\\) with \\(v \\approx 200\\) km/s: a few in a million. Watching \\(10^7\\) stars in the Large Magellanic Cloud for five years should then show about 100 events lasting half a year each if MACHOs weigh 1 M☉, or about 2,600 lasting a week if they weigh as much as Jupiter (\\(t_E \\propto \\sqrt{M}\\)). MACHO, EROS and OGLE found only a handful, of 0.1–1 M☉: compact objects make at most a few percent of the halo. Dark matter is something else (chapter 4).</p>' }
      ],
      practice: [
        { q: 'What is (v/c)² for v = 200 km/s, in units of 10⁻⁷?', a: 4.44, tol: 0.02, unit: '×10⁻⁷', sol: '\\((2\\times10^5/3\\times10^8)^2 = (6.67\\times10^{-4})^2 = 4.4\\times10^{-7}\\).' }
      ]
    },

    // ============================================================================== chapter 4 (lecture 4)
    kirkwood: {
      beats: [
        { after: 'Beyond Neptune', h: 'Seeing a 1 km body at 40 AU',
          eq: 'F = \\sqrt{\\frac{\\lambda a}{2}}',
          html: '<p>Most Kuiper-belt objects are far too small and faint to see. They can be detected when one passes in front of a distant star: an <b>occultation</b>. But a body only a kilometre across is smaller than the <b>Fresnel scale</b>, so it casts a diffraction pattern rather than a sharp shadow, and the dip lasts a fraction of a second as the Earth sweeps through it at about 25 km/s. Finding them needs small, distant stars and fast photometry (HW4 Q1).</p>',
          example: '<p>In green light, \\(\\lambda = 500\\) nm, at \\(a = 40\\) AU \\(= 6.0\\times10^{12}\\) m: \\(F = \\sqrt{5\\times10^{-7} \\times 6.0\\times10^{12}/2} = 1.2\\) km. Swept at 25 km/s, the shadow passes in \\(1.2/25 \\approx 0.05\\) s.</p>' }
      ],
      practice: [
        { q: 'What is the Fresnel scale for λ = 500 nm at 40 AU, in km? (1 AU = 1.496×10¹¹ m)', a: 1.22, tol: 0.03, unit: 'km', sol: '\\(\\sqrt{5\\times10^{-7} \\times 5.98\\times10^{12}/2} = \\sqrt{1.50\\times10^6} = 1.22\\times10^3\\) m.' }
      ]
    },
    shapley: {
      beats: [
        { after: 'Dust, and the true distance', h: 'The Milky Way in numbers',
          eq: 'P = \\frac{2\\pi R_0}{v_\\odot}, \\qquad M(<R_0) = \\frac{v_\\odot^2 R_0}{G}',
          html: '<p>The Sun orbits the centre at about 240 km/s, 8.2 kpc out. One orbit, a <b>Galactic year</b>, takes about 210 million years: the Sun has gone round about 22 times. The mass inside its orbit is about \\(10^{11}\\) M☉. The disk of stars thins out exponentially, with a scale length of about 2.5 kpc and a thickness of a few hundred parsecs (thinner for gas and young stars, thicker for old ones); the Sun sits about 25 pc above the mid-plane. Spiral arms are not fixed sets of stars but <b>density waves</b>, traffic jams that stars move through, where gas piles up and forms new stars: H II regions trace them. The central bulge, about 1 kpc across, holds old stars; around it lie the halo and some 160 globular clusters.</p>',
          example: '<p>\\(P = 2\\pi \\times 8.2 \\times 3.086\\times10^{19}\\ \\text{m} / 2.4\\times10^5\\ \\text{m/s} = 6.6\\times10^{15}\\) s \\(= 210\\) Myr. \\(M = (2.4\\times10^5)^2 \\times 2.53\\times10^{20}/6.67\\times10^{-11} = 2.2\\times10^{41}\\) kg \\(= 1.1\\times10^{11}\\,M_\\odot\\).</p>' }
      ],
      practice: [
        { q: 'With R₀ = 8.2 kpc and v = 240 km/s, how long is a Galactic year, in Myr?', a: 210, tol: 0.03, unit: 'Myr', sol: '\\(2\\pi \\times 2.530\\times10^{20}\\ \\text{m} / 2.4\\times10^{5}\\ \\text{m/s} = 6.62\\times10^{15}\\) s \\(= 2.10\\times10^8\\) yr.' }
      ]
    },
    rotation: {
      beats: [
        { after: 'Flat curves', h: 'A halo ten times the disk',
          eq: 'M(<r) \\propto r \\;\\Rightarrow\\; \\rho(r) = \\frac{1}{4\\pi r^2}\\frac{dM}{dr} \\propto r^{-2}',
          html: '<p>Rotation curves are measured from the Doppler shifts of Hα and other optical lines in the inner parts, the 21 cm line of hydrogen far out, and CO in molecular gas. In the Milky Way and most spirals they stay flat to 30 kpc and more, about ten disk scale lengths. A flat curve needs a mass growing in proportion to radius, which means a density falling as \\(r^{-2}\\): an isothermal halo, containing 80–90% of the mass, about \\(10^{12}\\) M☉ for the Milky Way.</p><p>Could it be ordinary matter we missed? Atomic gas would shine at 21 cm, molecular gas in CO, hot ionised gas in X-rays, dust in the far infrared, and none of these adds up. Dead stars would have spread their metals; brown dwarfs and planets were ruled out by microlensing (chapter 3); neutrons decay in 15 minutes; neutrinos are too light and fast (“hot”) to let galaxies form. What is left is a new kind of particle: WIMPs, axions or something else, still unseen.</p>' },
        { after: 'Or different gravity?', h: 'How small is a₀?',
          html: '<p>MOND changes gravity only where accelerations fall below \\(a_0 \\approx 1.2\\times10^{-10}\\) m/s². Compare: at the Earth’s surface, 9.8 m/s²; the Earth round the Sun, \\(6\\times10^{-3}\\) m/s²; Neptune round the Sun, \\(7\\times10^{-6}\\) m/s². Only the Sun’s own orbit round the Galaxy, \\(v^2/R = 2.3\\times10^{-10}\\) m/s², comes close to \\(a_0\\). That is why no experiment in the Solar System could have noticed it, and why galaxies are where the question is decided.</p>' }
      ],
      practice: [
        { q: 'What is the Sun’s acceleration round the Galaxy (v = 240 km/s, R = 8.2 kpc), in units of 10⁻¹⁰ m/s²?', a: 2.28, tol: 0.03, unit: '×10⁻¹⁰ m/s²', sol: '\\(v^2/R = (2.4\\times10^5)^2/2.53\\times10^{20} = 2.28\\times10^{-10}\\) m/s², about twice \\(a_0\\).' }
      ]
    },

    // ============================================================================== chapter 5 (lecture 5)
    redblue: {
      beats: [
        { after: 'Galaxies collide', h: 'How often galaxies collide',
          html: '<p>Stars inside a galaxy almost never collide; galaxies do. Take galaxies of radius 50 kpc moving at 500 km/s through a space holding 0.01 galaxies per Mpc³. A galaxy hits another when their centres pass within about two radii, a target area \\(\\pi (0.1\\ \\text{Mpc})^2 = 0.031\\) Mpc². In 10 billion years it travels about 5 Mpc and sweeps \\(0.031 \\times 5 = 0.16\\) Mpc³, which holds \\(0.0016\\) galaxies: about one galaxy in 600 has had a collision. In a cluster the density is \\(10^4\\) times higher and collisions come every few billion years, which is why clusters are full of ellipticals.</p>' },
        { after: 'Active galactic nuclei', h: 'Why the engine must be a black hole',
          eq: 'L = \\eta\\,\\dot M c^2, \\qquad \\eta_\\text{fusion} = 0.007, \\quad \\eta_\\text{accretion} \\approx 0.06\\text{–}0.4',
          html: '<p>Radio jets a megaparsec long need the engine to run for at least \\(10^6\\)–\\(10^7\\) years; quasars shine at \\(10^{39}\\)–\\(10^{40}\\) W from a region light-days across (they vary in days). Fusion turns only 0.7% of rest mass into energy; matter falling onto a black hole radiates 6% to 40%. So only accretion can supply a quasar’s energy without needing tens of billions of solar masses of fuel. Seyfert galaxies, radio galaxies, blazars (BL Lac objects) and quasars are the same engine seen with different power and from different angles: the <b>unified model</b>, in which a dusty torus hides the broad-line region (gas at \\(10^4\\) km/s) from some directions and a jet points at us in blazars.</p>',
          example: '<p>A quasar at \\(10^{39}\\) W for \\(10^8\\) yr emits \\(3.2\\times10^{54}\\) J. By fusion that needs \\(3.2\\times10^{54}/(0.007\\,c^2) = 5\\times10^{39}\\) kg \\(= 2.5\\times10^9\\,M_\\odot\\) of hydrogen; by accretion at \\(\\eta = 0.1\\), \\(1.8\\times10^8\\,M_\\odot\\).</p>' }
      ],
      practice: [
        { q: 'A quasar shines at 10³⁹ W for 10⁸ years with an accretion efficiency of 0.1. How much mass does it swallow, in units of 10⁸ M☉? (1 yr = 3.156×10⁷ s, M☉ = 1.989×10³⁰ kg)', a: 1.76, tol: 0.03, unit: '×10⁸ M☉', sol: '\\(E = 10^{39} \\times 3.156\\times10^{15} = 3.16\\times10^{54}\\) J; \\(M = E/(0.1\\,c^2) = 3.51\\times10^{38}\\) kg \\(= 1.76\\times10^8\\,M_\\odot\\).' }
      ]
    },
    zwicky: {
      beats: [
        { after: 'Where the mass is', h: 'The hot gas between the galaxies',
          html: '<p>X-ray telescopes see that rich clusters are filled with gas at \\(10^7\\)–\\(10^8\\) K (1–10 keV), shining by bremsstrahlung. There is about ten times more of it than mass in stars. Assuming the gas is in hydrostatic balance in the cluster’s gravity gives a mass independent of the galaxies’ speeds, and gravitational lensing (chapter 3) a third. All three agree: about 15% of a cluster’s mass is ordinary matter, close to the cosmic ratio \\(\\Omega_b/\\Omega_m\\) of chapter 6, and the rest is dark. Clusters also grow: distant clusters hold more blue, star-forming galaxies than nearby ones (the Butcher–Oemler effect).</p>' }
      ]
    },
    tully: {
      beats: [
        { after: 'How old is the Universe?', h: 'A clock in the rocks, and no centre',
          eq: 't = \\frac{\\ln\\left[(N_{235}/N_{238})_0 \\big/ (N_{235}/N_{238})\\right]}{\\lambda_{235} - \\lambda_{238}}',
          html: '<p>Two independent checks of the Hubble time. First, the elements themselves: uranium-235 decays faster (half-life 0.70 Gyr) than uranium-238 (4.47 Gyr), and today their ratio is 0.0072. If they were made in roughly equal amounts, the uranium on Earth was forged about 6 billion years ago, a lower limit to the age of the Galaxy, and with several generations of stars about 10 billion years. Second, the Hubble law has no centre: if \\(v = H_0 d\\) holds for us, then for any other galaxy the difference \\(v_1 - v_2 = H_0 (d_1 - d_2)\\), so every observer sees the same law. Expansion is of space itself, not an explosion from one place.</p>',
          example: '<p>\\(\\lambda_{235} = 0.693/0.704 = 0.985\\) /Gyr, \\(\\lambda_{238} = 0.693/4.468 = 0.155\\) /Gyr. \\(t = \\ln(1/0.0072)/0.830 = 4.93/0.830 = 5.9\\) Gyr.</p>' }
      ],
      practice: [
        { q: 'If ²³⁵U and ²³⁸U were made in equal amounts and their ratio is 0.0072 today, how long ago were they made, in Gyr? (half-lives 0.704 and 4.468 Gyr)', a: 5.94, tol: 0.03, unit: 'Gyr', sol: '\\(\\ln(1/0.0072) = 4.93\\); \\(\\lambda_{235} - \\lambda_{238} = 0.985 - 0.155 = 0.830\\) /Gyr; \\(t = 5.94\\) Gyr.' }
      ]
    },

    // ============================================================================== chapter 6 (lecture 7)
    accel: {
      beats: [
        { after: 'Cosmological redshift', h: 'Two wave crests in an expanding Universe',
          eq: '\\frac{\\lambda_\\text{obs}}{\\lambda_\\text{emit}} = \\frac{a(t_\\text{obs})}{a(t_\\text{emit})} = 1 + z',
          html: '<p>Picture two successive crests of a light wave leaving a galaxy. Both travel at \\(c\\) through space that keeps stretching. Each crest’s path is the same in comoving coordinates, so the time between their arrivals is stretched by the same factor as space itself: the interval between crests, the wavelength, grows in proportion to the scale factor \\(a(t)\\). The redshift is not a Doppler shift through space: it measures how much the Universe has expanded while the light was on its way. Clocks there seem to run slow by the same factor: distant supernovae rise and fade \\(1 + z\\) times more slowly.</p>' },
        { after: 'The surprise of 1998', h: 'From slowing down to speeding up',
          eq: 'q(z) = \\frac{\\Omega_m (1+z)^3/2 - \\Omega_\\Lambda}{E^2(z)} = 0 \\;\\Rightarrow\\; 1 + z = \\left(\\frac{2\\Omega_\\Lambda}{\\Omega_m}\\right)^{1/3}',
          html: '<p>Matter thins out as \\((1+z)^3\\) while a cosmological constant stays the same, so the Universe changed character. Early on, matter dominated and the expansion slowed; since then, dark energy has taken over and it speeds up. With Ωm = 0.315 and ΩΛ = 0.685 the switch came at z = 0.63, about 6 billion years ago, when the Universe was a little over half its present age. Supernovae up to z ≈ 2, when the Universe was a quarter of its age, show both eras.</p>',
          example: '<p>\\(1 + z = (2 \\times 0.685/0.315)^{1/3} = 4.35^{1/3} = 1.632\\): \\(z = 0.63\\).</p>' }
      ],
      practice: [
        { q: 'At what redshift did the expansion switch from slowing down to speeding up, for Ωm = 0.3 and ΩΛ = 0.7?', a: 0.671, tol: 0.02, unit: '', sol: '\\(1 + z = (2 \\times 0.7/0.3)^{1/3} = 4.667^{1/3} = 1.671\\).' }
      ]
    },
    cmb: {
      beats: [
        { after: 'Found by accident', h: 'One frequency, one temperature',
          eq: 'T_b = \\frac{c^2}{2 k \\nu^2}\\, I_\\nu \\quad \\text{(Rayleigh–Jeans)}',
          html: '<p>Penzias and Wilson measured at a single frequency, 4.08 GHz, far down the long-wavelength side of a 3 K blackbody, where intensity is proportional to temperature. Because the radiation filled the whole sky, its intensity per steradian could be compared directly with a blackbody’s: unlike a star’s flux, it does not depend on any distance or size. One number, an excess of about 3.5 K over what the antenna and the air explained, gave the temperature. COBE’s FIRAS later measured the whole curve (your exercise).</p>' },
        { after: 'More photons than anything', h: 'A redshifted blackbody stays a blackbody', ex: { id: 'cmbz' },
          html: '<p>As the Universe expands by a factor \\(a\\), the number of photons per volume falls as \\(a^{-3}\\) and each photon’s energy as \\(a^{-1}\\), its wavelength stretching with space. A Planck spectrum at temperature \\(T\\) then becomes exactly a Planck spectrum at \\(T/a\\): the shape is kept, only the temperature falls. That is why the CMB, emitted at about 3,000 K, is still a perfect blackbody at 2.725 K, and why \\(T = T_0 (1 + z)\\) at every redshift, a prediction since confirmed with molecules in distant gas clouds.</p>' }
      ]
    },
    peaks: {
      beats: [
        { after: 'What else the peaks weigh', h: 'Seeds older than the horizon',
          html: '<p>At recombination, sound had travelled only about 150 Mpc (comoving), yet the CMB shows coherent peaks: the waves all started in step, as if their seeds were laid down everywhere at once, even on scales no signal could yet have crossed. That is what inflation predicts: quantum fluctuations stretched beyond the horizon in the first \\(10^{-32}\\) s. Dark matter, which does not feel the photons’ pressure, began clumping right then; ordinary matter could only fall into those clumps after recombination released it. Without dark matter there would not yet be galaxies. To vary the peaks yourself, as the lecture suggests, try the Planck power-spectrum simulator (chrisnorth.github.io/planckapps/Simulator).</p>' }
      ]
    },
    forest: {
      beats: [
        { after: 'Light that passed through everything', h: 'Why the forest is not a wall',
          eq: '\\tau_{GP} \\approx 1.8\\times10^{5}\\, x_\\text{HI} \\left(\\frac{1+z}{4}\\right)^{3/2}',
          html: '<p>Neutral hydrogen absorbs Lyα so strongly that even one atom in \\(10^{4}\\) (\\(\\tau \\approx 20\\)) would black out everything blueward of a quasar’s Lyα line: the <b>Gunn–Peterson trough</b>. Quasars below z ≈ 6 show only a forest of separate lines, so the gas between galaxies must be almost completely ionised; only above z ≈ 6 do troughs appear. The light of the first galaxies and quasars <b>reionised</b> the Universe in its first billion years.</p>' }
      ]
    },

    // ============================================================================== chapter 7 (lectures 8-9)
    night: {
      beats: [
        { after: 'Coordinates and time', h: 'Clocks for astronomers',
          html: '<p>Several clocks are in use. <b>UTC</b> is atomic time (TAI) held within a second of the Earth’s rotation (UT1) by leap seconds. The <b>Julian Date</b> counts days continuously since 4713 BC, and the modified one, MJD = JD − 2,400,000.5, keeps the numbers small. Because light takes up to 8.3 minutes to cross the Earth’s orbit, timings of variable stars are corrected to the Sun: the <b>heliocentric</b> (or barycentric) Julian date. The Earth’s motion also tilts incoming starlight by up to 20.5″, the <b>aberration of light</b>, and its axis precesses once in 26,000 years, which is why catalogue positions are given for an epoch, J2000. Israel keeps UTC+2, and UTC+3 in summer.</p>' }
      ]
    },
    photometry: {
      beats: [
        { after: 'Magnitude systems', h: 'Per hertz or per nanometre, and why surface brightness is constant',
          eq: ['f_\\lambda = \\frac{c}{\\lambda^2} f_\\nu', 'I = \\frac{F}{\\Omega} \\propto \\frac{L/d^2}{A/d^2} = \\text{constant}'],
          html: '<p>A spectrum can be given per unit frequency, \\(f_\\nu\\) (Jy), or per unit wavelength, \\(f_\\lambda\\): the same light, spread over intervals of different width, so the two shapes differ and peak in different places. The AB system is flat in \\(f_\\nu\\); the Vega system sets Vega to zero in every band, so the two agree near V but differ by almost 2 magnitudes in the near infrared (K: AB − Vega = 1.85). And a key intuition: the brightness per unit area of an extended object, its <b>surface brightness</b>, does not change with distance (in a static space): flux falls as \\(1/d^2\\), but so does the solid angle it covers. A galaxy twice as far is four times fainter and four times smaller on the sky, but its pixels are just as bright. In an expanding Universe it falls as \\((1+z)^{-4}\\): Tolman’s test (chapter 9).</p>' },
        { after: 'Signal and noise', h: 'Counts, electrons and the gain',
          html: '<p>A CCD reports counts (ADU), not photons. The <b>gain</b>, a few electrons per count, converts one into the other; only electrons follow Poisson statistics. So a star that gives 10,000 counts with a gain of 2 e⁻/ADU has collected 20,000 electrons, with a noise of \\(\\sqrt{20{,}000} = 141\\) e⁻: S/N ≈ 141 from the star alone, before the sky and the read noise add theirs. Saturated pixels (beyond the <b>full well</b>, about 100,000 e⁻) and cosmic rays (rejected by comparing several exposures, or by sigma clipping) must be masked.</p>' },
        { after: 'Spectroscopy', h: 'What the slit and the grating do',
          eq: 'm \\lambda = d\\,(\\sin \\alpha + \\sin \\beta), \\qquad R = \\frac{\\lambda}{\\Delta\\lambda}',
          html: '<p>The slit keeps out the light of neighbouring stars and of the sky, and sets the resolution: its image on the detector is the narrowest line the spectrograph can show. The grating sends each wavelength into a different angle \\(\\beta\\) for each order \\(m\\). A low-resolution spectrograph (\\(R \\sim 1000\\), like SDSS’s at about 2,000) measures redshifts and broad features; separating the sodium doublet (589.0 and 589.6 nm) needs \\(R \\approx 1000\\); measuring a planet’s 1 m/s wobble needs \\(R \\sim 100{,}000\\) in an echelle, whose overlapping high orders are separated by a second disperser at right angles.</p>' }
      ],
      practice: [
        { q: 'A star gives 10,000 counts at a gain of 2 e⁻/ADU. Counting only the star’s own photon noise, what is its S/N?', a: 141, tol: 0.02, unit: '', sol: '\\(N = 20{,}000\\) electrons; \\(S/N = N/\\sqrt N = \\sqrt{20{,}000} = 141\\).' },
        { q: 'What resolving power R = λ/Δλ separates the sodium lines at 589.0 and 589.6 nm?', a: 982, tol: 0.03, unit: '', sol: '\\(R = 589.3/0.6 = 982\\), about 1,000.' }
      ]
    },

    // ============================================================================== chapter 8 (lectures 10-11)
    bursts: {
      beats: [
        { after: 'Infrared', h: 'How high must you go?',
          html: '<p>Water vapour in the lower atmosphere absorbs most infrared light. From a dry 4 km mountain (Mauna Kea, the Atacama) a few near- and mid-infrared windows open (J, H, K, L, M bands, and submillimetre ones for ALMA). Above 12 km, in an aircraft (SOFIA, 2010–2022), 99% of the water is below you and the far infrared opens. Balloons at 40 km reach further still. X-rays and gamma rays are absorbed high in the atmosphere, so they need rockets and satellites: the first cosmic X-ray source, Sco X-1, was found by a rocket in 1962.</p>' },
        { after: 'X-rays', h: 'Soft and hard, and the first black hole',
          html: '<p>X-rays of 0.1–2 keV (wavelengths above about 1 nm) are called soft; above 2–10 keV, hard. Soft X-rays can be focused by grazing-incidence mirrors (Chandra, XMM-Newton, eROSITA); hard ones only by special multilayer optics (NuSTAR) or not at all. Cygnus X-1, an X-ray source in a binary with a 15 M☉ blue supergiant, was in 1971 the first good black-hole candidate: its unseen companion weighs about 20 M☉, far above any neutron star. Hot cluster gas, supernova remnants (Cas A, Tycho) and accreting compact objects dominate the X-ray sky.</p>' },
        { after: 'Gamma rays', h: 'Two kinds of burst, and how gamma rays are made',
          html: '<p>Gamma rays come from the most violent processes: particle–antiparticle annihilation (the 511 keV line), radioactive decay (²⁶Al traced by COMPTEL across the Galaxy), inverse Compton scattering of light by fast electrons, and pion decay in cosmic-ray collisions. The Compton Gamma Ray Observatory (1991–2000) carried four instruments: BATSE, OSSE, COMPTEL and EGRET (271 sources). BATSE found that bursts come in two kinds: <b>short</b> ones, under about 2 s, with harder spectra, and <b>long</b> ones. Long bursts follow the collapse of massive stars (they sit in star-forming regions, and some arrive with supernovae); short ones come from merging neutron stars, as GW170817 proved in 2017 (chapter 8.3). Try a histogram of T₉₀ in the gamma-ray burst dataset: the two humps are there.</p>' }
      ]
    },
    gw: {
      beats: [
        { after: 'LIGO, Virgo and KAGRA', h: 'Twenty years to the first chirp',
          html: '<p>Initial LIGO ran from 2002 to 2010 and found nothing: the expected rate at its sensitivity was well under one event a year. Advanced LIGO, ten times more sensitive and so reaching a thousand times the volume, caught GW150914 within days of starting in September 2015. A second kind of gravitational wave is sought by very different means: waves from inflation would have left a twisting, “B-mode”, pattern in the CMB’s polarisation, not found so far. And pulsar timing arrays, watching dozens of millisecond pulsars (chapter 2), have found a background of nanohertz waves, probably from pairs of supermassive black holes.</p>' }
      ]
    },

    grbtwo: {
      full: {
        beats: [
          { h: 'A burst every day', sky: { ra: 266.4, dec: -28.9, fov: 200, mode: 'allsky', frame: 'gal', photo: true, band: 'gamma' },
            html: '<p>About once a day, somewhere in the sky, a gamma-ray burst outshines every other gamma-ray source for a few seconds. Mission 8.1 showed they come evenly from every direction, so they are far beyond the Milky Way: billions of light-years. To be seen so far they release, in seconds, as much energy as the Sun does in its whole life, beamed into narrow jets.</p>' },
          { h: 'Counting by duration',
            fig: { chart: { ds: 'grbs', x: 't90', y: '#', bins: 34, colourBy: 'kind', view: { x0: 0.01, x1: 1000 } }, caption: 'Fermi GBM bursts counted by T₉₀ on a logarithmic axis, coloured by the usual 2 s divide.' },
            html: '<p>The duration \\(T_{90}\\) is the time over which 90% of a burst’s counts arrive. Counted on a logarithmic axis, the durations do not spread evenly: they fall into two groups, with a dip between them near 2 s. The short bursts also have harder spectra (more high-energy photons). Two groups that differ in two independent properties suggest two kinds of source.</p>' },
          { h: 'Two engines',
            html: '<p>A <b>long</b> burst is the death of a massive star: its core collapses to a black hole while a jet punches out through the star in tens of seconds. A <b>short</b> burst is the merger of two neutron stars (or a neutron star and a black hole): the debris falls in within a fraction of a second. The test came on 17 August 2017, when a short burst arrived 1.7 s after the gravitational waves of a neutron-star merger.</p>',
            example: '<p>A burst at 1 Gpc with a fluence of \\(10^{-5}\\) erg/cm² released, if it shone equally in all directions, \\(4\\pi d^2 F = 4\\pi (3.1\\times10^{27}\\ \\text{cm})^2 \\times 10^{-5} = 1.2\\times10^{51}\\) erg: a supernova’s worth of energy in gamma rays alone, in seconds. Beaming into jets a few degrees wide lowers the true energy a hundredfold.</p>' }
        ],
        practice: [
          { q: 'A burst at 1 Gpc delivers a fluence of 10⁻⁵ erg/cm². Assuming it shone equally in all directions, how much energy did it release, in units of 10⁵¹ erg? (1 Gpc = 3.086×10²⁷ cm)', a: 1.2, tol: 0.03, unit: '×10⁵¹ erg', sol: '\\(4\\pi (3.086\\times10^{27})^2 \\times 10^{-5} = 1.20\\times10^{51}\\) erg.' }
        ]
      }
    },

    // ============================================================================== chapter 9 (lectures 12-13)
    redshift: {
      beats: [
        { after: 'Galaxies in numbers', h: 'How many galaxies of each brightness',
          eq: '\\phi(L)\\,dL = \\phi^* \\left(\\frac{L}{L^*}\\right)^{\\alpha} e^{-L/L^*}\\,\\frac{dL}{L^*}',
          html: '<p>Schechter’s function counts galaxies per volume and per luminosity: a power law of slope \\(\\alpha \\approx -1\\) at the faint end, cut off exponentially above a characteristic \\(L^*\\), about the Milky Way’s. It is hard to measure: faint galaxies are seen only nearby (Malmquist bias), so each luminosity needs its own volume. At higher redshift \\(L^*\\) was brighter and the faint end steeper, and even at z ≈ 10 no turnover is seen at the faint end. Ellipticals come in kinds: giant ellipticals, cD galaxies at the centres of clusters, with huge extended envelopes, and dwarfs. Cluster members can be picked out by colour alone: the old ellipticals of a cluster form a tight <b>red sequence</b>.</p>' },
        { after: 'Reading a spectrum', h: 'The spectrograph behind the spectra',
          html: '<p>SDSS fed each galaxy’s light into a fibre 3″ across (2″ for BOSS), 640 or 1,000 fibres at once, plugged into holes drilled at each object’s position. Their spectra have a resolving power of about 2,000: lines closer than about 3 Å blend, but redshifts are measured to better than 10 km/s. The fibre also explains a subtlety of chapter 9.2: for nearby galaxies it captures only the centre, so star formation rates are corrected for the light outside it.</p>' }
      ]
    },
    bpt: {
      beats: [
        { after: 'Two sequences', h: 'Measuring the metals in the gas',
          eq: 'R_{23} = \\frac{[\\text{O II}]\\,3727 + [\\text{O III}]\\,4959, 5007}{\\text{H}\\beta}, \\qquad N2 = \\log\\frac{[\\text{N II}]\\,6584}{\\text{H}\\alpha}',
          html: '<p>The oxygen abundance of the gas, 12 + log(O/H) (the Sun: 8.69), is read from ratios of its emission lines. Each indicator has a range where it works: \\(R_{23}\\) is double-valued; N2 and O3N2 saturate at high metallicity; different calibrations disagree by up to 0.5 dex. Plotted against mass, metallicity rises and flattens (the mass–metallicity relation). At a fixed mass, galaxies forming stars faster are poorer in metals: the <b>fundamental metallicity relation</b>, explained by inflowing pristine gas that both dilutes the metals and fuels new stars. Colour the mass–metallicity plane by star formation rate to see it.</p>' }
      ]
    },
    chain: {
      beats: [
        { after: 'Fitting a model', h: 'Local minima', ex: { id: 'costfn' },
          html: '<p>A misfit like χ² can have several valleys. A method that always walks downhill (gradient descent, Newton’s method) stops in whichever valley it starts in: a <b>local minimum</b>, not necessarily the best. A grid finds the global one, but costs \\(n^d\\) evaluations for \\(d\\) parameters (written as nested loops: <code>for x … for y … cost(x, y)</code>), so it is used only for one to three parameters; random or adaptive grids help a little. A Markov chain sometimes steps uphill, so it can climb out of a shallow valley, and it spends its time in proportion to probability rather than searching everywhere. Comparing two models, rather than fitting one, needs more: the evidence, the probability of the data under each model.</p>' },
        { after: 'Errors without formulas', h: 'Bootstrap and jackknife, by example',
          html: '<p>Suppose you measured the heights of 30 students and want the uncertainty of their mean. The <b>bootstrap</b>: draw 30 heights at random from your 30, with replacement (some appear twice, some not at all), take the mean, and repeat a thousand times; the spread of those means is the error. The <b>jackknife</b>: leave out one student at a time, compute 30 means of 29, and scale their spread by \\(\\sqrt{n-1}\\). Both work for any statistic, a median, a slope or a correlation, with no formula.</p>' },
        { after: 'Big data and machine learning', h: 'A neuron and a convolution',
          eq: ['y = f\\!\\left(\\sum_i w_i x_i + b\\right)', '(I * K)(x, y) = \\sum_{i,j} I(x - i, y - j)\\,K(i, j)'],
          html: '<p>A single artificial neuron multiplies its inputs by weights, adds them, and passes the sum through a non-linear function \\(f\\) (for example, zero below 0, unchanged above). A network stacks layers of them; training adjusts the weights to minimise a cost function, by the same gradient descent as above. Networks for images use <b>convolutions</b>: a small kernel, say 3 × 3, slides over the image and at each pixel sums the products. A kernel of \\(-1, 0, 1\\) across finds vertical edges; one of equal ninths blurs. A convolutional network learns its own kernels, which is how galaxies are classified by shape and lenses found among millions of images.</p>' }
      ]
    }
  };

  // explainers and pictures added to existing steps
  var SET = { hrd: [['Absolute magnitude', 'tenpc']], microlens: [['The Einstein radius', 'lensbuild']] };
  var FIGS = {
    radius: [['Sirius B', 'sirius_b', 'Sirius A and its white dwarf companion Sirius B, the faint dot at lower left (Hubble): ten thousand times fainter, yet hotter.']],
    yardstick: [['Island universes', 'hubble_var', 'Hubble’s 1923 plate of Andromeda, where he crossed out “N” (nova) and wrote “VAR!”: the Cepheid V1 that put M31 outside the Milky Way.']],
    pulsars: [['What a supernova leaves', 'crab', 'The Crab Nebula (Hubble), the debris of the supernova of 1054; at its centre spins the Crab pulsar, 30 times a second.']],
    turnoff: [['How stars end', 'wr124', 'WR 124 (Webb): a Wolf–Rayet star shedding its outer layers in dusty knots before it explodes.']],
    redshift: [['Deep fields', 'xdf', 'The Hubble eXtreme Deep Field: about 5,500 galaxies in a patch a tenth of the full Moon’s width, some seen 13 billion years ago.']],
    bpt: [['Star formation rates', 'm82', 'M82 (Hubble): a starburst galaxy. The red filaments are hydrogen glowing in Hα, blown out by thousands of supernovae.']],
    forest: [['Lensed quasars', 'h0licow', 'HE 0435−1223 (Hubble): one quasar seen four times around a lensing galaxy. The delays between the images measure H₀ (H0LiCOW).']],
    night: [['Telescopes', 'wise_obs', 'The Wise Observatory near Mitzpe Ramon, Israel’s optical observatory, at 875 m in the Negev.'], ['Resolution', 'neptune_ao', 'Neptune from the VLT with adaptive optics (MUSE/GALACSI): sharper than the atmosphere would allow.']],
    thermometer: [['Light and the windows of the air', 'opacity', 'How far each wavelength gets through the atmosphere: visible light and radio reach the ground; most of the rest needs space.']]
  };
  OBS.MISSIONS.forEach(function (m) {
    (SET[m.id] || []).forEach(function (p) { (m.lecture || []).forEach(function (b) { if (b.h === p[0] && !b.ex) b.ex = { id: p[1] }; }); });
    (FIGS[m.id] || []).forEach(function (p) {
      (m.lecture || []).concat((ADD[m.id] && ADD[m.id].beats) || []).forEach(function (b) {
        if (b.h !== p[0] || b._fig) return; b._fig = true;
        var f = { img: p[1], caption: p[2] }; b.fig = b.fig ? [].concat(b.fig, f) : f;
      });
    });
    var a = ADD[m.id]; if (!a) return;
    if (a.full) { m.lecture = a.full.beats; m.practice = a.full.practice || []; m.target = a.full.beats[0].sky || null; return; }
    if (!m.lecture) return;
    (a.beats || []).forEach(function (b) {
      var at = -1;
      if (b.after) for (var k = 0; k < m.lecture.length; k++) if (m.lecture[k].h === b.after) at = k + 1;
      var beat = {}; Object.keys(b).forEach(function (key) { if (key !== 'after') beat[key] = b[key]; });
      if (at < 0) m.lecture.push(beat); else m.lecture.splice(at, 0, beat);
    });
    m.practice = (m.practice || []).concat(a.practice || []);
  });
  OBS.LECTURE_ADDITIONS = ADD;      // for the tests
})(typeof window !== 'undefined' ? window : globalThis);
