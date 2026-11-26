# Observatorium

Learn General Astronomy 1 (BGU 203-1-3461) by retracing its discoveries with real data. A panoramic sky and a
"data lens" show the same catalogue objects. Each mission opens with a short lecture in the observatory (theory,
equations, a live explainer, and the dome turning to what is being discussed, in drawings or real photographs),
then puts you in the place of the astronomer who first made the measurement. Each finished mission unlocks a new
derived quantity (distance, luminosity, temperature, radius, mass…) that then appears as an axis for every dataset.

Why it is built this way, how it maps onto lectures 1–13 and the homework, and what comes next: [DESIGN.md](DESIGN.md).
The November 2026 review of every chapter (bugs, gaps against the slides, ideas): [audit/AUDIT.md](audit/AUDIT.md).

## Use it

- **Published page:** the artifact link from the session (data load per chapter).
- **Offline:** run `node build.mjs`, then open `observatorium.html` (one 37 MB file with every dataset and photograph
  inside; the equations need an internet connection for MathJax).

On a first visit a **two-minute tour** walks through the page: each part is lit up in turn with what it is for and how to
work it, with two small "try it" moments (turn the sky, select in the data lens) and a real lecture step on show. Leave it
any time with Esc; replay it from **Tour** at the top, or from *How this observatory works* under the missions.

| Part | What it does |
|---|---|
| **Logbook** (left) | Nine chapters follow lectures 1–13 (lectures 1–5 are chapters 1–5; lectures 7–13 are chapters 6–9; lecture 6 is not covered). A mission runs in four steps: *Lecture* (4–9 steps of theory in the right-hand pane, with typeset equations, live explainers, sky cues, photographs from Hubble, Webb, ESO and the EHT that enlarge on click, small live charts of real data, and worked examples with the numbers of the homework; the last step lists the variables, the relation and what it lets you deduce) → *Predict* (locked in) → *Exercise* (the page sets up the sky and lens and checks your measurement) → *Explain* (your numbers, what they mean, what really happened, where it is in the course, and practice problems with worked solutions). Also: *Notebook* (what you can measure so far; teacher mode unlocks everything), *Check yourself* (18 questions before and after, with normalised gain), *Data* (every source, cut and citation). |
| **The sky** (middle) | *Drawn* shows catalogue stars as glowing points over Gaia's star counts; *Photo* shows real photographs: ESO's all-sky Milky Way panorama with 26 Digitized Sky Survey, SDSS and Legacy Survey close-ups (among them strong lenses: the Cosmic Horseshoe, the Einstein Cross, Abell 370, Abell 1689, MACS J0416) blended in at full resolution (the credits appear under the sky). **Light** switches the photographed sky to another wavelength: radio (408 MHz), hydrogen (21 cm), the CMB (WMAP), far infrared (IRAS), Hα, X-rays (ROSAT) and gamma rays (EGRET), each a real all-sky map. Drag to turn, scroll to zoom, *Whole sky* for a Hammer map, *Galactic* to put the Milky Way along the equator, *Horizon* to stand at the Wise Observatory in Mitzpe Ramon with ground, twilight and the Sun at a date and time you choose. Shift-drag (or *Select circle*) selects. The **scale ladder** zooms out: sky → solar system → stars → clusters → Milky Way → galaxies → the Universe. |
| **Data lens** (right) | Any two variables of the open dataset, log or linear, or a histogram. Drag a box to select (it lights up on the sky), click a point for its card, scroll to zoom, double-click to fit. Missions add instruments here: a spectrograph, a parallax telescope, a binary-velocity reader, period folding, sliding reference populations and hand-fitted model curves with χ²; from chapter 6 also a redshift fitter for spectra, aperture photometry on a real CCD image, a night planner (altitude, airmass, twilight) and a Markov chain that samples cosmological parameters. |

*Night view* is the real sky; *Plate view* shows it as a photographic negative, as Leavitt saw it on glass plates.

## Missions

### Lectures 1–5

| # | Mission | Real data | Course |
|---|---|---|---|
| 1.1 | A sky without order | Hipparcos naked-eye sky (HYG) | L1 §1.1, §1.4 |
| 1.2 | A year-long wobble (Bessel) | 61 Cygni, Gaia parallax 286.0 mas | L1 §1.4, HW1 Q2 |
| 1.3 | Correct for distance (Hertzsprung, Russell) | Gaia DR3, 36,683 stars within 100 pc | L1 §1.5 |
| 1.4 | Colour is a thermometer (Wien, Planck) | Three Gaia stars with Gaia temperatures | L1 §1.2–1.4, HW1 Q2–3 |
| 1.5 | How big is a white dwarf? (Adams, Eddington) | Gaia 100 pc white dwarfs | L1 §1.5, HW2 Q2 |
| 1.6 | Weighing a star | Gaia DR3 double-lined eclipsing binary | L1 §1.4, HW2 Q4 |
| 1.7 | Mass sets the light (Eddington) | Torres, Andersen & Giménez 2010 | HW2 Q3 |
| 1.8 | A cluster hiding in the field (Trumpler; Hipparcos vs Gaia) | Gaia Pleiades field, 8,547 stars | L1 §1.6, HW1 Q4 |
| 2.1 | Clocks in clusters (Sandage) | Pleiades, Praesepe, M67, NGC 6397 (Gaia) | L2, HW2 Q3, HW3 Q1 |
| 2.2 | Leavitt's law | OGLE Cepheid light curve + 2,750 SMC Cepheids | L2, HW5 |
| 2.3 | A yardstick for the cosmos | Gaia Cepheids: Milky Way parallaxes vs the LMC | L4–5, HW5 |
| 2.4 | Lighthouses (Bell, Backer) | ATNF pulsar catalogue, P–Ṗ | L2 §2.4 |
| 3.1 | A shadow crosses a star | Kepler light curve of HAT-P-7 | L3, HW4 Q1 |
| 3.2 | Who finds which planets | NASA Exoplanet Archive | L3 |
| 3.3 | A star brightens for a month (Paczyński) | OGLE-2019-BLG-0148 | L3–4, HW3 Q2 |
| 4.1 | Gaps in the belt (Kirkwood) | JPL Small-Body Database | L4 |
| 4.2 | Where is the centre? (Shapley) | Harris globular-cluster catalogue | L4 |
| 4.3 | Too fast at the edge (Rubin, Bosma; MOND) | SPARC rotation curve of NGC 3198 | L4, HW5 |
| 5.1 | Red and blue galaxies (Hubble; Galaxy Zoo) | SDSS DR18 + Galaxy Zoo | L5, HW5 |
| 5.2 | Weighing a cluster of galaxies (Zwicky) | SDSS redshifts toward Coma | L5 |
| 5.3 | Spin tells power (Tully, Fisher) | SPARC galaxy table | L5, HW5 |

### Lectures 7–13 (lecture 6 is not part of the Observatorium)

| # | Mission | Real data | Course |
|---|---|---|---|
| 6.1 | Fainter than expected (Perlmutter, Schmidt, Riess) | Pantheon+, 1,466 type Ia supernovae | L7 |
| 6.2 | The afterglow (Penzias and Wilson; COBE) | COBE FIRAS spectrum of the CMB | L7 |
| 6.3 | The shape of space (BOOMERanG, WMAP, Planck) | Planck 2018 TT power spectrum and best-fit model | L7 |
| 6.4 | A forest of clouds (Schmidt, Lynds) | SDSS/BOSS spectrum of a quasar at z = 2.96 | L7 |
| 7.1 | Planning a night (Wise Observatory) | Five real targets (M42, M13, M31, Sgr A*, Pleiades) over the Hipparcos sky; sidereal time, Sun and airmass computed for Mitzpe Ramon | L8 |
| 7.2 | Counting photons | Legacy Surveys DR9 r-band image; SDSS magnitudes of its stars | L9 |
| 8.1 | Where do the bursts come from? (BATSE) | Fermi GBM, 4,430 gamma-ray bursts | L11 |
| 8.2 | Weighing the missing baryons (Lorimer, Macquart) | 183 fast radio bursts (FRBs/FRB compilation), with host redshifts | L10 |
| 8.3 | Hearing black holes merge (LIGO, Virgo, KAGRA) | GWOSC, 282 mergers through GWTC-5 | L11 |
| 9.1 | How far is that galaxy? (Slipher; SDSS) | Three SDSS galaxy spectra (star-forming, elliptical, distant red) | L9, L12 |
| 9.2 | Young stars or a black hole? (Baldwin, Phillips, Terlevich) | 20,000 SDSS galaxies with MPA–JHU line fluxes, masses, star-formation rates, metallicities | L12–13 |
| 9.3 | Let the chain find it (Metropolis) | Pantheon+ again, sampled by MCMC | L13 |

## Build, test, rebuild the data

```
node build.mjs            # dist/index.html + dist/data/ (published page), observatorium.html (offline file)
node --test               # unit tests: physics, CSV reader, every variable on the real data, mission plan
# end-to-end: plays all 33 missions in Chromium and checks each reaches "done"
(cd dist && python3 -m http.server 8765 &) ; node e2e/missions.js http://localhost:8765/preview.html
# the first-visit tour at desktop, tablet and phone widths: every step, both try-its, Tour, Esc, arrows
node e2e/tour.js http://localhost:8765/preview.html
```

The data in `data/` are generated by `pipeline/` (Python 3 with numpy, pandas, astropy):

```
cd pipeline
python3 fetch_gaia.py     # Gaia archive queries (anonymous jobs are limited to 60 s; it retries in pieces)
# other catalogues: the URLs are in data/manifest.json under "source"; save them in pipeline/cache/
python3 make_data.py      # cache/ -> ../data/*.csv, milkyway.json, constellations.json, manifest.json
python3 fetch_images.py   # ESO panorama + DSS/SDSS/Legacy close-ups -> ../data/img/, images.json (run after make_data)
python3 fetch_lecture_media.py  # lecture pictures (ESA/Hubble, ESA/Webb, ESO, NASA, Wikimedia) -> ../data/img/lec/,
                          # and sunspots.csv (SILSO), grbs.csv (Fermi GBM), spectra.csv (SDSS) (run after fetch_images)
python3 fetch_universe.py # chapters 6-9: Pantheon+, COBE FIRAS, Planck, SDSS spectra and MPA-JHU lines, Fermi GBM
                          # positions, FRBs, GWOSC, a Legacy Surveys image, SkyView all-sky maps, lecture pictures
                          # (needs matplotlib for the colour maps; run last, it rewrites manifest.json entries)
```

`data/manifest.json` records, for every file, the source, the query, the cuts, the citation and the licence.
Lecture pictures carry their credit under each figure (`data/lecture_images.json`): ESA/Hubble, ESA/Webb and ESO images
are CC BY 4.0; NASA and the 1920 eclipse plate are public domain. The SILSO sunspot series is CC BY-NC 4.0 (fine for
study, not for commercial use). From chapter 6: Pantheon+ is a public release; COBE FIRAS and Fermi GBM are NASA (public
domain); Planck data are from the ESA Planck Legacy Archive (free use with acknowledgement of ESA and the Planck
Collaboration); SDSS and the Legacy Surveys ask for their standard acknowledgement; the FRB compilation comes from the
FRBs/FRB repository (BSD-3); GWOSC data are CC BY 4.0; the all-sky maps come through NASA SkyView, and each survey
(Haslam 408 MHz, HI4PI, WMAP, IRIS, Finkbeiner Hα, ROSAT, EGRET) asks to be acknowledged, as the credits under the sky do.
One caution: **OGLE asks to be contacted before its Early Warning System photometry is used in publications.**
Using it for private study is fine; ask OGLE before sharing a page with their light curves publicly.

## Code

| File | Role |
|---|---|
| `src/js/core.js` | Namespace, storage, events, CSV parsing, data loading (inline or fetched) |
| `src/js/astro.js` | The physics: coordinates, magnitudes, Planck, Gaia colour calibrations, orbits, lensing, transits, rotation curves, pulsars; cosmological distances and ages, the CMB, FRB dispersion; sidereal time, the Sun, altitude and airmass |
| `src/js/catalog.js` | Every dataset and variable; which concept unlocks each derived variable |
| `src/js/chart.js` | The data lens |
| `src/js/sky.js` | The sky: stereographic dome and Hammer all-sky, equatorial, Galactic or horizon; drawn or photographed, in eight wavelengths |
| `src/js/instruments.js` | Controls; spectrograph, parallax telescope, binary reader, folding, model fitting, sliding references |
| `src/js/observing.js` | Chapter 6–9 instruments: redshift fitter, aperture photometry, night planner, sky isotropy, supernova χ², MCMC |
| `src/js/missions.js` | The 33 missions, the chapters and the concept check |
| `src/js/lectures.js`, `lectures2.js` | Each mission's lecture (theory, equations, sky cues, figures, live charts, worked examples) and practice problems, for chapters 1–5 and 6–9 |
| `src/js/lecture.js` | Runs a lecture in the right-hand pane, mounts its pictures (with credits and a lightbox) and charts, and turns the sky with it |
| `src/js/explainers.js`, `explainers2.js` | 33 live diagrams used in the lectures |
| `src/js/logbook.js` | Mission runner, notebook, quiz, data sources |
| `src/js/tour.js` | The first-visit tour: spotlight, step cards, try-it checks |
| `src/js/main.js` | Wiring, controls, object cards, theme |
| `pipeline/` | Data download and preparation |
