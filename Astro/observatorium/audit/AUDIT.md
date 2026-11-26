# Observatorium audit

26 November 2026. Ten reviewers each took one part of the game: chapters 1 to 9, plus the parts that every chapter
shares (tour, logbook, sky, data lens, lecture pane, quiz, layout). Each reviewer:

- played every mission through the menus, buttons, sliders and drags a student uses, with only the earlier missions
  done and teacher mode off;
- tested at 1600×950, 1100×800 and 390×844 (phone);
- compared each lecture with its slides (lectures 1–5 and 7–13), the tutorials and the homework;
- re-did the arithmetic of the worked examples and the practice problems.

The automated test (`e2e/missions.js`) had passed all 33 missions. It solves them by calling JavaScript directly,
which goes around locked menus, so it could not see most of what is below.

- [BUGS.md](BUGS.md): all 181 bug entries, with steps, the file and line, and a fix, most severe first.
- [reports/](reports/): the ten full reports, one per part, mission by mission. Each has gaps against the slides and
  suggestions.
- [tools/](tools/): the brief every reviewer followed and the Playwright harness that loads the page as a student who
  has done the earlier missions (reusable for a UI-driven test).
- [shots/](shots/): screenshots of the blocker and major bugs. The other screenshots and the reviewers' test
  scripts were left in the session's scratch space and are not kept here.

| Part | Blocker | Major | Minor | Cosmetic | Total |
|---|---|---|---|---|---|
| 1 · Lecture 1 (stars) | 1 | 4 | 12 | 3 | 20 |
| 2 · Stellar evolution, Cepheids, pulsars | 0 | 5 | 9 | 4 | 18 |
| 3 · Exoplanets, microlensing | 0 | 2 | 13 | 0 | 15 |
| 4 · Solar system, Milky Way, dark matter | 0 | 5 | 9 | 7 | 21 |
| 5 · Galaxies and clusters | 0 | 1 | 8 | 5 | 14 |
| 6 · The expanding Universe | 0 | 1 | 9 | 6 | 16 |
| 7 · The observer's toolkit | 0 | 2 | 7 | 4 | 13 |
| 8 · Every kind of light | 0 | 3 | 16 | 3 | 22 |
| 9 · Reading galaxies | 0 | 4 | 11 | 5 | 20 |
| Shared parts | 1 | 2 | 16 | 3 | 22 |
| **All** | **2** | **29** | **110** | **40** | **181** |

The two blockers are the same bug, found by two reviewers. A few other bugs also appear twice.

## The short version

- **One mission cannot be finished, and it blocks four more.** Mission 1.3 *Correct for distance* says "Switch the
  vertical axis to Absolute magnitude M_G". That option is locked, because 1.3 itself is the mission that unlocks it.
  The concept it would unlock (inverse square → luminosity) is needed by 1.5, 1.8, 2.1 and 2.3. So without teacher
  mode a student stops at 1.3. A check of all 33 missions found that this is the **only** locked item that any task,
  setup or lecture chart asks for. Every other mission can be finished through the UI.
- **Several missions check the answer badly.**

  | Mission | Problem |
  |---|---|
  | 9.1 | Passes if you sweep the redshift slider through the right values |
  | 9.3 | Is done after one click of *Run* |
  | 7.2 | Is done with three clicks at the default aperture |
  | 8.3 | Can be finished with nothing radiated |
  | 9.2 | A correct line scores 1% or 99% depending on which handle is on top |
  | 5.3 | Rejects the data's own best-fit slope |
  | 2.2 | The period window is too narrow to reach with a mouse |

  Elsewhere the *What does it mean?* button stays enabled after the answer has become wrong again.
- **Changing what is on screen during an exercise often breaks it.** Switching dataset or axis mid-exercise
  removes the lines and model curves. It can throw an error every 400 ms, or give a result built on the wrong
  data. Instruments from a finished mission keep running into the next one: 7.2 opens with the sky stuck on 7.1's
  night planner, and in 9.3 a click on the χ² map shows 7.2's CCD image.
- **The physics in the lectures is right.** The worked examples and the practice answers check out. The
  exceptions are a dozen numbers or captions that contradict the screen or each other (listed under each chapter).
  Two datasets need an honest label:
  - most transiting-planet masses are estimates from the radius;
  - white-dwarf temperatures are capped at 11,590 K by a calibration that does not apply to them.
- **The biggest gains for understanding are a handful of new views.** The reviewers keep asking for the same
  things:
  - the student's point on a χ² map;
  - a trace or a periodogram instead of a fiddly slider;
  - an animation that turns a formula into a picture, such as moving every star to 10 pc, a pulsar ageing on the
    P–Ṗ chart, the acoustic peaks forming, or the forest turning into a Gunn–Peterson trough.

  They also ask for lectures that show the slides' key figures and numbers. Each chapter's list is below.

## Ten root causes

Fixing these clears most of the major bugs and many of the minor ones.

1. **Unlocks arrive only when a mission ends.** `logbook.js` `finish()` sets the concepts, so the mission that
   teaches a variable cannot use it. This causes the 1.3 blocker: `ch1-B1`, `shell-B1`, plus `wd` (catalog.js:95).
   *Fix:* lend a mission its own `unlocks` from the start of its exercise. Add a unit test that checks every
   variable each task, setup and lecture chart uses against the course order.
   Related: some unlocks open nothing.
   - *Transit depth gives size* (3.1, `shell-B7`, `ch3-B7`) and *Period gives luminosity* (2.2, `ch2-B10`).
   - Chapters 4, 5, 7 and 8 unlock nothing at all. 4.2 could open R_gc, 4.3 M(<R), 2.2 a Leavitt distance, and 3.1
     planet radius and density.
2. **Instruments are never stopped.** Each instrument has `st.stop()`, but no code calls it. Their ResizeObservers,
   timers and `#imgview` click handlers live on into later missions (`ch7-B1`, `ch9-B18`). *Fix:* keep the current
   instrument on `cur.st`. Call its `stop()` in `app.reset()` and `LB.leave()`, and clear `#imgview.onclick`.
3. **Exercises do not survive a change of dataset or axis.** Overlays are cleared by `chart.setDataset`, but
   `progress` assumes they exist.
   - Affected: `ch4-B6` (all chapter-4 missions), `ch5` tully, `ch2-B8`, `ch1-B14`, `ch3` (folding goes off
     silently).
   - The model fitter computes χ² on whatever axes are shown (`ch6-B9`: χ²/N of 10⁵ when *Up* becomes μ).
   *Fix:* while an exercise needs a fixed dataset and axes, disable those menus (with a tooltip), or re-run `setup`
   when the student returns. Capture the columns at setup. Guard every `progress` against a missing overlay.
4. **Answer checks.** The cases are in the table under *The short version*. *Fix:*
   - Add a *Record this value* step where a slider sweep could pass (9.1). Start the chain with a badly chosen step
     (9.3). Start 7.2 at a 2 px aperture.
   - Define the AGN side of the BPT line by a fixed point (9.2).
   - Set the windows from fits to the shipped data: the 5.3 upper slope limit to about 5.3, the 2.2 fine-tune to at
     least one coarse step.
   - Re-check `progress` when *What does it mean?* is pressed.
5. **Galactic views point at RA 0, Dec 0.** `goTo(0, 0, …, {frame:'gal'})` turns the sky to l ≈ 96°. The readout
   still says "centre of the Galaxy in the middle". This happens in 6.2, 6.3, 6.4, 8.1, 8.2, 8.3 and 9.3
   (`ch8-B1`). Switching to *Galactic* in whole-sky mode does the same (`shell-B10`). *Fix:* `goTo(266.4, -28.9, …)`,
   as `lectures2.js` already does.
6. **An unescaped `<`.** `M(<r)` is inserted as HTML in four places in mission 4.3 (`ch4-B1`–`B4`). It breaks the
   first equation, loses half a paragraph, swallows the explainer readout and cuts the relation shown on the
   Predict screen. *Fix:* escape text when it is inserted (`lecture.js:24`, `:115`, `explainers.js:463`), or write
   `\lt`.
7. **The data lens.**
   - A box selection lets points with no x value through (`chart.js:454`, `ch3-B1`). It gives 3.2 the wrong
     transit share.
   - The autoscale trims the extreme 0.2%, which cuts Sirius and Rigel out of 1.1 (`ch1-B3`). It also squashes
     the asteroid and galaxy views (`shell-B15`).
   - The hover text wraps the read bar and resizes the canvas, so a line handle jumps away as the pointer
     reaches it (`ch2-B9`). This affects every hand-fitted line.
   - There is no log/lin switch, although the README promises one (`shell-B6`).
8. **The sky reloads its photos endlessly.** In the whole-sky visible view, six close-ups qualify for drawing, but
   the cache holds only five. Each load evicts another, about 10 loads and re-renders a second while idle
   (`ch8-B2`, `sky.js:200`). *Fix:* skip close-ups smaller than a few pixels, or enlarge the cache.
9. **Phone and laptop layouts.**
   - Pinch does nothing: there is no handler, and the canvases set `touch-action: none` (`shell-B16`).
   - On a phone the lecture opens about 2,000 px below the card, and the canvases swallow page scrolls
     (`shell-B21`).
   - At 1100×800 the sky is about 170 px tall, and instruments and their controls do not fit together
     (`shell-B22`, `ch9-B19`, and chapters 1 and 8).
10. **Data that needs an honest label or a better source.**
    - Planet masses are estimated from the radius for 64% of transiting planets (`ch3-B8`). Keep
      `pl_bmassprov` and offer a colour-by.
    - The white-dwarf temperature cap (`ch1-B11`).
    - The 1.3 exercise opens on the 100 pc sample, which already shows a sharp HR diagram in apparent magnitude.
      So its "smear" premise is false on screen (`ch1-B2`).
    - The OGLE light curve carries no visible permission or credit notice (chapter 3 report, "OGLE permission").
    - GWTC redshifts come from the distance with an assumed cosmology, so they must not be used for H₀ (chapter 8).

## Fix plan

1. **Unblock and grade honestly.** Root causes 1, 2, 4, 5, 6, the NaN box selection and the photo reload loop.
   These are small, local changes. Add the unit test for unlocks and an e2e mode that plays through the real menus,
   so these bugs cannot come back.
2. **Make exercises robust and the lens pleasant.**
   - Root cause 3.
   - The autoscale, the hover bar and a log/lin toggle.
   - Saving the open mission and its step on reload (`shell-B18`).
   - The tour text (it says twelve questions; there are 18).
   - The quiz should explain its answers after the second attempt (`shell-B19`).
3. **Phone mode.** Pinch zoom, the lecture as a sheet over the sky, and "tap to use" canvases.
4. **Lectures.** Close the slide gaps listed per chapter below. Add the key slide figures and numbers.
5. **New views and missions.** The list at the end.

## Chapter by chapter

Each section gives the state, the worst problems, what is missing against the slides, and the reviewer's top three
improvements. The full report is linked from each heading.

### 1 · Stars: Lecture 1 ([report](reports/ch1.md))

- **State:** seven of eight missions finish through the UI. 1.3 cannot, which blocks 1.5 and 1.8 (B1).
- **Worst:**
  - 1.3 opens on a sample that is already an HR diagram (B2).
  - 1.1's chart cuts off its own target stars (B3).
  - White dwarfs pile up at 11,590 K (B11).
  - Switching dataset in 1.8 makes the result claim that 137 pc and 78 pc "agree" (B14).
- **Missing against the slides:**
  - R ∝ T² along the main sequence.
  - The HR diagram drawn first for clusters.
  - α ≈ 3 and 5 in L ∝ M^α.
  - Luminosity classes.
  - The EM spectrum and the atmospheric windows (the *Light* menu exists but chapter 1 does not use it).
  - Distance from a blackbody fit needs a radius.
- **Top 3:**
  1. Lend a mission's unlocks during its exercise.
  2. Rebuild 1.3 on the naked-eye sky, where V against B−V is a real smear and M_V shows the bands, with a "move
     every star to 10 pc" slider.
  3. Fix the autoscale and the instrument-mode issues (no legend, the chart resizing on hover, the controls below
     the fold at 1100×800).

### 2 · Stellar evolution, Cepheids, pulsars: Lecture 2 and the ladder of Lectures 4–5 ([report](reports/ch2.md))

- **State:** all four missions finish. Leavitt (2.2) is hard to finish with a mouse alone.
- **Worst:**
  - 2.2's fold window is ±0.002 d, while the slider moves 0.2 d per pixel (B5).
  - 2.2's stage 2 keeps the fold controls, which can fold the period–luminosity diagram (B8).
  - Line handles jump on hover (B9).
  - In 2.3 the Milky Way Cepheids to slide are off screen at the start, then hidden under the LMC points (B11).
  - The turn-off explainer shows nothing below 0.8 Gyr (B3).
- **Missing against the slides:**
  - The mass limits of stars (0.08 and about 100 M☉).
  - The lecturer's 10 M☉ → 10⁴ L☉ → 10 Myr scaling.
  - The post-main-sequence stages and the evolutionary tracks.
  - Nucleosynthesis beyond "an onion ending in iron".
  - Supernova energetics and light curves.
  - **The pulsar minimum-density argument (ρ ≥ 3π/GP²)**, the slides' central calculation.
  - The P–Ṗ life of a pulsar.
  - Black-hole time dilation and tides (HW2 Q5).
  - The SN 1987A light-echo distance.
- **Top 3:**
  1. Fix the fold controls and the hover resize.
  2. Make 2.3's reference population visible, and use the student's own Leavitt slope there.
  3. Add live tools for the slide topics: isochrones and evolutionary tracks, the Crab density argument, the P–Ṗ
     death line and magnetars, and the SN 1987A ring.

### 3 · Exoplanets and microlensing: Lecture 3 ([report](reports/ch3.md))

- **State:** all three missions finish. 3.1's unlock does nothing (B7). 3.3 is fiddly with a mouse.
- **Worst:**
  - The box selection includes planets with no period, so 3.2 reports 69% found by transits where 92% is right
    (B1).
  - Masses estimated from the radius are shown as measurements (B8).
  - The χ² readout goes stale, and *What does it mean?* stays enabled after a change of event.
  - Wrong numbers: "Mercury ×13", "51 Peg ×10", "t_E 30–40 d".
- **Missing against the slides:**
  - HII regions: the Strömgren radius, forbidden lines, "nebulium".
  - The ISM phases, 21 cm, molecules, dust and extinction.
  - Shocks, supernova remnants and cosmic rays.
  - The radial-velocity numbers (Jupiter at 1 AU → 28 m/s).
  - The transit duration and impact parameter.
  - The MACHO optical depth.
- **Top 3:**
  1. Fix B1 and B8, and let the 3.1 unlock open planet radius and density.
  2. A microlensing builder (M, D_L, v, b → t_E), plus the MACHO optical-depth exercise from Lecture 4 and a
     planet anomaly.
  3. Give the IMF its own mission on the Gaia 100 pc sample, and add a radial-velocity exercise. 3.2 teaches no
     Doppler measurement at present.

### 4 · Solar system, Milky Way, dark matter: Lecture 4 ([report](reports/ch4.md))

- **State:** all three missions finish. The chapter unlocks nothing.
- **Worst:**
  - The `<` in M(<r) breaks 4.3 in four places (B1–B4).
  - Switching dataset deletes the exercise and loops a TypeError (B6).
  - *What does it mean?* stays enabled after the answer becomes wrong (B5).
- **Missing against the slides:**
  - Milky Way structure: the Sun's 240 km/s, the Galactic year, the enclosed mass, the exponential disk, stellar
    collisions, spiral density waves, the bulge.
  - The halo's ρ ∝ r⁻² and its mass.
  - How each kind of ordinary matter would have shown itself.
  - The MACHO microlensing numbers.
  - MOND's a₀ compared with everyday accelerations.
  - The Kuiper-belt occultation scale (HW4 Q1).
- **Top 3:**
  1. Fix the escaping.
  2. A top-down solar-system view for the *Solar system* rung and 4.1. The gaps are invisible in positions and
     obvious in `a`; Trojans and Hildas show too.
  3. Add the Milky Way structure and the MACHO numbers to 4.2–4.3, and let them unlock R_gc and M(<R).

### 5 · Galaxies and clusters: Lecture 5 ([report](reports/ch5.md))

- **State:** all three missions finish. Nothing in the chapter is locked, and it unlocks nothing.
- **Worst:**
  - 5.3 rejects slopes above 4.6, though the data's bisector slope is 4.57 and the orthogonal slope 4.78 (B1).
  - *Next* stays enabled after the measurement is undone.
  - Switching dataset in 5.3 throws an error.
  - "Colour by shape" does nothing while a selection is on.
  - The Tully–Fisher explainer says ×8 where it should be ×16.
  - The Olbers example uses galaxy discs rather than stars.
- **Missing against the slides:**
  - The galaxy collision rate.
  - AGN energetics, types and unification.
  - Cluster gas: about 10× the stellar mass, the baryon fraction, lensing against virial mass.
  - Distance methods: surface-brightness fluctuations, the Phillips relation, the SZ distance.
  - The uranium clock.
  - The "no centre" argument for Hubble's law.
- **Top 3:**
  1. Fix 5.3, and add a baryonic-mass axis so the student can test whether the baryonic relation is tighter.
  2. Give 5.2 the mass-to-light number (≈200, as in the slides) and a crossing time, and let the student choose R.
  3. Add the slides' quantitative arguments as worked examples, and a 21-cm line-profile explainer.

### 6 · The expanding Universe: Lecture 7 ([report](reports/ch6.md))

- **State:** all four missions finish, with no locked items.
- **Worst:**
  - The model fitter breaks when the axis or dataset changes (B9).
  - *What does it mean?* stays enabled after the answer becomes wrong (B10).
  - Texts contradict the screen or the physics:
    - supernovae "on the sky" that are not there;
    - a "black" curve that is blue;
    - a nucleosynthesis explainer that disagrees with itself;
    - BAO Δz 0.06 where it is 0.044;
    - the angular-scale numbers (0.6°, ≈1°, 0.82°, "twice the full Moon") disagree.
  - 6.4 measures only the quasar's redshift, and never uses the forest its prediction is about.
- **Missing against the slides:**
  - The two-crest derivation of cosmological redshift.
  - The turn from deceleration to acceleration (z ≈ 0.6).
  - A redshifted blackbody stays a blackbody.
  - Penzias and Wilson on the Rayleigh–Jeans tail.
  - The Planck map (only WMAP is shown).
  - The ruler worked out for a flat universe.
  - The Planck simulator the slides send students to.
  - Gunn–Peterson in the lecture itself.
- **Top 3:**
  1. A model fitter that follows (or locks) the axes, plus an Ωm–ΩΛ χ² map with the student's point. The code
     already exists in `OBS.snChi2` and the MCMC map.
  2. A second step for 6.4: click dips for absorber redshifts, measure the mean transmitted flux, and add a
     Gunn–Peterson slider.
  3. CMB layers in the *Light* menu (uniform → dipole → Galaxy → anisotropies) with a μK colour bar. Draw a ruler
     circle on the CMB map that follows Ω_total.

### 7 · The observer's toolkit: Lectures 8–9 ([report](reports/ch7.md))

- **State:** both missions finish. Neither uses the axis menus.
- **Astronomy:** checked against astropy.
  - Sidereal time agrees to 0.3 s and the Sun to 0.01°; sunset and twilight agree to the minute.
  - Target altitudes are off by up to 0.4°, because the targets use J2000 positions without precession.
- **Worst:**
  - The night planner leaks into 7.2: the sky stays in the horizon frame on M42 (B1).
  - The sky's own date box and *Now* button go out of step with the planner (B2).
  - Arrow keys cannot move the instrument sliders.
  - 7.2 is done in three clicks at the default aperture. Its result says "S/N of only a few" while the screen shows
    about 50.
  - The planner ignores Israeli summer time.
- **Missing against the slides:**
  - The hour angle as a readout.
  - The Moon on the visibility plot.
  - The time systems, heliocentric JD and aberration.
  - Seeing and PSF sampling.
  - Filter curves.
  - f_ν against f_λ, and **surface brightness does not change with distance**, a core intuition.
  - AB − Vega by band, CCD gain, the grating equation and the slit.
- **Top 3:**
  1. Make 7.1 test its own prediction: fix the clock at 00:00 and find the month.
  2. Start 7.2 at a 2 px aperture, with a star away from the image edge.
  3. Add the Moon and the three twilight stages to the night plot.

### 8 · Every kind of light: Lectures 10–11 ([report](reports/ch8.md))

- **State:** all three missions finish, at all three sizes. All seven wavelength maps line up in every frame,
  checked on Cas A, Cyg A, Vela, the Crab, Geminga, Sco X-1, M87, the LMC and the Galactic centre. Each map carries
  the right credit.
- **Worst:**
  - Each mission points the sky at l ≈ 96° (B1).
  - The photo reload loop (B2).
  - 8.3 plots final mass against total mass on log axes, where the 4% effect is invisible (B6).
  - You can finish 8.3 with ε = 0 and be told your prediction held.
  - The *windows* and *interferometer* explainers ignore their sliders.
  - A wrong credit on the Coma close-up, a caption that names the wrong colour, and a wrong error bar in 8.1.
- **Missing against the slides:**
  - **BATSE's short/long split of bursts.** The data are already loaded.
  - Sub-mm and ALMA.
  - SOFIA and balloons for "how high must you go".
  - Soft against hard X-rays, and Cyg X-1.
  - The gamma-ray emission mechanisms and CGRO's instruments.
  - The inflationary gravitational-wave background.
  - The "stellar graveyard" plot.
- **Top 3:**
  1. Centre every Galactic view on l = 0. Plot 8.3 as mass radiated against total mass.
  2. Add a short-against-long T₉₀ step to 8.1.
  3. Make the windows and fringes explainers respond to their controls: an altitude slider, fringes that change.

### 9 · Reading galaxies: Lectures 12–13 (and 9) ([report](reports/ch9.md))

- **State:** all three missions finish. The physics and every worked example check out.
- **Worst:** the answer checks and the χ² map:
  - 9.1 passes on a slider sweep (B1);
  - 9.2 scores by handle order (B2);
  - 9.3 is done in one click (B3);
  - 9.3's χ² map shows 7.2's image (B18).

  On the lecture side:
  - 9.3's lecture is all text: five steps with no figure or explainer.
  - The BPT step has no BPT figure.
  - A vertical line passes 9.2, so [O III]/Hβ is never needed.
- **Missing against the slides:**
  - The Schechter luminosity function.
  - Types of ellipticals and their profiles.
  - Red-sequence cluster selection.
  - The lecturer's own SFR against HI paper.
  - SFH fitting and colour degeneracies.
  - Metallicity indicators and the fundamental metallicity relation.
  - The 1-D cost function with local minima, the slides' central figure.
  - Grid pseudocode, bootstrap and jackknife, the single neuron, a convolution kernel.
- **Top 3:**
  1. A χ²(z) strip under the redshift fitter, with a *Record* button. It also fixes B1 and prepares for 9.3.
  2. A trace plot with the burn-in marker, and a start step outside the good range. It fixes B3 and teaches what
     burn-in is.
  3. The Kauffmann and Kewley curves on the BPT, with the composites in the score, plus a cost-function explainer
     for Lecture 13.

### Shared parts ([report](reports/shell.md))

- **State:** solid at desktop size. There were no console errors in any pane. All 30 lecture pictures, 8 sky bands,
  7 ladder rungs and 29 datasets load. Sky cues land exactly. The 18 quiz answers are correct.
- **Worst:**
  - The 1.3 blocker (B1).
  - The phone: no pinch (B16), the lecture far below the card, canvases that swallow scrolling (B21).
  - Pressing *Tour* closes the current mission without warning (B12).
  - A reload loses the open mission and its step (B18).
  - The quiz:
    - gives no feedback;
    - the right answer is the middle option in 10 of 18 questions;
    - a blank answer is saved silently;
    - the "before" attempt can never be retaken.
  - Star names: 7,245 naked-eye stars are called just "Ori" or "Cas". The 100 pc stars are all "Gaia star".
  - The tour says twelve questions; there are 18.
- **Top 3:**
  1. Lend unlocks during an exercise, and test them.
  2. A phone mode.
  3. Make the lens an exploration tool: a log/lin toggle, colour by a number with a colour bar, sensible default
     views, and a free line-fit tool.

## The strongest suggestions

Picked from the ten reports for what they would teach. Each report has more, with the files they would extend.

### Interactive views and simulations

| Where | What the student does and sees | The idea it fixes | Builds on |
|---|---|---|---|
| 1.3 | A slider moves every star to 10 pc: the naked-eye smear contracts into the main sequence and the giants | "Brightness tells what a star is" | `INS.slideRef` |
| 1.2 | The Earth's orbit beside the star's wobble, with an ecliptic-latitude slider (a line at β = 0, a circle at 90°) | "The shift is always 2p" | `INS.parallax` |
| 1.6 | Inclination slider: K shrinks by sin i, masses by sin³ i, the eclipse disappears below ~80° | Why eclipsing binaries matter | `INS.binary`, `orbit2` |
| 2.1 | An evolutionary-track player (0.6–15 M☉) with a clock for each phase; isochrones on the four Gaia clusters | "Stars move around the diagram" | new explainer, `INS.model` |
| 2.2 | A periodogram: scatter against trial period, with a draggable marker | Fiddly sliders; what folding measures | `INS.foldScatter` |
| 2.4 | Crab spin against a star's radius: a white dwarf flies apart; a pulsar ages along constant B to the death line, then is recycled | "Pulsars could be any spinning star" | new explainer, chart overlay |
| 3.3 | Microlensing builder: M, D_L, v, b → Einstein ring, track, light curve; add a planet | "A bigger peak means a heavier lens" | `X.lens` |
| 4.1 | A top-down solar system from `smallbodies`: gaps invisible in position, obvious in a; Trojans, Hildas | "The gaps are empty rings in space" | sky.js / image view |
| 4.3 | An acceleration ruler: Earth, Earth's orbit, the Sun round the Galaxy, a₀ | "Why didn't we notice MOND here?" | new explainer |
| 5.3 | 21-cm line profile with V and inclination sliders; a baryonic Tully–Fisher toggle | "Line width = speed" | new explainer, catalog.js |
| 6.1 | Ωm–ΩΛ χ² map with the student's point, the flat line and q₀ = 0 | Degeneracy; why the CMB is needed | `OBS.snChi2`, MCMC map |
| 6.2 | Redshift a 3,000 K blackbody from z = 1100 to 0; the CMB in layers (uniform, dipole, Galaxy, anisotropies) | "The CMB was always microwaves" | `explainers2.js`, sky bands |
| 6.3 | A ruler circle on the CMB sky that follows Ω_total; the acoustic explainer animated into peaks | Peaks as a ruler | sky markers, `X.acoustic` |
| 6.4 | A neutral-fraction slider turns the real forest into a Gunn–Peterson trough | "The IGM is mostly neutral" | `INS.zfit` |
| 7.1 | A date × hour altitude map with dark hours outlined; the Moon and twilight stages on the night plot | "A star rises at the same time every night" | `INS.night` |
| 7.2 | A seeing slider that blurs the real stars: the best aperture grows and S/N falls | Seeing costs S/N | `observing.js` |
| 8.1 | The windows explainer with an altitude slider (sea level → Mitzpe Ramon → Mauna Kea → SOFIA → orbit), linked to the *Light* menu | "Space telescopes only give sharper pictures" | `X.windows` |
| 8.2 | De-disperse a burst: slide a trial DM until the sweep stands upright | Where a DM comes from | new instrument |
| 8.3 | Hear the chirp (Web Audio) and overlay GW150914's strain from GWOSC | The chirp as a measurement | `X.chirp` |
| 9.1 | χ²(z) strip under the fitter, with a false dip for a wrong line ID | Matching as a minimum | `INS.zfit` |
| 9.3 | Trace plot and burn-in marker; an "add the CMB" prior; a bootstrap button; a 1-D cost-function explainer | "A minimiser always finds the best fit" | `INS.mcmc`, new explainer |
| Shared | "Powers of ten" scale ladder from 1 AU to 10 Gpc in 3-D | "The sky is a flat dome" | sky.js |
| Shared | The inference chain as a live graph in the Notebook | "Why is this locked?" | `renderNotebook` |
| Shared | Colour by a number with a colour bar; log/lin toggle; a free line-fit tool | Exploration beyond the missions | chart.js, main.js |

### Pictures

Licences are as the reviewers found them. Check each page before use. Items marked "not verified" in the reports
are left out here.

| Where | Picture | Licence |
|---|---|---|
| 1.5 Sirius B | ESA/Hubble heic0516a, Sirius A and B | CC BY 4.0 |
| 1.3 | ESA/Gaia/DPAC, Gaia's HR diagram (2018) | CC BY-SA 3.0 IGO |
| 1.6 | ESO eso1101a, SS Leporis by interferometry | CC BY 4.0 |
| 2.3 | ESA/Hubble opo1115a, Hubble's "VAR!" plate of V1 in M31 | CC BY 4.0 |
| 2.4 | ESA/Hubble heic0515a, the Crab Nebula; ESO eso1415a, magnetar in Westerlund 1 | CC BY 4.0 |
| 2.1 | ESA/Webb weic2307a, WR 124; Wikimedia nucleosynthesis periodic table; M5 colour–magnitude diagram | CC BY 4.0; CC BY-SA 4.0 |
| 2.3 / 5 | ESA/Hubble heic1704a, SN 1987A's ring (light-echo distance) | CC BY 4.0 |
| 5.1 | ESA/Hubble heic9902o, the Hubble tuning fork | CC BY 4.0 |
| 5.2 | ESA/Hubble potw1302a, the Great Attractor; Chandra Bullet Cluster composite | CC BY 4.0; NASA |
| 6.2 | ESA Planck CMB map (2013); COBE DMR three maps and the FIRAS spectrum (LAMBDA) | ESA terms; public domain |
| 6.3 | NASA/WMAP "shape of the Universe" triangles | public domain |
| 6.4 | ESA/Hubble heic1702a, H0LiCOW lensed quasars | CC BY 4.0 |
| 7.1 | Wikimedia `File:WiseObservatory.jpg` (Wise Observatory) | CC BY 3.0 |
| 7.1 | Wikimedia `File:Airy-pattern.svg`; ESO eso1824a, Neptune with and without adaptive optics | public domain; CC BY 4.0 |
| 8.1 | Wikimedia "Atmospheric electromagnetic opacity.svg" (the slides' figure); ESO eso1907a, the M87* EHT image; ESO eso1436a, ALMA's HL Tauri | public domain; CC BY 4.0 |
| 8.1 | NASA Fermi bubbles (the EGRET map cannot show them) | public domain |
| 9.1 | ESA/Hubble heic1214a, the eXtreme Deep Field; ESA/Webb weic2209a, Webb's first deep field; ESA/Hubble heic1401a, Abell 2744 | CC BY 4.0 |
| 9.2 | ESA/Hubble heic0604a, M82's Hα wind; ESA/Hubble opo0020a, the M87 jet | CC BY 4.0 |
| *Light* menu | New bands through SkyView: 2MASS K (near IR), WISE 12 μm, GALEX NUV (partial sky); Fermi LAT for gamma rays; Planck SMICA for the CMB | with acknowledgement |

### New missions and exercises

Each uses data that are already shipped or reachable from here.

| Chapter | Mission | Data |
|---|---|---|
| 1 | *Hot stars are big*: fit log R against log T on the main sequence (slope ≈ 2) | Gaia 100 pc (shipped) |
| 1 | Classify six real spectra OBAFGKM by their lines | `spectra.csv` (shipped) |
| 2 | *How dense is a pulsar?* 3π/GP² for the fastest pulsar and the Crab | ATNF (shipped) |
| 2 | *A ladder of my own*: the student's 2.2 slope gives every OGLE Cepheid a distance; the depth of the Clouds | OGLE, Gaia (shipped) |
| 3 | The IMF: count 100 pc stars per mass bin and fit Salpeter's slope (HW3 Q1) | Gaia 100 pc (shipped) |
| 3 | Radial velocity: fold 51 Peg's published velocities and read M sin i | published RVs |
| 3 | Planet densities: radius against measured mass with lines of constant density | `exoplanets.csv` (shipped) |
| 4 | *Did MACHOs fill the halo?* Predicted event counts against the observed ones | `microlensing.csv` (shipped) + slide numbers |
| 4 | *Weighing Sgr A\** from the orbit of S2 | published astrometry |
| 5 | Faber–Jackson on SDSS ellipticals; a Schechter fit (HW5 Q3) | `galaxies` (shipped) |
| 6 | *Read the forest*: absorber redshifts and the flux decrement; then a z ≈ 6 quasar's trough | SDSS (shipped + reachable) |
| 6 | *The age check*: the student's ΛCDM age against the globular-cluster ages of 2.1 | shipped |
| 7 | *Where is north?* Solve the 7.2 image's plate scale and orientation from three stars | `photfield.json` (shipped) |
| 7 | *Star or galaxy?* FWHM against magnitude in a larger Legacy Surveys cutout | Legacy Surveys (reachable) |
| 8 | *Two kinds of burst*: the T₉₀ histogram's dip at 2 s, linked to GW170817 | `grbs` (shipped) |
| 8 | *Too many electrons for the Milky Way*: DM against latitude, FRBs against pulsars | `frbs`, `pulsars` (shipped) |
| 8 | *The mass gap* in the GW masses; a *standard siren* H₀ from GW170817 alone | `gw` (shipped) |
| 9 | Photometric redshifts against spectroscopic ones; the luminosity function with Malmquist bias | SDSS (shipped or reachable) |
| Capstone | *Climb the ladder*: the student's own stored numbers from 61 Cygni to H₀ | shipped |
| 8 | *Many wavelengths, one sky*: find the same objects in every *Light* band | SkyView maps (shipped) |
