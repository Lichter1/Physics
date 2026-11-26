# Chapter 4 (the Sun's family and the Galaxy: kirkwood, shapley, rotation): audit

## Summary

- **Locked axes:** none. After only the earlier missions (`H.open({before: id})`), all three missions can be finished through the real UI (mouse drags, slider clicks and arrow keys). Every axis, dataset and tool they name is open, because no chapter-4 variable has `needs`. The other side of this: chapter 4 **unlocks nothing**.
- **Worst problems:**
  - (a) An unescaped `<` in `M(<r)` breaks mission 4.3 in four places: raw TeX in the first equation, half a paragraph lost, the explainer readout swallowed, and the mission's relation cut to "V² = G M(" on the summary and Predict screens (B1–B4).
  - (b) Switching dataset during an exercise deletes the instruments and makes the progress check throw every 400 ms (B6).
  - (c) *What does it mean?* stays enabled after the answer becomes wrong (B5).
- **Top 3 improvements:**
  1. Fix the `<` escaping.
  2. Give the solar-system rung and lecture 4.1 a real top-down solar-system view (orbits, asteroids, ecliptic).
  3. Add the Milky Way structure from the slides to lecture 4.2 (rotation speed, Galactic year, mass, exponential disk, MACHO microlensing numbers, a₀ against everyday accelerations). Let 4.2 and 4.3 unlock R_gc and M(<R) as new axes.

## Bugs

Shots are in `audit/shots/ch4/`. Scripts are in `audit/work/ch4/` (`kirkwood.js`, `shapley.js`, `rotation.js`, `edge.js`, `layout.js`, `bins.js`, outputs `*.out`). No console errors in the normal runs.

| ID | Severity | Mission | Where | What happens (steps) | Expected | Likely cause (file:line) | Fix |
|---|---|---|---|---|---|---|---|
| ch4-B1 | major | rotation | Lecture beat 1, "Speed tells mass" | `H.open({before:'rotation'})`, start 4.3. The equation box shows raw `\[\frac{v^2}{r} = \frac{G M(`, and the step dots and *Next* button render inside the equation box. MathJax does not typeset it (`rotation-L1.png`). | Typeset \(v^2/r = GM(<r)/r^2 \Rightarrow M(<r)=v^2r/G\) | `src/js/lectures.js:414`: `M(<r)` is inserted as raw HTML by `src/js/lecture.js:115`, so `<r` opens an HTML tag | `M(\\lt r)`, or escape `<` when building `.lec-eq` |
| ch4-B2 | major | rotation | Lecture beat 3, "Flat curves" | The text reads "A flat curve means \(M(dark matter around every galaxy." The sentence about enclosed mass growing where there is no light is lost (`rotation-L3.png`). | Full paragraph | `src/js/lectures.js:419` `\\(M(<r) \\propto r\\)` | `\\(M(\\lt r) \\propto r\\)` |
| ch4-B3 | major | rotation | Lecture beat 3, explainer readout | Under "Dark halo strength" the readout is "V² = G M(0 km/s." at every slider value. The whole explanation is swallowed (`rotation-L3.png`). | "V² = G M(<r)/r. The visible matter is concentrated…" | `src/js/explainers.js:463` puts `M(<r)` into `innerHTML` | `M(&lt;r)` |
| ch4-B4 | major | rotation | "Your mission" summary (last lecture beat, shown again on Predict) | The relation box shows only "V² = G M(" (`rotation-L6.png`; the script's text read-back ends there). | "V² = G M(<R) / R. Visible matter alone: … MOND changes the force law instead." | `src/js/missions.js:604` (`M(<R)`), inserted raw by `src/js/lecture.js:24` | `M(&lt;R)` in the string, or escape it in `summary()` |
| ch4-B5 | minor | all; verified in shapley | Exercise, `#m-next` | 4.2: drag the cross to (7.2, 0.2) and the status is "done". Drag it on to (15.3, −1.1): the status returns to "in progress" but the button stays enabled. Pressing it gives "Your prediction held … You put the centre of the Galaxy 15 kpc from the Sun" (`edge.out`). | The button is disabled again whenever progress is not done | `src/js/logbook.js:168` only ever enables (`if (r.done && b.disabled)`) | `b.disabled = !r.done`; pulse only on the first enable |
| ch4-B6 | major | kirkwood (and any mission with overlays) | Exercise, *Dataset* menu | Kirkwood exercise: choose *Globular clusters*, then *Asteroids…* again. The three gap rules are gone, the y axis is now eccentricity instead of the histogram, and the status freezes at "0 of 3 rules in a gap". The console logs `TypeError: Cannot read properties of null (reading 'value')` every 400 ms (7 in 3 s, `edge.out`). Only *Reset the view* recovers. | Either keep the mission's overlays, or warn and restore them on return. No error loop. | `chart.setDataset` clears overlays (`src/js/chart.js:32`); `progress` assumes they exist (`src/js/missions.js:561`; same pattern at :592 and :644) | Disable *Dataset* while an exercise needs a fixed dataset, or re-run `setup` when the student returns to it. Guard `progress` against a missing overlay and tell the student to press *Reset the view*. |
| ch4-B7 | minor | rotation | Exercise, *Galaxy* menu (175 galaxies offered) | Choose NGC2403 during the exercise. NGC 3198's model and "stars + gas" curves stay drawn over NGC 2403's data, and χ²/N stays at the stale 704 until a slider moves; after that it is computed against the wrong galaxy (`edge-rotation-other-galaxy.png`). | Rebuild the model for the chosen galaxy (that would be a good extension), or lock the menu | radius→row `map` built once in setup (`src/js/missions.js:612`); `setSubset` keeps overlays (`src/js/chart.js:38`) | Rebuild `map` and the model on the `sub-sel` change, or hide `#sub-wrap` during the exercise |
| ch4-B8 | minor | rotation | Exercise progress | On entering the exercise the status already reads "Best without dark matter: χ²/N = 704": `INS.model` calls `onchange` during construction with vh = 0. The "try without a halo first" step is passed before the student touches anything. A student who goes straight to the halo gets "no dark matter: χ²/N = 704" in the result, while the explanation says "about 200". | The no-halo step counts only after Υ has been moved with vh = 0 | `src/js/instruments.js:268` (`update()` → `onchange`) with `src/js/missions.js:638` | Ignore the construction-time `onchange`, or require Υ to have moved |
| ch4-B9 | minor | rotation | Exercise, MOND box | With MOND ticked the halo sliders still move but change nothing. With a good halo (Υ 0.58, 142 km/s, 3.1 kpc) the status stays "in progress, χ²/N = 29" and never says why. Unticking gives "done, 1.07" (`rotation.out`). | Halo sliders greyed out while MOND is on, and the status says "untick MOND to fit a halo" | `src/js/missions.js:617-619`, `:643-646` | Disable the `vh`/`rc` inputs when `st.mond`; add the hint to the progress text |
| ch4-B10 | minor | kirkwood | Lens *Bins* slider | The exercise uses 220 bins but the slider shows 40. One arrow-key press sets 41 bins, and the 5:2 gap bin goes from 1 object to 334: the narrow 5:2 and 7:3 gaps vanish (`bins.js` output, `kirkwood-bins-41.png`). | Slider shows 220 | `src/js/missions.js:552` sets `app.chart.bins = 220` after `app.open` already ran `refreshLens` (`src/js/main.js:99`) | Call `app.refresh()` after setting bins |
| ch4-B11 | minor | kirkwood | Lecture beat 5, "Resonance" explainer | It says "Gold: where the asteroid is each time Jupiter overtakes it". The asteroid is inside Jupiter's orbit and faster, so it overtakes Jupiter. Gold is also the colour of the Sun and of Jupiter in the same drawing (`kirkwood-L5.png`). | "each time it overtakes Jupiter"; marks in a distinct colour | `src/js/explainers.js:419-422` | Fix the wording; use another colour for the marks |
| ch4-B12 | minor | shapley | Lecture beat 3 explainer | The histogram's y range is fixed at 0–50. At R0 = 8 kpc the central bin holds about 60 clusters, so the bar runs out of the frame (`shapley-L3.png`). Neither this explainer nor the 4.3 one has tick labels. | Bars inside the frame; ticks on the axes | `src/js/explainers.js:441` (`[0, 50]`); `plotBox` (:43) draws no ticks | Scale y to the largest bin; add ticks to `plotBox` |
| ch4-B13 | minor | kirkwood | Phone, 390×844 | 220 bins in about 300 px are under 1.5 px each. The histogram turns into faint hatching and the 5:2 and 7:3 gaps are hard to see (`layout-kirkwood-390-exercise.png`). The ±0.03 AU window is about ±5 px. | Readable gaps on a phone | fixed `bins = 220` (`src/js/missions.js:552`) | Bins from the canvas width, or open zoomed on 2.4–3.0 AU on narrow screens |
| ch4-B14 | minor | rotation | Phone, 390×844 | The status line and *What does it mean?* are about 2,000 px above the sliders (status at y≈500, tools at y≈2,680). The student cannot see "done" while fitting. | Progress visible next to the instrument | page layout; `#m-status` lives only in the logbook card | Mirror the progress pill in `#tools` or make it sticky on narrow screens |
| ch4-B15 | cosmetic | rotation | Lens labels | At start (vh = 0) the labels "model" and "stars + gas" are drawn on top of each other ("model gas"), and cut off at the right on a phone (`rotation-exercise-start.png`, `layout-rotation-390-exercise.png`). | Separate labels | `labelAt` default 0.82 of the width for both curves (`src/js/chart.js:390`) | Give `vbar` its own `labelAt`, or a legend |
| ch4-B16 | cosmetic | kirkwood | Lecture beat 1 equation | "L☉ = …, R☉ = …, T_surface = 5772 K, T_co…" is cut at the right edge at 1600×950 (`kirkwood-L1.png`). | Whole equation visible | `src/js/lectures.js:367`, one long line | Split it into two `eq` lines |
| ch4-B17 | cosmetic | kirkwood | Dataset counts | The lens subtitle says "50,405 orbits", the brief "50,000 orbits", the readout "52,613 shown"; the manifest has 52,613 rows. | One number | `src/js/catalog.js:246`, `src/js/missions.js:543` | "52,613" |
| ch4-B18 | cosmetic | kirkwood | After *Keep exploring* | The three gap rules stay on the free lens. Following the explanation ("switch the vertical axis to eccentricity and widen the range to 6 AU") autoscales to 0–108 AU; the student has to find wheel-zoom alone. | Rules removed on leaving; a hint or button for "0–6 AU" | `leave()` (`src/js/logbook.js:63`) does not clear overlays | Clear mission overlays in `leave()`; add "scroll to zoom" to the explanation |
| ch4-B19 | cosmetic | rotation | Sky photo of M31 | The DSS close-up `data/img/m31.jpg` has a flat orange block near its top edge. It shows on the dome in 4.3 beats 2–6 and through the whole exercise (`rotation-L6.png`, `work/ch4/m31_cutout.png`). | Clean image | the cutout file (`data/images.json`, entry `m31`) | Re-fetch it or mask that region |
| ch4-B20 | cosmetic | shapley | Lecture beat 4 sky | At the Galactic centre (field 22°) the 0.4° `baade` close-up shows as a small black square near Alnasl (`shapley-L4.png`). | No visible patch | close-ups blended without tone matching, `src/js/sky.js:254-266` | Skip close-ups smaller than about 1/30 of the field, or match their levels to the panorama |
| ch4-B21 | cosmetic | all three | Homework labels | Chapter header "HW4 Q1, HW5 MOND"; rotation `hw: 'HW5 (MOND questions)'`; lecture "no halo is needed (HW5)"; explanation "(HW5)". In the course, MOND is **HW4 Q3** (Maoz ch. 7 problem 6) and HW5 Q4 is "the case for dark matter" (`HW4-5_solutions.txt`). | "HW4 Q3 (MOND), HW5 Q4 (dark matter)" | `src/js/missions.js:21, 601, 652`; `src/js/lectures.js:421` | Relabel |

Not bugs, but noted:
- Leaving an exercise half-way (start 4.2 from the logbook, then click 4.1 again) restarts 4.1 at Lecture step 0. The rule positions are lost and the lecture has to be skipped again. Acceptable, but a "resume exercise" would be kinder.
- *Reset the view* also resets the mission's rules and cross. The label suggests it only resets the zoom.

## Mission by mission

### 4.1 kirkwood (Gaps in the belt)
- **Can it be finished through the UI:** yes. The dataset, the histogram and the `a` axis are all open (the x menu offers a, e, i, H, P, P/P_J; none locked). I dragged the three rules with the mouse to 2.502, 2.825 and 2.958 AU and got "done 3 of 3". The result table reads 1/3, 2/5, 3/7. No console errors.
- **Lecture:** 8 beats: the Sun; the 11-year cycle (live SILSO chart, works); small bodies; Kepler III; resonance (explainer); beyond Neptune; comets and the Oort cloud; summary. The physics and arithmetic are right: spot flux 0.23, 2.5^1.5 = 3.95 yr = 11.86/3, Oort comet 1.1×10⁷ yr, Halley 75 yr.
- The Oort cloud is given as 2,000–100,000 AU; the slides say 1,000–10,000 AU. Say both, or match the slides.
- The sky does nothing for the whole lecture and exercise: RA 12h, Dec 0, field 120°, with no Sun, ecliptic or planets, while the text says "all move close to the ecliptic".
- **Exercise:** clear and fair. The window is ±0.03 AU; any 3 of the 4 gaps count (3:1, 5:2, 7:3, 2:1, and 4:1 too).
- Checked against the data: 2.499 AU has 4 objects per 0.0077 AU bin against about 200 nearby, 2.823 AU has 1, 2.955 AU has 54 against about 120, and 3.27 AU has 3. The 5:2 gap is one bin wide, which is why B10 matters.
- The prediction question is good, but the "Only close to Jupiter" distractor is weak. A stronger misconception to test: "fewer asteroids where Jupiter pulls hardest (nearest Jupiter)" against "at simple period ratios".
- **Explain & practice:** both practice answers check (2.50 AU; 0.308), and a wrong one gets "Too high". The table writes "1/3"; add the usual names "3:1, 5:2, 7:3".
- The explanation promises Hildas at 4.0 AU and Trojans at 5.2 AU, but only after a manual zoom (B18). Make this a one-click second view: a vs e, 1.5–5.5 AU, coloured by group.
- **Views:** the "Solar system" rung opens the asteroid table but the sky stays the star dome; asteroids have no positions. P/P_J as an x axis is the most direct view of the resonances and is not used.

### 4.2 shapley (Where is the centre?)
- **Can it be finished through the UI:** yes. X/Y are open, and l, b, R☉ and Z are also offered, all unlocked. I dragged the cross with the mouse to (7.2, 0.2): done, result 7.2 kpc. It also works by touch-size drag at 390×844.
- 8.2 kpc on the X axis is accepted (d = 1.2). No console errors.
- **Lecture:** 6 beats: edge-on all-sky photo (Magellanic Clouds correctly below and to the right; l increases to the left); M13 close-up; Shapley explainer; dust at the centre; S2 and Sgr A* (two ESO pictures, both load); summary. Numbers checked: RR Lyrae practice 10 kpc; S2 0.1255″ × 8178 pc = 1026 AU, giving 4.2×10⁶ M☉.
- Beat 3 ("they crowd toward Sagittarius") leaves the dome zoomed on M13 (field 0.3°). Here the sky should show the 157 clusters on the Galactic all-sky map, as the exercise later does.
- **Exercise:** the target is the median (7.0, 0.1) ± 1.5 kpc. The mean of the swarm, (6.0, −1.4), is rejected (d = 1.8) although "the middle of the swarm" could mean either. Accept within 2 kpc of the median, or say "where most clusters crowd".
- The prediction's "dust hides the clusters" option is a good distractor. The explanation could say why it is wrong: dust would hide clusters toward Sagittarius, not concentrate them there.
- **Explain & practice:** sound, with a nice historical note. There is only one practice problem.
- Missing: a Galactic-year or mass practice ("v = 230 km/s at R = 8.2 kpc: period and enclosed mass?"); slides: 240 km/s, 2×10⁸ yr.
- R_gc exists in the data (`globulars.csv`) but is not a variable. It is the natural unlock for this mission.
- **Views:** the sky circles the clusters on the all-sky Galactic map, but the map is small (about 450 px wide in a 600 px pane).
- The lens view "from above the Galactic plane" (X right, Y up) is right. A Z axis or an X–Z view would show the halo is round, not flat, which is half of Shapley's argument.

### 4.3 rotation (Too fast at the edge)
- **Can it be finished through the UI:** yes, with sliders driven by click plus arrow keys.
  - Stars and gas alone: best χ²/N = 196 at Υ = 1.6.
  - MOND: best 9.67 at Υ = 0.43.
  - Halo (Υ 0.58, 142 km/s, 3.13 kpc): 1.07, done.
  - Result: 2.3×10¹¹ against 4.6×10¹⁰ M☉ within 44.1 kpc (4.9×).
  - My own fit to `rotation_curves.csv` gives the same: 196 (Υ = 1.61), 9.67 (Υ = 0.43), 1.07, so the "≈200 / ≈10 / ≈1" claims in the explanation and DESIGN are right. No console errors.
- **Lecture:** 6 beats, with three of them damaged by B1–B4 (the first equation, the key sentence "M(<r) ∝ r", and the explainer's own explanation). The physics in the intact parts is right: v⁴ = GMa₀; a baryon fraction of about 1/6 from Big Bang nucleosynthesis; the Bullet-type clusters picture (heic1506a); Neptune 5 km/s.
- The flat-curve explainer has no axis numbers and uses one illustrative galaxy. Its "measured (typical)" dots are fake: a sine wiggle around 150. That is fine, but label them "illustrative".
- **Exercise:** the task is clear and the χ² feedback is live and helpful. The status line does not hint at what to change: "raise Υ", "the outer points need a halo". There are also B6–B9.
- **Explain & practice:** the explanation is good and honest about MOND. The single practice problem checks (9.0×10¹⁰ M☉).
- Missing: a MOND practice (HW4 Q3: v = (GMa₀)^{1/4} for 10¹¹ M☉ → 200 km/s; r_MOND = √(GM/a₀) ≈ 10.8 kpc) and a ρ ∝ r⁻² question.
- **Views:** the NGC 3198 SDSS close-up is lovely. The dome then stays on M31 (with the orange artifact) through the exercise; turning back to NGC 3198 would tie the curve to the galaxy. The 175-galaxy menu is a hidden treasure that the mission breaks (B7).

### Course order (`needs` / `unlocks`) across chapters
- None of the chapter-4 missions has `needs`, which is right: they use observed or catalogue quantities.
- Conceptually, 4.2 relies on standard candles (RR Lyrae; `invsq` from 1.3, `leavitt` from 2.2) and 4.1 on Kepler III (`kepler` from 1.6). All of these come earlier, so there is no ordering problem.
- None of the chapter-4 (or chapter-5) missions has `unlocks`. The chain "each deduction unlocks the next measurement" stops between 3.1 (`transit`) and 6.1 (`cosmo`).
- Natural unlocks:
  - 4.2 → `R_gc` (distance from the Galactic centre) for globulars, and Galactocentric radius for Cepheids, microlensing events and so on.
  - 4.3 → `M(<R)` (dynamical mass, v²R/G) and `M_dyn/M_bar` in the rotation and SPARC datasets. 5.3 Tully–Fisher could then use dynamical mass.

## Gaps against the slides (Lecture 4, HW4–5)

Already covered (elsewhere in the game):
- Great Debate and Hubble's Cepheids: 2.3.
- Habitable zone, snow line, protoplanetary discs: 3.2.
- Gamma-ray bursts: 2.4.

Missing from chapter 4:
1. **Milky Way structure (slides "שביל החלב – מבנה/הדסקה/הבאלג׳").**
   - The Sun's orbital speed (~240 km/s, plus 15 km/s peculiar), the Galactic year (~2×10⁸ yr), the mass inside the solar orbit (v²R/G ≈ 10¹¹ M☉).
   - The exponential disk (scale length 2.5 kpc; scale height 400 pc for old stars and 130 pc for gas), the Sun 25 pc above the plane.
   - Star density, mean distance between stars, and collision time longer than the age of the Universe, with gravitational focusing (620 km/s escape speed, cross-section ×1000).
   - Spiral arms as density waves (the traffic-jam analogy), H II regions tracing the arms.
   - The bulge (~1 kpc, ρ ∝ r⁻³, old metal-poor stars, ~200 globulars), cosmic rays.
   - Only "disk about 30 kpc, bulge, halo" appears now.
2. **The dark halo of the Milky Way.** The curve is flat to ~30 kpc (~10 disk scale lengths), 80–90% of the mass is dark, ~10¹² M☉, and **ρ ∝ r⁻²** follows from M ∝ r (also HW5 Q4). Rotation curves are measured from Balmer lines, 21 cm, CO and stellar absorption lines; the app names only 21 cm and Hα.
3. **Ruling out ordinary matter by its light.** Atomic gas would shine at 21 cm, molecular gas in CO/H₂, ionized gas in X-ray bremsstrahlung, dust in the infrared. Then the MACHO list with the reason each fails (neutron stars and white dwarfs: too many metals; brown dwarfs: microlensing) and the particle list (neutrons decay in about 15 min; massive neutrinos are "hot" and spoil galaxy formation; WIMPs, axions). The app has one paragraph.
4. **The microlensing test of MACHOs as numbers.** Toward the LMC (50 kpc): optical depth τ ~ v²/c² ~ 10⁻⁶; 10⁷ monitored stars give ~10 events at any moment; ~100 half-year events in 5 years for 1 M☉ MACHOs, or ~2,600 one-week events for Jupiter masses. The observed few events of 1 month to months mean 0.1–1 M☉, which is not the halo. This is a perfect worked example and links to mission 3.3.
5. **MOND intuition.** Compare a₀ ≈ 1.2×10⁻¹⁰ m/s² with Earth's centripetal acceleration (~6×10⁻³ m/s²) and the Sun's acceleration round the Galaxy (~2×10⁻¹⁰ m/s²): MOND only matters where accelerations are as small as the Sun's orbit round the Galaxy. Also F = m a²/a₀ (the slides' and HW4 Q3's form), Bekenstein's relativistic version, and "find a law valid in both regimes" (the interpolating function the code already uses, `src/js/astro.js:161`). HW4 Q3(b, c) (a 1/r force with r₀, and the Tully–Fisher trap) are not mentioned.
6. **HW4 Q1, Kuiper-belt occultations.** Fresnel scale F = √(λa/2) ≈ 1.2 km at 40 AU in green light; dip duration F/v ≈ 40 ms; the need for small, distant stars and high-cadence photometry. Beat "Beyond Neptune" only names it.
7. **Smaller slide topics:**
   - Classification of small bodies: centaurs (in the dataset's colour groups but never explained) and meteoroids.
   - Exocomets seen by variable Ca II absorption.
   - The solar wind's composition and energy (0.5–10 keV).
   - Biosignatures (the vegetation red edge at 7000 Å, O₂ as a sign of disequilibrium, CH₄/NH₃/CO₂ being ambiguous), SETI@home and the Wow! signal.
   - Sgr A* in X-rays and radio; supermassive black holes up to 10⁹ M☉ in every galaxy.

Tutorials 4–5 text is unreadable OCR of handwriting. The fragments show Sgr A* (4×10⁶ M☉, 8 kpc), escape-speed (v_esc) work at 8 kpc, and MOND V⁴ = GMa₀ (not verified further).

## Suggestions: simulations and interactive views

1. **Top-down solar system for the "Solar system" rung and 4.1.**
   - What the student does: plot every asteroid at its position on its orbit (a, e plus a random or real mean anomaly), with the planets' orbits and Jupiter's position; drag time.
   - What they see: the Kirkwood gaps are invisible in positions (eccentric orbits smear them) but obvious in `a`. The Trojans lead and trail Jupiter by 60°, and the Hildas make a rotating triangle.
   - Misconception it fixes: "the gaps are empty rings in space".
   - How hard: medium. A new drawing mode in `sky.js` (or the image view `app.imageView`) fed by `smallbodies`; Kepler's equation is already in `astro.js`.
2. **Resonance kick integrator (extends `X.resonance`).**
   - What the student does: add a "kick" toggle so each conjunction changes the asteroid's eccentricity by a small vector Δe pointing from the conjunction point, and plot e against time.
   - What they see: at 2.502 AU, e grows steadily until the orbit crosses Mars (marked); at 2.6 AU it random-walks.
   - Misconception it fixes: it shows why the gaps empty, not just where.
   - How hard: small, in `src/js/explainers.js:407`.
3. **Shapley from the inside (extends `X.shapley`).**
   - What the student does: use the real 157 clusters instead of random points; move a "Sun" slider along X; the explainer shows the l-histogram seen from there.
   - What they see: the histogram is flat only when the Sun reaches X ≈ 7–8.
   - It links the lecture and the exercise with the same data. Easy, since `OBS.open('globulars')` already exists.
4. **Dust slider in 4.2.** Add extinction A_V per kpc and recompute the distances Shapley would get. 1918's 15 kpc reappears with no dust correction for about 1 mag of dimming. This shows why his scale was 2× too big. Small, in the mission's `setup`.
5. **Rotation curves of the Solar System and the Galaxy side by side.**
   - What the student sees: the planets' speeds (Kepler, falling as r⁻¹ᐟ²) and the Milky Way's curve (flat at ~230 km/s), on the same log–log axes.
   - Misconception it fixes: "the Galaxy rotates like a solid disk" against "like the planets".
   - Data: planet speeds are 8 numbers; the Milky Way curve is in Eilers et al. 2019 (to fetch). Small (a lecture `fig.chart`).
6. **"Your halo, other galaxies".** Fix B7 properly: rebuild the model when the *Galaxy* menu changes. Show χ²/N for Newton, halo and MOND per galaxy, and keep a table of the student's fits. A dwarf (DDO 154), a big spiral (NGC 2841) and NGC 3198 show where MOND does well and badly.
7. **MOND acceleration ruler.** A log scale of accelerations from Earth's surface (9.8) and Earth's orbit (6×10⁻³) down to the Sun round the Galaxy (2×10⁻¹⁰) and NGC 3198's edge, with a₀ marked. Answers "why didn't we notice MOND in the Solar System?". Easy (a new explainer in `explainers2.js`).
8. **Microlensing rate calculator (for the MACHO beat).** Sliders for the MACHO mass and the halo fraction give expected events in 5 years, with their durations. The observed OGLE/EROS counts are drawn on top. Small; it reuses `A.magnification`.

## Suggestions: pictures

- **4.1 Sun:** an SDO AIA 171 Å corona image (NASA/SDO, public domain), next to the HMI disk, for "million-degree corona". A total-eclipse corona: ESO "eso1910a"-type eclipse photo from La Silla 2019 (CC BY 4.0) (not verified that this ID is the eclipse).
- **4.1 asteroids and Kuiper belt:** NASA/JPL "PIA22946"-style inner-solar-system asteroid map (public domain; ID not verified). Or better, generate it from `smallbodies.csv` (suggestion 1). New Horizons' Arrokoth (NASA/JHUAPL/SwRI, public domain) for "Kuiper-belt objects are small".
- **4.1 comets:** comet Hale–Bopp with separate dust and ion tails (ESO "Hale-Bopp" by E. Kolmhofer and H. Raab, CC BY-SA, on Wikimedia Commons) for the two-tail sentence.
- **4.2 Milky Way structure:** the NASA/JPL-Caltech/R. Hurt (SSC) artist's top-down Milky Way, the same picture the slides use (public domain, NASA). Put it next to the X–Y lens view with the Sun marked.
- **4.2 globulars:** ESA/Hubble's 47 Tucanae or M13 (heic-series, CC BY 4.0). The DSS M13 close-up is already on the dome.
- **4.3:** Vera Rubin at the Lowell/KPNO telescope (Carnegie, check the licence). The 21-cm velocity field of NGC 3198 from THINGS (Walter et al. 2008; credit THINGS/NRAO, check the licence) shows the "spider diagram" behind the curve. ESA/Hubble heic0615a (Bullet Cluster, CC BY 4.0) next to heic1506a.

## Suggestions: new missions or exercises

1. **4.x "The Galactic year".** Use Gaia DR3 proper motions and radial velocities of nearby stars (already partly in `nearby.csv`, if RVs are added), or the published Sgr A* proper motion (6.4 mas/yr, Reid & Brunthaler 2020). Measure the Sun's orbital speed (~240 km/s), then the period (2×10⁸ yr) and the enclosed mass (10¹¹ M☉). It covers the slides' first Milky Way numbers and reuses `massEnclosed`.
2. **4.x "Weighing Sgr A*" (already in DESIGN's open list).** Fit S2's orbit from GRAVITY/Keck astrometry (Gillessen et al. 2017 table) to get M = 4.3×10⁶ M☉ and R₀ = 8.2 kpc, the modern answer to 4.2. The lecture's worked example is there; make it a fitting exercise with the `model` instrument.
3. **4.x "Kuiper-belt occultations" (HW4 Q1).** Use the TNOs in `smallbodies.csv` (8,089 orbits, H magnitudes). Convert H to size (albedo slider), compare with the Fresnel scale (1.2 km at 40 AU), and estimate event durations (40 ms) and rates. A sliding-shadow explainer would show the diffraction pattern.
4. **4.x "Did MACHOs fill the halo?"** Use the OGLE EWS events already shipped (`microlensing.csv`, 2019 toward the bulge) and their t_E distribution. Compare with the predicted counts and durations for 10⁻³–1 M☉ MACHOs (slide numbers: 100 vs 2,600 events). It closes the loop the 4.3 lecture opens.
5. **4.1 add-on:** a second task "find the families that resonances protect". Show a vs e coloured by group; drag a rule onto the Hildas (3:2, 3.97 AU) and the Trojans (1:1, 5.20 AU). The data are shipped.
