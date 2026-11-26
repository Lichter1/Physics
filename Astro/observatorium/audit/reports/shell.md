# Shell (shared parts): audit

## Summary
The shell is solid at desktop size. In every pane I tested there were no console errors; all 30 lecture pictures, 8 sky wavelengths, 7 ladder rungs and 29 datasets load; sky cues land exactly; the quiz key is correct. **Worst problem: mission 1.3 asks for M_G, which only 1.3 itself unlocks (blocker, B1). Because 1.5, 1.8, 2.1 and 2.3 need that concept, a student without teacher mode is stuck at 1.3.** It is the only locked item in any task, setup or lecture chart across all 33 missions (unlock table below). Next worst is the phone: pinch does not zoom (and the tour promises it), the lecture opens two screens below the card, and the canvases swallow page scrolling (B16, B21).
Top 3 improvements: (1) let a mission lend its own unlocks during its exercise, and add a unit test that checks every task and setup variable against the course order; (2) a phone mode: pinch zoom, scroll to the lecture, “tap to use” canvases; (3) make the lens an exploration tool: a log/lin toggle, colour by numbers, sensible default views, and a quiz that explains its answers.

## Bugs
| ID | Severity | Area | Where | What happens (steps) | Expected | Likely cause (file:line) | Fix |
|---|---|---|---|---|---|---|---|
| shell-B1 | blocker | Unlocks / mission 1.3 | Lens *Up* menu | `H.open({before:'hrd'})`, click 1.3 in the list, read the lecture, predict, *Lock in and observe*. Task: “Switch the vertical axis to Absolute magnitude M_G”. `#y-sel` shows “Absolute magnitude M_G (locked)”, `disabled`. A student cannot pick it, so 1.3 never finishes and `invsq` never unlocks. Cascade (checked in the UI with 1.1, 1.2, 1.4 done): 1.5, 1.8, 2.1 and 2.3 all `need` `invsq`, so their *Make a prediction* and *Skip to the prediction* buttons stay disabled. Without teacher mode, chapters 1 and 2 stop at 1.3. Screenshot `shots/shell/hrd-locked-MG.png`. | M_G can be picked during 1.3 | `catalog.js:51` (MG `needs: ['parallax','invsq']`), `missions.js:92` (`invsq` unlocked only by `hrd` itself), `main.js:75-76` (disabled option), `lecture.js:29,121` and `logbook.js:81` (later missions gated) | Generic fix: while a mission’s exercise is open, treat its own `unlocks` as unlocked (in `OBS.unlocked`, also allow `cur.m.unlocks` when `cur.step === 2`). Or move `invsq` to mission 1.2. Add a unit test: every variable named in a task or set up by `setup` is unlocked by the missions before it, plus the mission’s own unlocks. |
| shell-B2 | minor | Logbook: mission list | `.mstate` label | With nothing done, 1.5 shows “needs 1.2”, although it needs 1.2, 1.3 and 1.4 (the card says all three). After 1.2 it shows “needs 1.3” while 1.4 is still missing. | List every missing mission, or at least “needs 1.2 +2” | `logbook.js:42` (`missingNames(m)[0]`) | Join all numbers (“needs 1.2, 1.3, 1.4”) |
| shell-B3 | minor | Notebook: *Reset my progress* | Notebook | Open mission 1.5 (card on step 0), Notebook → *Reset my progress* → *Clear everything*. Progress becomes `{}` and the confirmation line vanishes with no “cleared” message. The 1.5 card stays open, the lecture stays in the lens, and cached mass columns keep the old M–L slope. | A confirmation, the mission card closed, columns recomputed | `logbook.js:209` | After clearing, call `LB.leave()`, `OBS.forget()` on every dataset, and print “Cleared.” |
| shell-B4 | minor | Tour | Step “More in the logbook” and the last step | The text says “Twelve concept questions” and “answer the twelve questions”. *Check yourself* has 18. | 18 (or `OBS.QUIZ.length`) | `tour.js:61`, `tour.js:71` | Use `OBS.QUIZ.length` |
| shell-B5 | minor | Tour | Step “Choose what to plot” | It says “Quantities you have not learned to measure yet are not in the menus”. They are in the menus, greyed and marked “(locked)”. | Say what the screen shows | `tour.js:54` vs `main.js:76` | “…appear greyed out, marked (locked), until a mission teaches them” |
| shell-B6 | minor | Data lens | Axis menus | No log/linear switch exists. The README says the lens shows “any two variables … log or linear”, but each variable has a fixed scale (`scale: 'log'` or `'lin'`). | A log/lin toggle per axis | `catalog.js:28`, `chart.js:65-66`, `page.html:76-79` | Add `chart.logx/logy` overrides (default: the variable’s scale) and two small toggles |
| shell-B7 | minor | Unlocks | Mission 3.1 explanation | The Explain step says “Unlocked: Transit depth gives size. Transit depth now gives planet sizes.” No variable needs `transit` and no mission needs it, so nothing changes. `exoplanets.Rp` is open from the start. | The unlock opens something (for example Rp, or a planet-density variable) | `catalog.js:17`, `missions.js:448`, `missions.js:476` | Gate `exoplanets.Rp` (and a new density ρ = M/R³) on `transit`, or drop the concept |
| shell-B8 | minor | Data lens | Dataset “Stars weighed to 3%” | Before 1.6, choosing this dataset puts the *Across* axis on “Mass M (locked)”: the chart plots a quantity the menus say is locked. | Fall back to the first unlocked variable | `catalog.js:114` (`x: 'mass'`, needs `kepler`), `main.js:20` / `chart.js:35` | In `app.open`, replace a locked default axis by the first unlocked one |
| shell-B9 | cosmetic | Data lens / Data tab | Subtitles and counts | “Asteroids and Kuiper-belt objects” says “50,405 orbits”; the file has 52,613 rows (and so does the Data tab). The Data tab says “Pictures in the lectures: 19 rows”; there are 28 lecture pictures. | Counts that match | `catalog.js:246`, `data/manifest.json` (`lecture_images.rows`) | Fill the counts in from the data at build time |
| shell-B10 | minor | Sky: frames | Readout under the sky | *Whole sky*, then *Galactic*: the readout says “Galactic coordinates, centre of the Galaxy in the middle”, but the view keeps its old direction. In the screenshot the bulge sits at the left edge and the middle is near l ≈ 220° (`shots/shell/sky-allsky-gal.png`). | Recentre on l = 0, or say what is in the middle | `main.js:207` (fixed text), `sky.js:532-536` (`setFrame` keeps the direction) | In the `data-sframe` handler, `goTo(266.4, -28.9)` when switching to Galactic in whole-sky mode |
| shell-B11 | minor | Sky: selection | Shift-drag on the sky | With the naked-eye sky in the lens, shift-drag a circle (172 selected), then shift-drag another: the selection is *replaced* (49). Shift-drag in the lens *adds* (the lens hint says “shift-drag adds”). On the sky, adding needs the *Select circle* tool plus Shift. | Same rule in both panes | `sky.js:570` (`add: e.shiftKey && self.tool === 'circle'`) | Use Alt or Ctrl with Shift to add on the sky, or make the second circle add when the tool is *Select circle*; say so in the canvas label |
| shell-B12 | minor | Tour | *Tour* button during a mission | Start mission 1.1 (or be in any exercise), press *Tour*: the mission is closed (`logbook.current().m` becomes null), with no warning, and it is not reopened when the tour ends. | Ask first, or reopen the mission after the tour | `tour.js:164` | Remember `cur.m` and `cur.step` and restore them in `T.end()` |
| shell-B13 | minor | Sky / object cards | Star names | 7,245 of the 8,868 naked-eye stars have only a constellation in the `bayer` column, so hover and cards call them “Ori”, “Cas”, “Peg”. Bayer names show as “Tau Phe”, “Zet Scl”. Likewise 36,242 of the 36,683 “Stars within 100 pc” have no Gaia id and are all called “Gaia star”. | “HIP 27938 (Lep)”, “τ Phe”; a Gaia id or “star at …” | `catalog.js:79` (name function), `catalog.js:93`; `data/sky.csv` and `data/nearby.csv` columns | Use `HIP n` when `bayer` is three letters; map the Greek abbreviations; keep `source_id` in `nearby.csv` |
| shell-B14 | cosmetic | Object card on a phone | `#pop` at 390 px | Tap a star at 390×844: the card is wider than the lens and is clipped on the right; the × and the values (“locke…”, “KOII…”) are cut (`shots/shell/tour-phone-7.png`). It also covers most of the chart. | Card inside the pane | `style.css:322` and `style.css:358` (`width: calc(100% - 20px)` plus 28 px padding, no `box-sizing`) | `box-sizing: border-box` on `.pop` |
| shell-B15 | minor | Data lens: default axes | Asteroids; Nearby galaxies | “Asteroids and Kuiper-belt objects” opens on a *linear* orbit size from 0 to 108 AU: the whole main belt (2–3.5 AU) is a one-pixel sliver at the left (`shots/shell/ladder-solar.png`). The *Solar system* rung of the ladder lands exactly here. “Nearby galaxies” opens with u − r from 0.3 to 10.2 because of a few outliers (the data reach 20), so the red sequence and blue cloud are squeezed into the bottom third. | A view that shows the structure | `catalog.js:250` (a has no `scale:'log'`), `catalog.js:405`; `chart.js:83` (0.2% trimming, only above 3,000 points) | `scale:'log'` for `a`; trim 1% for the default view; or per-dataset default `view` |
| shell-B16 | major | Sky and lens on touch | Pinch | Real two-finger pinch at 390×844 (CDP touch events): on the sky it *turns* the view (longitude 53° → 313°, field unchanged); on the lens it changes nothing and *clears the selection* (7,409 → 0). No pinch handler exists: `sky.js:564-617` and `chart.js:493-561` react to pointers and the mouse wheel only, and the canvases set `touch-action: none` (`style.css:109`), so the browser cannot zoom either. The tour says “scroll or pinch to zoom”. On a phone the sky cannot be zoomed (only *Whole sky* and the ladder change the field), and the lens has *Fit view* and *Pan* but no zoom. (See the phone section for the run.) | Pinch to zoom both canvases | as listed | Track two active pointers; scale `fov` (sky) or the view (lens) by the change in their distance |
| shell-B17 | minor | Theme / lecture pane | Live lecture charts | Open 1.1, go to “Magnitudes” (live histogram), press *Plate view*. The page turns light, but the lecture chart keeps its dark panel (corner pixel 14,20,36 before and after; the light panel is #f6f7f3) until you leave the step (`shots/shell/theme-plate-lecture.png`). The explainer canvases are redrawn only if they animate. | Charts follow the theme | `main.js:266-270` (redraws only the main chart and the sky); `lecture.js:79-97` keeps no reference to its charts | Emit a `theme` event; `lecture.js` re-renders the current beat (or keeps its charts and calls `draw()`) |
| shell-B18 | minor | Persistence | Reload mid-mission | Start 5.1, lock a prediction, box the red sequence (“done”, 3,091 selected), reload before *What does it mean?*. No mission is open; the logbook shows chapter 1 (not 5); the lens is back on the naked-eye sky; the measurement is lost. Leaving a mission for another one and coming back also restarts it at the lecture (the prediction is kept). | Reopen the mission at its step (at least the chapter and the card) | `logbook.js:12` (only `chapter` is stored, and only when a chapter button is clicked, `logbook.js:31`); `cur.m`/`cur.step` are never stored | Store `{m, step, pred}` on every step change and reopen it on load; store `chapter` in `LB.start` too |
| shell-B19 | minor | *Check yourself* | Quiz | (1) No feedback at all: after the “after” attempt nobody learns which answers were wrong. (2) The right answer is the middle option in 10 of 18 questions (first option 6, last 2), so “always pick the middle” scores 56%. (3) A blank answer is saved silently (17 of 18 answered → “Before: 9/18”). (4) Once both attempts exist the form keeps offering “after” and overwrites it; “before” can never be retaken. Score and gain are right: 9/18 → 15/18 gives (15 − 9)/(18 − 9) = 0.67. All 18 keys are correct (checked by hand). | Show right/wrong with a one-line reason after the “after” attempt; shuffle or balance the answer positions; warn on blanks | `logbook.js:216-233`, `missions.js:1154-1173` | Add `why` to each question and show it after the “after” attempt; store a fixed per-question shuffle |
| shell-B20 | cosmetic | Lecture lightbox | Focus | Enlarge a picture (for example 2.1, step 6), press Esc: the lightbox closes but focus goes to `<body>`, not back to the picture. Keyboard users lose their place in the lecture. | Focus returns to the button | `lecture.js:70` | `bt.focus()` in `close()` |
| shell-B21 | major | Phone layout (390×844) | Mission card → lecture | Tap mission 1.3 in the list. The card opens at the top and says “The lecture is in the right-hand pane”, but on a phone the lecture is 1,973 px further down (below the logbook, the sky and its controls), and the page does not scroll there (`shots/shell/phone-after-tap-mission.png`). Getting there is hard too: the sky (64vh) and lens (60vh) canvases fill 124% of a screen height and take every swipe (`touch-action: none`). A swipe that starts on them does not scroll the page (measured: 0 px), so the student must find the thin margins to scroll. | The lecture next to the card, or the card scrolls to it; canvases that let the page scroll | `logbook.js:78` (text), `logbook.js:56` (scrolls to the card only); `style.css:109`, `style.css:342-344` | On narrow screens: scroll `#pane-lens` into view on *Lecture*, say “below” instead of “right-hand pane”, and give each canvas a “tap to interact” mode (or `touch-action: pan-y` until a long press) |
| shell-B22 | minor | Layout at 1100×800 | Medium layout | The chapter buttons wrap into three rows (header 172 px); the sky pane is 307 px tall with a canvas about 170 px high; the ladder shows only three rungs; the lecture pane is 283 px tall (`shots/shell/layout-1100-light-lecture.png`). The card again says “right-hand pane” while the lecture is below the sky. | A usable sky and lecture at a common laptop size | `style.css:333-335`, `main.js:297-299` | Collapse the chapter titles to numbers (as on phones) below 1280 px; let the lecture take the full right column while a lecture is open; make the ladder a horizontal strip |

## Mission by mission
Scripts: `work/shell/*.js` (Playwright, real clicks, drags and touch events; JS only to read state). Screenshots: `shots/shell/`.

### Course-wide unlock audit (all 33 missions)
How it was checked: statically (every lecture `fig.chart`, the mission’s `needs`/`unlocks`, the variables named in each `task`, `explain` and lecture text) and in the UI. Every mission was opened from the logbook as a student who finished all earlier missions, the lecture was stepped through with *Next*, the prediction was locked in, and then the lens dataset, axes, colour key and locked menu options were read after `setup` (`work/shell/unlock.js`, output `unlock.json`). “Concepts before” means the concepts unlocked by every earlier mission in course order.

| # | Mission | Declared needs → unlocks | What the task, setup and lecture use (needs) | Locked when used? |
|---|---|---|---|---|
| 1.1 | sky | – → – | sky: ci, mag; lecture chart sky mag histogram | ok |
| 1.2 | parallax | – → parallax | instrument | ok |
| **1.3** | **hrd** | parallax → invsq | setup nearby bp_rp, G; **task: MG (parallax, invsq)**; result: wd category (invsq, read directly) | **LOCKED: MG needs invsq, which only 1.3 itself unlocks (shell-B1).** Colour by “White dwarf” (invsq) is also missing from *Colour by* |
| 1.4 | thermometer | – → thermo | instrument; lecture spectra lam, f | ok |
| 1.5 | radius | parallax, invsq, thermo → stefan | nearby T (thermo), L (parallax, invsq), colour wd (invsq); result R (stefan, read directly) | ok in course order, **but unreachable in practice** (cascade of B1) |
| 1.6 | binary | – → kepler | instrument (masses shown only once kepler is unlocked, `instruments.js:207`) | ok |
| 1.7 | masslum | kepler → masslum | torres mass (kepler), L, colour stage | ok |
| 1.8 | pleiades | parallax, invsq → – | clusters pmra, pmdec; stage 2 bp_rp, G, nearby MG reference (read directly) | ok, unreachable in practice (B1) |
| 2.1 | turnoff | parallax, invsq, masslum → – | clusters bp_rp0 and MGc (parallax, invsq), colour cluster | ok, unreachable in practice (B1) |
| 2.2 | leavitt | – → leavitt | cep_lc t, mag; cep_ogle P, I | ok (the `leavitt` concept gates no variable) |
| 2.3 | yardstick | parallax, invsq, leavitt → – | cep_gaia P, W; reference MW (parallax, invsq, read directly) | ok, unreachable in practice (B1) |
| 2.4 | pulsars | – → – | pulsars P, Pdot, colour bin; lecture grbs t90 | ok |
| 3.1 | transit | – → transit | kepler t, f | ok (`transit` gates nothing, B7) |
| 3.2 | census | – → – | exoplanets P, Mp, method; **lecture chart nearby Mms (parallax, invsq, masslum)** | ok in course order. 3.2 declares no `needs`, so a student who opens chapter 3 first sees a “locked” quantity plotted in the lecture (lecture charts bypass locks, `lecture.js:87`) |
| 3.3 | microlens | – → – | microlensing t, mag | ok |
| 4.1 | kirkwood | – → – | smallbodies a, histogram; explain: e; lecture sunspots | ok |
| 4.2 | shapley | – → – | globulars X, Y | ok |
| 4.3 | rotation | – → – | rotation R, Vobs | ok |
| 5.1 | redblue | – → – | galaxies Mr, ur, colour morph | not locked, but **Mr is a Hubble-law distance plus the inverse square with no `needs`**: the “redshift gives distance” concept arrives only at 6.1 |
| 5.2 | zwicky | – → – | coma cz histogram; result d = cz/H₀ | same remark (Hubble law before 6.1) |
| 5.3 | tully | – → – | sparc Vflat, L36 | ok |
| 6.1 | accel | – → cosmo | supernovae z, dmu (tlb is locked during it, not used) | ok |
| 6.2 | cmb | – → – | cmbspec f, I | ok |
| 6.3 | peaks | – → – | cmbpower l, Dl | ok |
| 6.4 | forest | – → – | zspec qso; lecture zspec | ok |
| 7.1 | night | – → – | instrument (horizon frame) | ok |
| 7.2 | photometry | – → – | instrument | ok |
| 8.1 | bursts | – → – | grbs sinb, histogram; pulsars sinb | ok |
| 8.2 | frb | – → – | frbs z, dm; lecture pulsars b, dm | ok |
| 8.3 | gw | – → – | gw mtot, mfinal, colour kind; lecture gw m1 | ok |
| 9.1 | redshift | – → – | zspec; lecture galaxies Mr, ur, morph | ok |
| 9.2 | bpt | – → – | bpt n2ha, o3hb; task: colour by cls; explain: logm, logsfr, oh | ok |
| 9.3 | chain | cosmo → – | instrument | ok |

**Concepts that unlock too late or are never used**
- `invsq` unlocks too late: at the end of the mission whose task needs it (B1). Four later missions depend on it.
- `transit` is never used: no variable and no mission needs it, yet the Explain step announces it (B7).
- `leavitt` opens no axis. It only gates 2.3. The Notebook line “Each relation opens new axes in every dataset” is not true for it; a variable such as “Absolute magnitude from period” for `cep_ogle` (needs leavitt) would make it real.
- `cosmo` (“Redshift gives distance and age”) unlocks at 6.1 but only gates lookback time (`supernovae.tlb`, `frbs.tlb`) and `bpt.lha`. Distances from redshift (`galaxies.d`, `galaxies.Mr`, Coma’s cz/H₀) are open from mission 1.1 and used in 5.1 and 5.2. Either gate them on a “Hubble’s law” concept unlocked in 5.x, or accept that Hubble’s law is taught as a given in chapter 5.
- Inconsistent gating of the same physics: `nearby.MG` needs parallax and invsq, while `galaxies.Mr` and `supernovae.mu`/`dL` need nothing; `torres.radius` needs kepler while `torres.L` (computed from that radius) and `torres.teff` need nothing; `clusters.bp_rp0` (a dust correction, no distance involved) needs parallax and invsq.
- Missions’ own `needs` are incomplete: 1.3 declares `parallax` but uses `invsq`; 3.2’s lecture uses `masslum`. With chapter buttons always open, the lock warning is the only guard.

(test-harness caveat: Playwright `selectOption('#y-sel','MG')` *does* select the disabled option and sets `chart.yid = 'MG'`; real users cannot. Any e2e check of locked axes must assert `option.disabled`, not rely on `selectOption` failing.)

### Logbook: chapters, mission list, lock warnings
- Nine chapter buttons, counters correct (0/8 … 0/3), title tooltips carry the lecture line. Chapters are never locked (`c.locked` is never set).
- Lock warnings: the card (`#lb-card .warn`) and the summary step (`#lecture .warn`) both name all missing missions; *Skip to the prediction* and *Make a prediction* are disabled. The list label names only the first (B2).
- Fresh state: 1.3, 1.5, 1.7, 1.8, 2.1, 2.3 and 9.3 show “needs”. Every other mission is open even when it uses derived quantities (see the unlock table).

### First-visit tour (`work/shell/tour.js`, no `toured` key)
- It starts by itself on a first visit at 1600×950 and at 390×844. 14 cards; at desktop, no card covers its spotlight or leaves the screen (`shots/shell/tour-desk-*.png`). Both try-its work with real input (sky drag; lens box, 951 stars). The lecture step shows 1.1 “Magnitudes”, and the sky does turn to Sirius (centre 103°, −10°). *Start mission 1.1* opens it; `toured` is stored; Esc ends the tour.
- Wrong text: 12 questions instead of 18 (B4); “not in the menus” (B5); “pinch to zoom” (B16).
- Phone: the card is a sheet at the top or bottom. In steps 2, 3 and 9 it covers part of the spotlit pane (the lecture step hides the live chart it describes, `shots/shell/tour-phone-9.png`). Cosmetic.
- *Tour* mid-mission silently drops the mission (B12).

### The sky (`work/shell/sky.js`)
- Drag turns the dome naturally (grab-and-move); the wheel zooms 110° → 12°, and zooming out past 199° switches to *Whole sky*; the readout follows. *Dome*/*Whole sky* work.
- Frames: Galactic works, but the readout claims the centre is in the middle (B10). Horizon: the time box appears (UTC+2) and *Now* works. Checked Sun altitudes: 21 Jun 12:00 → +82° (expected ≈ 82.8°); 21 Dec 00:00 → −81° (lower culmination −82.8°); 20 Mar 18:00 → −2°, civil twilight (sunset ≈ 17:50). Horizon also works in *Whole sky* (ground, compass points, `shots/shell/sky-horizon-allsky.png`). The time is fixed at UTC+2; Israel is on UTC+3 in summer, so “local time” in mission 7.1 is an hour off between April and October (worth a note in the UI).
- *Light*: all 8 bands load and credit their survey under the sky (`shots/shell/sky-band-*.png`). Choosing a band while *Drawn* switches to *Photo* automatically, and *Drawn* resets the band to Visible. Consistent.
- Photo: the panorama plus close-ups, with a credit line listing which close-ups are on screen.
- Selection: shift-drag on the sky selects the stars in the circle and the lens readout updates (172 of 8,868); *Select circle* works without Shift. Shift does not add (B11).
- Object card: clicking Betelgeuse gives a full card (spectral type M2Ib, d = 152.7 pc, M_V = −5.47, T = 3,794 K); *Centre the sky on it* and *Select it* work. Names of most faint stars are just a constellation (B13).
- Scale ladder: all 7 rungs open their dataset with no console errors and turn the sky (sky → dome on Orion; solar → asteroids, whole sky; stars → 100 pc, whole sky; clusters → Pleiades at 9°; Milky Way → globulars, Galactic; galaxies → SDSS galaxies; Universe → Pantheon+ over the WMAP CMB map). The *Solar system* rung shows nothing of the solar system on the sky (no ecliptic or planets) and lands on a badly scaled lens (B15). The ladder is a dataset switcher, not a zoom.

### The data lens (`work/shell/lens.js`)
| Dataset | Default axes | Shown / plottable | Remark |
|---|---|---|---|
| sky | ci × mag | 8,868 / 8,829 | ok |
| smallbodies | a × e, colour by group | 52,613 | linear a to 108 AU (B15); subtitle says 50,405 (B9) |
| sunspots | year × sn | 326 | ok |
| nearby | bp_rp × G, Gaia colours | 36,683 | ok |
| binaries | log P × K1 | 195 | ok |
| torres | log mass × log L | 190 | mass is locked before 1.6 (B8) |
| pulsars | log P × log Ṗ | 4,440 / 2,898 | ok (Ṗ missing for the rest) |
| exoplanets | log P × log Mp, colour by method | 6,473 / 6,087 | ok |
| kepler, cep_lc, microlensing | time series with a subset menu | | ok |
| spectra, zspec | λ × f, joined lines | | ok |
| clusters | bp_rp × G, subset Pleiades | 8,547 | 6 subsets all work; the sky does not move to the chosen cluster |
| cep_gaia, cep_ogle | log P × W or I | | ok |
| globulars | X × Y | 157 | not equal-aspect outside mission 4.2 |
| grbs, coma | histograms | | ok |
| bpt | n2ha × o3hb | 20,000 | ok |
| rotation | R × Vobs, 175-galaxy subset menu | | ok |
| sparc | log Vflat × log L36 | 175 / 135 | ok |
| galaxies | Mr × ur | 14,000 | y range stretched to 10.2 (B15) |
| supernovae | log z × μ | 1,466 | ok |
| cmbspec, cmbpower, cmbtheory | | | ok |
| frbs | z × DM | 183 / 101 | ok (82 have no host z; nothing tells the student) |
| gw | log mtot × log mfinal | 282 / 279 | ok |
- No console errors on any dataset. No log/lin switch (B6). *Colour by* offers only categories: there is no colour by a number (temperature, metallicity, redshift), which would help a lot.
- Box select, the readout (count, medians, category shares), *Clear*, the wheel zoom (span 4.75 → 2.61), the histogram with the *Bins* slider (10–300), and switching dataset from a histogram (back to the dataset’s defaults) all work.
- Clicking a point opens its card on top of the middle of the chart. While it is open, later clicks and drags in that area hit the card, not the plot: a double-click or a pan aimed at the middle silently does nothing until the card is closed (seen in the run). Suggest placing the card in a corner or closing it on the next brush.
- Collapse and resize: *Hide* folds the lens (the sky takes 1,122 px); the divider drag resizes (353/813 px) and the chart redraws at the new size; a double-click on the divider resets it.

### Lecture pane (`work/shell/misc.js`)
- MathJax loads and typesets: 7 `mjx-container` in 1.1 “Magnitudes”, no raw `\[` left. (It needs the CDN: the offline file shows raw TeX without a network, as the README says.)
- All 30 lecture pictures referenced by the beats load (none broken); credits appear under each. The lightbox opens on click and focuses *Close*; Esc and a click close it; focus is not restored (B20).
- Live charts render (histograms, spectra, BPT). They are not redrawn on a theme change (B17). They are full `OBS.Chart` instances with brushing, so on a phone a swipe over a lecture chart does not scroll the lecture (`style.css:287`, `touch-action: none`).
- Sky cues: the first cue of 1.2, 2.2, 3.3, 5.2 and 7.1 lands exactly on its target, frame and field (separation 0.00°), including the Galactic frame for 3.3 and the horizon frame for 7.1.

### *Check yourself* and the *Data* tab
- Quiz: see B19. The 18 answers are all physically correct: 0.2″ in six months → 10 pc; same colour, 100× fainter → undetermined; Rayleigh–Jeans ratio; R = ¼; heavier star swings less; fainter turn-off older; m − M = 25 → 1 Mpc; 1% → 10%; v flat → M ∝ R; achromatic and symmetric; 2.5 AU → 3:1; red → ellipticals; fainter SNe → accelerating; 2.725 × 1101 ≈ 3,000 K; airmass sec 60° = 2; 0.21 m / 10 km ≈ 4″; isotropic → far; strong [O III] and [N II] → AGN.
- *Copy results*: not verified (headless clipboard).
- *Data* tab: 35 sources, each with source, query, cuts, citation, licence and date, except “Constellation lines” and “Pictures in the lectures” (no citation). No source is a clickable link (`logbook.js:251` escapes every field as text). Stale counts: B9. The microlensing entry carries OGLE’s request to be contacted before public use of its EWS photometry; this matters if the page is published.

### Persistence
- Stored per browser: progress, theme, layout, `toured`, teacher mode (and the M–L slope). Teacher mode on/off and reset: see the Notebook section and B3. Reload mid-mission and leave-and-return: B18.

### Theme and layouts (`work/shell/phone.js`, `work/shell/misc.js`)
- *Plate view*/*Night view* switches the page theme (it is the light/dark theme, labelled as a sky look). It is stored, it follows `prefers-color-scheme` until clicked, and it redraws the sky, the lens and the legend; lecture charts stay dark (B17). Light mode is legible at 1100×800 (`shots/shell/layout-1100-light-lecture.png`) and on the phone (`shots/shell/phone-light-top.png`).
- 1600×950: three columns, no overflow, everything reachable (`shots/shell/sky-allsky-gal.png`).
- 1100×800: works but cramped (B22).
- 390×844, real touch: no horizontal overflow; the page is 2,589 px tall (logbook, then the sky at y = 906, then the lens at y = 1,736). Tapping a mission, the lecture *Next*, and the ladder all work. A one-finger drag turns the sky; a finger box in the lens selects (7,409 stars) without scrolling the page. Pinch does not zoom either canvas (B16). The lecture sits two screens below the card (B21). The sky’s controls take about 230 px (CSS) before the canvas starts (`shots/shell/phone-sky.png`); the chapter strip scrolls sideways and hides chapters 6–9 off-screen with no hint. During a lecture the lens controls are hidden (the lens *is* the lecture), so the only way back to the data before the exercise is a ladder rung.

### Notebook and teacher mode
- Concepts list: 9 concepts, each with its relation, and “from mission 1.x” or “locked: mission …”. The variables table of the open dataset shows locked rows with “Locked until: …”.
- Teacher mode on: stored in `localStorage`, concept lines say “open in teacher mode”, the list drops “needs”, and the open mission’s prediction buttons unlock (the card re-renders). The axis menus refresh through the `unlock` event; I checked this only while a lecture was in the lens, where the menus are hidden (not verified with data showing).
- Reset: see B3.

## Gaps against the slides
(Shell only; the chapter reviewers cover each lecture’s content.)
- **Light menu vs lectures 10–11:** the slides treat infrared at length (Spitzer, WISE: 25 mentions in L10) and ultraviolet (about 30 in L11). The menu has far infrared (IRAS 100 µm) but no near- or mid-infrared and no UV. No lecture cue uses a wavelength band at all; only missions do (CMB 3×, radio 1×).
- **Coordinates (lectures 1 and 8):** the sky has no ecliptic, no Sun or planets in the equatorial view, no meridian or zenith marks, no altitude/azimuth grid labels in the horizon frame beyond the compass, and no time-lapse. The seasons, sidereal time and the ecliptic, which the slides teach, can only be read off numbers.
- **Check yourself** covers 18 ideas. Nothing on Kepler’s laws or orbits, black holes or the Schwarzschild radius (L2, HW2 Q5), CMB peaks and geometry, detectors and S/N (L9), or gravitational-wave energetics, although missions teach each of them.
- **Scale ladder vs lecture 1 §1.1 (“scales of the Universe”):** it switches datasets but never shows a distance scale; the solar-system rung shows nothing of the solar system on the sky.

## Suggestions: simulations and interactive views
1. **A real zoom-out (“powers of ten”) for the scale ladder.** The student drags one slider from 1 AU to 10 Gpc: planets’ orbits (Keplerian elements), the 100 pc stars in 3-D from their parallaxes, the five clusters, a top view of the Milky Way with the globulars and the Sun (Shapley), SDSS galaxies at cz/H₀, then the Pantheon+ supernovae inside the CMB sphere. Fixes “the sky is a flat dome” and shows why distance is the hard measurement. A new `space` mode next to `dome`/`allsky` in `sky.js`, fed by the datasets that have positions plus `dist`; medium-hard.
2. **The inference chain as a live map in the Notebook.** The DESIGN.md graph drawn as SVG: concepts light up as they unlock, and each shows the axes it opened and the missions that need it. This makes “why is this locked?” answerable on screen, and would have exposed B1 and B7. Easy (`logbook.js` `renderNotebook`).
3. **Colour by a number with a colour bar** (HR diagram coloured by distance or tangential speed; galaxies by σ; BPT by stellar mass) and a **log/lin toggle** per axis (B6). Easy–medium (`chart.js` `colourOf`, `main.js` `c-sel`).
4. **A free-exploration line and curve tool** in the lens (the mission line overlay with slope and scatter), so the student can test any power law they find. Easy (`chart.js` `add({type:'line'})`, a *Fit a line* button in `lens-ctl`).
5. **Sky time-lapse and ecliptic layer:** a *Play a night* / *Play a year* button in the horizon frame, the Sun and planets on the ecliptic, and the meridian. It fixes the confusion between the sidereal and the solar day. Medium (`sky.js` `_horizon`, `astro.js` `sunPos`; planets need mean elements).
6. **Phone mode:** pinch zoom, “tap to use” canvases, and the lecture as a bottom sheet over the sky (B16, B21). Medium (`chart.js`, `sky.js`, `style.css`).
7. **Quiz with reasons:** after the “after” attempt, show each answer with one line of why and a link to the mission that teaches it. Easy (`logbook.js` `renderQuiz`, a `why` and `m` field per question).

## Suggestions: pictures
- **Near-infrared all-sky (2MASS)** for the *Light* menu: SkyView survey “2MASS-K”, credit “2MASS, UMass/IPAC-Caltech, NASA/NSF”, public with acknowledgement. Same pipeline as the other bands (`pipeline/fetch_universe.py`).
- **Mid-infrared all-sky (WISE)**: SkyView “WISE 12”, NASA/JPL-Caltech/UCLA, public with acknowledgement. Matches L10’s WISE slides.
- **Ultraviolet (GALEX)**: SkyView “GALEX Near UV” (partial sky; mark the gaps), NASA/JPL-Caltech; for L11.
- **Gamma rays: Fermi LAT** all-sky map (NASA/DOE/Fermi LAT Collaboration, public domain) to replace EGRET, about 50 times sharper.
- **CMB: Planck SMICA map** (ESA and the Planck Collaboration, free with acknowledgement) to replace WMAP ILC for the *Universe* rung.
- **Milky Way rung:** “The Milky Way galaxy, artist’s concept” (NASA/JPL-Caltech/R. Hurt, ssc2008-10, public domain) as the backdrop of a top view, with the Sun marked: it directly supports mission 4.2.
- **Plate view:** a real Harvard plate of the Magellanic Clouds from DASCH (Harvard College Observatory; check the terms per plate) in the tour card that explains *Plate view*.

## Suggestions: new missions or exercises
- **Climb the ladder (capstone, end of chapter 5 or 6):** reuse the student’s own stored numbers (`mlslope`, `plslope`, LMC μ from 2.3, the Tully–Fisher slope) to go from 61 Cygni’s parallax to H₀ with Pantheon+ (shipped). The lens holds a “ladder” readout that propagates their errors. It ties chapters 1–6 together and makes the unlocks feel earned.
- **Many wavelengths, one sky (chapter 8):** with the *Light* menu, find the same objects in radio, 21 cm, Hα, X-rays and gamma rays (Cygnus, the LMC, the Galactic centre, the North Polar Spur). Ask which wavelength sees through dust and why the CMB map has no Milky Way after cleaning. Data: the shipped SkyView maps.
- **Where am I? (chapter 7):** in the horizon frame, read Polaris’ altitude to get the latitude, and use the time a known star crosses the meridian to get the longitude. Data: Hipparcos (shipped).
- **Free-exploration challenges** at the end of each chapter, checked like missions but with no setup: for example, “find the white dwarfs in the 100 pc sample using only colour and M_G”, or “show that hot Jupiters are found by transits”. They train the student to use the lens without step-by-step help.
