# Open items from the audit: checklist

Every item that was still open in AUDIT.md on 28 November 2026. An item is ticked only when it is built and verified:
by a unit test, by the browser test of all missions, or by a screenshot looked at. The note says how.

## Interactive views
- [x] 1. Night planner: civil, nautical and astronomical twilight shaded in three tones (7.1) — four tones from the Sun's altitude, screenshot checked
- [x] 2. A second step for 6.4: read the Lyman-α forest (mark absorbers, measure the mean transmitted flux) — stage 2 of 6.4: drag a marker onto dips, “Mark this absorber”, mean transmitted flux and neutral fraction; browser test marks 6 absorbers, a non-dip is refused
- [x] 3. De-disperse a fast radio burst: a waterfall and a trial-DM slider (8.2) — explainer in 8.2’s lecture; summed S/N 2.8 at DM 0, 27 at the true 180, 3.4 at 300
- [x] 4. A top-down view of the solar system from the real asteroid orbits (4.1) — explainer in 4.1’s lecture from 8,843 JPL orbits; the Trojan swarms show 60° from Jupiter
- [x] 5. A "powers of ten" zoom from 1 AU to the CMB, with real data at each scale (the scale ladder) — “Powers of ten” at the foot of the scale ladder, 1 AU to 15 Gpc; six scales checked by screenshot, Esc closes it

## Missions
- [x] 6. The initial mass function from the Gaia 100 pc stars (chapter 3) — mission 3.6; browser test passes (Γ = 1.40)
- [x] 7. Radial velocity: fold a real velocity curve and weigh the planet (chapter 3) — mission 3.4 on 354 Lick velocities of 51 Peg; browser test passes (0.48 M_J, published 0.46)
- [x] 8. Could the dark halo be MACHOs? Lens masses from microlensing durations (chapter 4) — mission 4.5 on 1,490 OGLE 2019 events; browser test passes (typical lens 0.28 M☉)
- [x] 9. Weighing Sgr A* from the orbit of S2 (chapter 4) — mission 4.3 on Gillessen et al. 2017; browser test passes (4.45 million M☉)
- [x] 10. Faber–Jackson, with a better galaxy sample (chapter 5) — mission 5.4 on 8,000 SDSS ellipticals, fitting σ against L (the unbiased direction) gives L ∝ σ^3.1–3.7; browser test passes

## Small behaviours
- [x] 11. Shift-drag on the sky adds to a selection the same way as in the lens (`shell-B11`) — with Select circle, shift-drag adds (115 → 224 stars), a plain drag replaces (90); browser check
- [x] 12. The scale ladder follows the lecture's sky (`ch5-B12`) — lecture cues set the rung (5.1 → Galaxies, 4.1 → Solar system); browser check
- [x] 13. 8.1 quotes a separate error for each statistic (`ch8-B10`) — ±0.009 for ⟨cos θ⟩ and ±0.004 for ⟨sin² b − ⅓⟩; browser test result text
- [x] 14. Negative radiated masses in the GW data are explained and kept out of the fit (`ch8-B13`) — the three events are explained in the variable’s note and left out of 8.3 (276 shown); browser check

## Cosmetic
- [x] 15. Overlay labels in charts no longer print on top of each other — chart labels step aside and stay inside the plot; 6.1’s two curve labels checked by screenshot; 9.3’s map labels moved along their lines
- [x] 16. Coverage gaps masked on the far-infrared and X-ray sky maps — IRAS stripes and ROSAT arcs painted grey, credits say “Grey: no data”; images checked
- [x] 17. The orange block on the M31 close-up — patch painted with the surrounding sky, also in the pipeline (clean_plate); image checked
- [x] 18. A correct picture of SS Leporis for mission 1.6 — ESO eso1148a (three epochs of the orbit) in 1.6 “Stars in pairs”; unit test and browser check

All 18 ticked on 28 November 2026. Unit tests pass (12 of 12); the browser test plays all 40 missions to the end.
