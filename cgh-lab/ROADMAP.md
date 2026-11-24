# cGħ World: next steps

Research notes and a build plan, November 2026. Companion to [RESEARCH.md](RESEARCH.md), which surveys the
existing simulators.

## Summary

1. **Make the view explain itself.** Add a probe that, for any pixel you point at, draws the light ray's path
   in a side view of the planet and lists its travel time and each shift factor. Add one switch per effect,
   so you can turn bending, delay, aberration and colour shift off one at a time and see what each does.
2. **Go up and down.** Add a lift in the clock tower and free vertical motion. Height is the variable that
   gravity's redshift, clock rates and the horizon all depend on, and the renderer already handles any eye
   height.
3. **Let things move.** Thrown balls, lifts and satellites carrying lamps, drawn exactly by evaluating each
   moving object at the time its light left it. Orbits around the small planet will precess visibly.
4. **Missions in predict–observe–explain form,** each tied to a real experiment: Pound–Rebka in the tower,
   the twin trip around the planet, the photon orbit, two doors, Heisenberg's microscope, the quantum bouncer,
   and a clock in two places at once.
5. **The uncertainty principle as an instrument.** Show Δx and Δp of your wave live, and make "Look" a
   microscope whose light you choose: a sharper look kicks you harder.
6. **New dials, later:** the fine-structure constant α (changes the colours of lamps and what atoms can
   exist), the cosmological constant Λ (a horizon in the distance), and a decoherence rate.

## Status

Phase 1 is built: flying up and down (E/Q, "Up to the tower clock"), seven effect switches, the light
probe with a spectrometer, and a first version of measurement (section 9).

Phase 2 is built: the mission framework (brief → predict → explore → explain, other dials locked,
progress saved) and eight missions: 1–5 and 7 from the table in section 6, plus "The bell nobody hears"
and "Look closer" (mission 8 there), brought forward because of the focus on measurement. Mission 5 asks
how far you can see rather than for a flag behind you: a ray at exactly 2/3 is unstable and drifts off
within a few hundred metres, so "see the back of your head" would be unreliable.

Phase 3 is built: a satellite drawn at its retarded position with the exact Doppler factor of an orbiting
clock, a radar echo in the probe, the Orbits and Hover labs, and missions 9–12 (satellite clocks, radar
echo, rosette orbits, hover). Thrown balls live in the Orbits lab rather than in the 3-D street.

Phase 4 is built: α and Λ dials, the Two staircases lab, a pocket black hole (exact when the planet's own
gravity is negligible; a superposition otherwise), and missions 13–15.

Phase 5 is ready to run: a ten-question quiz before and after, mission predictions recorded, and
[EVALUATION.md](EVALUATION.md). Running it with people is up to you.

Still open: a detector screen that records arrivals by itself; a bomb-tester room; moving objects in the
street besides the satellite; testing on real graphics cards and phones; and the sessions themselves.

## 1. What research says about teaching with simulations

| Finding | Source | What it means here |
|---|---|---|
| Good simulations guide without instructions, through what the controls allow, what they prevent, visual cues and immediate feedback ("implicit scaffolding"). Scaffolding can fade over successive tabs. | Podolefsky, Moore and Perkins, [arXiv:1306.6544](https://arxiv.org/abs/1306.6544); classroom test with no written instructions: [Moore, Herzog and Perkins 2013](https://pubs.rsc.org/en/content/articlehtml/2013/rp/c3rp20157k) | Unlock dials one at a time in missions; keep text short; make every control give visible feedback at once. |
| Asking for a prediction before showing the result improves explanations. The evidence is many small studies, no meta-analysis. | Zacharia 2005 and others, e.g. [Springer](https://link.springer.com/article/10.1007/s10956-013-9482-4) | Every mission starts with a prediction you commit to (a slider or a choice), then the reveal. |
| Several linked representations help when each adds something and learners get help translating between them. | Ainsworth, DeFT framework, [Nottingham repository](https://nottingham-repository.worktribe.com/output/23577631/deft-a-conceptual-framework-for-considering-learning-with-multiple-representations) | Link the first-person view to a side view of light paths, a spacetime diagram and the numbers. Pointing at one highlights the same thing in the others. |
| Students using Real Time Relativity sometimes blamed relativistic effects on "bugs". | [McGrath et al. 2010](https://arxiv.org/abs/0911.0226) | Always label whether you are seeing an effect or an artefact; the per-effect switches answer "is this real?". |
| "What you see" and "what you measure" are different and are a known source of confusion. | UvA thesis, [Seeing vs. Measuring](https://pure.uva.nl/ws/files/305314527/Seeing_vs._Measuring.pdf); review of special-relativity teaching, [arXiv:2012.15149](https://arxiv.org/abs/2012.15149) | Keep the "what is there" view from the first version and add it back to the world. |
| Games: *A Slower Speed of Light* teaches by gradually lowering c as you collect orbs. *Velocity Raptor* uses puzzles and introduces effects level by level, but players report getting stuck on arcade difficulty rather than physics. | [MIT Game Lab](https://gamelab.mit.edu/games/a-slower-speed-of-light/), [Kortemeyer et al. 2013](https://searchworks.jioinstitute.edu.in/articles/eric__EJ1025045), [Velocity Raptor](https://www.testtubegames.com/velocityraptor_old.html), [HN discussion](https://hn.svelte.dev/item/9247460) | Introduce one effect per mission; never make the challenge reflexes. |
| *Outer Wilds* makes observation a game rule: its quantum moon moves when you look away. Critics praise it; it is not physically accurate. | [Game Developer analysis](https://gamedeveloper.com/design/outer-wilds-critical-analysis), [PC Gamer](https://pcgamer.com/great-moments-in-pc-gaming-reaching-the-quantum-moon-in-outer-wilds) | Proof that "looking changes things" can be a satisfying mechanic. Ours can be the accurate version. |
| Quantum Flytrap pairs a sandbox with a guided game mode and publishes its visualization components as open source. | [Migdał et al. 2022](https://arxiv.org/abs/2203.13300) | Two modes: missions (guided) and the free world (sandbox). |
| Virtual-reality relativity exists (*Captain Einstein*); the Norwegian ReleQuant project built and studied school modules on general relativity and quantum physics. | [arXiv:1806.11085](https://arxiv.org/pdf/1806.11085), [ReleQuant](https://www.mn.uio.no/fysikk/english/research/projects/relequant/) | ReleQuant's research on student difficulties is a good source of prediction questions. |

Honest caveat: the evidence for game-based relativity learning is thin. The one controlled study I found is
Real Time Relativity's (small, one group). Plan to test with classmates (section 8).

## 2. Interface plan

**Three linked panels.**
- *View:* the first-person world, as now.
- *Probe:* point at any pixel (or hold Alt) to see a side view of the planet with that pixel's light path drawn
  on it. Beside it: how long the light took, the emitter's clock reading when it left, the gravitational
  factor, the Doppler factor, the brightness factor, and the formula for each. This answers "what process am
  I looking at?" directly. It runs the JavaScript copy of the ray tracer that the tests already use.
- *Logbook:* the current mission, your prediction, measurements you have taken.

**Per-effect switches** (an ablation panel): light bending · light delay · aberration · colour shift ·
brightness shift · time dilation · your wave · photon grain. Turn one off and the view changes by exactly
that effect.

**Instruments in the world** rather than readouts:
- *Spectrometer:* point at a lamp and see its sodium-yellow line move along a spectrum bar, with the measured
  ratio against the prediction.
- *Clock comparator:* carry a clock, compare with any other clock you stand next to.
- *Radar:* send a pulse, time the echo (Shapiro delay).
- *Hit wall:* marks where each run of the two-door experiment ended up.

**Formulas** rendered with KaTeX, shown on demand next to the number they produce.
**Units switch:** pure numbers or c, G and ħ.

**Libraries for the interface**
| Library | Use | Notes |
|---|---|---|
| [lil-gui](https://npmjs.com/package/lil-gui) / [Tweakpane](https://github.com/cocopon/tweakpane) | Developer panel for tuning | lil-gui about 5 KB; Tweakpane larger, more input types and plugins ([comparison](https://discourse.threejs.org/t/lib-why-changed-dat-gui-to-lil-gui/62124)) |
| KaTeX | Formulas | Not searched this round; widely used |
| [SceneryStack](https://scenerystack.github.io/community) | PhET's own framework: model–view–controller, accessible 2D panels | PhET simulations are GPLv3; check the framework's own licence before mixing in |
| [Lorentz interactive](https://www.github.com/vkuhlmann/lorentz-interactive) | Reference for an interactive spacetime diagram | Open source |

## 3. Moving up and down, and moving objects

**Vertical motion.** A lift in the clock tower (0–50 m) and a free "hover" mode. Nothing changes in the
renderer: it already traces from any eye position. What you would see going up on a compact planet:
the horizon drops away and the far side comes into view; lamps on the ground turn red, the tower-top clock
stops running fast relative to you. This is Pound and Rebka's 22.5 m tower in 1960
([Wikipedia](https://en.wikipedia.org/wiki/Pound%E2%80%93Rebka_experiment)) and Chou et al.'s 33 cm with
optical clocks in 2010 ([NIST](https://www.nist.gov/news-events/news/2010/09/nist-pair-aluminum-atomic-clocks-reveal-einsteins-relativity-personal)),
made visible.

**Things that fall and orbit.** Balls you throw follow timelike geodesics: the same equations as light with
a different starting condition, so the existing integrator nearly covers it. On a compact planet, a thrown
ball that goes into orbit will precess, drawing a rosette instead of an ellipse.

**Drawing moving objects exactly.** In the ray tracer, each moving object is tested at the position it had
when this ray's light left it: current time minus the light travel time accumulated so far along the curved
ray. That gives exact light delay along bent paths, at almost no cost for objects with simple motion.
Lorentz contraction is applied in the object's own frame; its Doppler factor comes from its velocity.
Candidates: a lift car with a lamp, a satellite with a lamp and clock, a tram round the equator, thrown balls.

**Your wave in the vertical.** With ħ on, vertical motion adds the quantum bouncer: a ball above the floor has
only certain allowed heights. Neutrons do this above a mirror in the qBounce experiments
([arXiv:1510.03078](https://arxiv.org/abs/1510.03078v1), [arXiv:1512.09134](https://www.arxiv.org/pdf/1512.09134)).
The 1-D solver for this already exists in `qcore.js`.

## 4. Effects worth adding

### Gravity and light (G with c)
| Effect | What you would do | Real experiment |
|---|---|---|
| Gravitational redshift | Ride the lift, point the spectrometer down and up | Pound–Rebka 1960 |
| Clocks and height | Go up, wait, come down, compare | Chou et al. 2010; Hafele–Keating 1971 ([Wikipedia](https://en.wikipedia.com/wiki/Hafele-Keating_experiment)) |
| Photon orbit | Plant a flag behind you; find the compactness where it appears ahead at eye level | — |
| Lensing by a small dense object | A heavy marble hovering above the street makes an Einstein ring of the lamps behind it | Exact only when the planet's own gravity is weak; with both strong it would be an approximation, and the lab should say so |
| Shapiro delay | Bounce a radar pulse off the far side | Shapiro 1964 (from memory) |
| Orbit precession | Throw a ball into orbit | Mercury's perihelion |
| Tides | Drop a long rod and watch it stretch | — |

### Quantum (ħ)
| Effect | What you would do |
|---|---|
| **Uncertainty principle** | A live meter shows Δx, Δp and their product against ħ/2. "Look" becomes a microscope: choose the light's wavelength; a sharper look gives a bigger momentum kick. This connects the two ħ dials: short-wavelength photons carry more momentum. Background: [Heisenberg's microscope](https://en.wikipedia.org/wiki/Heisenberg%27s_microscope) and a corrected textbook model, [arXiv:1712.08579](https://arxiv.org/pdf/1712.08579). I found no existing interactive that shows the kick. |
| Building up interference | Go through the two doors many times; each run leaves one mark on the hit wall |
| Tunnelling | Walk at a thin wall slower than it would take to climb it |
| Single photons | Lower the photon dial until you can no longer read the clock. People can detect a single photon slightly above chance ([Tinsley et al. 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4960318)) |
| Zero-point motion | Lamps on springs and light objects jitter even at rest |

### All three at once
**A clock in two places.** Your wave splits by height (two staircases), then rejoins. Each branch's clock
ages differently, so your watch records which way you went and the interference fades. Proposed by
Zych et al. in 2011 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3221301)); simulated in the lab by
Margalit et al. in 2015 using a magnetic field instead of gravity ([arXiv:1505.05765](https://arxiv.org/pdf/1505.05765)).
The gravitational version has not been done; analyses differ on which interferometers are sensitive to it
([arXiv:1905.09102](https://arxiv.org/pdf/1905.09102v1)). Best built as a two-path model with exact phases
rather than a full wave simulation.

## 5. Other constants

Only dimensionless combinations matter, so each new dial should be one.

| Dial | What it would change in the world | Notes |
|---|---|---|
| Fine-structure constant α | Atomic line wavelengths scale as 1/α²: sodium lamps change colour, the sky's colour changes; above a threshold no atoms | The cleanest "constant of nature" to add next |
| Electron-to-proton mass ratio β | Molecules, the hardness of materials | Together with α forms the habitable-region map in fine-tuning studies ([Adams, arXiv:1902.03928](https://arxiv.org/pdf/1902.03928v1)); could be a second map view |
| Gravitational coupling α_G = G m_p²/ħc | How massive stars are (about α_G^(−3/2) proton masses) | A "night sky" view |
| Cosmological constant Λ | A horizon at a set distance: far objects redden and recede | A de Sitter game already exists ([arXiv:2307.15814](https://arxiv.org/html/2307.15814v2)) |
| Decoherence rate | How fast the environment "looks" at you; turns the "Always watched" switch into a dial | Explains why footballs are never waves |

## 6. Missions

Each mission: a short brief → a prediction you commit to → exploration with only the needed dials unlocked
→ a measurement with an instrument → the explanation, revealed with the probe → the real experiment.

| # | Mission | Prediction asked | Dials unlocked | Real counterpart |
|---|---|---|---|---|
| 1 | The lamps flash together | Which lamp do you see flash first? | c | — |
| 2 | Round the planet | You sprint round and stop by the start clock. Which is ahead? | c | Hafele–Keating |
| 3 | The tower | Ride up 50 m. What colour is the street lamp below? | G | Pound–Rebka, Chou 2010 |
| 4 | Older upstairs | Wait at the top for a minute. How far ahead is your watch? | G | Chou 2010 |
| 5 | Light orbits | At what compactness does the flag behind you appear ahead? | G | — |
| 6 | The radar | Is the echo past the planet late or early? | G, c | Shapiro |
| 7 | Two doors | Where will the marks pile up? | ħ | Electron and molecule double slits |
| 8 | Look closer | Look with blue light, then red. Which makes you jump more? | ħ, photons | Heisenberg microscope |
| 9 | Hover | Drop a ball from 2 m. Does it come to rest on the floor? | ħ, G | qBounce |
| 10 | How few photons | When can you no longer read the clock? | photons | Tinsley 2016 |
| 11 | Two staircases | Will the stripes survive if the stairs differ in height? | c, G, ħ | Zych 2011, Margalit 2015 |

## 7. Libraries and engines

| Need | Option | Verdict |
|---|---|---|
| Rendering | Current raw WebGL 2 ray tracer | Keep. It is exact and small. |
| | three.js with its shading language (TSL), which targets WebGPU and falls back to WebGL; a TSL black-hole ray marcher exists as a reference ([Singularity](https://discourse.threejs.org/t/black-hole-singularity/92140.md)) | Worth it only if the town needs detailed models |
| Bigger wave simulations | WebGPU compute shaders (FFT on the GPU) | Needed for a 256² or 3-D wave, or several waves at once |
| Checking geodesics and orbits | [EinsteinPy](https://pypi.python.org/pypi/einsteinpy) (Python, MIT licence): timelike and null geodesics in Schwarzschild, Kerr, Kerr–Newman | Use offline to test the new orbit and lift code |
| Quantum references | [Schroeder's Weber State apps](https://physics.weber.edu/schroeder/software/) (open source); [Falstad's applets](https://www.falstad.com/mathphysics.html) (licence not stated); Quantum Flytrap's Quantum Tensors and BraKetVue (open source) | For checking and for ideas, not as dependencies |
| Interface | lil-gui or Tweakpane (developer panel), KaTeX (formulas), SceneryStack (accessible 2-D panels, licence to check) | See section 2 |

## 8. Build order

| Phase | Contents | Why first |
|---|---|---|
| 1 | Lift and hover; spectrometer; per-effect switches; the probe panel | Turns the current world into something that explains itself, using code that already exists |
| 2 | Mission framework (predict–observe–explain), missions 1–5 and 7 | The teaching structure; mostly interface work |
| 3 | Moving objects (thrown balls, orbits, satellite lamps); uncertainty meter and microscope; missions 6, 8, 9 | New physics, each piece testable against formulas |
| 4 | α and Λ dials; two-staircase clock interferometer; lens object; mission 11 | The research frontier and new constants |
| 5 | Try it on classmates: a short pre/post concept quiz and think-aloud sessions | The evidence base for this kind of tool is thin; your own data is worth having |

## 9. Measurement: what students find hard, and how the world teaches it

**What research finds.** Students of all levels struggle with quantum measurement. Common ideas are that
measurement needs a conscious observer, that it reveals a value the particle already had, and that after a
measurement the state stays collapsed or returns to the original if you wait
([Zhu and Singh 2012](https://arxiv.org/pdf/1602.05720); tools that helped: [part 2](https://arxiv.org/pdf/1602.06926)).
Two-state questionnaires on measurement now exist ([Bitzenbauer 2024](https://doaj.org/article/8e8dbaf6be6844fb973958b2a5f103b2)).
QuVis found that how single-photon superposition is drawn changes what students take from it
([Kohnle et al.](https://arxiv.org/pdf/1410.0867)).

**Good explainers.** [PBS Space Time on decoherence](https://www.pbs.org/video/how-decoherence-splits-the-quantum-multiverse-b29lll)
(addresses the conscious-observer idea head on); [Matt Strassler on why measurement destroys interference](https://profmattstrassler.com/?p=21478);
[Hsu, "the missing lecture"](https://arxiv.org/pdf/2212.02391) for after a first course. Interactive:
[Quantum Flytrap Virtual Lab](https://arxiv.org/abs/2203.13300) and QuVis cover photon interferometers,
the eraser and the [Elitzur–Vaidman bomb tester](https://en.wikipedia.org/wiki/Elitzur%E2%80%93Vaidman_bomb_tester).
None of these puts *you* in the interferometer, which is what the world can do.

**The ideas, and the control that shows each**

| Idea | In the world | Real experiment |
|---|---|---|
| Outcomes are random with probability \|ψ\|²; there was no single place before | Rewind and look again: marks pile up where the map is bright | Every single-particle double slit |
| A measurement is any interaction that leaves a record; nobody needs to read it | The bell at the left door fades the stripes | Which-path experiments; heated C70 molecules lose fringes by emitting light ([Hackermüller et al. 2004](https://arxiv.org/abs/quant-ph/0402146)) |
| How much you can know trades against how much interference survives, D² + V² ≤ 1 | Set the bell between 0 and 100% | [Schwindt, Kwiat and Englert 1999](https://arxiv.org/abs/quant-ph/9908072v1) |
| Records can sometimes be undone | Erase the bell's record | Quantum eraser |
| Measuring disturbs; Δx·Δp ≥ ħ/2 | Microscope sharpness, the live meter | Heisenberg's microscope (thought experiment) |
| After a look the state is updated, then evolves | Look twice quickly, then wait | — |
| Big things look definite because the world keeps measuring them | The world looks at you 16 times a second | Decoherence by collisions and thermal emission |
| Which outcome happens is not explained by decoherence | Stated in the page; "Look" uses the textbook rule | Interpretations differ |

**Still to do.** A real detector screen across the yard that records arrivals by itself; a visibility
meter computed from the marks; a bomb-tester room (detect a bomb behind a door without touching it);
and showing the bell's own state as part of the picture, so it is clear that you and the bell are now
one entangled system.
