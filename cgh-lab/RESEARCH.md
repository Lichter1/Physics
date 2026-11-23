# Building intuition by changing the constants: what exists and what is missing

Research notes, November 2026. Written to answer three questions:

1. Is "change a constant until the effect shows up at human scale" a recognised way to build intuition?
2. What already exists, and where should you go to use it?
3. What is missing, and what does the lab in this folder add?

## 1. Is this a recognised approach?

Yes. It is about 85 years old and it has some evidence behind it.

- **George Gamow, *Mr Tompkins in Wonderland* (1940).** The hero dreams of a town where light moves at
  bicycle speed, a billiard hall where Planck's constant is large, and so on. His initials, C. G. H.,
  are the three constants. This is the same idea as yours, in prose.
  [Wikipedia](https://en.wikipedia.org/wiki/Mr_Tompkins)
- **The cube of theories** (Gamow, Ivanenko and Landau 1928; Bronstein 1930s; Zel'manov; Okun). Put 1/c, G
  and ħ on three axes. Each corner of the cube is a theory: Newtonian mechanics at the origin, special
  relativity, Newtonian gravity, quantum mechanics on the axes, then general relativity, quantum field
  theory, quantum mechanics in a gravitational field, and at the far corner a theory nobody has.
  [Wikipedia: cGh physics](https://en.wikipedia.org/wiki/CGh_physics)
- **Games built on it.** MIT Game Lab's *A Slower Speed of Light* (2012) is exactly the first-person,
  slow-light game you describe. Real Time Relativity (ANU, 2007) is the academic predecessor.
- **Evidence that it teaches.** Real Time Relativity was studied over four semesters at two universities:
  students found relativity less abstract, concept-test scores on time dilation improved, and exam marks
  were slightly but significantly higher for the group that used it. The studies are small, from one
  research group, and mostly first-year cohorts, so treat them as encouraging, not conclusive.
  [Savage et al. 2007](https://arxiv.org/abs/physics/0701200),
  [McGrath et al. 2010](https://arxiv.org/abs/0911.0226),
  [Savage et al., Teaching physics using virtual reality](https://arxiv.org/abs/0910.5776)

One caveat matters for everything below. **Only dimensionless ratios are physical.** "Set c to 10 m/s" has
no meaning unless you say what stays fixed. If atoms stay as they are, you have really changed the
fine-structure constant α = e²/4πε₀ħc, and chemistry is gone. Gamow sidestepped this by decree, and so does
every game in this list. The standard reference is the three-way argument between Duff, Okun and Veneziano,
[Trialogue on the number of fundamental constants](https://arxiv.org/abs/physics/0110060). A 2022 paper
actually ran quantum chemistry in a "Mr Tompkins world" and found that familiar matter survives only in a
narrow range of the constants: [Dergachev et al.](https://arxiv.org/abs/2202.04228).

## 2. What exists

### Special relativity: well served. Use these.

| Tool | What it is | Platform | Status |
|---|---|---|---|
| [Relativity Park](https://rivendell.dmitrybrant.com/relativity/) (Dmitry Brant, 2026) | Walk through a park with c = 5 km/h, adjustable. Light delay, aberration, Doppler colour and beaming each have a toggle. Moving Ferris wheel, carousel and shuttle; watch and world clock. | Browser, WebGL 2, nothing to install | [Source on GitHub](https://github.com/dbrant/relativity). The pages I read state no licence, so ask before reusing code. |
| [A Slower Speed of Light](https://gamelab.mit.edu/games/a-slower-speed-of-light/) (MIT Game Lab, 2012) | First-person game: collect orbs, each one lowers c. The original. | Desktop download (Unity). A third-party [web edition](https://theandelope.itch.io/a-slower-speed-of-light) exists; I did not verify it. | Finished prototype |
| [OpenRelativity](https://github.com/MITGameLab/OpenRelativity) | The toolkit behind the MIT game. Drop it into a Unity project to make any scene relativistic. | Unity, MIT licence | Upstream dormant since 2020. An active fork, [vm6502q/OpenRelativity](https://github.com/vm6502q/OpenRelativity), adds acceleration, fixed curved backgrounds and approximate black-hole shaders (commits as recent as November 2026). Its README is explicit that it does not solve Einstein's equations. |
| Real Time Relativity (ANU) | First-person relativistic flight through a simple scene, with published classroom studies | Desktop | Research software, 2007 |
| Velocity Raptor (TestTubeGames) | 2-D puzzle game; switches between "what is there" and "what you see" | Browser | Finished game |
| [VirtualRelativity](https://arxiv.org/abs/2408.01442) (2024) | Special relativity in a VR headset | VR | Research prototype |
| [spacetimetravel.org](https://www.spacetimetravel.org) (Kraus and Zahn) | Films and papers: cycling through Tübingen at near light speed, and much more | Web | Long-running teaching project |
| [harrygifford/special-relativity](https://github.com/harrygifford/special-relativity), [Dmytry/space](https://github.com/Dmytry/space) | Small WebGL demos | Browser | Hobby projects |

**Recommendation:** for a polished slow-light walk, open Relativity Park today. For a headset or a native
game, start from the OpenRelativity fork.

### General relativity: good pictures, little lived experience

- **Black-hole renderers** are plentiful and beautiful: [oseiskar/black-hole](https://github.com/oseiskar/black-hole),
  [adriwin06's](https://adriwin06.github.io/black-hole/), [ESA's WebGL demo](https://www.esa.int/gsp/ACT/phy/Projects/Blackholes/WebGL/),
  [Gravy](https://github.com/portsmouth/gravy) for lensing, and research ray-tracers such as GYOTO.
  They show what a black hole looks like. They do not let you live through a clock running slow.
- ***Interstellar*** itself: the renderer is described in
  [James, von Tunzelmann, Franklin and Thorne 2015](https://arxiv.org/abs/1502.03808), and the physics in
  Thorne's book *The Science of Interstellar*. The code is not public.
- **Time-dilation play** is thin: a blog-post game, [Gravity Well Arena](https://sean-reid.github.io/blog/gravity.html),
  and assorted calculators. The OpenRelativity fork has approximate gravity.
- **Teaching without a screen:** Kraus and Zahn's [sector models](https://arxiv.org/abs/1405.0323) are paper
  cut-outs of curved space. A game in de Sitter space also exists: [arXiv 2307.15814](https://arxiv.org/abs/2307.15814).

I found nothing where you choose an orbit, wait, and compare watches with someone who stayed behind,
with the film's own numbers.

### Quantum mechanics: many tools, but ħ stays where it is

The field is surveyed in [Seskir et al. 2022, Quantum games and interactive tools](https://arxiv.org/abs/2202.07756).
Highlights:

- **Simulation suites:** PhET (Colorado), QuVis (St Andrews), and Daniel Schroeder's
  [Weber State simulations](https://physics.weber.edu/schroeder/software/). Clear, accurate, atomic scale.
- **Games:** Quantum Moves (move atoms with optical tweezers, [paper](https://arxiv.org/abs/1506.08761)),
  [Particle in a Box](https://learnqm.gatech.edu) (a platformer with a classical and a quantum world, with a
  measured gain in students' grasp of probability), Quantum Flytrap's Virtual Lab (optics table), Quantum Odyssey.
- **Closest to Gamow:** [Quantum Minigolf](https://quantumminigolf.sourceforge.net/) (2007, GPL): the ball is a
  wave packet, and looking at it collapses it.

None of the ones I found offers ħ as a dial, or lets a football and a neutron be the same problem at
different settings.

### All three constants on one panel

I searched for this specifically and found nothing: no tool where c, G and ħ are three controls acting on
the same everyday scene, and no interactive version of the cube of theories. That is "not found", not
"does not exist".

## 3. What is missing

1. **One control panel.** Each existing tool lives at one corner of the cube. Nothing shows that the corners
   are the same world with different dials, which is the whole content of the cube.
2. **Gravity you wait through.** Clocks at different heights or orbits, compared after the fact.
3. **ħ at the scale of a football,** including the place where it meets gravity (a ball that cannot rest on
   a floor), which is a real experiment done with neutrons.
4. **Honesty about the decree.** Tools change c without saying what happens to α, to atoms, or to the Earth
   (which becomes a black hole long before c reaches 10 m/s).
5. **The empty corner.** Nothing marks where known physics stops.

## 4. What this folder adds: cGħ Lab

A single web page, no installation, with three dials and four tabs:

- **Map.** The cube of theories with your dial settings plotted in it, and a table of what those settings do
  to a bike ride, a tenth-floor clock, the Earth, a football and an atom.
- **Light.** A first-person slow-light street. This overlaps with Relativity Park on purpose, so that the
  lab is complete; it adds overhead clocks and a "what is there" view beside "what you see".
- **Gravity.** A tower whose floors age at different rates, and the *Interstellar* mission with the film's
  numbers (it reproduces one hour for seven years, and the 1.7-hour orbit).
- **Quantum.** Two doors, a fence, and a dropped ball, for a football with ħ turned up.

It was written from scratch with no dependencies. OpenRelativity is Unity-only and could not be merged into a
web page; Relativity Park carries no stated licence.

Limits are listed in the README. The main ones: the tower is a uniform field, the black hole has no visuals
of lensing, the quantum scenes are one isolated particle, and nothing is simulated at the corner where all
three constants matter, because there is no theory to simulate.

## 5. Can this help with research at the boundaries?

A simulator computes consequences of equations someone already wrote down. It cannot discover a law that is
not in its code. What it can do:

- make you fluent in the dimensionless numbers that decide which theory you need;
- let you compare rival hypotheses in a regime no experiment reaches yet, as long as each is labelled as
  a hypothesis;
- show how big an effect would be at a given scale, which is how table-top experiments get designed.

The intersections are where experiments are live. These are standard references given from memory; check
them before citing.

| Corner | What has been measured | What is open |
|---|---|---|
| c and G | Pound and Rebka 1960 (redshift over 22 m). Chou et al., *Science* 2010 (time dilation at 10 m/s and over 33 cm). Bothwell et al., *Nature* 2022 (redshift across one millimetre). GPS. LIGO. | Strong fields, horizons |
| G and ħ | Colella, Overhauser and Werner 1975 (neutron interference shifted by Earth's gravity). Nesvizhevsky et al., *Nature* 2002 and the qBounce experiments (neutrons in discrete states above a mirror, the Drop scene). Overstreet et al., *Science* 2022 (gravitational Aharonov-Bohm effect). | Gravity as a classical field acting on quantum matter is confirmed. Whether gravity sourced by quantum matter is itself quantum is not. |
| G of small masses | [Westphal et al., *Nature* 2021](https://arxiv.org/abs/2009.09546) (gravity between millimetre gold spheres) | Pushing the source mass down to where it can be put in superposition |
| c, G and ħ | Nothing directly. Analogue Hawking radiation in a condensate (Steinhauer 2016) tests the mathematics, not gravity. | Everything. The leading table-top proposal is gravity-mediated entanglement (Bose et al. 2017; Marletto and Vedral 2017). It has not been done, and what a positive result would prove is disputed. |

Reviews: [Carney, Stamp and Taylor, a user's manual](https://arxiv.org/abs/1807.11494);
[Bose et al., Massive quantum systems as interfaces of quantum mechanics and gravity](https://arxiv.org/abs/2311.09218).
Critiques: [Huggett, Linnemann and Schneider](https://arxiv.org/abs/2205.09013);
[Martín-Martínez and Perche](https://arxiv.org/abs/2208.09489).

## 6. Where to take it next

Ordered by how much each would add for a study group:

1. **Guided missions.** Predict, then observe: the twin trip, the pole in the barn, which-door detection.
2. **A black hole you can look at.** A ray-traced view from the planet and the ship, and light signals between
   them, so that you see the other clock and not only its rate.
3. **Acceleration in the Light scene.** A horizon forms behind an accelerating rider, which is the same
   horizon as the one under the tower.
4. **Decoherence as a fourth control,** to show why footballs do not diffract even in principle.
5. **A hypothesis bench for the empty corner.** Small calculators for Schrödinger-Newton self-gravity,
   Diósi-Penrose collapse times and the entanglement phase in the Bose-Marletto-Vedral proposal, each
   labelled with the experiment that would test it.
6. **A headset version,** probably by porting scenes to the OpenRelativity fork.
