# cGħ Lab

Three dials, c, G and ħ, and a handful of everyday scenes that respond to them. The aim is intuition for
special relativity, gravity and quantum mechanics by bringing their effects to walking pace, the way
George Gamow did in *Mr Tompkins*.

Why it exists, what else is out there, and what to build next: [RESEARCH.md](RESEARCH.md).

## Use it

Open `index.html` in a browser. It is one self-contained file: no server, no installation.
The Light tab needs WebGL 2 (any current browser).

| Tab | What you do |
|---|---|
| Map | See which theory a person needs at the current dial settings, plotted in the cube of theories, and what the settings do to a bike ride, a tenth-floor clock, the Earth, a football and an atom. |
| Light | Ride down a street with a slow speed of light. W/S change effort, A/D steer, drag to look. Compare "what you see" with "what is there". |
| Gravity | Wait on different floors of a tower and compare watches; or send a crew to a planet skimming a spinning black hole (the *Interstellar* numbers). |
| Quantum | Roll a football at two doors, at a fence, or drop it on the floor, with ħ large enough to see. |

Each scene tells you when the current dials make its effect invisible and offers the setting that reveals it.

## Change it

```
src/page.html      markup
src/style.css      styles
src/js/physics.js  formulas, no DOM (tested)
src/js/qcore.js    Schrödinger solver, no DOM (tested)
src/js/app.js      dials, formatting, tabs, the cube, the Map tab
src/js/light.js    Light tab (WebGL 2)
src/js/gravity.js  Gravity tab
src/js/quantum.js  Quantum tab
```

```
node build.mjs                    # writes index.html and dist/cgh-lab.html
node --test test/physics.test.js  # 18 checks of the physics
```

No dependencies; Node 18 or later for the build and tests.

## What the physics is, and is not

**Light.** Every vertex is drawn at the event where its light left it (its worldline meets your past light
cone), transformed into your frame. That gives aberration, the apparent rotation and stretching of objects,
and light-travel delay in one step. Doppler shift and beaming follow from the same event. Exact for a
static street and for trams at constant velocity.
Not modelled: your own acceleration (speed changes are instant), shadows, and real spectra. Colour uses a
smooth guess at each surface's reflectance from ultraviolet to infrared, lit by a 5800 K black body.

**Gravity, tower.** A rigid tower at constant proper acceleration: rate(h) = 1 + gh/c², exact, with a horizon
c²/g below the floor. g is Earth's pull under your G. The planet itself is left out, because with a slow
speed of light it would be inside its own horizon. Light delay between floors is not drawn; rates are.

**Gravity, mission.** Circular, prograde, equatorial orbits in the Kerr metric. Tested against the film:
spin 1 − 1.3×10⁻¹⁴ gives a factor near 61,000 at the innermost stable orbit and a 1.7-hour orbit around a
hundred-million-solar-mass hole. There is no picture of the hole's lensing. The Hawking temperature and
lifetime are the standard semiclassical formulas.

**Quantum.** The time-dependent Schrödinger equation for one particle by the split-step Fourier method,
128 × 128 grid for the yard and 512 points for the drop (which separates). Walls are high finite
potentials; yard edges absorb. Tested: free spreading, norm conservation, tunnelling against the plane-wave
formula, free fall, and the stationary Airy state. One isolated particle with no environment: a real
football would decohere at once.

**All three at once.** Not simulated. No accepted theory exists, and the lab says so.

**The decree.** Changing c or ħ alone changes the fine-structure constant and every atom. The lab keeps
people, streets and footballs as they are and changes only their mechanics. The Map tab shows what that hides.

## Tested how

The formulas and the solver are covered by the Node tests. The page was run in headless Chromium at desktop
and phone widths, in both themes, with no script errors. That browser renders WebGL in software (about
4 frames a second for the Light tab), so frame rate on a real graphics card and behaviour on a real phone
are untested.
