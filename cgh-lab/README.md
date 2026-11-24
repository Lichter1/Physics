# cGħ World

A first-person walk round a small planet, with four dials that decide how strongly relativity, gravity
and quantum mechanics show up at human scale. Each dial is a pure number; the page translates it into
c, G and ħ.

Why it exists and what else is out there: [RESEARCH.md](RESEARCH.md). What comes next: [ROADMAP.md](ROADMAP.md).

## Use it

Open `index.html` in a browser (one self-contained file; needs WebGL 2).
W/S effort, A/D steer, E/Q up and down, drag to look, F to look where you are, Space to stop.

| Dial | What you see |
|---|---|
| **c**: light in walking speeds | Moving squeezes the view ahead, blue-shifts and brightens it; far clocks read early. |
| **G**: planet compactness r_s/R | Light falls. The horizon rises; at 2/3 light circles the planet at eye level; beyond, the ground wraps over you. Fly up and lamps below turn red; higher clocks run fast. |
| **ħ**: your wavelength at walking pace | You are a wavefunction: you spread, reflect, split at two doors; the view overlays every place you might be. |
| **ħ (light)**: photons per pixel | Light arrives in lumps; the picture turns to grain. |

**Effect switches** turn off one process at a time: light bending, light delay, aberration, colour shift,
gravity's shift, brightness shift, clock rates. "Newton's view" turns them all off.

**Light probe.** For the pixel under the crosshair: a slice through the planet with the light's actual
path and a straight line for comparison, a spectrometer (sent and received wavelength), and in words:
what it is, when its light left, how much it bent, and each shift factor.

**Measuring you** (when you are a wave):
- *Look where I am*, with a chosen microscope sharpness; *Rewind and look again* repeats the measurement
  on the same wave, so outcomes pile up as yellow marks on the map.
- *Only count looks behind the wall*: a detector screen for the two-door experiment.
- *Bell at the left door*: a detector that records who passes. The stripes fade even though nobody
  reads it. *Erase the bell's record* brings them back (quantum eraser).
- *The world looks at you*: repeated environmental looks, i.e. decoherence. You stay sharp but jitter.
- A live Δx, Δv and Δx·Δp meter (never below ħ/2).

## How it works

- `src/js/shaders.js`: one light ray per pixel, traced backwards along exact Schwarzschild null geodesics
  (`x'' = -(3/2) r_s L² x / r⁵`), sphere-traced against the street, with travel time and gravitational
  shift. Your motion enters as exact aberration and Doppler shift. Uniforms switch each effect off.
- `src/js/geo.js`: the planet, the street, and JavaScript copies of the scene and of the shader's ray march
  (used by the probe, by walking into things, and by the tests).
- `src/js/wave.js`: your 2-D wavefunction on a 32 m window that follows you (split-step Fourier with
  relativistic dispersion), a second component for the bell's "rang" state, Gaussian position looks,
  rewinding, and the eraser.
- `src/js/world.js`: walking and flying, the GPU passes, the HUD and the map. `src/js/probe.js`: the probe.

```
node build.mjs           # writes index.html and dist/cgh-lab.html
node --test test/*.js    # 29 checks
```

Tests cover light bending, the photon orbit, Shapiro delay, the horizon formula against traced rays,
aberration and Doppler, free spreading, tunnelling, the relativistic group velocity, Δx·Δp ≥ ħ/2 after
sharp and blurry looks, repeatable looks, and the two-door experiment: the bell lowers the stripes'
contrast and erasing its record restores it.

## Limits

Colours are a guessed spectrum. Sunlight is not bent. The planet's mass is set by decree. The overlay of
views is an average no single observer sees. Your wave lives on the ground (flying turns it off) on a
0.25 m grid, which caps your speed as a wave. Nothing in the town moves except you.

**Tested how:** the physics by the Node tests; the page in headless Chromium at desktop and phone widths,
light and dark, with no script errors. That browser renders on the CPU at a few frames a second, so speed
on a real graphics card and on a real phone is untested; the renderer lowers its own resolution when
frames are slow.
