# cGħ World

A first-person walk round a small planet, with four dials that decide how strongly relativity, gravity
and quantum mechanics show up at human scale. Each dial is a pure number; the page translates it into
c, G and ħ.

Why it exists, what else is out there, and what to build next: [RESEARCH.md](RESEARCH.md).

## Use it

Open `index.html` in a browser (one self-contained file; needs WebGL 2). W/S change effort, A/D steer,
drag to look, F to look where you are, Space to stop. Presets set the dials to interesting places.

| Dial | What you see |
|---|---|
| **c**: light in walking speeds | Moving squeezes the view ahead, blue-shifts and brightens it; far clocks read early. |
| **G**: planet compactness r_s/R | Light falls. The horizon rises; at 2/3 light circles the planet at eye level; beyond, the ground wraps over you and the sky shrinks overhead. Higher clocks run fast. |
| **ħ**: your wavelength at walking pace | You are a wavefunction. You spread, reflect from walls, split at two doors; the view overlays every place you might be. Looking collapses you. A map shows |ψ|². |
| **ħ (light)**: photons per pixel | Light arrives in lumps; the picture turns to grain. |

## How it works

- `src/js/shaders.js` traces one light ray per pixel backwards from the eye. Rays follow exact Schwarzschild
  null geodesics (`x'' = -(3/2) r_s L² x / r⁵`), sphere-traced against the street, with coordinate time
  accumulated for light delay and gravitational shift. Your motion enters as exact aberration and Doppler
  shift at the eye. A second pass averages frames; a third draws Poisson photon counts.
- `src/js/wave.js` solves your 2-D wavefunction on a 32 m window that follows you (split-step Fourier,
  relativistic dispersion `E = sqrt((ħkc)² + (mc²)²) - mc²`). Everything standing on the ground is a wall.
  Walking applies a uniform force, i.e. shifts the whole momentum distribution. Each frame the camera stands
  at one position drawn from |ψ|².
- `src/js/geo.js` holds the planet, the street layout and a JavaScript copy of the ray integrator for tests.

```
node build.mjs           # writes index.html and dist/cgh-lab.html
node --test test/*.js    # 27 checks
```

The tests cover: light bending against 2r_s/b plus the second-order term, the photon orbit at 1.5 r_s,
the Shapiro delay of a radial ray, the horizon formula against traced rays, aberration and Doppler,
the Kerr and Airy results kept from the first version, free spreading, tunnelling, the relativistic group
velocity of a wave packet, and the solver's window shift.

## Limits

Colours are a guessed spectrum. Sunlight is not bent. The planet's mass is set by decree (a real 382 m rock
has r_s/R ≈ 3×10⁻¹⁸). The overlay of views is an average that no single observer would see. The wave grid is
0.25 m, which caps your speed when you are a wave. Light rays do not see moving objects, so nothing in the
town moves except you.

**Tested how:** the physics by the Node tests; the page in headless Chromium at desktop and phone widths,
light and dark, with no script errors. That browser renders on the CPU (about 7 frames a second at reduced
resolution), so speed on a real graphics card and on a real phone is untested. The renderer lowers its own
resolution when frames are slow.
