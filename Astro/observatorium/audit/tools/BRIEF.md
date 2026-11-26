# Observatorium audit: brief for every reviewer

You are reviewing one part of the **Observatorium**, a browser game/simulation for the university course General
Astronomy 1 (BGU, lecturer Adi Zitrin). A student explores a real sky and a "data lens" (charts of real
catalogues), and retraces historical discoveries in **missions**. The goal of this audit, in the owner's words:
make it **more user-friendly, understandable, interactive and visual, bug-free, and give the student knowledge
and intuition**. The owner has seen exercises that ask the student to choose an axis or variable that is **locked**
even though the earlier missions were done, and other bugs. Find them all in your part.

**This is a review. Do not change anything in the repository.** Do not edit files under
`/home/user/Physics/`, do not run `node build.mjs`, do not run git commands that write (commit, push, checkout,
stash), and do not stop the web server. Write only inside your own folders under this audit directory:
`AUDIT=/tmp/claude-0/-home-user-Physics/af81278e-c676-5039-8223-e0e8de9aabda/scratchpad/audit`
(your scripts in `$AUDIT/work/<your-part>/`, screenshots in `$AUDIT/shots/<your-part>/`, the report in
`$AUDIT/reports/<your-part>.md`).

## The app

- Source: `/home/user/Physics/Astro/observatorium/` (read `README.md` and `DESIGN.md` first, briefly).
  Vanilla JS modules on `window.OBS`, Canvas 2D.
  - `src/js/missions.js`: the chapters (`OBS.CHAPTERS`), every mission (`OBS.MISSIONS`: brief, vars, relation,
    deduce, question/choices/answer, `setup(app, st)`, `task`, `progress`, `result`, `explain`, `real`,
    `unlocks`, `needs`), and the quiz.
  - `src/js/lectures.js` (chapters 1–5) and `src/js/lectures2.js` (chapters 6–9): each mission's lecture beats
    (`h`, `html`, `eq`, `ex` = live explainer, `sky` = sky cue, `fig` = picture or live chart, `example`) and
    `practice` problems.
  - `src/js/catalog.js`: datasets and variables; `OBS.CONCEPTS` and which mission unlocks each; a variable
    with `needs: [...]` is **locked** in the axis menus until those concepts are unlocked (or teacher mode is on).
  - `src/js/instruments.js`, `src/js/observing.js`: the exercise instruments. `src/js/explainers.js`,
    `src/js/explainers2.js`: live diagrams. `src/js/sky.js`, `src/js/chart.js`, `src/js/main.js`,
    `src/js/logbook.js` (mission runner), `src/js/lecture.js`, `src/js/tour.js`.
  - `data/`: the real datasets (`data/manifest.json` describes each).
- Running page: **http://localhost:8765/preview.html** (already served from `dist/`). If it does not answer,
  start it with `cd /home/user/Physics/Astro/observatorium/dist && nohup python3 -m http.server 8765 >/dev/null 2>&1 &`
  (if the port is taken someone else already did).
- Chromium and Playwright are installed: `require('playwright')` works from any directory. Run **one browser at a
  time**, close it when done, and wrap runs in `timeout`. Nine other reviewers share the machine.
- **Harness** (use it): `$AUDIT/harness.js`. Read it; it has `open({before: '<missionId>'})`, which loads the page as
  a student who has finished every mission *before* that one, in course order, with teacher mode off. It also has
  `start`, `readLecture`, `toExercise`, `lensMenus` (the axis menus with their locked options), and `shot`.
- `e2e/missions.js` shows how the automated test solves each mission. It **calls JavaScript directly**
  (`setAxes`, `setSel`, slider `.set`), which bypasses locked menus. That is why the locked-axis bugs slipped
  through. **You must play through the real UI**: `page.selectOption('#x-sel', ...)`, clicks, drags on the chart
  canvas, the sliders and buttons the student sees. Use JS only to diagnose.
- Mission flow: the mission card (step 0) → *Lecture* (beats, Next) → *Predict* (choose, locked in) →
  *Exercise* (`#m-status` shows progress; `#m-next` enables when done) → *Explain* (`.verdict`, `.result`,
  then the explanation and practice problems).

## What to check for every mission in your part

1. **Can a student finish it through the UI alone, after only the earlier missions?** Use `H.open({before: id})`.
   Do every instruction in the `task` text and in the lecture literally: every dataset, axis, colour-by
   option, subset, tool or button it names must exist, be findable and be **unlocked**. Record each locked or
   missing item with the mission that unlocks it. Check that the `needs` and `unlocks` order makes sense across
   the whole course, not just inside your chapter.
2. **Bugs**: console errors (the harness collects them), wrong or missing UI state after setup, instruments that
   misbehave (dragging, sliders, resizing the lens, switching dataset mid-mission), progress that never reaches
   done or reaches it too easily, results with wrong numbers or units, text that contradicts the screen, sky cues
   that point at the wrong place, broken pictures or charts in the lecture, and layout problems at **1600×950,
   1100×800 and 390×844** (phone). Also try leaving a mission half-way, starting another, and coming back.
3. **The lecture**: compare with the lecture slides for your part (text extracted in `$AUDIT/sources/`; PDFs for
   lectures 7–13 are there too: render a page with `pdftoppm -f N -l N -r 60 -png` if you need to see a figure).
   Is the physics right? Are the equations right? What important topics or figures from the slides are missing?
   Is it understandable for a second-year student? Is it too wordy? Does it build *intuition* (what to picture,
   orders of magnitude, why it matters), not only formulas?
4. **The exercise and the explanation**: is the task clear? Is the prediction question good (does it reveal a
   misconception)? Is the feedback helpful when the student is wrong or stuck? Are the thresholds fair? Do the
   result numbers make physical sense? Are the practice problems and worked solutions correct (check the arithmetic)?
5. **Views and interactivity**: sky cues, photo/drawn sky, wavelengths (*Light*), frames (equatorial, Galactic,
   horizon), scale ladder, data-lens charts, explainers. What confuses? What is static that should be interactive?

## Suggestions (be concrete)

- **New simulations or interactive views** that would give intuition for a concept in your part. For each, say
  what the student does, what they see and which misconception it fixes. Say how hard it is to build in this code
  (which file or instrument it would extend).
- **New pictures**: name the specific image (for example "ESO eso1907a, the M87 black hole", or "NASA/JPL PIA…"),
  where it would go, and its licence. You may check that a URL exists; do not download into the repository.
- **Missing missions or exercises** for topics the slides teach but the game skips, with the real dataset that
  could carry them.

## Report format (`$AUDIT/reports/<your-part>.md`)

```
# <Part>: audit
## Summary            (5 lines max: state of the part, the worst problems, the top 3 improvements)
## Bugs
| ID | Severity | Mission | Where | What happens (steps) | Expected | Likely cause (file:line) | Fix |
   Severity: blocker (cannot finish through the UI) / major / minor / cosmetic. ID: <part>-B1, B2...
## Mission by mission  (for each: Lecture · Exercise · Explain & practice · Views; short bullets, specific)
## Gaps against the slides
## Suggestions: simulations and interactive views
## Suggestions: pictures
## Suggestions: new missions or exercises
```

Every bug must have reproduction steps you actually ran, and a file:line where you can find one. Mark anything
you did not verify as "(not verified)". Keep the report specific and skimmable. Finish with a two-line reply that
gives the report path and the counts of bugs by severity.
