// Review harness for the Observatorium audit. Require it from your own script:
//   const H = require('./audit/tools/harness.js');
//   const { browser, page, errors } = await H.open({ before: 'hrd' });   // every mission before 'hrd' marked done, as a student who played in order
//   await H.start(page, 'hrd');            // opens the mission card (step 0)
//   console.log(await H.lensMenus(page));   // the axis menus as a student sees them, with locked options
//   await H.shot(page, 'ch1/hrd-lecture');  // screenshot into audit/shots/
//   await browser.close();
// Options: before (mission id, or null = nothing done), all (true = every mission done), sandbox (teacher mode on),
//          width/height (viewport), url, dark (default true).
const { chromium } = require('playwright');
const path = require('path');
const AUDIT = __dirname;
const URL = 'http://localhost:8765/preview.html';

async function open(o = {}) {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: o.width || 1600, height: o.height || 950 }, deviceScaleFactor: 1, colorScheme: o.dark === false ? 'light' : 'dark' });
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') errors.push(m.type() + ': ' + m.text()); });
  await page.addInitScript(() => { try { localStorage.setItem('observatorium.toured', 'true'); } catch (e) { /* none */ } });
  await page.goto(o.url || URL);
  await page.waitForTimeout(2000);
  const prog = await page.evaluate(({ before, all }) => {
    const P = { done: {}, concepts: {}, quiz: {} };
    for (const m of OBS.MISSIONS) {
      if (!all && (before === null || before === undefined || m.id === before)) break;
      P.done[m.id] = { pred: m.answer, right: true, at: Date.now() };
      (m.unlocks || []).forEach((k) => { P.concepts[k] = true; });
    }
    localStorage.setItem('observatorium.progress', JSON.stringify(P));
    return P;
  }, { before: o.before === undefined ? null : o.before, all: !!o.all });
  await page.evaluate((s) => localStorage.setItem('observatorium.sandbox', JSON.stringify(!!s)), o.sandbox);
  await page.reload();
  await page.waitForTimeout(2500);
  return { browser, page, errors, progress: prog };
}

// Open a mission from the logbook the way a student does (switches to its chapter and clicks its entry when it can).
async function start(page, id) {
  await page.evaluate((id) => OBS.logbook.start(OBS.MISSIONS.find((x) => x.id === id)), id);
  await page.waitForTimeout(800);
}

// Step through the lecture with the Next button; returns the headings it showed.
async function readLecture(page, perBeat) {
  const n = await page.evaluate(() => document.querySelectorAll('#lecture .dot').length);
  const heads = [];
  for (let k = 0; k < n; k++) {
    heads.push(await page.evaluate(() => { const h = document.querySelector('#lecture h3, #lecture h2'); return h ? h.textContent.trim() : ''; }));
    if (perBeat) await perBeat(k);
    if (k < n - 1) { await page.click('#lec-next'); await page.waitForTimeout(400); }
  }
  return heads;
}

// From the mission card to the exercise: read the lecture, lock in a prediction (index into the choices; default the
// right one), press Next. Afterwards: #m-status shows progress, #m-next finishes when it is enabled, then .verdict/.result.
async function toExercise(page, id, pred) {
  await readLecture(page);
  await page.click('#lec-pred'); await page.waitForTimeout(300);
  const v = pred === undefined ? await page.evaluate((id) => OBS.MISSIONS.find((m) => m.id === id).answer, id) : pred;
  await page.evaluate((v) => document.querySelector('input[name="pred"][value="' + v + '"]').click(), v);
  await page.click('#m-next');
  await page.waitForTimeout(2000);
}

// The data-lens menus as the student sees them.
async function lensMenus(page) {
  return page.evaluate(() => {
    const read = (id) => { const s = document.getElementById(id); if (!s) return null; return { value: s.value, options: [...s.options].map((o) => (o.disabled ? '[locked] ' : '') + o.value + ' = ' + o.textContent.trim()) }; };
    return { dataset: read('ds-sel'), x: read('x-sel'), y: read('y-sel'), colour: read('c-sel'), unlocked: Object.keys(OBS.state.concepts).filter((k) => OBS.state.concepts[k]), sandbox: OBS.state.sandbox };
  });
}

async function shot(page, name) {
  const p = path.join(AUDIT, 'shots', name + '.png');
  require('fs').mkdirSync(path.dirname(p), { recursive: true });
  await page.screenshot({ path: p });
  return p;
}

module.exports = { open, start, readLecture, toExercise, lensMenus, shot, URL, AUDIT };
