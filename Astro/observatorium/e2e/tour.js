// End-to-end check of the first-visit tour: it starts by itself once, every step finds its part of the page and
// keeps its card on screen (desktop and phone), the two "try it" steps notice the student doing it, the last step
// starts mission 1.1, and Tour / Esc / the arrow keys work.
//   node build.mjs && (cd dist && python3 -m http.server 8765 &) && node e2e/tour.js
const { chromium } = require('playwright');
const assert = require('node:assert/strict');

async function walk(b, base, W, H) {
  const p = await b.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1, colorScheme: 'dark' });
  const errs = []; p.on('pageerror', (e) => errs.push(e.message)); p.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()); });
  await p.goto(base); await p.waitForTimeout(3500);
  assert.ok(await p.evaluate(() => OBS.tour.on), 'the tour starts by itself on a first visit');
  const n = await p.evaluate(() => OBS.tour.steps.length);
  for (let k = 0; k < n; k++) {
    await p.waitForTimeout(600);
    const h = await p.textContent('#tour-h');
    const step = await p.evaluate(() => { const s = OBS.tour.steps[OBS.tour.k]; return { at: !!s.at, tried: !!s.try }; });
    if (step.at) assert.ok(await p.evaluate(() => !document.querySelector('.tour').classList.contains('centred')), `${W}px: "${h}" finds its part of the page`);
    const drag = async (sel, a, z) => {
      const r = await p.evaluate((sel) => { const r = document.querySelector(sel).getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; }, sel);
      await p.mouse.move(r[0] + r[2] * a[0], r[1] + r[3] * a[1]); await p.mouse.down();
      await p.mouse.move(r[0] + r[2] * z[0], r[1] + r[3] * z[1], { steps: 8 }); await p.mouse.up(); await p.waitForTimeout(400);
    };
    if (h === 'The sky') await drag('.skycanvas', [0.5, 0.5], [0.62, 0.56]);
    if (h === 'The data lens') await drag('#chart', [0.3, 0.3], [0.6, 0.7]);
    if (step.tried) assert.ok(await p.evaluate(() => document.querySelector('.tour-try').classList.contains('ok')), `${W}px: "${h}" notices the try-it`);
    const box = await p.evaluate(() => { const r = document.querySelector('.tour-pop').getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; });
    assert.ok(box[0] >= 0 && box[1] >= 0 && box[2] <= W && box[3] <= H, `${W}px: the card for "${h}" is on screen (${box.map(Math.round)})`);
    if (k < n - 1) await p.click('.tour-pop [data-t="next"]');
  }
  await p.click('.tour-pop [data-t="begin"]'); await p.waitForTimeout(1200);
  assert.equal(await p.evaluate(() => OBS.tour.on), false);
  assert.equal(await p.evaluate(() => OBS.logbook.current().m && OBS.logbook.current().m.id), 'sky', 'the last step starts mission 1.1');
  await p.click('#tour-btn'); await p.waitForTimeout(500);
  assert.ok(await p.evaluate(() => OBS.tour.on && !OBS.logbook.current().m), 'Tour replays it from a clean slate');
  await p.keyboard.press('ArrowRight'); await p.waitForTimeout(300);
  assert.equal(await p.evaluate(() => OBS.tour.k), 1, 'the right arrow moves on');
  await p.keyboard.press('Escape'); await p.waitForTimeout(300);
  assert.equal(await p.evaluate(() => OBS.tour.on), false, 'Esc ends it');
  await p.reload(); await p.waitForTimeout(3000);
  assert.equal(await p.evaluate(() => OBS.tour.on), false, 'it does not start again on the next visit');
  assert.deepEqual(errs, [], 'no errors');
  await p.close();
  console.log(`PASS tour at ${W}×${H}: ${n} steps`);
}

(async () => {
  const base = process.argv[2] || 'http://localhost:8765/preview.html';
  const b = await chromium.launch();
  try { for (const [W, H] of [[1600, 950], [1100, 800], [390, 844]]) await walk(b, base, W, H); }
  catch (e) { console.log('FAIL ' + e.message); process.exitCode = 1; }
  await b.close();
})();
