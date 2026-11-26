// End-to-end check: play every mission in a real browser, doing what a student would do through the app,
// and confirm the page says "done" and renders the explanation without errors.
//   node build.mjs && (cd dist && python3 -m http.server 8765 &) && node e2e/missions.js
const { chromium } = require('playwright');

const SOLVE = {
  sky: (OBS) => { const c = OBS.app.chart.ds.table.cols, idx = []; for (let i = 0; i < c.mag.length; i++) if (c.mag[i] < 1.0) idx.push(i); OBS.app.chart.setSel(idx); },
  parallax: (OBS, st) => {
    const s = st.scope; let best = 0, bt = 0;
    for (let t = 0; t < 0.5; t += 0.005) { const a = s.posAt(t), b = s.posAt(t + 0.5), d = Math.hypot(a[0] - b[0], a[1] - b[1]); if (d > best) { best = d; bt = t; } }
    const a = s.posAt(bt), b = s.posAt(bt + 0.5);
    s.state.marks = [{ x: a[0], y: a[1], label: 'A' }, { x: b[0], y: b[1], label: 'B' }]; s.state.sep = best;
    st.input.set(0.286);
  },
  hrd: async (OBS, st) => {
    OBS.app.chart.setAxes(null, 'MV'); st.t0 = Date.now() - 3000;
    await OBS.MISSIONS.find((m) => m.id === 'hrd').progress(OBS.app, st);        // stage 2 opens the 100 pc stars in M_G
    const ch = OBS.app.chart, wd = OBS.column(ch.ds, 'wd'), mg = OBS.column(ch.ds, 'MG'), idx = [];
    for (let i = 0; i < wd.length; i++) if (wd[i] === 'white dwarf' && mg[i] > 10) idx.push(i); ch.setSel(idx.slice(0, 300));
  },
  thermometer: (OBS, st) => { st.stars.forEach((s, k) => { st.spec.state.star = k; st.spec.slider.set(OBS.astro.bbTeffFromBpRp(s.bprp)); }); },
  radius: (OBS) => { const ch = OBS.app.chart, wd = OBS.column(ch.ds, 'wd'), idx = []; for (let i = 0; i < wd.length; i++) if (wd[i] === 'white dwarf') idx.push(i); ch.setSel(idx.slice(0, 300)); },
  binary: (OBS, st) => { const ch = OBS.app.chart, b = st.b; ch.get('t1').value = b.gamma + b.K1; ch.get('b1').value = b.gamma - b.K1; ch.get('t2').value = b.gamma + b.K2; ch.get('b2').value = b.gamma - b.K2; st.bin.read(); },
  masslum: (OBS) => { const o = OBS.app.chart.get('fit'); Object.assign(o, { x1: 1, y1: 1, x2: 10, y2: Math.pow(10, 3.8) }); },
  pleiades: async (OBS, st) => {
    const ch = OBS.app.chart, c = ch.ds.table.cols, idx = []; for (let i = 0; i < ch.ds.table.n; i++) if (OBS.isMember(c, i)) idx.push(i);
    ch.setSel(idx); await OBS.MISSIONS.find((m) => m.id === 'pleiades').progress(OBS.app, st); st.slide.set(5.78);
  },
  turnoff: (OBS) => { const ch = OBS.app.chart; ch.get('pleiades').value = -1; ch.get('praesepe').value = 0.6; ch.get('m67').value = 2.7; ch.get('ngc6397').value = 3.6; },
  leavitt: async (OBS, st) => {
    st.fold.coarse.set(11.8606); document.getElementById('fold-on').click();
    await OBS.MISSIONS.find((m) => m.id === 'leavitt').progress(OBS.app, st);
    Object.assign(OBS.app.chart.get('pl'), { x1: 1, y1: 17.39, x2: 10, y2: 14.29 });
  },
  yardstick: (OBS, st) => st.slide.set(18.6),
  pulsars: (OBS) => { const ch = OBS.app.chart, P = OBS.column(ch.ds, 'P'), idx = []; for (let i = 0; i < P.length; i++) if (P[i] < 0.03) idx.push(i); ch.setSel(idx); },
  transit: (OBS, st) => { st.fold.coarse.set(2.204737); document.getElementById('fold-on').click(); OBS.app.chart.get('depth').value = 1 - 0.0067; },
  census: (OBS) => { const ch = OBS.app.chart, P = OBS.column(ch.ds, 'P'), M = OBS.column(ch.ds, 'Mp'), idx = []; for (let i = 0; i < P.length; i++) if (P[i] < 10 && M[i] > 0.3) idx.push(i); ch.setSel(idx); },
  microlens: (OBS, st) => { const e = st.ews; const s = st.model.sliders; s.t0.set(e.tmax - 2450000); s.tE.set(e.tE); s.u0.set(e.u0); s.fbl.set(e.fbl); s.I0.set(e.I0); },
  kirkwood: (OBS) => { const ch = OBS.app.chart; ch.get('g1').value = 2.502; ch.get('g2').value = 2.825; ch.get('g3').value = 2.958; },
  shapley: (OBS) => { const p = OBS.app.chart.get('centre'); p.x = 7.2; p.y = 0.3; },
  rotation: (OBS, st) => { const s = st.model.sliders; s.Y.set(0.9); s.Y.set(0.58); s.vh.set(142); s.rc.set(3.12); },
  redblue: (OBS) => { const ch = OBS.app.chart, u = OBS.column(ch.ds, 'ur'), idx = []; for (let i = 0; i < u.length; i++) if (u[i] > 2.35 && u[i] < 3.2 && ch.mask[i]) idx.push(i); ch.setSel(idx); },
  zwicky: (OBS) => { const ch = OBS.app.chart, v = OBS.column(ch.ds, 'cz'), idx = []; for (let i = 0; i < v.length; i++) if (v[i] > 4300 && v[i] < 9700) idx.push(i); ch.setSel(idx); },
  tully: (OBS) => {
    const ch = OBS.app.chart, x = OBS.column(ch.ds, 'Vflat'), y = OBS.column(ch.ds, 'L36'); let n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (let i = 0; i < x.length; i++) if (ch.mask[i] && x[i] > 0 && y[i] > 0) { const X = Math.log10(x[i]), Y = Math.log10(y[i]); n++; sx += X; sy += Y; sxx += X * X; sxy += X * Y; }
    const m = (n * sxy - sx * sy) / (n * sxx - sx * sx), b = (sy - m * sx) / n;
    Object.assign(ch.get('tf'), { x1: 50, y1: Math.pow(10, b + m * Math.log10(50)), x2: 250, y2: Math.pow(10, b + m * Math.log10(250)) });
  },
  // chapters 6-9
  accel: (OBS, st) => { const s = st.model.sliders; s.Om.set(0.36); s.OL.set(0.64); s.H0.set(72.6); },
  cmb: (OBS, st) => { const s = st.model.sliders; s.T.set(2.725); s.dT.set(0); },
  peaks: (OBS, st) => { const s = st.model.sliders; s.Ot.set(1.0); s.dO.set(0); },
  forest: (OBS, st) => st.fit.set(st.meta.z),
  night: (OBS, st) => {
    const p = st.planner; p.sd.set(167); p.sh.set(24);
    let h = p.transit; while (h < 16) h += 24; while (h > 32) h -= 24; p.sh.set(Math.round(h * 4) / 4);
  },
  photometry: (OBS, st) => { st.phot.slider.set(10); for (let k = 0; k < st.field.stars.length; k++) st.phot.pick(k); },
  bursts: async (OBS, st) => {
    await st.go('pulsars'); await st.go('grbs');
    const ch = OBS.app.chart, sb = OBS.column(ch.ds, 'sinb'), idx = []; for (let i = 0; i < sb.length; i++) if (Math.abs(sb[i]) < 0.17) idx.push(i); ch.setSel(idx);
  },
  frb: (OBS, st) => { const s = st.model.sliders; s.Ob.set(0.048); s.a.set(250); },
  gw: (OBS, st) => st.model.sliders.eps.set(0.042),
  grbtwo: (OBS) => { OBS.app.chart.get('cut').value = 2; },
  redshift: async (OBS, st) => { for (const k of ['sf', 'ell', 'lrg']) { await st.go(k); st.fit.set(st.meta[k].z); [...document.querySelectorAll('#tools button')].find((b) => b.textContent === 'Record this redshift').click(); } },
  bpt: (OBS, st) => { Object.assign(OBS.app.chart.get('cut'), { x1: -0.22, y1: 1.2, x2: -0.22, y2: -1.2 }); st.score(); },
  chain: (OBS, st) => { const m = st.mc; m.sStep.set(0.05); m.reset(); for (let k = 0; k < 6000; k++) m.stepOnce(); m.sBurn.set(Math.ceil(((m.inAt || 0) + 100) / 50) * 50); m.report(); }
};

(async () => {
  const base = process.argv[2] || 'http://localhost:8765/preview.html', shots = process.argv[3];
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1600, height: 950 }, deviceScaleFactor: 1, colorScheme: 'dark' });
  await p.addInitScript(() => { try { localStorage.setItem('observatorium.toured', 'true'); } catch (e) { /* no storage */ } });   // no tour here: e2e/tour.js tests it
  const errs = [];
  p.on('pageerror', (e) => errs.push(e.message));
  p.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()); });
  await p.goto(base); await p.waitForTimeout(2500);
  const ids = await p.evaluate(() => OBS.MISSIONS.map((m) => m.id));
  let fails = 0;
  for (const id of ids) {
    errs.length = 0;
    await p.evaluate((id) => OBS.logbook.start(OBS.MISSIONS.find((x) => x.id === id)), id);
    const beats = await p.evaluate(() => document.querySelectorAll('#lecture .dot').length);
    for (let k = 1; k < beats; k++) await p.click('#lec-next');          // read the lecture step by step
    if (shots && ['sky', 'parallax', 'pleiades', 'microlens'].includes(id)) await p.screenshot({ path: `${shots}/lecture-${id}.png` });
    await p.click('#lec-pred');
    await p.evaluate((id) => document.querySelector('input[name="pred"][value="' + OBS.MISSIONS.find((m) => m.id === id).answer + '"]').click(), id);
    await p.click('#m-next');
    await p.waitForTimeout(1800);
    await p.evaluate(() => {
      window.__locked = [];
      const ch = OBS.app.chart, set = ch.setAxes;
      ch.setAxes = function (x, y) {
        [['across', x], ['up', y]].forEach(([id, v]) => { if (!v || v === '#') return; const q = this.ds.vars.find((w) => w.id === v); if (!q || !OBS.unlocked(q.needs)) window.__locked.push(id + ' = ' + v + ' in ' + this.ds.id); });
        return set.call(this, x, y);
      };
      window.__restore = () => { ch.setAxes = set; };
    });
    await p.evaluate('(' + SOLVE[id].toString() + ')(OBS, OBS.logbook.current().st)');
    const locked = await p.evaluate(() => { window.__restore(); return window.__locked; });
    if (locked.length) errs.push('the solution needs a locked menu option: ' + locked.join(', '));
    await p.waitForTimeout(1500);
    const status = (await p.textContent('#m-status')).trim();
    const done = status.startsWith('done');
    if (done) await p.click('#m-next');
    await p.waitForTimeout(900);
    const verdict = done ? (await p.textContent('.verdict')).trim() : '';
    const result = done ? (await p.textContent('.result')).trim() : '';
    if (shots) await p.screenshot({ path: `${shots}/done-${id}.png` });
    if (!done || errs.length) fails++;
    console.log((done && !errs.length ? 'PASS ' : 'FAIL ') + id.padEnd(12) + ' ' + (done ? result.slice(0, 230) : status) + (errs.length ? '\n     errors: ' + errs.join(' | ') : ''));
  }
  await b.close();
  console.log(fails ? fails + ' mission(s) failed' : 'all missions completed');
  process.exit(fails ? 1 : 0);
})();
