// Unit tests: physics, the CSV reader, every variable of every dataset on the real data, and the mission plan.
//   node --test test/
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const ctx = { console, Math, JSON, Promise, Float64Array, Uint8Array, Array, Object, Number, String, isFinite, parseFloat, setTimeout };
ctx.globalThis = ctx; ctx.window = ctx;
ctx.OBS_INLINE = {};
for (const f of readdirSync(join(root, 'data')).filter((x) => /\.(csv|json)$/.test(x))) ctx.OBS_INLINE[f] = readFileSync(join(root, 'data', f), 'utf8');
vm.createContext(ctx);
for (const s of ['core', 'astro', 'catalog', 'instruments', 'observing', 'explainers', 'explainers2', 'missions', 'lectures', 'lectures2', 'lectures3']) vm.runInContext(readFileSync(join(root, 'src/js', s + '.js'), 'utf8'), ctx, { filename: s + '.js' });
const OBS = ctx.OBS, A = OBS.astro;
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b} (±${tol})`);

test('coordinates: the Galactic centre and pole', () => {
  const gc = A.equToGal(266.40499, -28.93617);
  assert.ok(Math.min(gc[0], 360 - gc[0]) < 0.01 && Math.abs(gc[1]) < 0.01);
  const ngp = A.galToEqu(0, 90);
  close(ngp[0], 192.8595, 0.01, 'NGP RA'); close(ngp[1], 27.1283, 0.01, 'NGP Dec');
  close(A.sep(0, 0, 90, 0), 90, 1e-9, 'separation');
});

test('blackbodies and colours', () => {
  close(A.wienPeak(5772) * 1e9, 502, 1, 'Wien peak of the Sun');
  close(A.bbBpRp(5772), 0.82, 1e-9, 'Sun colour zero point');
  close(A.bbTeffFromBpRp(A.bbBpRp(9000)), 9000, 5, 'colour temperature inverts');
  close(A.teffFromBpRp(0.82), 5700, 150, 'empirical Sun-like temperature');
  assert.ok(A.teffFromBpRp(2.5) > A.bbTeffFromBpRp(2.5), 'red dwarfs are hotter than their blackbody colour says');
  close(A.radius(1, 5772), 1, 1e-12, 'Sun radius');
  close(A.lumFromMG(4.67, 0.82), 1, 0.1, 'Sun luminosity from M_G');
});

test('orbits, lensing, pulsars, galaxies', () => {
  const m = A.binaryMasses(365.25, 29.78, 0, 0, 90);
  close(m.total, 1, 0.01, 'Earth–Sun total mass from Earth’s orbital speed');
  close(A.magnification(1), 1.3416, 1e-4, 'A(u = 1)');
  close(A.keplerP(1, 1), 1, 1e-12, 'Kepler III');
  close(Math.log10(A.pulsarB(0.0334, 4.21e-13)), 12.6, 0.1, 'Crab field');
  close(A.pulsarAge(0.0334, 4.21e-13), 1257, 10, 'Crab characteristic age');
  close(A.vBaryon(3, 4, 0, 1, 0), 5, 1e-12, 'speeds add in quadrature');
  const s = A.parallaxShift(270, 66.56, 0.3);
  close(Math.hypot(s.x, s.y), 1, 0.01, 'a star at the ecliptic pole traces a circle of radius p');
  for (let k = 0; k < 20; k++) { const ph = k / 20, v1 = A.rv(ph, 0.3, 40, 50, 10), v2 = A.rv(ph, 0.3, 220, 100, 10); close(100 * (v1 - 10) + 50 * (v2 - 10), 0, 1e-6, 'momentum balance: M1 v1 + M2 v2 = 0 with M1/M2 = K2/K1'); }
});

test('CSV reader keeps long identifiers as text', () => {
  const t = OBS.parseCSV('id,x,name\n3684484600212307712,1.5,"a, b"\n12,,c\n');
  assert.equal(t.cols.id[0], '3684484600212307712');
  assert.ok(Number.isNaN(t.cols.x[1]));
  assert.equal(t.cols.name[0], 'a, b');
});

test('every variable of every dataset gives real values', async () => {
  for (const id of Object.keys(OBS.DATASETS)) {
    const ds = await OBS.open(id);
    for (const v of ds.vars) {
      const col = OBS.column(ds, v.id);
      const n = v.cat ? col.filter((x) => x !== '' && x !== undefined).length : Array.from(col).filter(Number.isFinite).length;
      assert.ok(n > 0.2 * ds.table.n || n > 50, `${id}.${v.id}: only ${n} of ${ds.table.n} values`);
    }
    assert.ok(OBS.varOf(ds, ds.x) && (ds.y === '#' || OBS.varOf(ds, ds.y)), `${id} default axes exist`);
  }
});

test('the data show what the missions teach', async () => {
  await OBS.open('clusters');
  close(OBS.CLUSTERS.pleiades.dist, 135.5, 3, 'Pleiades distance from member parallaxes');
  assert.ok(OBS.CLUSTERS.pleiades.n > 500, 'Pleiades members found by proper motion');
  const order = ['hyades', 'pleiades', 'praesepe', 'm67', 'ngc6397'].map((k) => OBS.CLUSTERS[k].dist);
  assert.deepEqual([...order].sort((a, b) => a - b), order, 'cluster distances in the expected order');
  const ogle = await OBS.open('cep_ogle'), P = OBS.column(ogle, 'P'), I = OBS.column(ogle, 'I'), g = ogle.table.cols.gal;
  let n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < P.length; i++) if (g[i] === 'smc') { const x = Math.log10(P[i]); n++; sx += x; sy += I[i]; sxx += x * x; sxy += x * I[i]; }
  close((n * sxy - sx * sy) / (n * sxx - sx * sx), -3.1, 0.2, 'Leavitt slope in the SMC (I band)');
  const lc = await OBS.open('cep_lc'), t = [], y = [], id = lc.table.cols.id;
  for (let i = 0; i < lc.table.n; i++) if (id[i] === 'OGLE-LMC-CEP-1109') { t.push(lc.table.cols.t[i]); y.push(lc.table.cols.mag[i]); }
  const sharp = OBS.instruments.foldScatter(t, y, 11.8606), off = OBS.instruments.foldScatter(t, y, 11.9);
  assert.ok(sharp < 0.5 * off, 'folding on the true period is sharpest');
  const nb = await OBS.open('nearby'), wd = OBS.column(nb, 'wd'), mg = OBS.column(nb, 'MG');
  const wmg = []; for (let i = 0; i < wd.length; i++) if (wd[i] === 'white dwarf') wmg.push(mg[i]);
  wmg.sort((a, b) => a - b); assert.ok(wmg[wmg.length >> 1] > 11, 'white dwarfs are faint');
});

test('the mission plan is consistent', () => {
  const ids = new Set(), have = new Set();
  for (const m of OBS.MISSIONS) {
    assert.ok(!ids.has(m.id), 'unique id ' + m.id); ids.add(m.id);
    for (const k of ['title', 'brief', 'relation', 'deduce', 'question', 'task', 'explain', 'real', 'course']) assert.ok(m[k], `${m.id} has ${k}`);
    assert.ok(m.choices.some((c) => c[0] === m.answer), m.id + ': the answer is one of the choices');
    for (const k of m.needs || []) assert.ok(have.has(k), `${m.id} needs ${k}, which an earlier mission must unlock`);
    for (const k of m.unlocks || []) { assert.ok(OBS.CONCEPTS[k], m.id + ' unlocks a known concept'); assert.equal(OBS.CONCEPTS[k].by, m.id, `concept ${k} names ${m.id} as its source`); have.add(k); }
    assert.ok(!OBS.CHAPTERS[m.ch - 1].locked, m.id + ' is in an open chapter');
  }
  for (const k of Object.keys(OBS.CONCEPTS)) assert.ok(have.has(k), 'some mission unlocks ' + k);
  for (const q of OBS.QUIZ) assert.ok(q.a >= 0 && q.a < q.c.length);
});

test('nothing a mission asks for is locked when a student gets there in course order', () => {
  // Concepts open at a mission: everything earlier missions unlock, plus what the mission itself lends while open.
  const have = new Set(), problems = [];
  const findVar = (dsid, id) => { const ds = OBS.DATASETS[dsid]; return ds && ds.vars.find((v) => v.id === id); };
  const check = (m, dsid, id, where) => {
    if (!id || id === '#' || id === 'auto' || id === 'none') return;
    const v = findVar(dsid, id);
    if (!v) { problems.push(`${m.id}: ${where} names ${dsid}.${id}, which does not exist`); return; }
    const miss = (v.needs || []).filter((k) => !have.has(k) && !(m.unlocks || []).includes(k));
    if (miss.length) problems.push(`${m.id}: ${where} uses “${v.label}”, locked until ${miss.join(', ')}`);
  };
  for (const m of OBS.MISSIONS) {
    const src = [m.setup, m.progress].filter(Boolean).map((f) => f.toString()).join('\n');
    const opened = [];
    // app.open('dataset', { x: 'a', y: 'b', colourBy: 'c' }) in setup or progress
    for (const o of src.matchAll(/\.open\('(\w+)'(?:\s*,\s*\{([^}]*)\})?/g)) {
      opened.push(o[1]);
      for (const k of ['x', 'y', 'colourBy']) { const r = new RegExp(k + ":\\s*'([\\w#]+)'").exec(o[2] || ''); if (r) check(m, o[1], r[1], 'the exercise setup (' + k + ')'); }
    }
    // chart.setAxes(x, y) on the first dataset opened
    for (const a of src.matchAll(/setAxes\((?:'(\w+)'|null)\s*,\s*(?:'([\w#]+)'|null)\)/g)) for (const id of [a[1], a[2]]) check(m, opened[0], id, 'setAxes');
    // a variable named in quotes in the task text must be one the student can choose
    for (const q of (m.task || '').matchAll(/“([^”]+)”/g)) for (const dsid of new Set(opened)) {
      const v = OBS.DATASETS[dsid] && OBS.DATASETS[dsid].vars.find((x) => x.label.toLowerCase() === q[1].toLowerCase());
      if (v) check(m, dsid, v.id, 'the task text');
    }
    // live charts in the lecture
    for (const b of m.lecture || []) for (const f of [].concat(b.fig || [])) if (f && f.chart) for (const k of ['x', 'y', 'colourBy']) check(m, f.chart.ds, f.chart[k], 'lecture chart “' + b.h + '” (' + k + ')');
    for (const k of m.unlocks || []) have.add(k);
  }
  assert.deepEqual(problems, []);
});

test('every mission has a lecture with valid sky cues, explainers and practice', () => {
  for (const m of OBS.MISSIONS) {
    assert.ok(m.lecture && m.lecture.length >= 3, m.id + ' has a lecture');
    assert.ok(m.target && isFinite(m.target.ra) && isFinite(m.target.dec) && m.target.fov > 0, m.id + ' turns the sky somewhere when it opens');
    for (const b of m.lecture) {
      assert.ok(b.h && b.html, m.id + ': a step has a heading and text');
      if (b.ex) assert.equal(typeof OBS.EXPLAINERS[b.ex.id], 'function', m.id + ' explainer ' + b.ex.id);
      if (b.sky) assert.ok(isFinite(b.sky.ra) && isFinite(b.sky.dec) && b.sky.fov > 0, m.id + ' sky cue');
      for (const s of [b.html, ...(Array.isArray(b.eq) ? b.eq : b.eq ? [b.eq] : [])]) assert.ok(!/\\\(([^)]*)$/.test(s), m.id + ': unclosed inline maths');
    }
    assert.ok(m.practice.length >= 1, m.id + ' has practice');
    for (const p of m.practice) assert.ok(isFinite(p.a) && p.q && p.sol, m.id + ' practice entry');
  }
});

test('practice answers agree with the physics module', () => {
  const ans = (id, k) => OBS.MISSIONS.find((m) => m.id === id).practice[k].a;
  const near = (a, b, msg) => assert.ok(Math.abs(a / b - 1) < 0.01, `${msg}: page ${a} vs computed ${b}`);
  near(ans('sky', 0), Math.pow(10, 0.4 * 3), 'magnitudes');
  near(ans('parallax', 1), 1 / 0.768 * 3.26156, 'Proxima in light-years');
  near(ans('thermometer', 0), A.wienPeak(1) / 290e-9, 'Wien');
  near(ans('radius', 1), A.radius(0.003, 10000), 'white dwarf radius');
  near(ans('binary', 0), A.binaryMasses(10, 50, 100, 0, 90).total, 'binary total mass');
  near(ans('binary', 1), A.binaryMasses(10, 50, 100, 0, 90).M1, 'heavier star');
  near(ans('masslum', 0), 1e4 * Math.pow(5, -2.5), '5 Msun lifetime (Myr)');
  near(ans('turnoff', 0), 1e4 * Math.pow(2, -2.5), '2 Msun lifetime (Myr)');
  near(ans('leavitt', 0), A.distFromMu(15 - (-2.43 * (1 - 1) - 4.05)) / 1000, 'Cepheid distance');
  near(ans('yardstick', 0), A.distFromMu(18.48) / 1000, 'LMC');
  near(ans('pulsars', 0), A.pulsarAge(1, 1e-15) / 1e6, 'pulsar age');
  near(ans('microlens', 0), A.magnification(0.1), 'A(0.1)');
  near(ans('kirkwood', 0), 5.2026 * Math.pow(1 / 3, 2 / 3), '3:1 resonance');
  near(ans('rotation', 0), A.massEnclosed(220, 8) / 1e10, 'enclosed mass');
  near(ans('zwicky', 0), A.virialMass(1000, 1.5) / 1e15, 'virial mass');
  near(ans('census', 1), 28.4 * Math.pow(4 / 365.25, -1 / 3), 'RV amplitude');
  near(ans('census', 2), 30 / 40, 'angular separation');
  near(ans('census', 3), Math.pow(20, 2.35), 'Salpeter ratio');
  near(ans('kirkwood', 1), Math.pow(4300 / 5772, 4), 'sunspot flux');
  near(ans('redblue', 1), A.C.c * 2 * 86400 / A.C.AU, 'light-crossing size (AU)');
  near(ans('redblue', 2), 1e39 / 1.26e31 / 1e7, 'Eddington mass');
  near(ans('zwicky', 1), 1150 / 70, 'Hubble distance');
  near(ans('tully', 1), A.C.pc * 1e6 / 1e3 / 73 / (365.25 * 86400 * 1e9), 'Hubble time');
  // chapters 6-9
  near(ans('accel', 1), Math.pow(10, (24.0 + 19.3) / 5 + 1) / 1e6, 'SN distance');
  near(ans('accel', 2), A.q0(0.3, 0.7), 'q0');
  near(ans('cmb', 1), A.wienNuGHz(2.725), 'CMB peak frequency');
  near(ans('cmb', 2), A.C.c / 1e3 * 3.36e-3 / 2.725, 'CMB dipole speed');
  near(ans('night', 2), 1.22 * 550e-9 / 8.2 * 206264.8, 'VLT diffraction limit');
  near(ans('photometry', 0), -2.5 * Math.log10(1e-3 / 3631), 'AB magnitude of 1 mJy');
  near(ans('bursts', 0), 1.22 * 0.21 / 100 * 206264.8 / 60, 'GBT resolution');
  near(ans('bursts', 1), 1.3e-3 / 1e7 * 206264.8e6, 'EHT resolution');
  near(ans('frb', 0), 4.149 * 500 * (1 / 0.16 - 1 / 0.64) / 1000, 'dispersion delay');
  near(ans('gw', 2), Math.pow(1.4 * 1.4, 0.6) / Math.pow(2.8, 0.2), 'chirp mass');
  near(ans('redshift', 1), 2.5 * Math.log10(81), 'Tolman dimming');
  near(ans('chain', 1), Math.exp(-1), 'Metropolis acceptance');
});

test('chapters 6-9: the real data give the textbook answers', () => {
  const t = (id) => OBS.loadTable(OBS.DATASETS[id].file).then(() => OBS.DATASETS[id]);
  return Promise.all(['cmbspec', 'frbs', 'gw', 'grbs', 'supernovae'].map((id) => OBS.open(id))).then(([cmb, frb, gw, grb, sn]) => {
    // the CMB: a 2.725 K blackbody to chi2/N ~ 1
    const c = cmb.table.cols; let chi = 0;
    for (let i = 0; i < cmb.table.n; i++) chi += ((c.I[i] - A.planckNu(c.f_ghz[i], 2.725)) / c.err[i]) ** 2;
    assert.ok(chi / cmb.table.n < 3, 'FIRAS fits 2.725 K: ' + chi / cmb.table.n);
    // fast radio bursts: Omega_b near 0.049 fits the DM-z relation
    const f = frb.table.cols; let s0 = 0, s1 = 0, n = 0;
    for (let i = 0; i < frb.table.n; i++) if (f.z[i] > 0) { const r = f.dm[i] - A.dmCosmic(f.z[i], 0.049); s0 += r; n++; s1 += (f.dm[i] - A.dmCosmic(f.z[i], 0)) ; }
    assert.ok(s0 / n > 100 && s0 / n < 400, 'FRB offset from the Milky Way and hosts: ' + s0 / n);
    // gravitational waves: a few percent of the mass is radiated
    const g = gw.table.cols, fr = []; for (let i = 0; i < gw.table.n; i++) if (g.mfinal[i] > 0) fr.push(1 - g.mfinal[i] / (g.m1[i] + g.m2[i]));
    fr.sort((a, b) => a - b); const med = fr[fr.length >> 1];
    assert.ok(med > 0.03 && med < 0.055, 'median fraction radiated ' + med);
    // gamma-ray bursts are isotropic
    const iso = OBS.isotropy(grb); assert.ok(Math.abs(iso.sinb2) < 3 * iso.err && Math.abs(iso.plane - 0.174) < 0.03, 'bursts isotropic');
    // supernovae: accelerating beats matter-only by a wide margin
    const chi2 = OBS.snChi2(sn.table.cols);
    assert.ok(chi2(1, 0).chi2 - chi2(0.33, 0.67).chi2 > 300, 'supernovae reject a matter-only universe');
    assert.ok(chi2(0.33, 0.67).chi2 - chi2(0, 0).chi2 < 0, 'and prefer acceleration to an empty universe');
  });
});

test('lecture figures point at real pictures and real datasets', () => {
  const pics = JSON.parse(ctx.OBS_INLINE['lecture_images.json']);
  const ids = new Set(pics.map((p) => p.id));
  for (const p of pics) {
    assert.ok(p.credit && p.source, p.id + ' has a credit and a source');
    readFileSync(join(root, 'data', 'img', 'lec', p.id + '.jpg'));   // throws if the file is missing
  }
  let n = 0, used = new Set();
  for (const m of OBS.MISSIONS) for (const b of m.lecture) {
    for (const f of b.fig ? [].concat(b.fig) : []) {
      n++;
      assert.ok(f.caption, m.id + ': every figure has a caption');
      if (f.img) { assert.ok(ids.has(f.img), m.id + ': picture ' + f.img + ' exists'); used.add(f.img); continue; }
      const o = f.chart, ds = OBS.DATASETS[o.ds];
      assert.ok(ds, m.id + ': chart dataset ' + o.ds);
      for (const v of [o.x || ds.x, o.y || ds.y]) assert.ok(v === '#' || OBS.varOf(ds, v), m.id + ': chart variable ' + v);
      if (o.sub) assert.ok(o.sub === '*' || ds.subset.names[o.sub], m.id + ': chart subset ' + o.sub);
    }
    if (b.example) assert.ok(!/\\\(([^)]*)$/.test(b.example), m.id + ': unclosed maths in an example');
  }
  assert.ok(n >= 25, 'the lectures carry figures (' + n + ')');
  for (const id of ids) assert.ok(used.has(id), 'picture ' + id + ' is used by some lecture');
});
