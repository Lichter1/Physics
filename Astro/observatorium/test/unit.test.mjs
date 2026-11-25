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
for (const f of readdirSync(join(root, 'data'))) ctx.OBS_INLINE[f] = readFileSync(join(root, 'data', f), 'utf8');
vm.createContext(ctx);
for (const s of ['core', 'astro', 'catalog', 'instruments', 'missions']) vm.runInContext(readFileSync(join(root, 'src/js', s + '.js'), 'utf8'), ctx, { filename: s + '.js' });
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
