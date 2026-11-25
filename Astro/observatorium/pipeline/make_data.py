"""Turn the downloaded catalogues (cache/) into the compact CSV files the Observatorium loads (../data/).

Every file gets an entry in ../data/manifest.json: where it came from, the query or URL, the cuts, the
citation and the licence. Run fetch_gaia.py and the downloads listed in README.md first, then:
    python3 make_data.py
"""
import base64, glob, gzip, io, json, math, os, re, tarfile, zipfile
from datetime import date
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')
OUT = os.path.join(HERE, '..', 'data')
os.makedirs(OUT, exist_ok=True)
MANIFEST = {}
RNG = np.random.default_rng(20261125)
FETCHED = '2026-11-25'


def c(p):
    return os.path.join(CACHE, p)


def save(name, df, fmt, **meta):
    """Write df as CSV with per-column decimal places (fmt: {col: decimals or None for text})."""
    df = df[list(fmt)].copy()
    for col, dp in fmt.items():
        if dp is not None:
            df[col] = df[col].round(dp)
    path = os.path.join(OUT, name + '.csv')
    df.to_csv(path, index=False, na_rep='')
    meta.update(rows=len(df), columns=list(fmt), fetched=FETCHED, bytes=os.path.getsize(path))
    MANIFEST[name] = meta
    print(f'{name:16s} {len(df):7d} rows {os.path.getsize(path)/1e3:8.0f} kB')


GAIA_CITE = 'Gaia Collaboration, Vallenari et al. 2023, A&A 674, A1 (Gaia DR3)'
GAIA_LIC = 'ESA/Gaia/DPAC, CC BY-SA 3.0 IGO'

# --------------------------------------------------------------------------------------------- the naked-eye sky
hyg = pd.read_csv(c('hyg_v44.csv.gz'), low_memory=False)
hyg = hyg[hyg.id != 0]  # drop the Sun
named = hyg.proper.notna()
sky = hyg[(hyg.mag < 6.5) | (named & (hyg.mag < 8))].copy()
sky['ra'] = sky.ra * 15.0  # hours -> degrees
sky['name'] = sky.proper.fillna('')
sky['bayer'] = (sky.bayer.fillna('') + ' ' + sky.con.fillna('')).str.strip()
sky.loc[sky.dist >= 100000, 'dist'] = np.nan  # HYG uses 100000 for "unknown"
sky['hip'] = sky.hip.fillna(0).astype(int)
sky['spect'] = sky.spect.fillna('')
save('sky', sky, {'hip': None, 'ra': 3, 'dec': 3, 'mag': 2, 'ci': 3, 'dist': 2, 'name': None, 'bayer': None, 'spect': None},
     title='The naked-eye sky (HYG v4.4)',
     source='HYG database v4.4 (Hipparcos, Yale Bright Star and Gliese catalogues combined), codeberg.org/astronexus/hyg',
     cuts='V < 6.5 (naked-eye limit), plus named stars to V < 8. Distances from Hipparcos parallaxes (van Leeuwen 2007).',
     citation='Nash, D. (astronexus) HYG v4.4; ESA 1997 Hipparcos; van Leeuwen 2007, A&A 474, 653',
     license='CC BY-SA 4.0', columns_note='ra, dec in degrees (J2000); mag = Johnson V; ci = B-V colour index; dist in parsecs')

# Constellation stick figures (Stellarium "modern" sky culture), as pairs of rows in sky.csv.
st = json.load(open(c('stellarium_modern.json')))
hip_row = {h: i for i, h in enumerate(sky.hip.values) if h}
lines = []
for con in st['constellations']:
    for poly in con.get('lines', []):
        pts = [p for p in poly if isinstance(p, int)]
        for a, b in zip(pts, pts[1:]):
            if a in hip_row and b in hip_row:
                lines.append([hip_row[a], hip_row[b]])
labels = []
for con in st['constellations']:
    rows = [hip_row[p] for poly in con.get('lines', []) for p in poly if isinstance(p, int) and p in hip_row]
    if rows:
        name = con.get('common_name', {}).get('english') or con['id'].split()[-1]
        labels.append([name, rows])
json.dump({'lines': lines, 'labels': [[n, r[len(r)//2]] for n, r in labels]}, open(os.path.join(OUT, 'constellations.json'), 'w'), separators=(',', ':'))
MANIFEST['constellations'] = {'title': 'Constellation lines', 'source': 'Stellarium, skycultures/modern/index.json (github.com/Stellarium/stellarium)',
                              'license': 'GPL-2.0 (Stellarium)', 'rows': len(lines), 'fetched': FETCHED,
                              'columns_note': 'pairs of row numbers into sky.csv'}
print('constellations', len(lines), 'segments')

# ------------------------------------------------------------------------------- the Milky Way: Gaia star counts
def pix2ang_nest(nside, pix):
    """HEALPix nested pixel centres -> (theta, phi). Vectorised port of the standard algorithm."""
    pix = np.asarray(pix, dtype=np.int64)
    npface = nside * nside
    face = pix // npface
    ipf = pix % npface
    def compress(v):
        r = np.zeros_like(v)
        for b in range(30):
            r |= ((v >> (2 * b)) & 1) << b
        return r
    ix, iy = compress(ipf), compress(ipf >> 1)
    jrll = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]); jpll = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7])
    jr = jrll[face] * nside - ix - iy - 1
    nr = np.where(jr < nside, jr, np.where(jr > 3 * nside, 4 * nside - jr, nside)).astype(float)
    z = np.where(jr < nside, 1 - nr**2 / (3 * npface), np.where(jr > 3 * nside, -1 + nr**2 / (3 * npface), (2 * nside - jr) * 2.0 / (3 * nside)))
    kshift = np.where((jr >= nside) & (jr <= 3 * nside), (jr - nside) & 1, 0)
    jp = (jpll[face] * nr + ix - iy + 1 + kshift) / 2
    jp = np.where(jp > 4 * nside, jp - 4 * nside, jp); jp = np.where(jp < 1, jp + 4 * nside, jp)
    phi = (jp - (kshift + 1) * 0.5) * (np.pi / 2) / nr
    return np.arccos(np.clip(z, -1, 1)), phi

def ang2pix_nest(nside, theta, phi):
    z = np.cos(theta); za = np.abs(z); tt = np.mod(phi, 2 * np.pi) / (np.pi / 2)
    out = np.zeros(theta.shape, dtype=np.int64)
    eq = za <= 2.0 / 3
    t1 = nside * (0.5 + tt); t2 = nside * z * 0.75
    jp = (t1 - t2).astype(np.int64); jm = (t1 + t2).astype(np.int64)
    ifp, ifm = jp // nside, jm // nside
    face_eq = np.where(ifp == ifm, ifp | 4, np.where(ifp < ifm, ifp, ifm + 8))
    ix_eq = jm & (nside - 1); iy_eq = nside - (jp & (nside - 1)) - 1
    ntt = np.minimum(3, tt.astype(np.int64)); tp = tt - ntt; tmp = nside * np.sqrt(3 * (1 - za))
    jp2 = np.minimum((tp * tmp).astype(np.int64), nside - 1); jm2 = np.minimum(((1 - tp) * tmp).astype(np.int64), nside - 1)
    face_p = np.where(z >= 0, ntt, ntt + 8)
    ix_p = np.where(z >= 0, nside - jm2 - 1, jp2); iy_p = np.where(z >= 0, nside - jp2 - 1, jm2)
    face = np.where(eq, face_eq, face_p); ix = np.where(eq, ix_eq, ix_p); iy = np.where(eq, iy_eq, iy_p)
    def spread(v):
        r = np.zeros_like(v)
        for b in range(30):
            r |= ((v >> b) & 1) << (2 * b)
        return r
    return face * nside * nside + spread(ix) + (spread(iy) << 1)

NSIDE = 64
test = RNG.integers(0, 12 * NSIDE**2, 2000)
th, ph = pix2ang_nest(NSIDE, test)
assert (ang2pix_nest(NSIDE, th, ph) == test).all(), 'HEALPix round trip failed'
sm = pd.read_csv(c('gaia_skymap.csv'))
counts = np.zeros(12 * NSIDE**2); sm = sm.dropna(); counts[sm.hpx6.values.astype(np.int64)] = sm.n.values
W, H = 720, 360
ra_g = (np.arange(W) + 0.5) / W * 360.0; dec_g = 90 - (np.arange(H) + 0.5) / H * 180.0
RA, DEC = np.meshgrid(np.radians(ra_g), np.radians(dec_g))
grid = counts[ang2pix_nest(NSIDE, np.pi / 2 - DEC, RA)]
lg = np.log10(np.maximum(grid, 1))
lo, hi = np.percentile(lg, 1), np.percentile(lg, 99.7)
img = np.clip((lg - lo) / (hi - lo), 0, 1)
gc_px = counts[ang2pix_nest(NSIDE, np.array([np.radians(90 + 28.94)]), np.array([np.radians(266.40)]))][0]
pole_px = counts[ang2pix_nest(NSIDE, np.array([np.radians(90 - 27.13)]), np.array([np.radians(192.86)]))][0]
assert gc_px > 20 * pole_px, 'Milky Way map is not brighter toward the Galactic centre: HEALPix ordering is wrong'
json.dump({'w': W, 'h': H, 'lo': float(lo), 'hi': float(hi), 'data': base64.b64encode((img * 255).astype(np.uint8).tobytes()).decode()},
          open(os.path.join(OUT, 'milkyway.json'), 'w'))
MANIFEST['milkyway'] = {'title': 'Star counts across the whole sky (Gaia DR3)', 'source': 'Gaia archive, gaiadr3.gaia_source',
                        'query': 'SELECT GAIA_HEALPIX_INDEX(6, source_id), COUNT(*) FROM gaiadr3.gaia_source WHERE random_index < 36000000 GROUP BY 1',
                        'cuts': 'A 2% random subsample of the 1.8 billion Gaia sources, counted in 49,152 HEALPix cells, resampled to a 720 x 360 RA/Dec grid; log scale.',
                        'citation': GAIA_CITE, 'license': GAIA_LIC, 'fetched': FETCHED, 'rows': int(sm.n.sum()),
                        'columns_note': f'uint8 image, value = (log10 count - {lo:.3f}) / ({hi:.3f} - {lo:.3f})'}
print('milkyway map', W, H, 'GC/pole contrast', round(gc_px / pole_px))

# ------------------------------------------------------------------------------- the solar neighbourhood (100 pc)
nb = pd.read_csv(c('nearby_clean.csv'))
wd = nb[nb.wd_prob > 0.5]; giants = nb[(nb.MG < 3) & (nb.bp_rp > 0.9) & (nb.wd_prob <= 0.5)]
rest = nb.drop(wd.index).drop(giants.index)
keep = pd.concat([wd.sample(min(len(wd), 4000), random_state=1), giants, rest.sample(32000, random_state=1)])
keep = keep.sample(frac=1, random_state=2)  # shuffle so partial draws look uniform
keep['rv'] = keep.radial_velocity
keep['wd'] = (keep.wd_prob > 0.5).astype(int)
ap = pd.read_csv(c('gaia_nearby_ap.csv'))
keep = keep.merge(ap[['source_id', 'teff_gspphot', 'radius_flame', 'mass_flame', 'lum_flame']], on='source_id', how='left')
keep['id'] = np.where(keep.teff_gspphot.notna(), keep.source_id.astype(str), '')  # ids only where Gaia has extra parameters, to keep the file small
save('nearby', keep, {'id': None, 'ra': 4, 'dec': 4, 'parallax': 3, 'phot_g_mean_mag': 3, 'bp_rp': 3, 'pmra': 2, 'pmdec': 2, 'rv': 1, 'wd': 0,
                      'teff_gspphot': 0, 'radius_flame': 3, 'mass_flame': 3, 'lum_flame': 4},
     title='Stars within 100 parsecs (Gaia DR3 / Gaia Catalogue of Nearby Stars)',
     source='Gaia archive, external.gaiaedr3_gcns_main_1 (Gaia Collaboration, Smart et al. 2021, A&A 649, A6), plus gaiadr3.astrophysical_parameters',
     query='parallax > 10 mas, then the HR-diagram quality cuts of Gaia Collaboration, Babusiaux et al. 2018: parallax/error > 10, G flux S/N > 50, BP and RP S/N > 20, BP/RP excess factor between 1.0+0.015(BP-RP)^2 and 1.3+0.06(BP-RP)^2, RUWE < 1.4.',
     cuts=f'{len(nb)} stars pass. Shipped: every red giant (M_G < 3, BP-RP > 0.9; {len(giants)}), 4000 of {len(wd)} white dwarfs (GCNS probability > 0.5) and 32000 of the other stars, chosen at random. Gaia temperatures, radii, masses and luminosities attached where Gaia has them (a random 6000 were looked up).',
     citation=GAIA_CITE + '; Gaia Collaboration, Smart et al. 2021, A&A 649, A6', license=GAIA_LIC,
     columns_note='parallax in mas; G magnitude; BP-RP colour; proper motions in mas/yr; rv km/s; teff K; radius, mass, lum in solar units (Gaia FLAME)')

# ------------------------------------------------------------------------------------------------------ clusters
CL = {  # name: (cache, label, E(B-V), age [Myr] and distance [pc] from the literature, for checking answers only)
    'pleiades': ('gaia_pleiades_field.csv', 'Pleiades (M45)', 0.045, 112, 135.2),
    'hyades': ('gaia_hyades.csv', 'Hyades', 0.0, 680, 47.5),
    'praesepe': ('gaia_praesepe.csv', 'Praesepe (M44)', 0.027, 650, 186.2),
    'm67': ('gaia_m67.csv', 'M67', 0.041, 4000, 860),
    'ngc6397': ('gaia_ngc6397.csv', 'NGC 6397 (globular)', 0.18, 13000, 2480),
}
parts = []
for key, (f, label, ebv, age, dist) in CL.items():
    d = pd.read_csv(c(f))
    if len(d) > 6000:
        d = d.sample(6000, random_state=3) if key != 'pleiades' else d
    d['cl'] = key
    parts.append(d)
cl = pd.concat(parts)
cl['rv'] = cl.radial_velocity
save('clusters', cl, {'cl': None, 'ra': 4, 'dec': 4, 'parallax': 3, 'pmra': 3, 'pmdec': 3, 'phot_g_mean_mag': 3, 'bp_rp': 3, 'rv': 1},
     title='Star clusters (Gaia DR3)', source='Gaia archive, gaiadr3.gaia_source, cone searches',
     query='Cone around each cluster centre with loose parallax and proper-motion windows (see fetch_gaia.py). The Pleiades file is the whole 3-degree field with parallax > 2 mas, so members must be picked out by the player.',
     cuts='NGC 6397 randomly thinned to 6000 stars.',
     clusters={k: {'label': v[1], 'ebv': v[2], 'age_myr': v[3], 'dist_pc': v[4]} for k, v in CL.items()},
     literature='E(B-V), ages and distances, used only to check answers: Gaia Collaboration, Babusiaux et al. 2018 (A&A 616, A10); Baumgardt & Vasiliev 2021 (MNRAS 505, 5957) for NGC 6397.',
     citation=GAIA_CITE, license=GAIA_LIC)

# ---------------------------------------------------------------------------------------------------- binaries
sb = pd.read_csv(c('sb2_eclipsing.csv'))
sb = sb[(abs(sb.Pecl / sb.period - 1) < 0.002) & (sb.semi_amplitude_primary > 20) & (sb.semi_amplitude_secondary > 20)].copy()
sb['id'] = sb.source_id.astype(str)
sb['K1'] = sb.semi_amplitude_primary; sb['K2'] = sb.semi_amplitude_secondary; sb['e'] = sb.eccentricity.fillna(0)
sb['w'] = sb.arg_periastron.fillna(90); sb['gamma'] = sb.center_of_mass_velocity; sb['T0'] = sb.t_periastron
sb['depth'] = sb.derived_primary_ecl_depth
save('binaries', sb, {'id': None, 'ra': 4, 'dec': 4, 'parallax': 3, 'phot_g_mean_mag': 3, 'bp_rp': 3, 'period': 6, 'e': 4, 'w': 2,
                      'K1': 2, 'K2': 2, 'gamma': 2, 'T0': 4, 'depth': 3},
     title='Double-lined spectroscopic binaries that also eclipse (Gaia DR3)',
     source='Gaia archive: gaiadr3.nss_two_body_orbit (nss_solution_type = SB2) matched to gaiadr3.vari_eclipsing_binary',
     cuts='Kept when the eclipse period agrees with the orbit period to 0.2% and both velocity amplitudes exceed 20 km/s. Eclipses mean the orbit is seen nearly edge-on, so sin i is close to 1.',
     citation=GAIA_CITE + '; Gaia Collaboration, Arenou et al. 2023, A&A 674, A34; Mowlavi et al. 2023, A&A 674, A16',
     license=GAIA_LIC, columns_note='period days; K1, K2, gamma km/s; w degrees; T0 = time of periastron (days, Gaia reference epoch 2016.0); depth = primary eclipse depth (mag)')

# Torres, Andersen & Gimenez 2010: masses and radii good to 3% for 95 eclipsing binaries (190 stars)
tex = open(c('torres_src/Torres_2009_Final.tex'), encoding='latin-1').read().split('\n')
start = next(i for i, l in enumerate(tex) if 'V3903 Sgr' in l and '27.27' in l)
end = next(i for i in range(start, len(tex)) if 'end{longtable}' in tex[i])
def num(s):
    s = s.replace('$-$', '-').replace('\\phn', '').replace('\\phs', '').replace('\\pht', '')
    m = re.match(r'\s*(-?[0-9.]+)', s.replace('$', ''))
    return float(m.group(1)) if m else np.nan
rows, system = [], ''
for l in tex[start:end]:
    if '&' not in l or 'noalign' in l:
        continue
    f = [x.strip() for x in l.rstrip('\\ ').split('&')]
    if len(f) < 11:
        continue
    if f[1] and f[0]:
        system = f[1].replace('$-$', '-')
    f = [x.replace('\\pppp', '$\\pm$') for x in f]
    rows.append({'system': system, 'star': f[3], 'spt': f[4], 'mass': num(f[5].split('$\\pm$')[0]), 'radius': num(f[6].split('$\\pm$')[0]),
                 'teff': num(f[7].split('$\\pm$')[0]), 'logg': num(f[8].split('$\\pm$')[0]), 'logL': num(f[9].split('$\\pm$')[0]),
                 'period': num(f[2]) if f[0] else np.nan})
tor = pd.DataFrame(rows)
tor['period'] = tor.period.ffill()
assert len(tor) == 190, len(tor)
save('torres', tor, {'system': None, 'star': None, 'spt': None, 'period': 2, 'mass': 3, 'radius': 3, 'teff': 0, 'logg': 3, 'logL': 3},
     title='Accurate stellar masses (Torres, Andersen & Gimenez 2010)', source='arXiv:0908.2624, Table 1 (LaTeX source)',
     citation='Torres, Andersen & Gimenez 2010, A&ARv 18, 67', license='Published data; used for teaching with citation',
     columns_note='mass, radius, logL in solar units; teff K; period days')

# ---------------------------------------------------------------------------------------------------- Cepheids
og = []
for g in ('smc', 'lmc'):
    cat = pd.read_csv(c(f'ogle_{g}_cepF.dat'), sep=r'\s+', header=None, na_values='-',
                      names=['id', 'I', 'V', 'P', 'eP', 'T0', 'Iamp', 'R21', 'phi21', 'R31', 'phi31'])
    ident = pd.read_fwf(c(f'ogle_{g}_ident.dat'), header=None, colspecs=[(0, 17), (18, 27), (28, 39), (40, 51)], names=['id', 'mode', 'ra_s', 'dec_s'])
    cat = cat.merge(ident, on='id')
    h = cat.ra_s.str.split(':', expand=True).astype(float); d_ = cat.dec_s.str.split(':', expand=True).astype(float)
    cat['ra'] = 15 * (h[0] + h[1] / 60 + h[2] / 3600)
    cat['dec'] = np.sign(d_[0] + 0.0 * d_[1]) * (abs(d_[0]) + d_[1] / 60 + d_[2] / 3600)
    cat.loc[cat.dec_s.str.startswith('-'), 'dec'] = -abs(cat.dec)
    cat['gal'] = g
    og.append(cat)
og = pd.concat(og).dropna(subset=['I', 'P'])
save('cepheids_ogle', og, {'id': None, 'gal': None, 'ra': 4, 'dec': 4, 'P': 6, 'I': 3, 'V': 3, 'Iamp': 3},
     title='Classical Cepheids in the Magellanic Clouds (OGLE-IV)', source='OGLE Collection of Variable Stars, www.astrouw.edu.pl/ogle/ogle4/OCVS/{smc,lmc}/cep/cepF.dat and ident.dat',
     cuts='Fundamental-mode classical Cepheids with a mean I magnitude.', citation='Soszynski et al. 2015, Acta Astron. 65, 297 (OGLE-IV Cepheids)',
     license='OGLE data, free to use with citation', columns_note='P days; I, V intensity-mean magnitudes; Iamp = I-band amplitude')

gc = pd.read_csv(c('gaia_cepheids.csv'))
gc['region'] = 'mw'
lmc = (gc.ra > 55) & (gc.ra < 105) & (gc.dec > -78) & (gc.dec < -58)
smc = ((gc.ra < 35) | (gc.ra > 345)) & (gc.dec > -78) & (gc.dec < -64)
m31 = (abs(gc.ra - 10.7) < 2) & (abs(gc.dec - 41.3) < 2); m33 = (abs(gc.ra - 23.5) < 1) & (abs(gc.dec - 30.7) < 1)
gc.loc[lmc, 'region'] = 'lmc'; gc.loc[smc, 'region'] = 'smc'; gc.loc[m31 | m33, 'region'] = 'm31'
gc = gc[(gc.region != 'm31') & gc.int_average_g.notna() & gc.int_average_bp.notna() & gc.int_average_rp.notna()]
gc = gc[(gc.region != 'mw') | ((gc.parallax / gc.parallax_error > 5) & (gc.ruwe < 1.4))]
gc['id'] = gc.source_id.astype(str); gc['P'] = gc.pf; gc['G'] = gc.int_average_g; gc['BP'] = gc.int_average_bp; gc['RP'] = gc.int_average_rp
save('cepheids_gaia', gc, {'id': None, 'region': None, 'ra': 4, 'dec': 4, 'parallax': 4, 'parallax_error': 4, 'P': 5, 'G': 3, 'BP': 3, 'RP': 3},
     title='Classical Cepheids measured by Gaia (DR3)', source='Gaia archive: gaiadr3.vari_cepheid joined to gaiadr3.gaia_source',
     cuts="type_best_classification = 'DCEP', fundamental mode. Milky Way Cepheids kept only with parallax/error > 5 and RUWE < 1.4.",
     citation=GAIA_CITE + '; Ripepi et al. 2023, A&A 674, A17', license=GAIA_LIC,
     columns_note='parallax mas (no zero-point correction applied here; the app adds +0.017 mas, Lindegren et al. 2021); P days; G, BP, RP intensity-averaged magnitudes')

lc = []
for f in sorted(glob.glob(c('ogle_cep/*.I.dat'))):
    d = pd.read_csv(f, sep=r'\s+', header=None, names=['t', 'mag', 'err'])
    d['id'] = os.path.basename(f).replace('.I.dat', '')
    lc.append(d)
lc = pd.concat(lc)
save('cepheid_lc', lc, {'id': None, 't': 5, 'mag': 3, 'err': 3},
     title='Cepheid light curves (OGLE-IV, I band)', source='www.astrouw.edu.pl/ogle/ogle4/OCVS/lmc/cep/phot/I/',
     citation='Soszynski et al. 2015, Acta Astron. 65, 297', license='OGLE data, free to use with citation', columns_note='t = HJD - 2450000; mag = I')

# ----------------------------------------------------------------------------------------------------- pulsars
db = tarfile.open(c('psrcat_pkg.tar.gz')).extractfile('psrcat_tar/psrcat.db').read().decode('latin-1')
recs = []
for block in db.split('@-'):
    r = {}
    for line in block.split('\n'):
        p = line.split()
        if len(p) >= 2:
            r[p[0]] = p[1]
    if 'PSRJ' in r:
        recs.append(r)
def fl(r, k):
    try:
        return float(r[k].replace('D', 'E'))
    except (KeyError, ValueError):
        return np.nan
def hms(s):
    if not s: return np.nan
    p = [float(x) for x in s.split(':')] + [0, 0]
    return 15 * (p[0] + p[1] / 60 + p[2] / 3600)
def dms(s):
    if not s: return np.nan
    sg = -1 if s.startswith('-') else 1
    p = [abs(float(x)) for x in s.split(':')] + [0, 0]
    return sg * (p[0] + p[1] / 60 + p[2] / 3600)
ps = []
for r in recs:
    P0, P1 = fl(r, 'P0'), fl(r, 'P1')
    F0, F1 = fl(r, 'F0'), fl(r, 'F1')
    if np.isnan(P0) and F0 > 0:
        P0 = 1 / F0
    if np.isnan(P1) and not np.isnan(F1) and F0 > 0:
        P1 = -F1 / F0**2
    typ = r.get('TYPE', '')
    ps.append({'name': r['PSRJ'], 'P': P0, 'Pdot': P1, 'binary': int('BINARY' in r or 'PB' in r), 'type': typ,
               'assoc': r.get('ASSOC', '').split('[')[0].split('(')[0], 'dist': fl(r, 'DIST_DM'),
               'ra': hms(r.get('RAJ')), 'dec': dms(r.get('DECJ')), 'dm': fl(r, 'DM')})
ps = pd.DataFrame(ps)
ps = ps[(ps.P > 0)]
version = re.search(r'#CATALOGUE\s+(\S+)', db)
save('pulsars', ps, {'name': None, 'P': 9, 'Pdot': None, 'binary': 0, 'type': None, 'assoc': None, 'dist': 3, 'ra': 5, 'dec': 5, 'dm': 4},
     title='Pulsars (ATNF Pulsar Catalogue)', source='www.atnf.csiro.au/research/pulsar/psrcat, psrcat_pkg.tar.gz' + (f' (catalogue {version.group(1)})' if version else ''),
     citation='Manchester, Hobbs, Teoh & Hobbs 2005, AJ 129, 1993', license='ATNF Pulsar Catalogue, free to use with citation',
     columns_note='P seconds; Pdot s/s (blank when unmeasured); binary 1 = in a binary; type: AXP, HE, NRAD, RRAT etc.; dist kpc from dispersion measure; ra, dec J2000 degrees; dm pc/cm^3')

# ----------------------------------------------------------------------------------------------------- exoplanets
ex = pd.read_csv(c('exoplanets.csv'))
ex['method'] = ex.discoverymethod
save('exoplanets', ex, {'pl_name': None, 'hostname': None, 'method': None, 'disc_year': 0, 'pl_orbper': 5, 'pl_orbsmax': 5, 'pl_bmassj': 5,
                        'pl_radj': 4, 'pl_eqt': 0, 'pl_orbeccen': 3, 'st_teff': 0, 'st_mass': 3, 'st_rad': 3, 'sy_dist': 3, 'ra': 4, 'dec': 4},
     title='Confirmed exoplanets (NASA Exoplanet Archive)', source='exoplanetarchive.ipac.caltech.edu TAP, table pscomppars (one row per planet)',
     citation='NASA Exoplanet Archive, operated by Caltech under contract with NASA (DOI 10.26133/NEA13)', license='Public (NASA)',
     columns_note='orbital period days; semi-major axis AU; mass and radius in Jupiter units; eqt K; star teff K, mass and radius solar; distance pc')

from astropy.io import fits
kp = []
for star, fn in (('HAT-P-7', 'hatp7_q2.fits'), ('HAT-P-11', 'hatp11_q2.fits')):
    d = fits.open(c(fn))[1].data
    t, fx = d['TIME'], d['PDCSAP_FLUX']; ok = np.isfinite(t) & np.isfinite(fx)
    t, fx = t[ok], fx[ok]
    r = fx / pd.Series(fx).rolling(97, center=True, min_periods=20).median().values  # ~2-day running median removes spots and drifts
    s = 1.4826 * np.median(abs(r - 1)); keep_ = r < 1 + 5 * s                       # drop upward outliers only: transits are downward
    kp.append(pd.DataFrame({'star': star, 't': t[keep_], 'f': r[keep_]}))
kp = pd.concat(kp)
save('kepler', kp, {'star': None, 't': 5, 'f': 6}, title='Two transiting planets seen by Kepler (quarter 2): HAT-P-7 b and HAT-P-11 b',
     source='MAST: archive.stsci.edu/missions/kepler/lightcurves/0106/010666592/kplr010666592-2009259160929_llc.fits (HAT-P-7 = KIC 10666592) and .../0107/010748390/kplr010748390-2009259160929_llc.fits (HAT-P-11 = KIC 10748390)',
     cuts='PDCSAP flux, finite values, divided by a 2-day running median; points more than 5 sigma above the median removed.',
     citation='Borucki et al. 2010, Science 327, 977 (Kepler); Pal et al. 2008, ApJ 680, 1450 (HAT-P-7 b); Bakos et al. 2010, ApJ 710, 1724 (HAT-P-11 b)',
     license='Public (NASA/MAST)', columns_note='t = BJD - 2454833 (Kepler time, days); f = relative flux')

# -------------------------------------------------------------------------------------- microlensing (OGLE EWS)
ews = open(c('ews2019.html'), encoding='latin-1').read()
def ews_params(ev):
    row = re.search(r'blg-' + ev + r'\.html.*?</TR>', ews, flags=re.S).group(0)
    cells = [re.sub('<[^>]+>', '', x).strip() for x in re.split(r'<TD[^>]*>', row)]
    return {'tmax': float(cells[5]), 'tE': float(cells[7]), 'u0': float(cells[8]), 'Amax': float(cells[9]), 'fbl': float(cells[11]), 'I0': float(cells[13])}
ml, evmeta = [], {}
for ev in ('0148', '0628'):
    d = pd.read_csv(c(f'ogle/blg-{ev}.dat'), sep=r'\s+', header=None, names=['t', 'mag', 'err', 'seeing', 'sky'])
    p = ews_params(ev); evmeta['2019-BLG-' + ev] = p
    d = d[(d.t > p['tmax'] - 200) & (d.t < p['tmax'] + 200)].copy()
    d['t'] = d.t - 2450000; d['event'] = '2019-BLG-' + ev
    ml.append(d)
ml = pd.concat(ml)
save('microlensing', ml, {'event': None, 't': 5, 'mag': 3, 'err': 3}, title='Microlensing light curves (OGLE-IV Early Warning System, 2019)',
     source='ogle.astrouw.edu.pl/ogle4/ews/2019/ (events 2019-BLG-0148 and 2019-BLG-0628, phot.dat)', events=evmeta,
     cuts='Points within 200 days of the peak.', citation='Udalski 2003, Acta Astron. 53, 291 (EWS); Udalski, Szymanski & Szymanski 2015, Acta Astron. 65, 1 (OGLE-IV)',
     license='OGLE asks to be contacted before its EWS photometry is used in publications. Fine for private study; ask OGLE before sharing this page publicly.',
     columns_note='t = HJD - 2450000; mag = I')

# ---------------------------------------------------------------------------------------------- solar system
a1 = json.load(open(c('sbdb_ast.json'))); a2 = json.load(open(c('sbdb_tno.json')))
ast = pd.DataFrame(a1['data'], columns=a1['fields'])
tno = pd.DataFrame(a2['data'], columns=a2['fields'])
for d_ in (ast, tno):
    for k in ('a', 'e', 'i', 'H'):
        d_[k] = pd.to_numeric(d_[k], errors='coerce')
ast = ast[ast['class'] != 'TNO']
belt = ast['class'].isin(['MBA', 'IMB', 'OMB'])   # thin the whole belt by one factor, so no class boundary shows up as a step
sample = pd.concat([ast[belt].sample(frac=0.3, random_state=4), ast[~belt], tno])
sample = sample[(sample.a > 0) & (sample.a < 120) & (sample.e < 1)].copy()
sample.loc[sample['class'].isin(['MBA', 'IMB', 'OMB']), 'pdes'] = ''  # names only for the rarer classes, to keep the file small
save('smallbodies', sample, {'pdes': None, 'class': None, 'a': 4, 'e': 4, 'i': 2, 'H': 2},
     title='Asteroids and trans-Neptunian objects (JPL Small-Body Database)', source='ssd-api.jpl.nasa.gov/sbdb_query.api',
     cuts=f'Numbered asteroids brighter than H = 15.5 ({len(ast)}): a random 30% of the main belt (inner, middle and outer alike) plus every other class; every TNO ({len(tno)}). a < 120 AU, bound orbits.',
     citation='JPL Solar System Dynamics, Small-Body Database', license='Public (NASA/JPL)', columns_note='a AU; e; i degrees; H absolute magnitude')

# ------------------------------------------------------------------------------------- the Galaxy: globulars
txt = open(c('mwgc.dat'), encoding='latin-1').read().split('\n')
i0 = next(i for i, l in enumerate(txt) if l.strip().startswith('NGC 104'))
rows = []
for l in txt[i0:]:
    if not l.strip():
        break
    try:
        rows.append({'id': l[:9].strip(), 'name': l[9:25].strip(), 'ra_s': l[25:36].strip(), 'dec_s': l[38:49].strip(),
                     'l': float(l[50:58]), 'b': float(l[58:66]), 'R_sun': float(l[67:73]), 'R_gc': float(l[73:79]),
                     'X': float(l[79:85]), 'Y': float(l[85:91]), 'Z': float(l[91:97])})
    except ValueError:
        pass
gcl = pd.DataFrame(rows)
h = gcl.ra_s.str.split(expand=True).astype(float); dd = gcl.dec_s.str.split(expand=True).astype(float)
gcl['ra'] = 15 * (h[0] + h[1] / 60 + h[2] / 3600)
gcl['dec'] = (abs(dd[0]) + dd[1] / 60 + dd[2] / 3600) * np.where(gcl.dec_s.str.startswith('-'), -1, 1)
assert 140 < len(gcl) < 170, len(gcl)
save('globulars', gcl, {'id': None, 'name': None, 'ra': 3, 'dec': 3, 'l': 2, 'b': 2, 'R_sun': 2, 'R_gc': 2, 'X': 2, 'Y': 2, 'Z': 2},
     title='Globular clusters of the Milky Way (Harris catalogue, 2010 edition)', source='physics.mcmaster.ca/~harris/mwgc.dat',
     citation='Harris 1996, AJ 112, 1487 (2010 edition, arXiv:1012.3224)', license='Free to use with citation',
     columns_note='l, b Galactic degrees; distances kpc; X toward the Galactic centre, Y in the direction of rotation, Z to the north Galactic pole (Sun-centred, kpc)')

# --------------------------------------------------------------------------------------------- rotation curves
z = zipfile.ZipFile(c('Rotmod_LTG.zip'))
rc = []
for n in z.namelist():
    if not n.endswith('_rotmod.dat'):
        continue
    raw = z.read(n).decode()
    dist = float(re.search(r'Distance\s*=\s*([0-9.]+)', raw).group(1))
    d = pd.read_csv(io.StringIO(raw), sep=r'\s+', comment='#', header=None,
                    names=['R', 'Vobs', 'eV', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul'])
    d['gal'] = os.path.basename(n).replace('_rotmod.dat', '')
    rc.append(d)
rc = pd.concat(rc)
save('rotation_curves', rc, {'gal': None, 'R': 3, 'Vobs': 2, 'eV': 2, 'Vgas': 2, 'Vdisk': 2, 'Vbul': 2},
     title='Rotation curves of 175 disk galaxies (SPARC)', source='astroweb.case.edu/SPARC/Rotmod_LTG.zip',
     citation='Lelli, McGaugh & Schombert 2016, AJ 152, 157', license='Free to use with citation',
     columns_note='R kpc; Vobs, eV observed speed and error km/s; Vgas, Vdisk, Vbul: speeds the gas, the stellar disk (for M/L = 1 at 3.6 um) and the bulge would give on their own, km/s')
mrt = open(c('SPARC_Lelli2016c.mrt')).read().split('\n')
i0 = max(i for i, l in enumerate(mrt) if l.startswith('-----')) + 1
gal = []
for l in mrt[i0:]:
    p = l.split()
    if len(p) >= 18:
        gal.append({'gal': p[0], 'T': int(p[1]), 'D': float(p[2]), 'inc': float(p[5]), 'L36': float(p[7]), 'Reff': float(p[9]),
                    'MHI': float(p[13]), 'Vflat': float(p[15]), 'eVflat': float(p[16]), 'Q': int(p[17])})
gal = pd.DataFrame(gal)
assert len(gal) == 175, len(gal)
save('sparc', gal, {'gal': None, 'T': 0, 'D': 2, 'inc': 1, 'L36': 4, 'Reff': 2, 'MHI': 4, 'Vflat': 1, 'eVflat': 1, 'Q': 0},
     title='SPARC galaxy table', source='astroweb.case.edu/SPARC/SPARC_Lelli2016c.mrt', citation='Lelli, McGaugh & Schombert 2016, AJ 152, 157',
     license='Free to use with citation', columns_note='T Hubble type code (0 = S0 ... 11 = BCD); D Mpc; L36 = 3.6 um luminosity, 1e9 Lsun; Reff kpc; MHI 1e9 Msun; Vflat km/s (0 = no flat part); Q quality 1-3')

# ------------------------------------------------------------------------------------------------- galaxies
sd = pd.read_csv(c('sdss_local.csv'))
sd = sd[(sd.u > 0) & (sd.g > 0) & (sd.r > 0) & (sd.r < 17.77) & (sd.r50 > 0)].copy()
sd['sigma'] = np.where((sd.velDisp > 30) & (sd.velDispErr > 0) & (sd.velDispErr < 0.2 * sd.velDisp), sd.velDisp, np.nan)
sd = sd.sample(min(14000, len(sd)), random_state=5)
save('galaxies', sd, {'ra': 4, 'dec': 4, 'z': 5, 'u': 3, 'g': 3, 'r': 3, 'r50': 2, 'sigma': 1, 'p_el': 3, 'p_sp': 3},
     title='Nearby galaxies with Galaxy Zoo classifications (SDSS DR18)', source='skyserver.sdss.org/dr18, SpecObj x PhotoObj x zooSpec',
     query="class = 'GALAXY', zWarning = 0, 0.02 < z < 0.05, a quarter of objects (objID mod 4 = 0)",
     cuts='Petrosian magnitudes corrected for Galactic extinction; r < 17.77 (the spectroscopic survey limit); 14000 kept at random. Velocity dispersions kept when error < 20%.',
     citation='SDSS DR18 (Almeida et al. 2023, ApJS 267, 44); Galaxy Zoo: Lintott et al. 2011, MNRAS 410, 166', license='SDSS data are public',
     columns_note='u, g, r magnitudes; r50 = half-light radius arcsec; sigma = stellar velocity dispersion km/s; p_el, p_sp = Galaxy Zoo vote fractions (debiased) for elliptical and spiral')
co = pd.read_csv(c('sdss_coma.csv'))
save('coma', co, {'ra': 4, 'dec': 4, 'z': 5, 'r': 3, 'g': 3}, title='Galaxies toward the Coma cluster (SDSS DR18)',
     source='skyserver.sdss.org/dr18, SpecObj x PhotoObj', query="class = 'GALAXY', zWarning = 0, 0.003 < z < 0.08, 192 < RA < 198, 25.4 < Dec < 30.6",
     citation='SDSS DR18 (Almeida et al. 2023, ApJS 267, 44)', license='SDSS data are public', columns_note='z redshift; r, g extinction-corrected magnitudes')

MANIFEST['_about'] = {'built': str(date.today()), 'note': 'Every number in the Observatorium comes from these files. See pipeline/README.md to rebuild them.'}
json.dump(MANIFEST, open(os.path.join(OUT, 'manifest.json'), 'w'), indent=1)
print('total', round(sum(os.path.getsize(p) for p in glob.glob(os.path.join(OUT, '*'))) / 1e6, 2), 'MB')
