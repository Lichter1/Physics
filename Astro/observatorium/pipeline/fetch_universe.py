"""Data for chapters 6-9 (lectures 7-13): the expanding Universe, the observer's toolkit, light beyond the visible,
and reading galaxies.

  supernovae.csv   type Ia supernovae: redshift and distance modulus (Pantheon+, Scolnic et al. 2022; Brout et al. 2022)
  cmb_spectrum.csv the CMB spectrum measured by COBE/FIRAS (Fixsen et al. 1996, via NASA LAMBDA)
  cmb_power.csv    the CMB temperature power spectrum (Planck 2018 release 3, low-l and binned high-l)
  cmb_theory.csv   the Planck 2018 best-fit LCDM TT spectrum, for the model curve
  zspec.csv        four SDSS spectra for measuring redshifts: a star-forming galaxy, an elliptical, a distant red
                   galaxy and a quasar at z ~ 3 with its Lyman-alpha forest (SDSS DR17 spec-lite, 0.4 nm bins)
  bpt.csv          emission lines, stellar masses, star formation and metallicity of SDSS galaxies (MPA-JHU)
  frbs.csv         fast radio bursts with dispersion measures, and host redshifts where localized
                   (FRBs/FRB repository, Prochaska et al.)
  gw.csv           compact-binary mergers from the gravitational-wave catalogues (GWOSC, GWTC-1 to GWTC-5)
  photfield.json   a real r-band image cutout in calibrated units (DESI Legacy Surveys DR9) with SDSS stars, for
                   aperture photometry
  img/sky_*.jpg    the whole sky at other wavelengths (NASA SkyView: Haslam 408 MHz, HI4PI, WMAP ILC, IRIS 100 um,
                   Finkbeiner H-alpha, ROSAT RASS, Fermi LAT)
  img/lec/*.jpg    more lecture pictures
Run after fetch_lecture_media.py:  python3 fetch_universe.py
"""
import io, json, os, sys, time, urllib.parse
import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fetch_lecture_media as FLM
from sdss import query

CACHE = os.path.join(HERE, 'cache', 'u'); os.makedirs(CACHE, exist_ok=True)
DATA = os.path.join(HERE, '..', 'data'); IMG = os.path.join(DATA, 'img'); LEC = os.path.join(IMG, 'lec')
TODAY = time.strftime('%Y-%m-%d')


def get(url, name):
    FLM.CACHE = CACHE
    return FLM.get(url, name)


# ---------------------------------------------------------------------------------------------- chapter 6
def supernovae():
    raw = get('https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat', 'pantheon.dat')
    d = pd.read_csv(io.BytesIO(raw), sep=r'\s+')
    d = d[(d.zHD > 0.01) & (d.IS_CALIBRATOR == 0)]
    w = 1 / d.MU_SH0ES_ERR_DIAG ** 2
    d = d.assign(w=w, wmu=w * d.MU_SH0ES, wmb=w * d.m_b_corr)
    g = d.groupby('CID').agg(ra=('RA', 'first'), dec=('DEC', 'first'), z=('zHD', 'mean'), w=('w', 'sum'), wmu=('wmu', 'sum'), wmb=('wmb', 'sum'), n=('w', 'size'))
    out = pd.DataFrame({'name': g.index, 'ra': g.ra.values, 'dec': g.dec.values, 'z': g.z.values,
                        'mu': (g.wmu / g.w).values, 'emu': (1 / np.sqrt(g.w)).values, 'mb': (g.wmb / g.w).values})
    out = out.sort_values('z')
    out.to_csv(os.path.join(DATA, 'supernovae.csv'), index=False, float_format='%.5g')
    return len(out)


def cmb():
    t = get('https://lambda.gsfc.nasa.gov/data/cobe/firas/monopole_spec/firas_monopole_spec_v1.txt', 'firas.txt').decode()
    rows = [list(map(float, l.split())) for l in t.splitlines() if l.strip() and not l.startswith('#')]
    f = pd.DataFrame(rows, columns=['nu_icm', 'I', 'resid', 'sig', 'gal'])
    s = pd.DataFrame({'f_ghz': f.nu_icm * 29.9792458, 'nu_icm': f.nu_icm, 'I': f.I, 'err': f.sig / 1000, 'resid': f.resid})
    s.to_csv(os.path.join(DATA, 'cmb_spectrum.csv'), index=False, float_format='%.6g')
    base = 'https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/'
    lo = pd.read_csv(io.BytesIO(get(base + 'COM_PowerSpect_CMB-TT-full_R3.01.txt', 'planck_tt_full.txt')), sep=r'\s+', comment='#', header=None, names=['l', 'Dl', 'lo', 'hi'])
    hi = pd.read_csv(io.BytesIO(get(base + 'COM_PowerSpect_CMB-TT-binned_R3.01.txt', 'planck_tt_binned.txt')), sep=r'\s+', comment='#', header=None, names=['l', 'Dl', 'lo', 'hi', 'best'])
    lo = lo[lo.l < 30]
    p = pd.concat([pd.DataFrame({'l': lo.l, 'Dl': lo.Dl, 'err': (lo.lo + lo.hi) / 2, 'binned': 0}),
                   pd.DataFrame({'l': hi.l.round(1), 'Dl': hi.Dl, 'err': (hi.lo + hi.hi) / 2, 'binned': 1})])
    p.to_csv(os.path.join(DATA, 'cmb_power.csv'), index=False, float_format='%.5g')
    th = pd.read_csv(io.BytesIO(get(base + 'COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt', 'planck_theory.txt')), sep=r'\s+', comment='#', header=None).iloc[:, :2]
    th.columns = ['l', 'tt']
    th = th[th.l <= 2600]
    th.to_csv(os.path.join(DATA, 'cmb_theory.csv'), index=False, float_format='%.5g')
    return len(s), len(p), len(th)


def spectrum(plate, mjd, fib, run2d='26', step=0.4, lo=380, hi=922):
    from astropy.io import fits
    sub = 'sdss' if run2d == '26' else 'eboss'
    url = f'https://data.sdss.org/sas/dr17/{sub}/spectro/redux/{run2d}/spectra/lite/{plate:04d}/spec-{plate:04d}-{mjd}-{fib:04d}.fits'
    h = fits.open(io.BytesIO(get(url, f'spec_{plate}_{mjd}_{fib}.fits'))); c = h[1].data
    lam = 10 ** c['loglam'] / 10.0; f = c['flux']; ok = (c['ivar'] > 0) & (c['and_mask'] == 0)
    bins = np.arange(lo, hi, step); idx = np.digitize(lam[ok], bins)
    fb = np.array([np.median(f[ok][idx == k]) if np.any(idx == k) else np.nan for k in range(1, len(bins))])
    return bins[:-1] + step / 2, fb


def zspec():
    """Four spectra; each candidate must show its defining feature where its redshift puts it (SDSS's highest-S/N
    'quasars' and 'galaxies' at high z are often misclassified stars)."""
    def mean(lam, f, a, b): m = (lam > a) & (lam < b); return np.nanmedian(f[m]) if m.any() else np.nan
    def ok(key, lam, f, z):
        if key == 'qso':     # a strong Lyman-alpha emission line, and a forest (darker) blueward of it
            peak = mean(lam, f, 1210 / 10 * (1 + z), 1225 / 10 * (1 + z)); red = mean(lam, f, 1300 / 10 * (1 + z), 1400 / 10 * (1 + z))
            return peak > 1.8 * red
        if key in ('ell', 'lrg'):   # the 4000 A break
            return mean(lam, f, 405 * (1 + z), 425 * (1 + z)) > 1.6 * mean(lam, f, 375 * (1 + z), 395 * (1 + z))
        return mean(lam, f, 655.5 * (1 + z), 657.5 * (1 + z)) > 2.5 * mean(lam, f, 640 * (1 + z), 650 * (1 + z))   # H-alpha in emission
    picks = [
        ('sf', 'A star-forming galaxy', '26', "class = 'GALAXY' AND subClass LIKE 'STARFORMING%' AND z BETWEEN 0.06 AND 0.10 AND snMedian > 20"),
        ('ell', 'An elliptical galaxy', '26', "class = 'GALAXY' AND (subClass IS NULL OR subClass = '') AND z BETWEEN 0.09 AND 0.13 AND snMedian > 25 AND velDisp > 200"),
        ('lrg', 'A distant red galaxy', 'v5_13_2', "class = 'GALAXY' AND (subClass IS NULL OR subClass = '') AND z BETWEEN 0.45 AND 0.55 AND snMedian > 6 AND velDisp > 150"),
        ('qso', 'A quasar', 'v5_13_2', "class = 'QSO' AND z BETWEEN 2.9 AND 3.2 AND snMedian > 8"),
    ]
    out, meta = [], []
    for key, label, run2d, where in picks:
        df = query(f"SELECT TOP 25 plate, mjd, fiberID, z, snMedian, ra, dec FROM SpecObj WHERE {where} AND zWarning = 0 AND run2d = '{run2d}' ORDER BY snMedian DESC",
                   cache=os.path.join(CACHE, f'zspec2_{key}.csv'))
        found = False
        for _, r in df.iterrows():
            try:
                lam, f = spectrum(int(r.plate), int(r.mjd), int(r.fiberID), run2d=run2d)
            except Exception as e:
                print('  skip', e); continue
            if np.isfinite(f).mean() > 0.95 and ok(key, lam, f, r.z):
                found = True; break
        if not found: raise RuntimeError('no good spectrum for ' + key)
        f = f / np.nanpercentile(f, 99.5)
        name = f'{label} (SDSS {int(r.plate)}-{int(r.mjd)}-{int(r.fiberID)})'
        out.append(pd.DataFrame({'obj': key, 'lam': lam, 'f': f}))
        meta.append({'obj': key, 'name': name, 'z': float(r.z), 'ra': float(r.ra), 'dec': float(r.dec), 'sn': float(r.snMedian)})
        print('spectrum', key, name, 'z =', round(r.z, 4), flush=True)
    d = pd.concat(out).dropna()
    d.to_csv(os.path.join(DATA, 'zspec.csv'), index=False, float_format='%.4g')
    json.dump(meta, open(os.path.join(DATA, 'zspec_meta.json'), 'w'), indent=1)
    return len(d)


# ---------------------------------------------------------------------------------------------- chapter 9
def bpt():
    d = query("""SELECT s.ra, s.dec, s.z, l.h_alpha_flux AS ha, l.h_beta_flux AS hb, l.oiii_5007_flux AS oiii, l.nii_6584_flux AS nii,
        e.lgm_tot_p50 AS logm, e.sfr_tot_p50 AS logsfr, e.oh_p50 AS oh, e.bptclass
      FROM SpecObj s JOIN galSpecLine l ON s.specObjID = l.specObjID JOIN galSpecExtra e ON s.specObjID = e.specObjID
      WHERE s.class = 'GALAXY' AND s.zWarning = 0 AND s.z BETWEEN 0.02 AND 0.2 AND (s.fiberID % 5) = 0
        AND l.h_alpha_flux_err > 0 AND l.h_beta_flux_err > 0 AND l.oiii_5007_flux_err > 0 AND l.nii_6584_flux_err > 0
        AND l.h_alpha_flux > 3 * l.h_alpha_flux_err AND l.h_beta_flux > 3 * l.h_beta_flux_err
        AND l.oiii_5007_flux > 3 * l.oiii_5007_flux_err AND l.nii_6584_flux > 3 * l.nii_6584_flux_err""",
              cache=os.path.join(CACHE, 'bpt.csv'))
    d = d[(d.logm > 7) & (d.logsfr > -5)]
    d['oh'] = d.oh.where(d.oh > 7, np.nan)
    d = d.sample(n=min(len(d), 20000), random_state=1)   # plenty for the diagrams, a third of the size
    d.to_csv(os.path.join(DATA, 'bpt.csv'), index=False, float_format='%.5g')
    return len(d)


# ---------------------------------------------------------------------------------------------- chapter 8
def grbs():
    q = urllib.parse.urlencode({'tablehead': 'name=heasarc_fermigbrst', 'Fields': 'name,t90,fluence,ra,dec,lii,bii', 'ResultMax': 0, 'displaymode': 'PureTextDisplay'})
    t = get('https://heasarc.gsfc.nasa.gov/db-perl/W3Browse/w3query.pl?' + q, 'fermigbrst_pos.txt').decode()
    rows = []
    for l in t.splitlines():
        p = [x.strip() for x in l.split('|')]
        if len(p) >= 8 and p[1].startswith('GRB'):
            try:
                rows.append({'name': p[1], 't90': float(p[2]), 'fluence': float(p[3]) if p[3] else np.nan,
                             'ra': ra_deg(p[4]), 'dec': dec_deg(p[5]), 'l': float(p[6]), 'b': float(p[7])})
            except ValueError:
                pass
    d = pd.DataFrame(rows)
    d = d[d.t90 > 0]
    d.to_csv(os.path.join(DATA, 'grbs.csv'), index=False, float_format='%.6g')
    return len(d)


def ra_deg(s):
    p = s.split()
    return 15 * (float(p[0]) + float(p[1]) / 60 + float(p[2]) / 3600) if len(p) == 3 else float(s)


def dec_deg(s):
    p = s.split()
    if len(p) != 3: return float(s)
    sg = -1 if p[0].startswith('-') else 1
    return sg * (abs(float(p[0])) + float(p[1]) / 60 + float(p[2]) / 3600)


def frbs():
    d = pd.read_csv(io.BytesIO(get('https://raw.githubusercontent.com/FRBs/FRB/main/frb/data/FRBs/FRBs_base.csv', 'frbs_base.csv')))
    out = pd.DataFrame({'name': d.Name, 'ra': d.ra, 'dec': d.dec, 'dm': d.DM, 'z': d.z,
                        'repeater': d.repeater.astype(str).str.upper().eq('TRUE').astype(int), 'telescope': d.telescope})
    out.to_csv(os.path.join(DATA, 'frbs.csv'), index=False, float_format='%.6g')
    return len(out), int(out.z.notna().sum())


def gw():
    ev = json.loads(get('https://gwosc.org/eventapi/json/GWTC/', 'gwtc.json'))['events']
    rows = []
    for e in ev.values():
        if not e.get('mass_1_source') or not e.get('mass_2_source'): continue
        if e.get('p_astro') is not None and e['p_astro'] < 0.5: continue
        rows.append({'name': e['commonName'], 'catalog': e['catalog.shortName'], 'm1': e['mass_1_source'], 'm2': e['mass_2_source'],
                     'mfinal': e.get('final_mass_source'), 'mchirp': e.get('chirp_mass_source'), 'dl': e.get('luminosity_distance'),
                     'z': e.get('redshift'), 'snr': e.get('network_matched_filter_snr'), 'chi_eff': e.get('chi_eff'),
                     'year': 2000 + int(e['commonName'][2:4])})
    d = pd.DataFrame(rows).drop_duplicates('name').sort_values('name')
    d.to_csv(os.path.join(DATA, 'gw.csv'), index=False, float_format='%.5g')
    return len(d)


# ---------------------------------------------------------------------------------------------- chapter 7
def photfield():
    """A Legacy Survey r-band cutout with several unsaturated SDSS stars and no bright ones."""
    from astropy.io import fits
    cands = [(248.0, 15.0), (250.0, 10.0)]   # towards the inner Galaxy at b ~ 35: several unsaturated stars in 52"
    size, ps = 200, 0.262
    half = size * ps / 2 / 3600
    for ra, dec in cands:
        st = query(f"""SELECT ra, dec, psfMag_r AS r, psfMagErr_r AS er FROM Star
                       WHERE ra BETWEEN {ra - half} AND {ra + half} AND dec BETWEEN {dec - half} AND {dec + half} AND psfMag_r < 22""",
                   cache=os.path.join(CACHE, f'phot_{ra}_{dec}.csv'))
        good = st[(st.r > 17) & (st.r < 21)]
        if len(good) >= 4 and not (st.r < 16.5).any():
            break
    print('photometry field', ra, dec, len(st), 'SDSS stars', flush=True)
    url = f'https://www.legacysurvey.org/viewer/fits-cutout?ra={ra}&dec={dec}&layer=ls-dr9&pixscale={ps}&bands=r&size={size}'
    h = fits.open(io.BytesIO(get(url, f'ls_{ra}_{dec}_r.fits')))
    img = np.asarray(h[0].data, float); hd = h[0].header
    img = np.nan_to_num(img)
    from astropy.wcs import WCS
    w = WCS(hd)
    xs, ys = w.all_world2pix(st.ra.values, st.dec.values, 0)
    stars = [{'x': round(float(x), 2), 'y': round(float(size - 1 - y), 2), 'ra': float(a), 'dec': float(b), 'r': round(float(m), 3), 'er': round(float(e), 3)}
             for x, y, a, b, m, e in zip(xs, ys, st.ra, st.dec, st.r, st.er) if 3 < x < size - 4 and 3 < y < size - 4]
    flip = img[::-1]                       # row 0 = north (top), as drawn on screen
    scale = 1000.0
    q = np.clip(np.round(flip * scale), -32000, 32000).astype(int)
    json.dump({'ra': ra, 'dec': dec, 'pixscale': ps, 'w': size, 'h': size, 'band': 'r', 'zp': 22.5, 'unit': 'nanomaggies',
               'scale': scale, 'data': q.ravel().tolist(), 'stars': stars,
               'credit': 'DESI Legacy Imaging Surveys DR9 (legacysurvey.org), r band; star magnitudes from SDSS DR18'},
              open(os.path.join(DATA, 'photfield.json'), 'w'), separators=(',', ':'))
    return len(stars)


# ---------------------------------------------------------------------------------------------- the sky at other wavelengths
BANDS = [  # id, survey file, label, colour map, transform, credit
    ('radio', 'sky_radio.fits', 'Radio 408 MHz', 'inferno', 'log', 'Haslam et al. 1982 (408 MHz all-sky survey), via NASA SkyView'),
    ('hi', 'sky_hi.fits', 'Hydrogen 21 cm', 'bone', 'log', 'HI4PI Collaboration 2016, via NASA SkyView'),
    ('cmb', 'sky_cmb.fits', 'Microwave: the CMB', 'coolwarm', 'lin', 'NASA/WMAP Science Team (9-year ILC map), via NASA SkyView'),
    ('fir', 'sky_fir.fits', 'Far infrared 100 µm', 'afmhot', 'log', 'IRIS (Miville-Deschenes & Lagache 2005, reprocessed IRAS), via NASA SkyView'),
    ('halpha', 'sky_halpha.fits', 'Hydrogen-alpha 656 nm', 'gist_heat', 'log', 'Finkbeiner 2003 (H-alpha composite), via NASA SkyView'),
    ('xray', 'alt_xray_r6.fits', 'X-rays 0.7-1.6 keV', 'magma', 'logsmooth', 'ROSAT All-Sky Survey diffuse background, R6 band (Snowden et al. 1997), via NASA SkyView'),
    ('gamma', 'alt_egret.fits', 'Gamma rays above 100 MeV', 'viridis', 'logsmooth', 'CGRO/EGRET (NASA), via NASA SkyView'),   # SkyView's Fermi maps have two empty patches
]


SKYVIEW = {'sky_radio.fits': '408MHz', 'sky_hi.fits': 'HI4PI', 'sky_cmb.fits': 'WMAP ILC', 'sky_fir.fits': 'IRIS 100',
           'sky_halpha.fits': 'H-Alpha Comp', 'alt_xray_r6.fits': 'RASS Background 6', 'alt_egret.fits': 'EGRET >100 MeV'}


def skyview(fn):
    """Whole-sky Car projection in Galactic coordinates, 0.25 deg pixels (slow: minutes per survey)."""
    q = urllib.parse.urlencode({'Survey': SKYVIEW[fn], 'Position': '0,0', 'Coordinates': 'Galactic', 'Projection': 'Car',
                                'Size': '360,180', 'Pixels': '1440,720', 'Return': 'FITS'})
    return get('https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl?' + q, fn)


def skymaps():
    import matplotlib; matplotlib.use('Agg')
    from matplotlib import cm
    from astropy.io import fits
    from scipy.ndimage import gaussian_filter
    meta = json.load(open(os.path.join(DATA, 'images.json')))
    meta['bands'] = []
    for key, fn, label, cmap, tr, credit in BANDS:
        path = os.path.join(CACHE, fn)
        skyview(fn)
        d = np.asarray(fits.open(path)[0].data, float)
        d = np.nan_to_num(d, nan=np.nanmedian(d))
        if tr == 'logsmooth':
            d = gaussian_filter(np.clip(d, 0, None), 2.0 if key == 'gamma' else 0.8)
        if tr.startswith('log'):
            pos = d[d > 0]
            floor = np.percentile(pos, 2) if len(pos) else 1e-12
            d = np.log10(np.clip(d, floor, None))
            lo, hi = np.percentile(d, [1, 99.7])
        else:
            lim = np.percentile(np.abs(d), 99.5); lo, hi = -lim, lim
        x = np.clip((d - lo) / (hi - lo), 0, 1)[::-1]          # FITS row 0 is the south pole; images start at the top
        rgb = (matplotlib.colormaps[cmap](x)[..., :3] * 255).astype(np.uint8)
        Image.fromarray(rgb).save(os.path.join(IMG, f'sky_{key}.jpg'), quality=85, optimize=True)
        meta['bands'].append({'id': key, 'label': label, 'file': f'img/sky_{key}.jpg', 'w': rgb.shape[1], 'h': rgb.shape[0],
                              'frame': 'gal', 'proj': 'car', 'lon_center': 0, 'lon_left': True, 'credit': credit})
        print('band', key, rgb.shape, os.path.getsize(os.path.join(IMG, f'sky_{key}.jpg')) // 1000, 'kB', flush=True)
    json.dump(meta, open(os.path.join(DATA, 'images.json'), 'w'), indent=1)
    return len(meta['bands'])


# ---------------------------------------------------------------------------------------------- pictures
ESAH, ESO = FLM.ESAH, FLM.ESO
WIKI = 'https://commons.wikimedia.org/wiki/Special:FilePath/{}?width=1400'
IMAGES = [
    ('wmap_cmb', WIKI.format('Ilc_9yr_moll4096.png'), 'NASA/WMAP Science Team (9-year ILC map, Wikimedia Commons), public domain'),
    ('horn_antenna', WIKI.format('Horn_Antenna-in_Holmdel,_New_Jersey.jpeg'), 'NASA (Wikimedia Commons), public domain'),
    ('tuning_fork', ESAH.format('heic9902o'), 'NASA & ESA (ESA/Hubble heic9902o), CC BY 4.0'),
    ('hudf', ESAH.format('heic0406a'), 'NASA, ESA, S. Beckwith (STScI) and the HUDF Team (ESA/Hubble heic0406a), CC BY 4.0'),
    ('bullet', WIKI.format('1e0657_scale.jpg'), 'X-ray: NASA/CXC/CfA/M. Markevitch et al.; optical: NASA/STScI, Magellan/U. Arizona/D. Clowe et al.; lensing map: NASA/STScI, ESO WFI, Magellan/U. Arizona/D. Clowe et al. (Wikimedia Commons), public domain'),
    ('vla', WIKI.format('USA.NM.VeryLargeArray.02.jpg'), 'Hajor (Wikimedia Commons), CC BY-SA 3.0'),
    ('eht_array', ESO.format('ann17015a'), 'ESO/O. Furtak (ESO ann17015a), CC BY 4.0'),
    ('kilonova', ESO.format('eso1733a'), 'ESO/L. Calçada/M. Kornmesser (ESO eso1733a, artist\'s impression), CC BY 4.0'),
    ('carina_ir', ESAH.format('heic0910f'), 'NASA, ESA and the Hubble SM4 ERO Team (ESA/Hubble heic0910f), CC BY 4.0'),
]


def images():
    FLM.CACHE = CACHE
    FLM.IMAGES = IMAGES
    new = FLM.images()
    path = os.path.join(DATA, 'lecture_images.json')
    old = [x for x in json.load(open(path)) if x['id'] not in {n['id'] for n in new}]
    json.dump(old + new, open(path, 'w'), indent=1)
    return len(new)


def manifest(counts):
    p = os.path.join(DATA, 'manifest.json'); m = json.load(open(p))
    def put(k, title, rows, source, cuts, citation, license):
        m[k] = {'title': title, 'rows': rows, 'fetched': TODAY, 'source': source, 'cuts': cuts, 'citation': citation, 'license': license}
    put('supernovae', 'Type Ia supernovae (Pantheon+)', counts['sn'], 'github.com/PantheonPlusSH0ES/DataRelease, Pantheon+SH0ES.dat',
        'z_HD > 0.01, Cepheid calibrators removed, repeat observations of one supernova averaged (inverse-variance)',
        'Scolnic et al. 2022, ApJ 938, 113; Brout et al. 2022, ApJ 938, 110', 'Public release')
    put('cmb_spectrum', 'The CMB spectrum (COBE FIRAS)', counts['firas'], 'lambda.gsfc.nasa.gov, firas_monopole_spec_v1.txt',
        'Monopole spectrum with 1-sigma uncertainties (converted from kJy/sr to MJy/sr)', 'Fixsen et al. 1996, ApJ 473, 576', 'NASA, public domain')
    put('cmb_power', 'The CMB temperature power spectrum (Planck 2018)', counts['power'], 'irsa.ipac.caltech.edu Planck release 3, COM_PowerSpect_CMB-TT-full/binned_R3.01',
        'l < 30 unbinned; l >= 30 in bins of 30; error = mean of the lower and upper 1-sigma', 'Planck Collaboration 2020, A&A 641, A5 and A6', 'ESA Planck Legacy Archive, free use with acknowledgement')
    put('cmb_theory', 'Planck 2018 best-fit LCDM TT spectrum', counts['theory'], 'COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01',
        'l <= 2600', 'Planck Collaboration 2020, A&A 641, A6', 'ESA Planck Legacy Archive')
    put('zspec', 'SDSS spectra for measuring redshifts', counts['zspec'], 'data.sdss.org DR17 spec-lite files (see zspec_meta.json)',
        'Highest S/N spectrum of each kind with no masked pixels; median-binned to 0.4 nm; scaled to the 99.5th percentile', 'Abdurro\'uf et al. 2022 (SDSS DR17)', 'SDSS, public with acknowledgement')
    put('bpt', 'Emission lines and properties of SDSS galaxies (MPA-JHU)', counts['bpt'], 'SkyServer DR18: SpecObj + galSpecLine + galSpecExtra',
        '0.02 < z < 0.2, every fifth fibre, H-alpha, H-beta, [OIII] 5007 and [NII] 6584 all detected at 3 sigma, log M > 7; a random 20,000 of them', 'Brinchmann et al. 2004; Kauffmann et al. 2003; Tremonti et al. 2004', 'SDSS, public with acknowledgement')
    put('grbs', 'Gamma-ray bursts (Fermi GBM)', counts['grb'], 'HEASARC fermigbrst', 'T90 > 0; positions from the GBM catalogue (typical error a few degrees)',
        'von Kienlin et al. 2020, ApJ 893, 46', 'NASA, public domain')
    put('frbs', 'Fast radio bursts', counts['frb'], 'github.com/FRBs/FRB, frb/data/FRBs/FRBs_base.csv', 'All bursts in the table; z only for localized hosts',
        'Prochaska & Zheng 2019; Macquart et al. 2020; Heintz et al. 2020; Gordon et al. 2023 (as compiled in the FRBs repository)', 'BSD-3 (repository)')
    put('gw', 'Gravitational-wave mergers (GWTC)', counts['gw'], 'gwosc.org/eventapi/json/GWTC/', 'Events with source-frame masses and p_astro >= 0.5; medians of the posteriors',
        'LIGO Scientific, Virgo and KAGRA Collaborations, GWTC-1 to GWTC-5 (GWOSC)', 'CC BY 4.0 (GWOSC)')
    put('photfield', 'An r-band image for photometry', counts['phot'], 'legacysurvey.org fits-cutout (ls-dr9, r, 0.262"/pixel, 200 x 200); SDSS Star table',
        'Pixel values in nanomaggies (AB zero point 22.5), multiplied by 1000 and rounded', 'Dey et al. 2019 (Legacy Surveys); SDSS DR18', 'Public with acknowledgement')
    m['skybands'] = {'title': 'The whole sky at other wavelengths', 'rows': counts['bands'], 'fetched': TODAY, 'source': 'NASA SkyView all-sky Car projections, Galactic, 0.25 deg/pixel',
                     'cuts': 'Log stretch (radio, HI, IR, H-alpha), smoothed log stretch (X-ray and gamma counts), symmetric linear stretch (CMB)',
                     'citation': '; '.join(b[5] for b in BANDS), 'license': 'NASA SkyView; each survey with acknowledgement'}
    json.dump(m, open(p, 'w'), indent=1)


if __name__ == '__main__':
    c = {}
    c['sn'] = supernovae(); print('supernovae', c['sn'], flush=True)
    c['firas'], c['power'], c['theory'] = cmb(); print('cmb', c['firas'], c['power'], c['theory'], flush=True)
    c['zspec'] = zspec()
    c['bpt'] = bpt(); print('bpt', c['bpt'], flush=True)
    c['grb'] = grbs(); print('grbs', c['grb'], flush=True)
    c['frb'] = frbs()[0]; print('frbs', c['frb'], flush=True)
    c['gw'] = gw(); print('gw', c['gw'], flush=True)
    c['phot'] = photfield(); print('phot stars', c['phot'], flush=True)
    c['bands'] = skymaps()
    c['img'] = images()
    manifest(c)
    print('done', c)
