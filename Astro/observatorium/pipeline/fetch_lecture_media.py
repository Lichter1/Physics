"""Pictures and small datasets for the lectures.

Images (CC BY 4.0 from ESA/Hubble, ESA/Webb and ESO; public domain from NASA and Wikimedia Commons) go to
../data/img/lec/. Data for the lecture graphs:
  sunspots.csv   yearly mean sunspot number since 1700 (SILSO, Royal Observatory of Belgium)
  grbs.csv       durations of gamma-ray bursts seen by Fermi GBM (HEASARC table fermigbrst)
  spectra.csv    real stellar spectra across the spectral sequence (SDSS DR17, binned to 2 nm)
Run:  python3 fetch_lecture_media.py
"""
import io, json, os, re, time, urllib.parse, urllib.request
import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache', 'lec'); os.makedirs(CACHE, exist_ok=True)
DATA = os.path.join(HERE, '..', 'data'); IMG = os.path.join(DATA, 'img', 'lec'); os.makedirs(IMG, exist_ok=True)

def get(url, name, tries=3):
    path = os.path.join(CACHE, name)
    if os.path.exists(path) and os.path.getsize(path) > 500:
        return open(path, 'rb').read()
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'observatorium-teaching-tool'})
            with urllib.request.urlopen(req, timeout=300) as r:
                b = r.read()
            open(path, 'wb').write(b); return b
        except Exception as e:
            print('  retry', name, e, flush=True); time.sleep(4)
    raise RuntimeError(url)

ESAH = 'https://cdn.esahubble.org/archives/images/screen/{}.jpg'
ESAW = 'https://cdn.esawebb.org/archives/images/screen/{}.jpg'
ESO = 'https://cdn.eso.org/images/screen/{}.jpg'
IMAGES = [  # id, url, credit
    ('abell370', ESAH.format('heic0910b'), 'NASA, ESA, the Hubble SM4 ERO Team and ST-ECF (ESA/Hubble heic0910b), CC BY 4.0'),
    ('webb_deep_field', ESAW.format('weic2209a'), 'NASA, ESA, CSA, STScI (ESA/Webb weic2209a), CC BY 4.0'),
    ('horseshoe', ESAH.format('potw1151a'), 'ESA/Hubble & NASA (potw1151a), CC BY 4.0'),
    ('cluster_dm_maps', ESAH.format('heic1506a'), 'NASA, ESA, D. Harvey (EPFL), R. Massey (Durham) (ESA/Hubble heic1506a), CC BY 4.0'),
    ('antennae', ESAH.format('heic0615a'), 'NASA, ESA, and the Hubble Heritage Team (ESA/Hubble heic0615a), CC BY 4.0'),
    ('df2', ESAH.format('heic1806a'), 'NASA, ESA, and P. van Dokkum (Yale) (ESA/Hubble heic1806a), CC BY 4.0'),
    ('m87_jet', ESAH.format('opo0020a'), 'NASA and the Hubble Heritage Team (STScI/AURA) (ESA/Hubble opo0020a), CC BY 4.0'),
    ('sn1987a', ESAH.format('heic0704a'), 'NASA, ESA, P. Challis and R. Kirshner (Harvard-Smithsonian CfA) (ESA/Hubble heic0704a), CC BY 4.0'),
    ('cats_eye', ESAH.format('heic0414a'), 'NASA, ESA, HEIC and the Hubble Heritage Team (ESA/Hubble heic0414a), CC BY 4.0'),
    ('hr8799', ESAH.format('opo1129a'), 'NASA, ESA, and R. Soummer (STScI) (ESA/Hubble opo1129a), CC BY 4.0'),
    ('pillars', ESAW.format('weic2216a'), 'NASA, ESA, CSA, STScI; J. DePasquale, A. Koekemoer, A. Pagan (ESA/Webb weic2216a), CC BY 4.0'),
    ('m87_bh', ESO.format('eso1907a'), 'EHT Collaboration (ESO eso1907a), CC BY 4.0'),
    ('sgra_bh', ESO.format('eso2208-eht-mwa'), 'EHT Collaboration (ESO eso2208-eht-mwa), CC BY 4.0'),
    ('hltau', ESO.format('eso1436a'), 'ALMA (ESO/NAOJ/NRAO) (ESO eso1436a), CC BY 4.0'),
    ('s2_motion', ESO.format('eso1825e'), 'ESO/GRAVITY Collaboration (eso1825e), CC BY 4.0'),
    ('eclipse1919', 'https://upload.wikimedia.org/wikipedia/commons/3/37/1919_eclipse_positive.jpg', 'F. W. Dyson, A. S. Eddington and C. Davidson 1920, Phil. Trans. R. Soc. A 220 (Wikimedia Commons), public domain'),
    ('einstein_cross', 'https://thumb.wikimedia.org/wikipedia/commons/thumb/c/c8/Einstein_cross.jpg/1280px-Einstein_cross.jpg', 'NASA, ESA and STScI (Wikimedia Commons), public domain'),
    ('pioneer_plaque', 'https://thumb.wikimedia.org/wikipedia/commons/thumb/0/02/Pioneer_plaque.svg/1280px-Pioneer_plaque.svg.png', 'NASA; designed by C. Sagan and F. Drake, artwork by L. Salzman Sagan (Wikimedia Commons), public domain'),
    ('sun_hmi', 'https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_HMIIC.jpg', 'NASA/SDO and the HMI science team, ' + time.strftime('%d %B %Y') + ', public domain'),
]

def images():
    out = []
    for key, url, credit in IMAGES:
        try:
            b = get(url, key + os.path.splitext(url.split('?')[0])[1])
        except Exception as e:
            print('FAILED', key, e); continue
        im = Image.open(io.BytesIO(b))
        if im.mode in ('RGBA', 'LA', 'P'):
            bg = Image.new('RGB', im.size, (255, 255, 255)); bg.paste(im.convert('RGBA'), mask=im.convert('RGBA').split()[-1]); im = bg
        im = im.convert('RGB')
        if im.width > 1400:
            im = im.resize((1400, round(im.height * 1400 / im.width)), Image.LANCZOS)
        im.save(os.path.join(IMG, key + '.jpg'), quality=82, optimize=True, progressive=True)
        out.append({'id': key, 'file': 'img/lec/' + key + '.jpg', 'w': im.width, 'h': im.height, 'credit': credit, 'source': url})
        print(key, im.size, os.path.getsize(os.path.join(IMG, key + '.jpg')) // 1000, 'kB', flush=True)
    return out

def sunspots():
    t = get('https://www.sidc.be/SILSO/DATA/SN_y_tot_V2.0.txt', 'SN_y_tot_V2.0.txt').decode()
    rows = [l.split() for l in t.splitlines() if l.strip()]
    d = pd.DataFrame({'year': [float(r[0]) for r in rows], 'sn': [float(r[1]) for r in rows]})
    d = d[d.sn >= 0]
    d.to_csv(os.path.join(DATA, 'sunspots.csv'), index=False)
    return len(d)

def grbs():
    q = urllib.parse.urlencode({'tablehead': 'name=heasarc_fermigbrst', 'Fields': 'name,t90,t90_error,fluence', 'ResultMax': 0, 'displaymode': 'PureTextDisplay'})
    t = get('https://heasarc.gsfc.nasa.gov/db-perl/W3Browse/w3query.pl?' + q, 'fermigbrst.txt').decode()
    rows = []
    for l in t.splitlines():
        p = [x.strip() for x in l.split('|')]
        if len(p) >= 5 and p[1].startswith('GRB'):
            try:
                rows.append({'name': p[1], 't90': float(p[2]), 'fluence': float(p[4]) if p[4] else np.nan})
            except ValueError:
                pass
    d = pd.DataFrame(rows)
    d = d[d.t90 > 0]
    d.to_csv(os.path.join(DATA, 'grbs.csv'), index=False, float_format='%.4g')
    return len(d)

def spectra():
    from astropy.io import fits
    import sys; sys.path.insert(0, HERE)
    from sdss import query
    types = [('B', 'B6%'), ('A', 'A0%'), ('F', 'F5%'), ('G', 'G2%'), ('K', 'K3%'), ('M', 'M2%')]
    out = []
    for label, pat in types:
        df = query(f"""SELECT TOP 6 plate, mjd, fiberID, subClass, snMedian FROM SpecObj
          WHERE class = 'STAR' AND subClass LIKE '{pat}' AND run2d = '26' AND snMedian > 25 AND zWarning = 0 ORDER BY snMedian DESC""",
                   cache=os.path.join(CACHE, f'sdss_star6_{label}.csv'))
        bins = np.arange(380, 922, 2.0)
        for _, r in df.iterrows():   # the best spectrum that covers the whole range (some lose their red half to masking)
            plate, mjd, fib = int(r.plate), int(r.mjd), int(r.fiberID)
            url = f'https://data.sdss.org/sas/dr17/sdss/spectro/redux/26/spectra/lite/{plate:04d}/spec-{plate:04d}-{mjd}-{fib:04d}.fits'
            b = get(url, f'spec_{plate}_{mjd}_{fib}.fits')
            h = fits.open(io.BytesIO(b)); c = h[1].data
            lam = 10 ** c['loglam'] / 10.0; f = c['flux']; ok = (c['ivar'] > 0) & ((c['and_mask'] == 0) if 'and_mask' in c.names else True)
            idx = np.digitize(lam[ok], bins)
            fb = np.array([np.median(f[ok][idx == k]) if np.any(idx == k) else np.nan for k in range(1, len(bins))])
            if np.isfinite(fb).sum() >= 0.95 * len(fb): break
        fb = fb / np.nanpercentile(fb, 99)
        name = f'{label} star ({str(r.subClass).strip()}), SDSS {plate}-{mjd}-{fib}'
        out.append(pd.DataFrame({'type': label, 'name': name, 'lam': bins[:-1] + 1, 'f': fb}))
        print('spectrum', name, flush=True)
    d = pd.concat(out).dropna()
    d.to_csv(os.path.join(DATA, 'spectra.csv'), index=False, float_format='%.4g')
    return len(d)

if __name__ == '__main__':
    imgs = images()
    n_sun, n_grb, n_spec = sunspots(), grbs(), spectra()
    man_path = os.path.join(DATA, 'manifest.json'); man = json.load(open(man_path))
    man['lecture_images'] = {'title': 'Pictures in the lectures', 'rows': len(imgs), 'fetched': time.strftime('%Y-%m-%d'),
                             'source': '; '.join(i['id'] + ': ' + i['credit'] for i in imgs),
                             'license': 'ESA/Hubble, ESA/Webb and ESO images CC BY 4.0; NASA and Wikimedia Commons items public domain.'}
    man['sunspots'] = {'title': 'Sunspot number since 1700', 'rows': n_sun, 'fetched': time.strftime('%Y-%m-%d'), 'source': 'www.sidc.be/SILSO/DATA/SN_y_tot_V2.0.txt',
                       'citation': 'SILSO World Data Center, Royal Observatory of Belgium, Brussels: International Sunspot Number, version 2.0', 'license': 'CC BY-NC 4.0',
                       'columns_note': 'year (mid-year); sn = yearly mean total sunspot number'}
    man['grbs'] = {'title': 'Gamma-ray bursts seen by Fermi', 'rows': n_grb, 'fetched': time.strftime('%Y-%m-%d'), 'source': 'HEASARC table fermigbrst (Fermi GBM burst catalog)',
                   'citation': 'von Kienlin et al. 2020, ApJ 893, 46 (Fermi GBM burst catalog)', 'license': 'Public (NASA/HEASARC)', 'columns_note': 't90 = time containing 90% of the burst counts, seconds; fluence erg/cm2'}
    man['spectra'] = {'title': 'Stellar spectra across the spectral sequence', 'rows': n_spec, 'fetched': time.strftime('%Y-%m-%d'),
                      'source': 'SDSS DR17 spec-lite files (data.sdss.org), brightest-S/N star of each subclass B6, A0, F5, G2, K3, M2',
                      'cuts': 'Flux binned to 2 nm (median), divided by its 99th percentile.', 'citation': 'SDSS DR17 (Abdurro’uf et al. 2022, ApJS 259, 35)', 'license': 'SDSS data are public'}
    json.dump(man, open(man_path, 'w'), indent=1)
    json.dump(imgs, open(os.path.join(DATA, 'lecture_images.json'), 'w'), indent=1)
