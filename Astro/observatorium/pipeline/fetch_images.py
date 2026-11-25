"""Real images of the sky for the Observatorium's photo layer.

- The whole sky: ESO's Milky Way panorama (S. Brunier), an equirectangular map in Galactic coordinates.
- Close-ups: Digitized Sky Survey 2 plates (red and blue) from NASA SkyView, stretched and combined into colour,
  or DESI Legacy Imaging Surveys colour cutouts for galaxies.
Writes ../data/img/*.jpg and ../data/images.json. Run:  python3 fetch_images.py
"""
import io, json, os, sys, time, urllib.parse, urllib.request
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache', 'img')
OUT = os.path.join(HERE, '..', 'data', 'img')
os.makedirs(CACHE, exist_ok=True); os.makedirs(OUT, exist_ok=True)

DSS_CREDIT = 'Digitized Sky Survey 2 (STScI/NASA; POSS-II Caltech/Palomar, UK Schmidt AAO), via NASA SkyView'
LS_CREDIT = 'DESI Legacy Imaging Surveys DR9 (legacysurvey.org)'
SDSS_CREDIT = 'Sloan Digital Sky Survey (SDSS DR18 image cutout service)'
# id, label, ra, dec, field of view [deg], source
TARGETS = [
    ('pleiades', 'The Pleiades (M45)', 56.75, 24.12, 2.6, 'dss'),
    ('praesepe', 'Praesepe (M44)', 130.10, 19.67, 2.0, 'dss'),
    ('m67', 'M67', 132.85, 11.81, 0.7, 'dss'),
    ('ngc6397', 'NGC 6397', 265.175, -53.674, 0.6, 'dss'),
    ('61cyg', '61 Cygni', 316.725, 38.749, 0.4, 'dss'),
    ('orion_neb', 'The Orion Nebula (M42)', 83.82, -5.39, 1.2, 'dss'),
    ('betelgeuse', 'Betelgeuse', 88.793, 7.407, 0.6, 'dss'),
    ('rigel', 'Rigel', 78.634, -8.202, 0.6, 'dss'),
    ('binary', 'Gaia DR3 3684484600212307712', 197.1246, -2.6789, 0.3, 'dss'),
    ('smc', 'The Small Magellanic Cloud', 13.19, -72.83, 4.5, 'dss'),
    ('crab', 'The Crab Nebula (M1)', 83.633, 22.0145, 0.25, 'dss'),
    ('hatp7', 'HAT-P-7', 292.2474, 47.9695, 0.3, 'dss'),
    ('baade', 'Toward the Galactic bulge: OGLE-2019-BLG-0148', 269.8946, -29.0638, 0.4, 'dss'),
    ('47tuc', '47 Tucanae', 6.024, -72.081, 0.7, 'dss'),
    ('m13', 'M13', 250.423, 36.461, 0.4, 'dss'),
    ('m31', 'The Andromeda galaxy (M31)', 10.685, 41.269, 3.2, 'dss'),
    ('ngc3198', 'NGC 3198', 154.979, 45.549, 0.25, 'sdss'),
    ('coma', 'The Coma cluster', 194.953, 27.981, 1.6, 'ls'),
    ('m87', 'M87, a giant elliptical', 187.706, 12.391, 0.2, 'ls'),
    ('m101', 'M101, a spiral', 210.802, 54.349, 0.4, 'sdss'),
    ('m51', 'M51, the Whirlpool', 202.470, 47.195, 0.2, 'sdss'),
]
PIX = 1600


def get(url, path, tries=3):
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return open(path, 'rb').read()
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=600) as r:
                b = r.read()
            open(path, 'wb').write(b)
            return b
        except Exception as e:
            print('  retry', k, e, flush=True); time.sleep(5)
    raise RuntimeError(url)


def stretch(a):
    """A log-scaled plate (SkyView 'Log') -> background black, faint detail kept, bright cores not blown out."""
    a = a.astype(float)
    bg, hi = np.percentile(a, 35), np.percentile(a, 99.95)
    x = np.clip((a - bg) / max(1, hi - bg), 0, 1)
    return x ** 1.35


def dss(t):
    key, label, ra, dec, fov, _ = t
    bands = {}
    for band, alt in (('DSS2 Red', 'DSS1 Red'), ('DSS2 Blue', 'DSS1 Blue')):
        for survey in (band, alt):   # some fields have no DSS2 plate in one colour; the first-epoch survey fills in
            q = urllib.parse.urlencode({'Survey': survey, 'Position': f'{ra},{dec}', 'Size': fov, 'Pixels': PIX, 'Return': 'JPEG', 'Scaling': 'Log'})
            name = f'{key}_{band[-3:].strip()}.jpg' if survey == band else f'{key}_{survey.replace(" ", "")}.jpg'
            raw = np.array(Image.open(io.BytesIO(get('https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl?' + q, os.path.join(CACHE, name)))).convert('L'))
            if raw.std() > 2:
                break
        bands[band] = stretch(raw) if raw.std() > 2 else None
    R, B = bands['DSS2 Red'], bands['DSS2 Blue']
    if B is None:      # no blue plate anywhere: show the red plate in neutral grey rather than false orange
        B = R
    rgb = np.dstack([R, 0.5 * (R + B), B])
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)), DSS_CREDIT


def ls(t):
    key, label, ra, dec, fov, _ = t
    size = PIX
    q = urllib.parse.urlencode({'ra': ra, 'dec': dec, 'layer': 'ls-dr9', 'pixscale': round(fov * 3600 / size, 4), 'size': size})
    b = get('https://www.legacysurvey.org/viewer/jpeg-cutout?' + q, os.path.join(CACHE, f'{key}_ls.jpg'))
    return Image.open(io.BytesIO(b)).convert('RGB'), LS_CREDIT


def sdss(t):
    key, label, ra, dec, fov, _ = t
    q = urllib.parse.urlencode({'ra': ra, 'dec': dec, 'scale': round(fov * 3600 / PIX, 4), 'width': PIX, 'height': PIX})
    b = get('https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg?' + q, os.path.join(CACHE, f'{key}_sdss.jpg'))
    return Image.open(io.BytesIO(b)).convert('RGB'), SDSS_CREDIT


def main():
    meta = {'allsky': None, 'cutouts': []}
    # the whole sky
    src = os.path.join(CACHE, 'eso0932a_large.jpg')
    get('https://cdn.eso.org/images/large/eso0932a.jpg', src)
    im = Image.open(src).convert('RGB').resize((4096, 2048), Image.LANCZOS)
    im.save(os.path.join(OUT, 'allsky.jpg'), quality=86, optimize=True, progressive=True)
    meta['allsky'] = {'file': 'img/allsky.jpg', 'w': 4096, 'h': 2048, 'frame': 'gal', 'proj': 'car', 'lon_center': 0, 'lon_left': True,
                      'credit': 'ESO/S. Brunier, the Milky Way panorama (eso0932a), CC BY 4.0', 'license': 'CC BY 4.0'}
    print('allsky', os.path.getsize(os.path.join(OUT, 'allsky.jpg')) // 1000, 'kB', flush=True)
    only = set(sys.argv[1:])
    for t in TARGETS:
        if only and t[0] not in only:
            continue
        try:
            img, credit = {'dss': dss, 'ls': ls, 'sdss': sdss}[t[5]](t)
        except Exception as e:
            print('FAILED', t[0], e, flush=True); continue
        f = f'img/{t[0]}.jpg'
        img.save(os.path.join(OUT, '..', f), quality=84, optimize=True, progressive=True)
        meta['cutouts'].append({'id': t[0], 'label': t[1], 'ra': t[2], 'dec': t[3], 'fov': t[4], 'w': img.width, 'h': img.height,
                                'file': f, 'proj': 'tan', 'credit': credit})
        print(t[0], img.size, os.path.getsize(os.path.join(OUT, '..', f)) // 1000, 'kB', flush=True)
    json.dump(meta, open(os.path.join(OUT, '..', 'images.json'), 'w'), indent=1)
    # record the photographs in the data manifest shown in the page's Data tab
    mpath = os.path.join(OUT, '..', 'manifest.json')
    man = json.load(open(mpath))
    srcs = sorted({c['credit'] for c in meta['cutouts']})
    man['photographs'] = {'title': 'Photographs of the sky (photo layer)', 'rows': len(meta['cutouts']) + 1, 'fetched': time.strftime('%Y-%m-%d'),
                          'source': 'All-sky: ' + meta['allsky']['credit'] + '. Close-ups (' + ', '.join(c['label'] for c in meta['cutouts']) + '): ' + '; '.join(srcs) + '.',
                          'cuts': 'DSS2 red and blue plates stretched and combined into colour (green = their mean); first-epoch DSS plates fill fields with no DSS2 plate in one colour. Each close-up is a gnomonic projection, north up, east left.',
                          'license': 'ESO panorama CC BY 4.0. DSS: free for research and education with acknowledgement. SDSS and Legacy Surveys images: public, with acknowledgement.',
                          'citation': 'Brunier (ESO) 2009, eso0932a; Lasker et al. 1990 / Reid et al. 1991 (DSS); SDSS DR18; Dey et al. 2019 (Legacy Surveys)'}
    json.dump(man, open(mpath, 'w'), indent=1)


if __name__ == '__main__':
    main()
