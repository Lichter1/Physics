"""Lecture pictures added after the November 2026 audit (see audit/AUDIT.md, "Pictures").

ESA/Hubble, ESA/Webb and ESO images are CC BY 4.0; the Wikimedia files carry the licence given with each.
Appends to ../data/lecture_images.json (replacing entries with the same id).
Run:  python3 fetch_audit_images.py
"""
import json, os
import fetch_lecture_media as FLM

ESAH, ESAW, ESO = FLM.ESAH, FLM.ESAW, FLM.ESO
WIKI = 'https://commons.wikimedia.org/wiki/Special:FilePath/{}?width=1400'
IMAGES = [
    ('sirius_b', ESAH.format('heic0516a'), 'NASA, ESA, H. Bond (STScI) and M. Barstow (University of Leicester) (ESA/Hubble heic0516a), CC BY 4.0'),
    ('hubble_var', ESAH.format('opo1115a'), 'NASA, ESA, the Hubble Heritage Team (STScI/AURA); Hubble’s 1923 plate: Carnegie Observatories (ESA/Hubble opo1115a), CC BY 4.0'),
    ('crab', ESAH.format('heic0515a'), 'NASA, ESA, J. Hester and A. Loll (Arizona State University) (ESA/Hubble heic0515a), CC BY 4.0'),
    ('xdf', ESAH.format('heic1214a'), 'NASA, ESA, G. Illingworth, D. Magee, P. Oesch, R. Bouwens and the HUDF09 Team (ESA/Hubble heic1214a), CC BY 4.0'),
    ('m82', ESAH.format('heic0604a'), 'NASA, ESA and the Hubble Heritage Team (STScI/AURA) (ESA/Hubble heic0604a), CC BY 4.0'),
    ('h0licow', ESAH.format('heic1702a'), 'ESA/Hubble, NASA, Suyu et al. (ESA/Hubble heic1702a), CC BY 4.0'),
    ('ss_lep', ESO.format('eso1148a'), 'ESO/PIONIER/IPAG (ESO eso1148a), CC BY 4.0'),
    ('neptune_ao', ESO.format('eso1824a'), 'ESO/P. Weilbacher (AIP) (ESO eso1824a), CC BY 4.0'),
    ('wr124', ESAW.format('weic2307a'), 'ESA/Webb, NASA, CSA, STScI, Webb ERO Production Team (ESA/Webb weic2307a), CC BY 4.0'),
    ('wise_obs', 'https://commons.wikimedia.org/wiki/Special:FilePath/WiseObservatory.jpg?width=1280', 'Etan J. Tal (Wikimedia Commons, File:WiseObservatory.jpg), CC BY 3.0'),
    ('opacity', WIKI.format('Atmospheric_electromagnetic_opacity.svg'), 'NASA (Wikimedia Commons, File:Atmospheric electromagnetic opacity.svg), public domain'),
]

if __name__ == '__main__':
    FLM.IMAGES = IMAGES
    new = FLM.images()
    path = os.path.join(FLM.DATA, 'lecture_images.json')
    old = [x for x in json.load(open(path)) if x['id'] not in {n['id'] for n in new}]
    json.dump(old + new, open(path, 'w'), indent=1)
    print(len(new), 'of', len(IMAGES), 'pictures')
