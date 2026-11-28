"""Data for the missions added after the November 2026 audit (see audit/CHECKLIST.md).
  rv51peg.csv      Lick radial velocities of 51 Pegasi (Fischer et al. 2014, VizieR J/ApJS/210/5)
  s2.csv           S2's positions and velocities around Sgr A* (Gillessen et al. 2017, VizieR J/ApJ/837/30)
  ellipticals.csv  SDSS DR18 Galaxy Zoo ellipticals with velocity dispersions (SkyServer SQL)
  ogle_events.csv  OGLE-IV EWS 2019 events with fitted time scales (from the cached EWS page, cache/ews2019.html)
Run: python3 fetch_audit_data.py  (fetch_orbits.py makes orbits.csv)"""
import json, os, re, urllib.parse, urllib.request
from datetime import date
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, '..', 'data'); CACHE = os.path.join(HERE, 'cache', 'audit'); os.makedirs(CACHE, exist_ok=True)
VIZ = 'https://vizier.cfa.harvard.edu/viz-bin/asu-tsv'
def get(url, name):
    p = os.path.join(CACHE, name)
    if not os.path.exists(p): open(p, 'wb').write(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'observatorium-teaching-tool'}), timeout=600).read())
    return open(p, encoding='utf-8', errors='ignore').read()
def tsv(text):
    rows = [l.rstrip('\n').split('\t') for l in text.split('\n') if l and not l.startswith('#')]
    return [x.strip() for x in rows[0]], rows[3:]

def rv51peg():
    h, d = tsv(get(VIZ + '?-source=J/ApJS/210/5/table2&-out.max=9999&-out.all&Name=217014', 'lick_51peg.tsv'))
    out = ['t,rv,erv'] + ['%.5f,%.2f,%.2f' % (float(r[h.index('JD')]), float(r[h.index('RVel')]), float(r[h.index('e_RVel')])) for r in d]
    open(os.path.join(DATA, 'rv51peg.csv'), 'w').write('\n'.join(out) + '\n'); return len(out) - 1

def s2():
    h, d = tsv(get(VIZ + '?-source=J/ApJ/837/30/table5&-out.max=9999&-out.all', 'gc_orbits.tsv'))
    out = ['year,x,y,ex,ey,rv,erv,inst']
    for r in d:
        g = lambda k: r[h.index(k)].strip()
        if g('oRA-S2') or g('RV-S2'): out.append(','.join([g('Date'), g('oRA-S2'), g('oDE-S2'), g('e_oRA-S2'), g('e_oDE-S2'), g('RV-S2'), g('e_RV-S2'), g('Inst')]))
    open(os.path.join(DATA, 's2.csv'), 'w').write('\n'.join(out) + '\n'); return len(out) - 1

FJQ = """select top 8000 s.ra, s.dec, s.z, s.veldisp, s.veldispErr, p.petroMag_r, p.extinction_r, p.petroR50_r, z.p_el_debiased
from SpecObj s join PhotoObj p on s.bestObjID = p.objID join zooSpec z on z.specobjid = s.specObjID
where s.class = 'GALAXY' and s.zWarning = 0 and s.z between 0.02 and 0.08 and s.veldisp between 60 and 420 and s.veldispErr between 0 and 15
and p.petroMag_r < 17.7 and z.p_el_debiased > 0.8"""
def ellipticals():
    p = os.path.join(CACHE, 'fj.json')
    if not os.path.exists(p):
        u = 'https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch?' + urllib.parse.urlencode({'cmd': FJQ, 'format': 'json'})
        json.dump(json.load(urllib.request.urlopen(u, timeout=600))[0]['Rows'], open(p, 'w'))
    fj = json.load(open(p))
    open(os.path.join(DATA, 'ellipticals.csv'), 'w').write('ra,dec,z,sigma,esigma,r,r50\n' + ''.join('%.5f,%.5f,%.5f,%.1f,%.1f,%.3f,%.2f\n' % (q['ra'], q['dec'], q['z'], q['veldisp'], q['veldispErr'], q['petroMag_r'] - q['extinction_r'], q['petroR50_r']) for q in fj))
    return len(fj)

def ogle_events():
    s = open(os.path.join(HERE, 'cache', 'ews2019.html'), errors='ignore').read()
    rows = re.findall(r'<A HREF="blg-\d+\.html" TARGET="event">(2019-BLG-\d+)</A>\s*<TD>[^<]*<TD ALIGN="RIGHT">[^<]*<TD>([^<]*)<TD>([^<]*)<TD>[^<]*<TD>[^<]*<TD>([^<]*)<TD>([^<]*)<TD>([^<]*)', s)
    def hms(x): h, m, sx = map(float, x.strip().split(':')); return 15 * (h + m / 60 + sx / 3600)
    def dms(x): x = x.strip(); sg = -1 if x.startswith('-') else 1; d, m, sx = map(float, x.lstrip('+-').split(':')); return sg * (d + m / 60 + sx / 3600)
    out = ['event,ra,dec,tE,u0,Amax']
    for ev, ra, dec, t, u, A in rows:
        try: t, u, A = float(t), float(u), float(A)
        except ValueError: continue
        if 0.3 < t < 600 and u > 0: out.append('%s,%.5f,%.5f,%.3f,%.4f,%.3f' % (ev, hms(ra), dms(dec), t, u, A))
    open(os.path.join(DATA, 'ogle_events.csv'), 'w').write('\n'.join(out) + '\n'); return len(out) - 1

if __name__ == '__main__':
    n = {'rv51peg': rv51peg(), 's2': s2(), 'ellipticals': ellipticals(), 'ogle_events': ogle_events(), 'orbits': sum(1 for _ in open(os.path.join(DATA, 'orbits.csv'))) - 1}
    mp = os.path.join(DATA, 'manifest.json'); m = json.load(open(mp)); T = date.today().isoformat()
    def put(k, title, source, cuts, citation, license): m[k] = {'title': title, 'rows': n[k], 'fetched': T, 'source': source, 'cuts': cuts, 'citation': citation, 'license': license}
    put('rv51peg', 'Radial velocities of 51 Pegasi (Lick)', 'VizieR J/ApJS/210/5, table2, Name = 217014', 'All Lick velocities of HD 217014; JD − 2400000', 'Fischer et al. 2014, ApJS 210, 5', 'VizieR / CDS, free with citation')
    put('s2', 'The star S2 around Sgr A*', 'VizieR J/ApJ/837/30, table5 (S2 columns)', 'Epochs with a position or a velocity of S2', 'Gillessen et al. 2017, ApJ 837, 30', 'VizieR / CDS, free with citation')
    put('ellipticals', 'SDSS ellipticals with velocity dispersions', 'SkyServer DR18: SpecObj + PhotoObj + zooSpec', '0.02 < z < 0.08, Galaxy Zoo p_el (debiased) > 0.8, 60 < σ < 420 km/s with error < 15, r < 17.7 (extinction corrected); first 8,000', 'Abdurro\'uf et al. 2022 (SDSS); Lintott et al. 2011 (Galaxy Zoo)', 'SDSS, public with acknowledgement')
    put('ogle_events', 'OGLE-IV EWS microlensing events, 2019', 'ogle.astrouw.edu.pl/ogle4/ews/2019/ews.html', 'Events with 0.3 < t_E < 600 d and u0 > 0 from OGLE\'s single-lens fits', 'Udalski et al. 2015, Acta Astron. 65, 1 (OGLE-IV); OGLE EWS', 'OGLE: ask before use in publications')
    put('orbits', 'Full orbits of bright asteroids and TNOs', 'ssd-api.jpl.nasa.gov/sbdb_query.api', 'Numbered objects with H < 13, a < 120 AU, bound; mean anomaly moved to JD 2461041.5 (2026 Jan 1)', 'JPL Solar System Dynamics, Small-Body Database', 'Public (NASA/JPL)')
    json.dump(m, open(mp, 'w'), indent=1, ensure_ascii=False); print(n)
