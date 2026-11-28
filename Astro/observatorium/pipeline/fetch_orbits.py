"""Full orbits of the bright numbered asteroids and trans-Neptunian objects, for the top-down solar-system view.
Mean anomalies are moved to one date (2026 Jan 1.0 TT, JD 2461041.5). Source: JPL SBDB query API (public).
Run: python3 fetch_orbits.py  ->  ../data/orbits.csv"""
import json, os, urllib.parse, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, '..', 'data')
JD0 = 2461041.5
def query(cdata, limit):
    q = {'fields': 'class,a,e,i,om,w,ma,epoch', 'sb-kind': 'a', 'sb-ns': 'n', 'sb-cdata': json.dumps(cdata), 'limit': str(limit)}
    u = 'https://ssd-api.jpl.nasa.gov/sbdb_query.api?' + urllib.parse.urlencode(q)
    return json.load(urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'observatorium-teaching-tool'}), timeout=600))['data']
rows = query({'AND': ['H|LT|13.0']}, 30000)
seen = len(rows)
out = ['class,a,e,i,om,w,M']
for c, a, e, i, om, w, ma, ep in rows:
    try: a, e, i, om, w, ma, ep = map(float, (a, e, i, om, w, ma, ep))
    except (TypeError, ValueError): continue
    if not (0 < a < 120 and e < 1): continue
    n = 0.9856076686 / a ** 1.5
    M = (ma + n * (JD0 - ep)) % 360
    out.append('%s,%.4f,%.4f,%.2f,%.2f,%.2f,%.2f' % (c, a, e, i, om, w, M))
open(os.path.join(DATA, 'orbits.csv'), 'w').write('\n'.join(out) + '\n')
print(seen, 'fetched', len(out) - 1, 'kept')
