"""Minimal SDSS SkyServer SQL client (DR18)."""
import io, os, urllib.parse, urllib.request
import pandas as pd
URL = 'https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch'
def query(sql, cache=None):
    if cache and os.path.exists(cache):
        return pd.read_csv(cache)
    u = URL + '?' + urllib.parse.urlencode({'cmd': ' '.join(sql.split()), 'format': 'csv'})
    with urllib.request.urlopen(u, timeout=600) as r:
        raw = r.read().decode()
    lines = raw.splitlines()
    if lines and lines[0].startswith('#Table'):
        lines = lines[1:]
    df = pd.read_csv(io.StringIO('\n'.join(lines)))
    if cache:
        df.to_csv(cache, index=False)
    return df
