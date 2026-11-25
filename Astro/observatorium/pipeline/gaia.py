"""Tiny client for the ESA Gaia archive TAP service (no astroquery needed)."""
import io, sys, time, urllib.parse, urllib.request
import pandas as pd

TAP = 'https://gea.esac.esa.int/tap-server/tap'

def query(adql, cache=None, timeout=600):
    """Run an ADQL query (async job), return a DataFrame. Cached to CSV if cache is a path."""
    import os
    if cache and os.path.exists(cache):
        return pd.read_csv(cache)
    data = urllib.parse.urlencode({'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'csv',
                                   'PHASE': 'RUN', 'QUERY': adql}).encode()
    req = urllib.request.Request(TAP + '/async', data=data)
    opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())
    with opener.open(req, timeout=60) as r:
        job = r.geturl()
    t0 = time.time()
    while True:
        with urllib.request.urlopen(job + '/phase', timeout=60) as r:
            phase = r.read().decode().strip()
        if phase == 'COMPLETED':
            break
        if phase in ('ERROR', 'ABORTED'):
            with urllib.request.urlopen(job + '/error', timeout=60) as r:
                raise RuntimeError(r.read().decode()[:2000])
        if time.time() - t0 > timeout:
            raise TimeoutError(job)
        time.sleep(3)
    with urllib.request.urlopen(job + '/results/result', timeout=300) as r:
        raw = r.read()
    df = pd.read_csv(io.BytesIO(raw))
    if cache:
        df.to_csv(cache, index=False)
    return df

if __name__ == '__main__':
    print(query(sys.argv[1]).head())
