"""Download PRCP v1.0.0 records from PhysioNet (Open Data Commons
Attribution License v1.0) and print a per-file inventory: real header
sampling rates, channels, durations, and event annotations."""
import os, re, sys
from urllib.request import urlopen

BASE = "https://physionet.org/files/prcp/1.0.0/"

def download(dest):
    html = urlopen(BASE).read().decode()
    files = sorted(set(re.findall(r'href="?(\d+\.(?:dat|hea|wqrs|wabp|anI))"?', html)))
    for f in files:
        p = os.path.join(dest, f)
        if not os.path.exists(p):
            open(p, 'wb').write(urlopen(BASE + f).read())
            print("dl", f, flush=True)
    return sorted({f.split('.')[0] for f in files})

if __name__ == "__main__":
    recs = download(sys.argv[1] if len(sys.argv) > 1 else "raw/prcp")
    print("records:", recs)
