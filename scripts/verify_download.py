#!/usr/bin/env python3
# Verify a local HF snapshot against the HF API (sizes + LFS sha256), parallel.
import hashlib, json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
repo, root = sys.argv[1], sys.argv[2]
url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
items = json.load(urllib.request.urlopen(url, timeout=60))
files = [i for i in items if i["type"] == "file"]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while (b := f.read(64 << 20)):
            h.update(b)
    return h.hexdigest()
def check(i):
    p = os.path.join(root, i["path"])
    if not os.path.exists(p):
        return (i["path"], "MISSING")
    sz = os.path.getsize(p)
    if sz != i["size"]:
        return (i["path"], f"SIZE {sz} != {i['size']}")
    lfs = i.get("lfs")
    if lfs:
        d = sha(p)
        if d != lfs["oid"]:
            return (i["path"], f"SHA {d} != {lfs['oid']}")
        return (i["path"], "OK sha")
    return (i["path"], "OK size")
with ThreadPoolExecutor(12) as ex:
    res = list(ex.map(check, files))
bad = [r for r in res if not r[1].startswith("OK")]
for r in res: print(*r)
tot = sum(i["size"] for i in files)
print(f"FILES {len(files)} TOTAL_BYTES {tot} BAD {len(bad)}")
for b in bad: print("BAD", *b)
