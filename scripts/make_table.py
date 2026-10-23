#!/usr/bin/env python3
"""make_table.py RESULTS.json: markdown table from a speed_bench.py run (one row per prompt size and concurrency)."""
import json, sys

r = json.load(open(sys.argv[1]))
f = lambda v, n=1: "-" if v is None else f"{v:,.{n}f}"
print("| prompt | users | prompt tokens (median) | TTFT p50 (s) | prefill tok/s | decode tok/s per user | decode tok/s, all users | total tok/s | DSpark acceptance | tokens per step |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for c in r["cells"]:
    if not c.get("n_ok"):
        print(f"| {c['size']} | {c['conc']} | failed: {c.get('errors')} |||||||")
        continue
    d = c["dspark"]
    print(f"| {c['size']} | {c['conc']} | {f(c['prompt_tokens_med'],0)} | {f(c['ttft_p50'],2)} | {f(c['prefill_tps_agg'],0)} | "
          f"{f(c['decode_tps_per_stream_p50'])} | {f(c['decode_tps_agg'])} | {f(c['total_tps'],0)} | "
          f"{f(d['acceptance_rate'],2)} | {f(d['mean_accepted_len'],2)} |"
          + ("" if c["n_err"] == 0 else f" ({c['n_err']} failed)"))
