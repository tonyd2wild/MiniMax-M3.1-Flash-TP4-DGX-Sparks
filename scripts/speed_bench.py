#!/usr/bin/env python3
"""speed_bench.py: C1..Cn concurrency sweep over several prompt sizes against an OpenAI-compatible
/v1/chat/completions endpoint (stdlib only). Per request: TTFT, prefill tok/s (prompt/TTFT), decode tok/s
(per stream, after the first token), plus aggregate throughput per cell and DSpark acceptance from /metrics.

usage: speed_bench.py --base http://127.0.0.1:8003 --model minimax-m3.1-flash --corpus DIR --out results.json
         [--sizes short,8k,32k,100k] [--conc 1,2,3,4,5,6] [--max-tokens 256]
Every request gets a unique nonce at the very start of the prompt, so the prefix cache never helps.
"""
import argparse, http.client, json, os, random, re, statistics, sys, threading, time, urllib.parse

SIZES = {"short": 0, "8k": 8000, "32k": 32000, "100k": 100000}
CHARS_PER_TOKEN = 4.1   # measured on this corpus (vLLM python source) with the MiniMax tokenizer; usage.prompt_tokens is what gets reported


def load_corpus(d):
    txt = []
    for root, _, files in sorted(os.walk(d)):
        for f in sorted(files):
            if f.endswith(".py"):
                try:
                    txt.append(open(os.path.join(root, f), encoding="utf-8", errors="ignore").read())
                except OSError:
                    pass
    return "\n\n".join(txt)


def make_prompt(corpus, size, nonce):
    if size == "short":
        return (f"[{nonce}] Write a detailed, well structured explanation of how a mixture-of-experts "
                "transformer routes tokens to experts, why it saves compute, and what can go wrong. Use plain prose.")
    n = int(SIZES[size] * CHARS_PER_TOKEN)
    off = random.randrange(0, max(1, len(corpus) - n))
    return (f"[{nonce}] Here is a large chunk of source code:\n\n{corpus[off:off + n]}\n\n"
            "Explain in detail what this code does, module by module, in plain prose.")


def metrics(base):
    u = urllib.parse.urlparse(base)
    try:
        c = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
        c.request("GET", "/metrics")
        body = c.getresponse().read().decode()
    except Exception:
        return {}
    out = {}
    for line in body.splitlines():
        m = re.match(r"^(vllm:spec_decode_num_(?:accepted_tokens|draft_tokens|drafts)_total)(?:\{[^}]*\})?\s+([0-9.eE+]+)$", line)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def one(base, model, prompt, max_tokens, res, extra):
    u = urllib.parse.urlparse(base)
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 1.0, "top_p": 0.95, "stream": True, "stream_options": {"include_usage": True}}
    body.update(extra)
    r = {"ok": False}
    t0 = time.time()
    try:
        c = http.client.HTTPConnection(u.hostname, u.port, timeout=3600)
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        resp = c.getresponse()
        if resp.status != 200:
            r["error"] = f"HTTP {resp.status} {resp.read()[:300]!r}"
            res.append(r); return
        first = None; last = None; usage = None; nchunks = 0
        buf = b""
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    now = time.time()
                    if first is None:
                        first = now
                    last = now; nchunks += 1
        t1 = time.time()
        if first is None or not usage:
            r["error"] = "no tokens/usage"; res.append(r); return
        ct, pt = usage.get("completion_tokens", 0), usage.get("prompt_tokens", 0)
        ttft = first - t0
        r.update(ok=True, prompt_tokens=pt, completion_tokens=ct, ttft=ttft, total=t1 - t0, t_start=t0, t_first=first, t_end=t1,
                 prefill_tps=pt / ttft if ttft > 0 else None,
                 decode_tps=(ct - 1) / (last - first) if last and last > first and ct > 1 else None)
    except Exception as e:
        r["error"] = repr(e)
    res.append(r)


def cell(a, corpus, size, conc, extra):
    res = []
    prompts = [make_prompt(corpus, size, f"{random.getrandbits(64):016x}") for _ in range(conc)]
    m0 = metrics(a.base)
    ths = [threading.Thread(target=one, args=(a.base, a.model, p, a.max_tokens, res, extra)) for p in prompts]
    t0 = time.time()
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.time() - t0
    m1 = metrics(a.base)
    ok = [r for r in res if r.get("ok")]
    acc = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
    out = {"size": size, "conc": conc, "n_ok": len(ok), "n_err": len(res) - len(ok), "wall_s": wall,
           "errors": [r.get("error") for r in res if not r.get("ok")][:3], "requests": res}
    if ok:
        med = lambda xs: statistics.median([x for x in xs if x is not None]) if any(x is not None for x in xs) else None
        out.update(
            prompt_tokens_med=med([r["prompt_tokens"] for r in ok]),
            ttft_p50=med([r["ttft"] for r in ok]),
            ttft_max=max(r["ttft"] for r in ok),
            prefill_tps_c1=ok[0]["prefill_tps"] if conc == 1 else None,
            prefill_tps_agg=sum(r["prompt_tokens"] for r in ok) / max(r["t_first"] - t0 for r in ok),
            decode_tps_per_stream_p50=med([r["decode_tps"] for r in ok]),
            decode_tps_agg=sum(r["completion_tokens"] for r in ok) /
                max(1e-9, max(r["t_end"] for r in ok) - min(r["t_first"] for r in ok)),
            total_tps=sum(r["prompt_tokens"] + r["completion_tokens"] for r in ok) / wall,
            gen_tps_wall=sum(r["completion_tokens"] for r in ok) / wall,
        )
    d, ac = acc.get("vllm:spec_decode_num_draft_tokens_total", 0), acc.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    nd = acc.get("vllm:spec_decode_num_drafts_total", 0)
    out["dspark"] = {"draft_tokens": d, "accepted": ac, "drafts": nd,
                     "acceptance_rate": ac / d if d else None, "mean_accepted_len": 1 + ac / nd if nd else None}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True); p.add_argument("--model", required=True)
    p.add_argument("--corpus", required=True); p.add_argument("--out", required=True)
    p.add_argument("--sizes", default="short,8k,32k,100k"); p.add_argument("--conc", default="1,2,3,4,5,6")
    p.add_argument("--max-tokens", type=int, default=256); p.add_argument("--extra", default="{}")
    p.add_argument("--label", default="")
    a = p.parse_args()
    corpus = load_corpus(a.corpus)
    extra = json.loads(a.extra)
    allres = {"label": a.label, "base": a.base, "model": a.model, "max_tokens": a.max_tokens, "extra": extra,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cells": []}
    one(a.base, a.model, "Say hi.", 8, [], extra)   # warm-up
    for size in a.sizes.split(","):
        for conc in [int(c) for c in a.conc.split(",")]:
            c = cell(a, corpus, size, conc, extra)
            allres["cells"].append(c)
            json.dump(allres, open(a.out, "w"), indent=1)
            f = lambda v, n=1: "-" if v is None else f"{v:.{n}f}"
            print(f"{size:>5} C{conc}: ok {c['n_ok']}/{conc} prompt {f(c.get('prompt_tokens_med'),0)} "
                  f"TTFT p50 {f(c.get('ttft_p50'),2)}s prefill agg {f(c.get('prefill_tps_agg'),0)} tok/s "
                  f"decode/stream {f(c.get('decode_tps_per_stream_p50'))} agg decode {f(c.get('decode_tps_agg'))} "
                  f"total {f(c.get('total_tps'),0)} tok/s accept {f(c['dspark']['acceptance_rate'],3)} "
                  f"len {f(c['dspark']['mean_accepted_len'],2)} {c['errors'] if c['n_err'] else ''}", flush=True)
    allres["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(allres, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
