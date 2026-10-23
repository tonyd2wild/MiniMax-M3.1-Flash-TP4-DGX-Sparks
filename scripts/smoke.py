#!/usr/bin/env python3
"""smoke.py BASE MODEL [IMAGE] [CORPUS_TXT]: chat, tool call, image input and a ~32K-token prompt against /v1/chat/completions.
Prints one PASS/FAIL line per check and writes smoke.json next to this script's cwd."""
import base64, json, sys, time, urllib.request

BASE, MODEL = sys.argv[1].rstrip("/"), sys.argv[2]
IMG = sys.argv[3] if len(sys.argv) > 3 else None
CORPUS = sys.argv[4] if len(sys.argv) > 4 else None
out = {}


def chat(messages, **kw):
    body = {"model": MODEL, "messages": messages, "max_tokens": kw.pop("max_tokens", 1024), "temperature": 1.0, "top_p": 0.95}
    body.update(kw)
    t = time.time()
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    return r, time.time() - t


def check(name, fn):
    try:
        ok, info = fn()
    except Exception as e:
        ok, info = False, repr(e)[:300]
    out[name] = {"ok": ok, "info": info}
    print(("PASS " if ok else "FAIL ") + name + ": " + str(info)[:400], flush=True)


def t_chat():
    r, dt = chat([{"role": "user", "content": "What is 17 x 19? Answer with the number only."}], max_tokens=2048)
    m = r["choices"][0]["message"]
    c = (m.get("content") or "")
    return "323" in c, {"content": c[:100], "reasoning_chars": len(m.get("reasoning_content") or m.get("reasoning") or ""),
                        "usage": r.get("usage"), "s": round(dt, 2)}


def t_tool():
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                             "required": ["city"]}}}]
    msgs = [{"role": "user", "content": "What's the weather in Nashville right now? Use the tool."}]
    r, dt = chat(msgs, tools=tools, tool_choice="auto", max_tokens=2048)
    m = r["choices"][0]["message"]
    tc = m.get("tool_calls") or []
    if not tc:
        return False, {"content": (m.get("content") or "")[:200]}
    args = json.loads(tc[0]["function"]["arguments"])
    msgs += [m, {"role": "tool", "tool_call_id": tc[0]["id"], "content": json.dumps({"temp_f": 71, "sky": "clear"})}]
    r2, dt2 = chat(msgs, tools=tools, max_tokens=2048)
    final = r2["choices"][0]["message"].get("content") or ""
    return tc[0]["function"]["name"] == "get_weather" and "nashville" in args.get("city", "").lower() and "71" in final, \
        {"call": tc[0]["function"], "final": final[:160], "s": round(dt + dt2, 2)}


def t_image():
    b64 = base64.b64encode(open(IMG, "rb").read()).decode()
    r, dt = chat([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
        {"type": "text", "text": "What object is in this photo? One short sentence."}]}], max_tokens=2048)
    c = (r["choices"][0]["message"].get("content") or "")
    return any(w in c.lower() for w in ("shoe", "sneaker", "jordan", "footwear")), {"content": c[:200], "usage": r.get("usage"), "s": round(dt, 2)}


def t_32k():
    text = open(CORPUS, encoding="utf-8", errors="ignore").read()[: int(32000 * 3.3)]
    needle = "The secret launch code for the Tech2Wild vault is MANGO-4471."
    mid = len(text) // 2
    doc = text[:mid] + "\n# " + needle + "\n" + text[mid:]
    r, dt = chat([{"role": "user", "content": doc + "\n\nWhat is the secret launch code for the Tech2Wild vault? Reply with the code only."}], max_tokens=2048)
    c = r["choices"][0]["message"].get("content") or ""
    return "MANGO-4471" in c, {"content": c[:80], "prompt_tokens": r["usage"]["prompt_tokens"], "s": round(dt, 2)}


check("chat", t_chat)
check("tool_call_roundtrip", t_tool)
if IMG:
    check("image_input", t_image)
if CORPUS:
    check("prompt_32k_needle", t_32k)
json.dump(out, open("smoke.json", "w"), indent=1)
