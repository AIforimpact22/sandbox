from __future__ import annotations
import os, re, json, hashlib, math, random
from typing import Any, Dict, List
from flask import Flask, request, jsonify, send_from_directory, make_response
from openai import OpenAI

# =========================
# Config
# =========================
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")   # set in env
PORT = int(os.getenv("PORT", "5000"))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
app = Flask(__name__, static_folder="static", template_folder="static")

# =========================
# Strict JSON-UI instructions
# =========================
SCHEMA_GUIDE = """
You are a UI generator. Return ONLY one JSON object (no backticks, no prose).

Schema:
{
  "page": {"title": string, "layout": "wide" | "centered"},
  "blocks": [
    { "type": "heading", "level": 1|2|3, "text": string },
    { "type": "text", "text": string },
    { "type": "metric", "label": string, "value": string, "delta": string? },
    { "type": "card", "title": string?, "body": string?, "children": [<blocks>]? },
    { "type": "table", "columns": [string], "rows": [[string|number|boolean,...], ...] },
    { "type": "chart", "kind": "line"|"bar", "data": {"x":[number|string], "y":[number]} },
    { "type": "input", "inputType": "text"|"number"|"password", "label": string, "id": string, "placeholder": string?, "value": string|number? },
    { "type": "select", "label": string, "id": string, "options": [{"label": string, "value": string}], "value": string? },
    { "type": "button", "text": string, "id": string },
    { "type": "columns", "ratio": [number, ...], "columns": [ [<blocks>], ... ] }
  ]
}

Rules:
- Valid JSON only (no comments).
- Use only the types listed above; no HTML.
- Avoid random demo numbers; keep structure compact. The server will provide numbers.
- IDs must be unique and URL-safe.
- If the user mentions employee check-in/out, include 2–4 KPIs, a by-hour chart, and a small table.
"""

# =========================
# Helpers
# =========================
def normalize_prompt(p: str) -> str:
    return re.sub(r"\s+", " ", (p or "").strip().lower())

def prompt_seed(p: str) -> int:
    h = hashlib.sha256(normalize_prompt(p).encode("utf-8")).hexdigest()
    return int(h[:16], 16)

def _extract_json(text: str) -> Dict[str, Any] | None:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except Exception:
                    return None
    return None

def _fallback_schema(prompt: str) -> Dict[str, Any]:
    return {
        "page": {"title": (prompt[:60] or "Your UI"), "layout": "wide"},
        "blocks": [
            {"type":"heading","level":1,"text":"Your UI"},
            {"type":"text","text":f"Prompt: {prompt or '(empty)'}"},
            {"type":"card","title":"Tip","body":"Try: 3 KPIs, a by-hour chart, and a 5-row table."}
        ]
    }

def _sanitize_schema(obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return _fallback_schema("")
    page = obj.get("page") or {}
    blocks = obj.get("blocks") or []
    out: Dict[str, Any] = {
        "page": {
            "title": str(page.get("title") or "Generated UI")[:80],
            "layout": "centered" if page.get("layout") == "centered" else "wide",
        },
        "blocks": []
    }
    def txt(x, n=4000): return ("" if x is None else str(x))[:n]
    def short(x, n=120): return ("" if x is None else str(x))[:n]
    def safe_id(s): return (re.sub(r"[^a-zA-Z0-9_.-]", "_", s or "") or "id")[:60]

    if isinstance(blocks, list):
        blocks = blocks[:100]
    else:
        blocks = []

    for b in blocks:
        if not isinstance(b, dict): continue
        t = b.get("type")

        if t == "heading":
            lvl = 1 if b.get("level")==1 else 3 if b.get("level")==3 else 2
            out["blocks"].append({"type":"heading","level":lvl,"text":txt(b.get("text"))})

        elif t == "text":
            out["blocks"].append({"type":"text","text":txt(b.get("text"))})

        elif t == "metric":
            out["blocks"].append({"type":"metric","label":short(b.get("label")), "value":short(b.get("value")), "delta":short(b.get("delta") or "")})

        elif t == "card":
            node = {"type":"card","title":short(b.get("title") or ""), "body":txt(b.get("body") or "")}
            if isinstance(b.get("children"), list):
                node["children"] = _sanitize_schema({"page":{}, "blocks":b["children"]})["blocks"]
            out["blocks"].append(node)

        elif t == "table":
            cols = [short(c) for c in (b.get("columns") or [])][:12]
            rows = []
            for r in (b.get("rows") or [])[:200]:
                if isinstance(r, list):
                    rows.append([short(x, 200) for x in r[:len(cols)]])
            out["blocks"].append({"type":"table","columns":cols,"rows":rows})

        elif t == "chart":
            kind = "bar" if b.get("kind") == "bar" else "line"
            data = b.get("data") or {}
            X = (data.get("x") or [])[:200]
            Y = (data.get("y") or [])[:200]
            Y = [float(y) if _isnum(y) else 0.0 for y in Y]
            out["blocks"].append({"type":"chart","kind":kind,"data":{"x":X,"y":Y}})

        elif t == "input":
            it = b.get("inputType") if b.get("inputType") in ("text","number","password") else "text"
            out["blocks"].append({"type":"input","inputType":it,"label":short(b.get("label")), "id":safe_id(b.get("id") or "input"),
                                  "placeholder":short(b.get("placeholder") or ""), "value":b.get("value")})

        elif t == "select":
            opts = []
            for o in (b.get("options") or [])[:50]:
                if isinstance(o, dict) and "label" in o and "value" in o:
                    opts.append({"label":short(o["label"]), "value":short(o["value"])})
            val = b.get("value") or (opts[0]["value"] if opts else "")
            out["blocks"].append({"type":"select","label":short(b.get("label")), "id":safe_id(b.get("id") or "select"),
                                  "options":opts, "value":short(val)})

        elif t == "button":
            out["blocks"].append({"type":"button","text":short(b.get("text") or "Submit"), "id":safe_id(b.get("id") or "action")})

        elif t == "columns":
            ratio = b.get("ratio") or [1,1]
            if not isinstance(ratio, list) or not ratio: ratio = [1,1]
            cols = b.get("columns") or [[] for _ in ratio]
            new_cols = []
            for col in cols[:len(ratio)]:
                if isinstance(col, list):
                    new_cols.append(_sanitize_schema({"page":{}, "blocks":col})["blocks"])
            out["blocks"].append({"type":"columns","ratio":[int(x) for x in ratio][:6], "columns": new_cols})

        # ignore others
    return out

def _isnum(v: Any) -> bool:
    try:
        float(v); return True
    except Exception:
        return False

# =========================
# Stable demo data per prompt
# =========================
def stable_demo_for_prompt(prompt: str) -> Dict[str, Any]:
    rng = random.Random(prompt_seed(prompt))
    # Hourly day-shape values
    hours = list(range(24))
    vals = []
    for h in hours:
        base = 20 + 15 * math.sin((h - 8) / 24 * 2 * math.pi)
        noise = rng.uniform(-3, 3)
        vals.append(max(0, round(base + noise, 2)))
    ins = [round(v + rng.uniform(-1.5, 1.5), 2) for v in vals]
    outs = [round(v * rng.uniform(0.85, 0.95), 2) for v in vals]
    total_in, total_out = int(sum(ins)), int(sum(outs))
    net = total_in - total_out
    # Recent events
    names = ["Alex","Sam","Taylor","Jordan","Morgan","Casey","Riley","Quinn","Avery","Drew","Jamie","Parker","Reese","Shawn","Emery"]
    rows = []
    for i in range(10):
        n = names[(i + rng.randrange(0, len(names))) % len(names)]
        hh = (8 + i) % 24
        mm = (rng.randrange(0, 59)//5)*5
        action = "check-in" if rng.random() > 0.45 else "check-out"
        rows.append([f"EMP-{1000 + i}", n, action, f"{hh:02d}:{mm:02d}"])
    return {
        "hours": [f"{h:02d}:00" for h in hours],
        "ins": ins,
        "outs": outs,
        "kpis": {"Check-ins": total_in, "Check-outs": total_out, "Net": net},
        "events": rows
    }

def canonicalize(schema: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    """
    Make block order stable for common 'dashboard' prompts:
    Heading -> (KPIs row) -> Chart -> Table -> (others)
    """
    blocks = schema.get("blocks", [])
    norm = normalize_prompt(prompt)
    looks_like_dashboard = any(k in norm for k in ["dashboard","check in","check-in","checkin","kpi","chart","table"])
    if not looks_like_dashboard:
        return schema

    # Collect blocks by type
    headings, metrics, charts, tables, others = [], [], [], [], []
    for b in blocks:
        t = b.get("type")
        if t == "heading": headings.append(b)
        elif t == "columns":
            # keep columns as-is, but bucket inside
            metrics.append(b) if any(bb.get("type") == "metric" for col in b.get("columns", []) for bb in col) else others.append(b)
        elif t == "metric": metrics.append(b)
        elif t == "chart": charts.append(b)
        elif t == "table": tables.append(b)
        else: others.append(b)

    ordered: List[Dict[str, Any]] = []
    ordered += headings[:1]                       # first heading only
    # If we have standalone metrics, group first three into a 3-col row
    solo_metrics = [m for m in metrics if m.get("type") == "metric"]
    if solo_metrics:
        trio = solo_metrics[:3]
        cols = [[trio[0]]] if len(trio)==1 else [[trio[0]],[trio[1]]] if len(trio)==2 else [[trio[0]],[trio[1]],[trio[2]]]
        ordered.append({"type":"columns","ratio":[1]*len(cols),"columns":cols})
    # Include any pre-existing columns blocks with metrics
    ordered += [m for m in metrics if m.get("type") == "columns"]
    ordered += charts[:1]
    ordered += tables[:1]
    # Finally add any remaining others
    ordered += [b for b in others if b.get("type") not in {"heading","metric","chart","table"}]

    schema["blocks"] = ordered
    return schema

def stabilize_numbers(schema: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    """
    Overwrite any numeric/demo content with deterministic values derived from the prompt.
    Ensures same prompt => same figures; new prompt => new figures.
    """
    data = stable_demo_for_prompt(prompt)
    blocks = schema.get("blocks", [])

    # Fill KPIs
    kpis = [("Check-ins", data["kpis"]["Check-ins"]),
            ("Check-outs", data["kpis"]["Check-outs"]),
            ("Net", data["kpis"]["Net"])]
    kpi_used = 0
    for b in blocks:
        if b.get("type") == "columns":
            for col in b.get("columns", []):
                for bb in col:
                    if bb.get("type") == "metric" and kpi_used < 3:
                        bb["label"], bb["value"], bb["delta"] = kpis[kpi_used][0], str(kpis[kpi_used][1]), bb.get("delta") or ""
                        kpi_used += 1
        elif b.get("type") == "metric" and kpi_used < 3:
            b["label"], b["value"], b["delta"] = kpis[kpi_used][0], str(kpis[kpi_used][1]), b.get("delta") or ""
            kpi_used += 1

    # If none were present, add a KPI row
    if kpi_used == 0:
        schema["blocks"].insert(0, {
            "type":"columns","ratio":[1,1,1],
            "columns":[
                [{"type":"metric","label":"Check-ins","value":str(kpis[0][1]),"delta":""}],
                [{"type":"metric","label":"Check-outs","value":str(kpis[1][1]),"delta":""}],
                [{"type":"metric","label":"Net","value":str(kpis[2][1]),"delta":""}],
            ]
        })

    # Ensure at least one chart
    has_chart = any(b.get("type") == "chart" for b in blocks) or any(bb.get("type") == "chart" for b in blocks if b.get("type")=="columns" for col in b.get("columns", []) for bb in col)
    if not has_chart:
        schema["blocks"].append({"type":"chart","kind":"line","data":{"x": data["hours"], "y": data["ins"]}})
    else:
        for b in blocks:
            if b.get("type") == "chart":
                b["kind"] = "line" if b.get("kind") not in ("line","bar") else b["kind"]
                b["data"] = {"x": data["hours"], "y": data["ins"]}

    # Ensure at least one table
    has_table = any(b.get("type") == "table" for b in blocks) or any(bb.get("type") == "table" for b in blocks if b.get("type")=="columns" for col in b.get("columns", []) for bb in col)
    if not has_table:
        schema["blocks"].append({"type":"table","columns":["Employee ID","Name","Action","Time"], "rows": data["events"]})

    return schema

# =========================
# Routes
# =========================
@app.get("/")
def index():
    return send_from_directory("static", "index.html")

@app.after_request
def no_store(resp):
    # Prevent intermediary caching of API responses
    if request.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp

@app.post("/api/interpret")
def api_interpret():
    if client is None:
        return jsonify({"error":"Set OPENAI_API_KEY"}), 400

    payload = request.get_json(force=True) or {}
    prompt = (payload.get("prompt") or "").strip()

    # Always compute fresh (no server cache)
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            instructions=SCHEMA_GUIDE,
            input=f"User request:\n{prompt}\n\nReturn ONLY the JSON UI object.",
            max_output_tokens=2000,   # do not send temperature for this model
        )
        raw = resp.output_text or ""
    except Exception:
        base = _fallback_schema(prompt)
        base = canonicalize(base, prompt)
        stable = stabilize_numbers(base, prompt)
        return jsonify({"schema": stable}), 200

    obj = _extract_json(raw)
    base = _sanitize_schema(obj) if obj else _fallback_schema(prompt)
    base = canonicalize(base, prompt)
    stable = stabilize_numbers(base, prompt)
    return jsonify({"schema": stable}), 200

# Optional demo action
@app.post("/api/submit")
def api_submit():
    payload = request.get_json(force=True) or {}
    return jsonify({"ok": True, "echo": payload})

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("⚠️  Set OPENAI_API_KEY (export OPENAI_API_KEY=...)")
    app.run(host="0.0.0.0", port=PORT, debug=True)
