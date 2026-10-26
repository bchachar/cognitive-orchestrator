"""Render one trace JSON (from CO_TRACE=true) as a single self-contained HTML file.

    python tools/view_trace.py traces/run-<timestamp>.json [-o out.html]
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

_CSS = """
body{font:14px system-ui,sans-serif;margin:2rem auto;max-width:900px;color:#1f2933}
h1{font-size:1.2rem} .meta{color:#52606d;margin-bottom:1.5rem}
.timeline{border-left:3px solid #cbd2d9;padding-left:1.2rem}
details{margin:0 0 .8rem;border:1px solid #cbd2d9;border-radius:6px;padding:.4rem .7rem;
 background:#fff;position:relative}
details::before{content:"";position:absolute;left:-1.7rem;top:.8rem;width:.7rem;height:.7rem;
 border-radius:50%;background:#3ebd93}
details.failure,details.error{border-color:#e12d39;background:#ffe3e3}
details.failure::before,details.error::before{background:#e12d39}
summary{cursor:pointer;font-weight:600}
.dur{float:right;font-weight:400;color:#52606d}
.err{color:#ab091e;font-weight:600;margin:.4rem 0}
table{border-collapse:collapse;width:100%;margin:.3rem 0}
td,th{border-top:1px solid #e4e7eb;padding:.2rem .4rem;text-align:left;vertical-align:top;
 font-family:ui-monospace,monospace;font-size:12px;word-break:break-all}
th{width:9rem;color:#52606d}h4{margin:.6rem 0 0}
"""


def _table(data: dict) -> str:
    if not data:
        return "<p><em>(empty)</em></p>"
    rows = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
                   for k, v in data.items())
    return f"<table>{rows}</table>"


def _entry(e: dict) -> str:
    status = e.get("status", "ok")
    parts = [f'<details class="{html.escape(status)}"><summary>{html.escape(e["node"])}'
             f'<span class="dur">{e.get("duration_ms", 0):.1f} ms'
             f'{"" if status == "ok" else " · " + html.escape(status.upper())}</span></summary>',
             f'<div>{html.escape(e.get("start", ""))} → {html.escape(e.get("end", ""))}</div>']
    if e.get("error"):
        parts.append(f'<div class="err">{html.escape(e["error"])}</div>')
    parts.append("<h4>Input</h4>" + _table(e.get("input", {})))
    parts.append("<h4>Output</h4>" + _table(e.get("output", {})))
    if e.get("model_calls"):
        parts.append("<h4>Model calls</h4>" + _table(
            {f"#{i + 1}": json.dumps(c) for i, c in enumerate(e["model_calls"])}))
    parts.append("</details>")
    return "".join(parts)


def render(trace: dict) -> str:
    entries = trace.get("entries", [])
    total = sum(e.get("duration_ms", 0) for e in entries)
    bad = sum(1 for e in entries if e.get("status", "ok") != "ok")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>Trace</title>"
            f"<style>{_CSS}</style></head><body><h1>{html.escape(trace.get('command', ''))}</h1>"
            f"<div class='meta'>{len(entries)} nodes · {total:.1f} ms · {bad} failed · "
            f"started {html.escape(trace.get('started_at', ''))}</div>"
            f"<div class='timeline'>{''.join(_entry(e) for e in entries)}</div></body></html>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="default: <trace>.html")
    args = ap.parse_args()
    out = args.output or args.trace.with_suffix(".html")
    out.write_text(render(json.loads(args.trace.read_text())))
    print(out)


if __name__ == "__main__":
    main()
