#!/usr/bin/env python3
"""Render report.md as a standalone, phone-friendly HTML page (report.html, figures referenced from figs/).
  python report/build_html.py        (needs: pip install markdown)"""
import re
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
md = (HERE / "report.md").read_text()
title_line, body_md = md.split("\n", 1)
title = title_line.lstrip("# ").strip()
body = markdown.markdown(body_md, extensions=["tables", "sane_lists"])
body = re.sub(r"<table>", '<div class="tbl"><table>', body).replace("</table>", "</table></div>")
# a figure image followed by its bold caption paragraph -> one <figure>
body = re.sub(r'<p><img alt="([^"]*)" src="([^"]+)" ?/?></p>\s*<p><strong>(Figure \d+\.[^<]*)</strong>(.*?)</p>',
              r'<figure><div class="plate"><img src="\2" alt="\1"></div><figcaption><strong>\3</strong>\4</figcaption></figure>',
              body, flags=re.S)

CSS = """
/* Layout: single reading column (max 46rem); figures sit on a white plate in both themes. */
:root {
  --bg: #f3f5f7; --paper: #ffffff; --ink: #1b2229; --muted: #5b6672; --line: #d8dde3; --accent: #2a78d6; --mark: #e9f1fb;
  --font-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-head: "IBM Plex Serif", Georgia, "Times New Roman", serif;
  --font-mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #111519; --paper: #191f25; --ink: #e3e7eb; --muted: #9aa5b0; --line: #2c343c; --accent: #6aa8e8; --mark: #1d2a38; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #111519; --paper: #191f25; --ink: #e3e7eb; --muted: #9aa5b0; --line: #2c343c; --accent: #6aa8e8; --mark: #1d2a38; color-scheme: dark }
* { box-sizing: border-box }
body { background: var(--bg); color: var(--ink); font-family: var(--font-body); font-size: 16px; line-height: 1.6;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  padding-block: 20px 48px }
main { max-width: 46rem; margin: 0 auto; background: var(--paper); border: 1px solid var(--line); border-radius: 10px;
  padding: clamp(18px, 4vw, 44px); min-width: 0 }
header { display: grid; gap: 6px; padding-bottom: 14px; border-bottom: 1px solid var(--line); margin-bottom: 8px }
.kicker { font: 500 .72rem var(--font-mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted) }
h1 { font: 600 clamp(1.45rem, 4.6vw, 2rem)/1.2 var(--font-head); margin: 0; text-wrap: balance }
h2 { font: 600 1.25rem/1.3 var(--font-head); margin: 1.8em 0 .4em; text-wrap: balance }
p, li { max-width: 65ch } p { margin: .6em 0 } ul, ol { padding-left: 1.3em; margin: .5em 0 } li { margin: .3em 0 }
em { color: var(--muted) } a { color: var(--accent) } strong { font-weight: 600 }
.tbl { overflow-x: auto; margin: .8em 0 } table { border-collapse: collapse; font-size: .9rem; min-width: 30rem }
th, td { text-align: left; vertical-align: top; padding: 6px 10px; border-bottom: 1px solid var(--line) }
th { font-weight: 600; font-size: .78rem; text-transform: uppercase; letter-spacing: .05em; color: var(--muted) }
figure { margin: 1.4em 0 .4em; display: grid; gap: 8px }
.plate { background: #ffffff; border: 1px solid var(--line); border-radius: 8px; padding: 6px }
.plate img { display: block; width: 100%; height: auto; max-width: 100% }
figcaption { font-size: .92rem; color: var(--ink) }
.meta { font: .78rem var(--font-mono); color: var(--muted); overflow-wrap: anywhere }
"""
html = f"""<title>ADAM-U Nut Pick Report</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Serif:wght@600&display=swap">
<style>{CSS}</style>
<main>
<header><div class="kicker">Report · MuJoCo · 50 held-out seeds per result</div><h1>{title}</h1></header>
{body}
</main>
"""
(HERE / "report.html").write_text(html)
print("wrote", HERE / "report.html")
