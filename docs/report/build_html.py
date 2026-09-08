"""
Build docs/report/methods_and_validation.html from the Markdown source, with the figures embedded and MathJax for the
equations. With --pdf, also print it to methods_and_validation.pdf with headless Chromium (the Playwright Chromium in
PLAYWRIGHT_BROWSERS_PATH, or `chromium` on the PATH); MathJax is fetched with curl and inlined so that the equations
render offline.  Run from anywhere:  python docs/report/build_html.py [--pdf]
"""
import os, re, sys, glob, base64, shutil, subprocess, tempfile, markdown
HERE = os.path.dirname(os.path.abspath(__file__))
MATHJAX = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.js"
src = open(os.path.join(HERE, "methods_and_validation.md"), encoding="utf-8").read()

# protect LaTeX from the Markdown converter
maths = []
def keep(m):
    maths.append(m.group(0)); return f"@@MATH{len(maths) - 1}@@"
src = re.sub(r"\$\$.*?\$\$", keep, src, flags=re.S)
src = re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", keep, src)

body = markdown.markdown(src, extensions=["tables", "fenced_code"])
for i, m in enumerate(maths):
    body = body.replace(f"@@MATH{i}@@", m)

# embed the figures
def embed(m):
    path = os.path.join(HERE, m.group(2))
    if os.path.exists(path):
        uri = "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()
        return f'<img alt="{m.group(1)}" src="{uri}"'
    return m.group(0)
body = re.sub(r'<img alt="([^"]*)" src="([^"]+)"', embed, body)
body = re.sub(r"<p><em>(.*?)</em></p>", r'<p class="caption"><em>\1</em></p>', body, flags=re.S)      # figure and table captions

html = f"""<title>VLM Ventilation Methods</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono&display=swap">
<script>window.MathJax = {{ tex: {{ inlineMath: [['$', '$']], displayMath: [['$$', '$$']] }}, svg: {{ fontCache: 'none' }} }};</script>
@@MATHJAX@@
<style>
:root {{ --paper:#FAFAF7; --ink:#1B1F24; --muted:#5C6570; --accent:#0B4F8A; --rule:#D8DCE1; --stripe:#F0F2F4; --note:#EEF3F8;
  --serif:"Newsreader",Georgia,serif; --sans:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif; --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --paper:#14171B; --ink:#E4E7EA; --muted:#9AA3AD; --accent:#6DB0E6; --rule:#2B3138; --stripe:#1B2026; --note:#1A222B; }} }}
:root[data-theme="dark"] {{ --paper:#14171B; --ink:#E4E7EA; --muted:#9AA3AD; --accent:#6DB0E6; --rule:#2B3138; --stripe:#1B2026; --note:#1A222B; }}
body {{ background:var(--paper); color:var(--ink); font-family:var(--serif); font-size:17px; line-height:1.55; margin:0; }}
main {{ max-width:72ch; margin:0 auto; padding:3rem 1.25rem 5rem; }}
h1 {{ font-size:2rem; font-weight:500; line-height:1.15; margin:0 0 1rem; text-wrap:balance; }}
h2 {{ font-size:1.35rem; font-weight:600; margin:2.4rem 0 .7rem; border-top:1px solid var(--rule); padding-top:1.2rem; }}
h3 {{ font-size:1.08rem; font-weight:600; margin:1.6rem 0 .4rem; }}
p {{ margin:0 0 .9rem; }} ol, ul {{ padding-left:1.4rem; margin:0 0 .9rem; }} li {{ margin-bottom:.35rem; }}
em {{ color:var(--muted); }}
img {{ max-width:100%; height:auto; display:block; margin:1.2rem auto .4rem; }}
p.caption {{ font-family:var(--sans); font-size:.82rem; }}
table {{ border-collapse:collapse; width:100%; font-family:var(--sans); font-size:.84rem; font-variant-numeric:tabular-nums; margin:1rem 0 1.6rem; display:block; overflow-x:auto; }}
th, td {{ padding:.4rem .6rem; text-align:left; border-bottom:1px solid var(--rule); vertical-align:top; }}
th {{ font-weight:600; color:var(--muted); border-bottom:2px solid var(--rule); }}
tbody tr:nth-child(even) {{ background:var(--stripe); }}
code {{ font-family:var(--mono); font-size:.85em; background:var(--stripe); padding:.05em .3em; border-radius:3px; }}
pre {{ background:var(--stripe); padding:.8rem 1rem; overflow-x:auto; font-size:.8rem; border-radius:3px; line-height:1.4; }}
pre code {{ background:none; padding:0; }}
a {{ color:var(--accent); }}
@page {{ size: A4; margin: 18mm 17mm 20mm 17mm; }}
@media print {{
  :root {{ --paper:#FFFFFF; --ink:#000000; --muted:#444B52; --accent:#0B4F8A; --rule:#BFC5CB; --stripe:#F3F4F6; }}
  body {{ font-size:10.5pt; line-height:1.45; background:#FFFFFF; -webkit-print-color-adjust:exact; print-color-adjust:exact; }}
  main {{ max-width:none; padding:0; }}
  h1 {{ font-size:20pt; }} h2 {{ font-size:14pt; break-after:avoid; }} h3 {{ font-size:11.5pt; break-after:avoid; }}
  table {{ display:table; font-size:8pt; break-inside:auto; }} tr {{ break-inside:avoid; }} thead {{ display:table-header-group; }} th, td {{ padding:.25rem .35rem; }}
  img {{ break-inside:avoid; max-height:9cm; width:auto; max-width:100%; }}
  p.caption {{ font-size:8.5pt; break-before:avoid; break-after:avoid; }}
  pre {{ font-size:7.8pt; white-space:pre-wrap; break-inside:avoid; }}
  a {{ color:inherit; text-decoration:none; }}
}}
</style>
<main>
{body}
</main>
"""
out = os.path.join(HERE, "methods_and_validation.html")
open(out, "w", encoding="utf-8").write(html.replace("@@MATHJAX@@", f'<script src="{MATHJAX}"></script>'))
print("written", out, len(html)//1024, "KB")

if "--pdf" in sys.argv:
    tmp = tempfile.mkdtemp()
    js = os.path.join(tmp, "tex-svg.js")
    try:
        subprocess.run(["curl", "-sS", "-o", js, MATHJAX], check=True)
        ext = os.path.join(tmp, "input", "tex", "extensions"); os.makedirs(ext)      # \boldsymbol is loaded on demand from this path
        subprocess.run(["curl", "-sS", "-o", os.path.join(ext, "boldsymbol.js"), MATHJAX.replace("tex-svg.js", "input/tex/extensions/boldsymbol.js")], check=True)
        script = '<script src="tex-svg.js"></script>'        # local copy next to the page: the headless browser may have no network
    except Exception as e:
        print("MathJax download failed, using the CDN:", e); script = f'<script src="{MATHJAX}"></script>'
    page = os.path.join(tmp, "report.html")
    print_html = re.sub(r'<link rel="stylesheet"[^>]*>', "", html)          # web fonts are not reachable offline and a pending stylesheet blocks the scripts
    open(page, "w", encoding="utf-8").write('<!doctype html>\n<meta charset="utf-8">\n' + print_html.replace("@@MATHJAX@@", script))
    chrome = shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome")
    for c in glob.glob(os.path.join(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", ""), "chromium-*", "chrome-linux", "chrome")):
        chrome = c
    pdf = os.path.join(HERE, "methods_and_validation.pdf")
    subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--virtual-time-budget=30000",
                    "--run-all-compositor-stages-before-draw", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", "file://" + page],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("written", pdf, os.path.getsize(pdf)//1024, "KB", "from", page)
