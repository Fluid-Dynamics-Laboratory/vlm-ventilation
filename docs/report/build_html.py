"""Build docs/report/methods_and_validation.html from the Markdown source, with the figures embedded and MathJax for the equations."""
import os, re, base64, markdown
HERE = os.path.dirname(os.path.abspath(__file__))
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

html = f"""<title>VLM Ventilation Methods</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono&display=swap">
<script>window.MathJax = {{ tex: {{ inlineMath: [['$', '$']], displayMath: [['$$', '$$']] }}, svg: {{ fontCache: 'none' }} }};</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.js"></script>
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
p > em:only-child {{ font-family:var(--sans); font-size:.82rem; display:block; }}
table {{ border-collapse:collapse; width:100%; font-family:var(--sans); font-size:.84rem; font-variant-numeric:tabular-nums; margin:1rem 0 1.6rem; display:block; overflow-x:auto; }}
th, td {{ padding:.4rem .6rem; text-align:left; border-bottom:1px solid var(--rule); vertical-align:top; }}
th {{ font-weight:600; color:var(--muted); border-bottom:2px solid var(--rule); }}
tbody tr:nth-child(even) {{ background:var(--stripe); }}
code {{ font-family:var(--mono); font-size:.85em; background:var(--stripe); padding:.05em .3em; border-radius:3px; }}
pre {{ background:var(--stripe); padding:.8rem 1rem; overflow-x:auto; font-size:.8rem; border-radius:3px; line-height:1.4; }}
pre code {{ background:none; padding:0; }}
a {{ color:var(--accent); }}
</style>
<main>
{body}
</main>
"""
out = os.path.join(HERE, "methods_and_validation.html")
open(out, "w", encoding="utf-8").write(html)
print("written", out, len(html)//1024, "KB")
