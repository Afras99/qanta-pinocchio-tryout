"""Render WRITEUP.md to WRITEUP.pdf (markdown-it-py -> HTML -> headless Chrome). Run from the repo root."""
import pathlib, subprocess, sys
from markdown_it import MarkdownIt

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """
@page { size: Letter; margin: 0.36in 0.5in; }
body { font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 8.8pt; line-height: 1.3; color: #111; }
h1 { font-size: 15pt; margin: 0 0 4px; } h2 { break-after: avoid; font-size: 11pt; margin: 6px 0 2px; border-bottom: 1px solid #ddd; padding-bottom: 1px; }
p { margin: 3px 0; } ul { margin: 2px 0; padding-left: 16px; } li { margin: 1px 0; }
code { font-family: Menlo, monospace; font-size: 8pt; }
tr { break-inside: avoid; } table { break-inside: avoid; } table { border-collapse: collapse; margin: 3px 0 4px; width: 100%; } th, td { border: 1px solid #bbb; padding: 1.5px 5px; text-align: left; vertical-align: top; font-size: 8.3pt; } th { background: #f3f3f3; }
.figs { display: flex; gap: 8px; margin: 3px 0; } .figs figure { flex: 1; margin: 0; } .figs img { width: 100%; max-height: 2.05in; object-fit: contain; }
.figs.one figure { flex: 0 0 56%; margin: 0 auto; } .figs.one img { max-height: none; }
figcaption { font-size: 7.6pt; color: #444; font-style: italic; margin-top: 1px; }
"""

def main():
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    html = f"<!doctype html><meta charset='utf-8'><style>{CSS}</style>{md.render(pathlib.Path('WRITEUP.md').read_text())}"
    tmp = pathlib.Path("_writeup_tmp.html"); tmp.write_text(html)
    try:
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pathlib.Path('WRITEUP.pdf').resolve()}", tmp.resolve().as_uri()],
                       check=True, capture_output=True, timeout=120)
    finally:
        tmp.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
