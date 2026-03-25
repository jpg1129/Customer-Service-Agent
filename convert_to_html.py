"""Convert DESIGN_DOC.md to a self-contained, styled HTML file."""

import markdown

with open("DESIGN_DOC.md") as f:
    md_text = f.read()

# Convert markdown to HTML (tables, fenced code, inline formatting)
html_body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "md_in_html"],
)

html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Bookly Support Agent: Design Document</title>
<style>
  @page {{
    size: letter;
    margin: 0.75in 0.85in;
  }}
  @media print {{
    body {{ font-size: 10pt; }}
    pre {{ font-size: 7.5pt; }}
    table {{ font-size: 9pt; }}
    h1 {{ font-size: 16pt; }}
    h2 {{ font-size: 12pt; }}
    details {{ open: true; }}
    details > summary {{ display: none; }}
    details > *:not(summary) {{ display: block; }}
  }}
  * {{
    box-sizing: border-box;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.55;
    color: #1a1a2e;
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem 1.5rem;
  }}
  h1 {{
    font-size: 20pt;
    font-weight: 700;
    color: #111;
    border-bottom: 2.5px solid #2c3e50;
    padding-bottom: 0.3em;
    margin-bottom: 0.6em;
  }}
  h2 {{
    font-size: 14pt;
    font-weight: 600;
    color: #2c3e50;
    margin-top: 1.6em;
    margin-bottom: 0.5em;
    padding-bottom: 0.2em;
    border-bottom: 1px solid #e0e0e0;
  }}
  p {{
    margin: 0.5em 0;
  }}
  strong {{
    font-weight: 600;
  }}
  em {{
    font-style: italic;
  }}
  code {{
    font-family: 'SF Mono', 'Menlo', 'Consolas', 'Liberation Mono', monospace;
    font-size: 0.88em;
    background-color: #f0f4f8;
    padding: 0.15em 0.4em;
    border-radius: 4px;
    color: #c7254e;
  }}
  pre {{
    background-color: #f6f8fa;
    border: 1px solid #d1d9e0;
    border-radius: 6px;
    padding: 14px 16px;
    font-size: 8.5pt;
    line-height: 1.45;
    overflow-x: auto;
    margin: 0.8em 0;
  }}
  pre code {{
    background: none;
    padding: 0;
    color: #24292f;
    font-size: inherit;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 0.8em 0;
    font-size: 10pt;
  }}
  th, td {{
    border: 1px solid #d1d9e0;
    padding: 8px 12px;
    text-align: left;
    vertical-align: top;
  }}
  th {{
    background-color: #f0f4f8;
    font-weight: 600;
    color: #2c3e50;
  }}
  tr:nth-child(even) {{
    background-color: #fafbfc;
  }}
  ol, ul {{
    margin: 0.4em 0;
    padding-left: 1.6em;
  }}
  li {{
    margin: 0.3em 0;
  }}
  li p {{
    margin: 0.2em 0;
  }}
  details {{
    margin: 1em 0;
    border: 1px solid #d1d9e0;
    border-radius: 6px;
    overflow: hidden;
  }}
  summary {{
    font-weight: 600;
    padding: 10px 14px;
    background-color: #f0f4f8;
    cursor: pointer;
    color: #2c3e50;
    border-bottom: 1px solid #d1d9e0;
  }}
  summary:hover {{
    background-color: #e8ecf0;
  }}
  details[open] summary {{
    border-bottom: 1px solid #d1d9e0;
  }}
  details > pre {{
    margin: 0;
    border: none;
    border-radius: 0;
  }}
  details > p {{
    padding: 10px 14px;
    margin: 0;
  }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

with open("DESIGN_DOC.html", "w") as f:
    f.write(html_doc)

print("Created DESIGN_DOC.html")
print("To get a PDF: open in your browser and use File > Print > Save as PDF")
