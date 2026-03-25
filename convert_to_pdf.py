"""Convert DESIGN_DOC.md to a styled PDF using fpdf2 with Unicode support."""

import re
from fpdf import FPDF, XPos, YPos


# Map Unicode box-drawing and special chars to ASCII equivalents
UNICODE_MAP = {
    "\u2500": "-",  # ─
    "\u2502": "|",  # │
    "\u250c": "+",  # ┌
    "\u2510": "+",  # ┐
    "\u2514": "+",  # └
    "\u2518": "+",  # ┘
    "\u251c": "+",  # ├
    "\u2524": "+",  # ┤
    "\u252c": "+",  # ┬
    "\u2534": "+",  # ┴
    "\u253c": "+",  # ┼
    "\u2550": "=",  # ═
    "\u2551": "|",  # ║
    "\u2192": "->", # →
    "\u2190": "<-", # ←
    "\u2014": "--", # —
    "\u2013": "-",  # –
    "\u201c": '"',  # "
    "\u201d": '"',  # "
    "\u2018": "'",  # '
    "\u2019": "'",  # '
    "\u2026": "...",# …
}


def sanitize(text):
    """Replace Unicode chars that latin-1 can't encode."""
    for char, replacement in UNICODE_MAP.items():
        text = text.replace(char, replacement)
    # Catch any remaining non-latin-1 chars
    return text.encode("latin-1", errors="replace").decode("latin-1")


class DesignDocPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def parse_table(lines):
    """Parse markdown table lines into rows of cells."""
    rows = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            continue
        if re.match(r"^\|[\s\-:|]+\|$", line):
            continue
        cells = [sanitize(c.strip()) for c in line.split("|")[1:-1]]
        rows.append(cells)
    return rows


def render_table(pdf, rows):
    """Render a table with auto-sized columns."""
    if not rows:
        return

    page_width = pdf.w - pdf.l_margin - pdf.r_margin
    num_cols = len(rows[0])

    # Calculate column widths
    col_widths = [0.0] * num_cols
    pdf.set_font("Helvetica", "", 8.5)
    for row in rows:
        for i, cell in enumerate(row):
            if i < num_cols:
                w = pdf.get_string_width(cell) + 8
                col_widths[i] = max(col_widths[i], w)

    # Scale to fit
    total = sum(col_widths)
    if total > page_width:
        col_widths = [w * page_width / total for w in col_widths]

    row_h = 7

    # Header
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_fill_color(240, 240, 240)
    for i, cell in enumerate(rows[0]):
        if i < num_cols:
            pdf.cell(col_widths[i], row_h, cell, border=1, fill=True)
    pdf.ln()

    # Data rows
    pdf.set_font("Helvetica", "", 8.5)
    for row in rows[1:]:
        if pdf.get_y() + row_h > pdf.h - 25:
            pdf.add_page()

        # Use multi_cell for long content, cell for short
        y_before = pdf.get_y()
        x_start = pdf.l_margin
        max_y = y_before

        for i, cell in enumerate(row):
            if i >= num_cols:
                break
            pdf.set_xy(x_start + sum(col_widths[:i]), y_before)
            pdf.multi_cell(col_widths[i], 6, cell, border=1)
            max_y = max(max_y, pdf.get_y())

        pdf.set_y(max_y)


def render_code_block(pdf, code_text):
    """Render a code block with monospace font and gray background."""
    pdf.set_font("Courier", "", 7.5)
    pdf.set_fill_color(245, 245, 245)
    page_width = pdf.w - pdf.l_margin - pdf.r_margin

    for line in code_text.split("\n"):
        if pdf.get_y() > pdf.h - 20:
            pdf.add_page()
            pdf.set_font("Courier", "", 7.5)
            pdf.set_fill_color(245, 245, 245)
        line = sanitize(line)
        # Truncate if too long
        display = "  " + line
        while pdf.get_string_width(display) > page_width - 4:
            display = display[:-1]
        pdf.cell(page_width, 5, display, fill=True,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(2)


def clean_md(text):
    """Strip markdown formatting for plain-text PDF rendering."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return sanitize(text)


def build_pdf():
    with open("DESIGN_DOC.md") as f:
        content = f.read()

    # Strip <details> block (full prompt is too long for the PDF)
    content = re.sub(
        r"<details>.*?</details>",
        "_See bookly/prompts.py for the full 338-line system prompt._\n",
        content,
        flags=re.DOTALL,
    )

    lines = content.split("\n")
    pdf = DesignDocPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    i = 0
    while i < len(lines):
        line = lines[i]

        # H1
        if line.startswith("# ") and not line.startswith("## "):
            pdf.set_font("Helvetica", "B", 18)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 12, clean_md(line[2:]),
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            y = pdf.get_y()
            pdf.set_draw_color(60, 60, 60)
            pdf.set_line_width(0.5)
            pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
            pdf.ln(4)
            i += 1
            continue

        # H2
        if line.startswith("## "):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(44, 62, 80)
            pdf.cell(0, 9, clean_md(line[3:]),
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)
            i += 1
            continue

        # Code block
        if line.strip().startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            render_code_block(pdf, "\n".join(code_lines))
            continue

        # Table
        if line.strip().startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            rows = parse_table(table_lines)
            render_table(pdf, rows)
            pdf.ln(3)
            continue

        # Numbered list
        if re.match(r"^\d+\.\s", line.strip()):
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 30, 30)
            text = clean_md(line.strip())
            page_width = pdf.w - pdf.l_margin - pdf.r_margin
            pdf.multi_cell(page_width, 6, "  " + text)
            i += 1
            continue

        # Blank line
        if not line.strip():
            pdf.ln(2)
            i += 1
            continue

        # Regular paragraph — collect consecutive lines
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 30, 30)
        para_lines = []
        while (i < len(lines)
               and lines[i].strip()
               and not lines[i].startswith("#")
               and not lines[i].strip().startswith("|")
               and not lines[i].strip().startswith("```")
               and not re.match(r"^\d+\.\s", lines[i].strip())):
            para_lines.append(lines[i].strip())
            i += 1

        if para_lines:
            text = clean_md(" ".join(para_lines))
            page_width = pdf.w - pdf.l_margin - pdf.r_margin
            pdf.multi_cell(page_width, 6, text)
            pdf.ln(1)
            continue

        i += 1

    pdf.output("DESIGN_DOC.pdf")
    print("Created DESIGN_DOC.pdf")


if __name__ == "__main__":
    build_pdf()
