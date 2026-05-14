"""Render docs/APPROACH.md into a concise two-page PDF."""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/APPROACH.md"
OUTPUT = ROOT / "docs/approach.pdf"


def clean_markdown(line: str) -> str:
    line = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", line)
    line = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", line)
    return line


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=20,
        spaceAfter=8,
        textColor=colors.HexColor("#1f3a5f"),
    )
    heading = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13,
        spaceBefore=5,
        spaceAfter=3,
        textColor=colors.HexColor("#1f3a5f"),
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.2,
        leading=10.4,
        spaceAfter=2,
    )
    bullet = ParagraphStyle(
        "Bullet",
        parent=body,
        leftIndent=12,
        firstLineIndent=-7,
        bulletIndent=0,
    )

    story = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            story.append(Spacer(1, 2))
            continue
        if line.startswith("# "):
            story.append(Paragraph(clean_markdown(line[2:]), title))
        elif line.startswith("## "):
            story.append(Paragraph(clean_markdown(line[3:]), heading))
        elif line.startswith("- "):
            story.append(Paragraph(clean_markdown(line[2:]), bullet, bulletText="-"))
        else:
            story.append(Paragraph(clean_markdown(line), body))

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
    )
    doc.build(story)
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
