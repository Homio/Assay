"""
Output formatting — Markdown tables and Docling-compatible JSON.

Dual output format:
  - Markdown: human-readable, easy to diff and review
  - JSON: machine-readable, compatible with docling export structure

Coordinate system: all bbox values are in PDF points (native PDF units).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_CODE_REV: Optional[str] = None


def code_rev() -> str:
    """
    Fingerprint of the code that produced an output artifact.

    git short hash when available; otherwise the newest mtime among the
    assay package modules.  Embedded in JSON exports so an evaluation
    snapshot can always be traced back to (and checked against) the exact
    code state that generated it.
    """
    global _CODE_REV
    if _CODE_REV is not None:
        return _CODE_REV
    rev = ""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            rev = r.stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain", "--", "assay/"],
                cwd=Path(__file__).resolve().parent.parent,
                capture_output=True, text=True, timeout=5,
            )
            if dirty.returncode == 0 and dirty.stdout.strip():
                rev += "+dirty"
    except Exception:
        rev = ""
    if not rev:
        pkg = Path(__file__).resolve().parent
        rev = f"mt{int(max(f.stat().st_mtime for f in pkg.glob('*.py')))}"
    _CODE_REV = rev
    return rev


@dataclass
class CellOutput:
    """A single cell in a parsed table."""
    text: str = ""
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    row: int = 0
    col: int = 0
    row_span: int = 1
    col_span: int = 1
    column_header: bool = False
    row_header: bool = False


@dataclass
class TableOutput:
    """A single parsed table."""
    page_no: int
    num_rows: int
    num_cols: int
    cells: List[CellOutput] = field(default_factory=list)
    caption: str = ""
    table_bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    """Overall table bbox in PDF points (YOLO + magnetic alignment result)."""
    completed: bool = False
    """True when the cell grid was geometrically rebuilt by
    assay.completion (SLANet under-segmentation rescue), not by SLANet."""

    def cell_grid(self) -> List[List[CellOutput]]:
        """Return cells as a 2D grid (row × col) using row/col attributes."""
        if not self.cells or self.num_rows == 0 or self.num_cols == 0:
            return []
        grid: List[List[CellOutput]] = [
            [CellOutput() for _ in range(self.num_cols)]
            for _ in range(self.num_rows)
        ]
        for cell in self.cells:
            r, c = cell.row, cell.col
            if 0 <= r < self.num_rows and 0 <= c < self.num_cols:
                grid[r][c] = cell
        return grid


@dataclass
class TextOutput:
    """A text segment (paragraph, heading, etc.)."""
    text: str
    page_no: int
    label: str = "text"  # 'text', 'section_heading', 'title'


@dataclass
class PageOutput:
    """Metadata about a parsed page."""
    page_no: int
    width: float = 0.0
    height: float = 0.0


@dataclass
class DocumentOutput:
    """Complete output for one PDF file."""
    file_name: str
    pages: List[PageOutput] = field(default_factory=list)
    tables: List[TableOutput] = field(default_factory=list)
    texts: List[TextOutput] = field(default_factory=list)
    status: str = "SUCCESS"
    error: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Markdown export
# ──────────────────────────────────────────────────────────────────────────────

def export_to_markdown(doc: DocumentOutput) -> str:
    """Export document as Markdown string with text and tables interleaved by page."""
    lines: List[str] = []

    # Group content by page — texts first, then tables
    page_texts: Dict[int, List[TextOutput]] = {}
    for t in doc.texts:
        page_texts.setdefault(t.page_no, []).append(t)

    page_tables: Dict[int, List[TableOutput]] = {}
    for t in doc.tables:
        page_tables.setdefault(t.page_no, []).append(t)

    # Use all page numbers from both texts and tables
    all_pages = sorted(set(list(page_texts.keys()) + list(page_tables.keys())))

    for page_no in all_pages:
        if len(lines) == 0:
            lines.append(f"<!-- Page {page_no} -->\n")
        else:
            lines.append(f"\n<!-- Page {page_no} -->\n")

        # Output text blocks for this page
        for elem in page_texts.get(page_no, []):
            label = elem.label
            if label == "section_heading":
                lines.append(f"\n## {elem.text}\n")
            elif label == "subsection_heading":
                lines.append(f"\n### {elem.text}\n")
            elif label == "title":
                lines.append(f"\n# {elem.text}\n")
            elif elem.text.strip():
                lines.append(f"{elem.text}\n")

        # Output tables for this page
        for table in page_tables.get(page_no, []):
            table_md = _table_to_md(table)
            lines.append(table_md)
            lines.append("")

    return "\n".join(lines)


def _table_to_md(table: TableOutput) -> str:
    """Convert a parsed table to Markdown table format."""
    grid = table.cell_grid()
    if not grid or not grid[0]:
        return ""

    num_cols = len(grid[0])
    num_rows = len(grid)
    md_lines: List[str] = []

    for ri, row in enumerate(grid):
        md_row: List[str] = []
        for cell in row:
            text = cell.text.replace("\n", " ").replace("|", "\\|")
            md_row.append(text)
        # Pad to full width
        while len(md_row) < num_cols:
            md_row.append("")
        md_lines.append("| " + " | ".join(md_row) + " |")

        # Add header separator after first row if:
        #   - The first row has explicit column_header flags, OR
        #   - The table has at least 2 rows (heuristic: first row = header)
        if ri == 0 and (any(c.column_header for c in row) or num_rows >= 2):
            sep = "| " + " | ".join(["---"] * num_cols) + " |"
            md_lines.append(sep)

    return "\n".join(md_lines)


# ──────────────────────────────────────────────────────────────────────────────
# JSON export (Docling-compatible)
# ──────────────────────────────────────────────────────────────────────────────

def export_to_dict(doc: DocumentOutput) -> dict:
    """Export to Docling-compatible dict structure."""
    return {
        "schema_name": "docling",
        "version": "0.1.0",
        "code_rev": code_rev(),
        "name": doc.file_name,
        "pages": {
            str(p.page_no): {
                "page_no": p.page_no,
                "size": {"width": p.width, "height": p.height},
            }
            for p in doc.pages
        },
        "texts": [
            {
                "text": t.text,
                "label": t.label,
                "prov": [{"page_no": t.page_no, "bbox": None}],
            }
            for t in doc.texts if t.text.strip()
        ],
        "tables": [
            {
                "label": "table",
                "data": {
                    "num_rows": t.num_rows,
                    "num_cols": t.num_cols,
                    "grid": _build_grid_json(t),
                },
                "prov": [{
                    "page_no": t.page_no,
                    # Detection-level bbox (PDF points): the trusted region
                    # evidence — cell bboxes may cover less (SLANet under-
                    # segmentation); evaluators need this to score both.
                    "bbox": [round(v, 2) for v in t.table_bbox],
                    # True when the grid was geometrically rebuilt by
                    # assay.completion rather than read by SLANet.
                    "completed": t.completed,
                }],
            }
            for t in doc.tables
        ],
    }


def _build_grid_json(table: TableOutput) -> List[List[dict]]:
    """Build the grid JSON (list of rows, each row is list of cells)."""
    grid = table.cell_grid()
    result: List[List[dict]] = []
    for row in grid:
        json_row: List[dict] = []
        for cell in row:
            json_row.append({
                "bbox": {
                    "l": cell.bbox[0],
                    "t": cell.bbox[1],
                    "r": cell.bbox[2],
                    "b": cell.bbox[3],
                },
                "text": cell.text,
                "row": cell.row,
                "col": cell.col,
                "row_span": cell.row_span,
                "col_span": cell.col_span,
                "column_header": cell.column_header,
                "row_header": cell.row_header,
            })
        result.append(json_row)
    return result


def write_markdown(doc: DocumentOutput, path: str) -> None:
    """Write Markdown output to file."""
    content = export_to_markdown(doc)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def write_json(doc: DocumentOutput, path: str, indent: int = 0) -> None:
    """Write JSON output to file."""
    data = export_to_dict(doc)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


# ──────────────────────────────────────────────────────────────────────────────
# Debug PDF: overlay table bboxes for manual verification
# ──────────────────────────────────────────────────────────────────────────────

def write_debug_pdf(
    doc: DocumentOutput,
    pdf_path: str,
    output_path: str,
) -> str:
    """
    Overlay detected table bboxes onto the original PDF for manual verification.

    Draws each table's magnet-aligned bbox as a green rectangle with label.
    Only page(s) that actually contain table detections are modified.

    Args:
        doc:         DocumentOutput with tables (each must have table_bbox set).
        pdf_path:    Path to the original PDF file.
        output_path: Path for the output debug PDF (e.g. ``{name}_layout.pdf``).

    Returns:
        The output file path on success.
    """
    import fitz

    # Group tables by page
    page_tables: Dict[int, List[TableOutput]] = {}
    for table in doc.tables:
        page_tables.setdefault(table.page_no, []).append(table)

    if not page_tables:
        _log_debug = __import__('logging').getLogger(__name__)
        _log_debug.warning("No tables to annotate — skipping debug PDF")
        return output_path

    src = fitz.open(pdf_path)
    try:
        for page_no, tables in page_tables.items():
            if page_no < 0 or page_no >= len(src):
                continue
            page = src[page_no]

            for ti, table in enumerate(tables):
                bbox = table.table_bbox
                if bbox == (0, 0, 0, 0):
                    continue  # no valid bbox stored

                rect = fitz.Rect(*bbox)

                # Green rectangle (stroke only)
                page.draw_rect(rect, color=(0, 0.69, 0.28), width=1.5)

                # Small label at top-left corner
                label = f"T{ti+1} | {table.num_rows}r×{table.num_cols}c"
                fontsize = 7
                # Place label just above the table bbox
                label_rect = fitz.Rect(
                    rect.x0,
                    max(0, rect.y0 - fontsize - 4),
                    rect.x0 + len(label) * 3.5,
                    max(0, rect.y0 - 2),
                )
                page.draw_rect(label_rect, color=(0, 0.69, 0.28),
                              fill=(0, 0.69, 0.28), width=0)
                page.insert_text(
                    label_rect.bl + (2, 0),
                    label,
                    fontname="Helv",
                    fontsize=fontsize,
                    color=(1, 1, 1),
                )

        # garbage=1 (basic compaction) NOT 3: the cross-object dedup pass of
        # garbage>=2 is pathologically slow on image-heavy PDFs (observed 167s
        # on a 4MB report vs 0.1s here) for no benefit on this throwaway debug
        # artifact.  deflate stays on so content streams are still compressed.
        src.save(output_path, garbage=1, deflate=True)
    finally:
        src.close()

    return output_path
