"""Assay — High-performance financial report PDF table parser."""

__version__ = "1.0.0"

from assay.models import TableModel
from assay.extractor import AssayExtractor
from assay.detection import detect_tables_on_page
from assay.pipeline import run
from assay.output import (
    CellOutput,
    DocumentOutput,
    PageOutput,
    TableOutput,
    TextOutput,
    write_json,
    write_markdown,
    write_debug_pdf,
)

__all__ = [
    "TableModel",
    "AssayExtractor",
    "detect_tables_on_page",
    "run",
    "CellOutput",
    "DocumentOutput",
    "PageOutput",
    "TableOutput",
    "TextOutput",
    "write_json",
    "write_markdown",
    "write_debug_pdf",
]
