#!/usr/bin/env python3
"""
Assay — High-performance financial report PDF table parser.

Usage:
    # Process a directory of PDFs (default: 12 workers)
    ./assay.py -p /path/to/pdfs -o /path/to/output

    # Custom worker count
    ./assay.py -p /path/to/pdfs -o /path/to/output -w 8

    # Single file
    ./assay.py -p /path/to/file.pdf -o /path/to/output
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Assay — High-performance financial report PDF table parser"
    )
    parser.add_argument(
        "-p", "--path", required=True,
        help="PDF file or directory containing PDFs",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output directory",
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=12,
        help="Number of CPU workers (default: 12)",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device ID (default: 0)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose logging",
    )
    parser.add_argument(
        "--no-json", action="store_true",
        help="Skip JSON output (markdown only, slightly faster)",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU-only mode (disable GPU acceleration)",
    )
    parser.add_argument(
        "--debug-pdf", action="store_true",
        help="Generate layout-annotated PDF with table bboxes for manual verification",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    input_path = Path(args.path)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    # Collect PDF paths
    pdf_files = []
    if input_path.is_file():
        pdf_files.append(input_path)
    else:
        pdf_files = sorted(input_path.glob("*.pdf"))

    total = len(pdf_files)
    if total == 0:
        print("Error: no PDF files found")
        sys.exit(1)

    # ── Single file: process in-process ──────────────────────────────────
    if total == 1:
        from assay.models import TableModel
        from assay.extractor import AssayExtractor
        from assay.output import write_json, write_markdown, write_debug_pdf

        pdf_path = str(pdf_files[0])
        file_stem = pdf_files[0].stem

        model = TableModel(use_gpu=not args.cpu)
        extractor = AssayExtractor(model)
        doc = extractor.extract(pdf_path)

        write_markdown(doc, str(output_path / f"{file_stem}.md"))
        if not args.no_json:
            write_json(doc, str(output_path / f"{file_stem}.json"), indent=2)

        if args.debug_pdf:
            debug_path = str(output_path / f"{file_stem}_layout.pdf")
            write_debug_pdf(doc, pdf_path, debug_path)

        print(f"\nOutput:")
        print(f"  Markdown: {output_path / f'{file_stem}.md'}")
        if not args.no_json:
            print(f"  JSON:     {output_path / f'{file_stem}.json'}")
        if args.debug_pdf:
            print(f"  Debug:    {output_path / f'{file_stem}_layout.pdf'}")
        print(f"  Tables:   {len(doc.tables)}")
        print(f"  Status:   {doc.status}")

        if doc.status == "ERROR":
            print(f"  Error:    {doc.error}")
            sys.exit(1)
        return

    # ── Directory: multi-worker pipeline ─────────────────────────────────
    from assay.pipeline import run

    elapsed = run(
        pdf_dir=str(input_path),
        output_dir=str(output_path),
        num_workers=args.workers,
        use_gpu=not args.cpu,
        no_json=args.no_json,
        debug_pdf=args.debug_pdf,
    )

    if elapsed == 0.0:
        sys.exit(1)


if __name__ == "__main__":
    # Must use spawn for CUDA compatibility
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
