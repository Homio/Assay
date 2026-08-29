#!/usr/bin/env python3
"""
Create a Label Studio annotation project for one PDF and import its
pre-annotated tasks (produced by make_tasks.py).

Automates what used to be an ad-hoc sequence of API calls:
  1. Log in with the user's credentials -> obtain the legacy API token.
  2. Create a project titled ``Assay-GT-<name>`` with eval/label_config.xml.
  3. Register a Local Files storage pointing at the render/<stem>/ directory
     (required or the PNGs 404 even with local-file serving enabled -- the
     directory must match a registered storage path).  We do NOT sync: syncing
     would create blank image-only tasks, but we import tasks *with* the
     pipeline's prediction rectangles instead.
  4. Import <stem>_tasks.json via /api/projects/{id}/import.
  5. Verify: task count + one image URL resolves 200.

Usage:
    python3 eval/make_ls_project.py <pdf_stem> "<project_title>"
e.g.
    python3 eval/make_ls_project.py \
        20250331_601857.SH_中国石油_..._e9f55911 "Assay-GT-中国石油"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

from ls_auth import base_url as _ls_base, get_session as _ls_get_session
BASE = _ls_base()
LABEL_ROOT = Path("/home/homio/sda/1111/label")
CONFIG = Path(__file__).resolve().parent / "label_config.xml"


def login() -> requests.Session:
    return _ls_get_session()


def main(stem: str, title: str) -> None:
    render_dir = LABEL_ROOT / "render" / stem
    tasks_file = LABEL_ROOT / "tasks" / f"{stem}_tasks.json"
    if not render_dir.is_dir():
        sys.exit(f"render dir missing: {render_dir}")
    if not tasks_file.is_file():
        sys.exit(f"tasks file missing: {tasks_file}")

    s = login()
    # Legacy Token auth is disabled by default in recent Label Studio, so we
    # drive the REST API with the session cookie + an X-CSRFToken header
    # (Django's CSRF cookie) on every unsafe request.
    h = {"X-CSRFToken": s.cookies.get("csrftoken"), "Referer": BASE}

    # 1. create project
    r = s.post(f"{BASE}/api/projects", headers=h, json={
        "title": title,
        "label_config": CONFIG.read_text(),
    })
    r.raise_for_status()
    pid = r.json()["id"]
    print(f"project created: id={pid} title={title!r}")

    # 2. register local storage (serve PNGs; do NOT sync)
    r = s.post(f"{BASE}/api/storages/localfiles", headers=h, json={
        "project": pid,
        "title": "render",
        "path": str(render_dir),
        "use_blob_urls": False,
    })
    r.raise_for_status()
    print(f"local storage registered: {render_dir}")

    # 3. import tasks with predictions
    tasks = json.loads(tasks_file.read_text())
    r = s.post(f"{BASE}/api/projects/{pid}/import", headers=h, json=tasks)
    r.raise_for_status()
    imp = r.json()
    print(f"import: {imp.get('task_count')} tasks, "
          f"{imp.get('annotation_count')} annotations, "
          f"{imp.get('prediction_count')} predictions")

    # 4. verify one image URL resolves
    sample = sorted(render_dir.glob("page_*.png"))[0].name
    url = f"{BASE}/data/local-files/?d=render/{stem}/{sample}"
    rv = s.get(url, headers=h)
    print(f"image probe: {url} -> http {rv.status_code}")
    print(f"\nOpen {BASE}/projects/{pid}/ to start annotating.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit('usage: make_ls_project.py <pdf_stem> "<project_title>"')
    main(sys.argv[1], sys.argv[2])
