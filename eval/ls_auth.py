#!/usr/bin/env python3
"""
Label Studio credentials loader — keeps auth OUT of the repository.

Credentials live in /home/homio/sda/1111/label/.ls-credentials.json
(outside this repository, so they can never be committed):

    {"url": "http://localhost:8080", "email": "...", "password": "..."}

Environment variables LS_URL / LS_EMAIL / LS_PASSWORD are the fallback.
"""
import json
import os
from pathlib import Path

CREDS_FILE = Path("/home/homio/sda/1111/label/.ls-credentials.json")


def _creds():
    if CREDS_FILE.exists():
        c = json.loads(CREDS_FILE.read_text())
        return (c.get("url", "http://localhost:8080"),
                c.get("email", ""), c.get("password", ""))
    url = os.environ.get("LS_URL", "http://localhost:8080")
    email = os.environ.get("LS_EMAIL")
    password = os.environ.get("LS_PASSWORD")
    if not (email and password):
        raise SystemExit(
            f"Label Studio credentials not found: create {CREDS_FILE} "
            "or set LS_EMAIL / LS_PASSWORD")
    return url, email, password


def base_url() -> str:
    return _creds()[0]


def get_session():
    """Logged-in requests.Session for the local Label Studio server."""
    import requests
    url, email, password = _creds()
    s = requests.Session()
    r = s.get(f"{url}/user/login/")
    r.raise_for_status()
    csrf = s.cookies.get("csrftoken")
    r = s.post(
        f"{url}/user/login/",
        data={"csrfmiddlewaretoken": csrf, "email": email,
              "password": password},
        headers={"Referer": f"{url}/user/login/"},
    )
    r.raise_for_status()
    if "sessionid" not in s.cookies:
        raise SystemExit("Label Studio login failed")
    return s
