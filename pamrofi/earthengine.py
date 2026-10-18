"""Earth Engine initialisation that works locally, in CI and on Streamlit Cloud.

Order of attempts:
1. Service-account JSON in the env var ``EE_SERVICE_ACCOUNT_JSON`` (raw JSON text)
   or a path in ``GOOGLE_APPLICATION_CREDENTIALS``.
2. Your personal credentials from ``earthengine authenticate`` / ``ee.Authenticate()``.
"""
from __future__ import annotations

import json
import os

import ee

_initialised = False


def init(project: str | None = None, service_account_json: str | None = None,
         interactive: bool = True) -> None:
    """``interactive=False`` (web app) never opens a browser login; it just raises."""
    global _initialised
    if _initialised:
        return

    sa_json = service_account_json or os.environ.get("EE_SERVICE_ACCOUNT_JSON")
    key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    if sa_json:
        info = json.loads(sa_json)
        creds = ee.ServiceAccountCredentials(info["client_email"], key_data=sa_json)
        ee.Initialize(creds, project=project or info.get("project_id"))
    elif key_path and os.path.exists(key_path):
        with open(key_path, encoding="utf-8") as fh:
            info = json.load(fh)
        creds = ee.ServiceAccountCredentials(info["client_email"], key_file=key_path)
        ee.Initialize(creds, project=project or info.get("project_id"))
    else:
        try:
            ee.Initialize(project=project)
        except Exception:  # noqa: BLE001 – first run on a new machine
            if not interactive:
                raise
            ee.Authenticate()
            ee.Initialize(project=project)
    _initialised = True
