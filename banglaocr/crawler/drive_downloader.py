"""
Downloads individual files from Google Drive share links recorded in the
manifest. Deliberately kept SEPARATE from the WordPress crawler above,
because Drive is a different service with different rules:

  - Google's Terms of Service restrict automated/bulk access to Drive.
  - Shared files have their own per-file download quota; Google will start
    returning "too many users have viewed or downloaded this file" errors
    if a single file gets hit heavily in a short window - this is Google
    throttling the FILE, not necessarily your IP, but it can still affect you.
  - There's no way to fully avoid this risk with anonymous/public share-link
    downloading at real scale. This module minimizes risk (slow, sequential,
    resumable, backs off on errors) but cannot eliminate it.

If this project ever moves from "personal research experiment" to "download
tens of thousands of files regularly", the right long-term move is asking
the archive maintainers for a bulk export, or using the Google Drive API
with your own OAuth credentials and respecting their per-user quota - not
scraping public share links at volume.
"""
from __future__ import annotations

import os
import time

import gdown

from . import manifest_db

DEFAULT_DELAY_SECONDS = 5.0  # deliberately slow - see module docstring


def download_pending(
    manifest_db_path: str,
    output_dir: str,
    limit: int = 20,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    pending = manifest_db.get_pending_drive_links(manifest_db_path, limit=limit)

    if not pending:
        print("Nothing pending to download.")
        return

    print(f"Downloading up to {len(pending)} files, {delay_seconds}s between each...")

    for item in pending:
        dest = os.path.join(output_dir, f"{item['id']}.jpg")
        try:
            result = gdown.download(url=item["drive_url"], output=dest, quiet=False)
            if result is None:
                raise RuntimeError("gdown returned None - download likely blocked or file inaccessible")
            manifest_db.mark_downloaded(manifest_db_path, item["id"], dest)
            print(f"[ok] {item['paper_name']} / {item['issue_date']} -> {dest}")
        except Exception as e:
            # Fallback to thumbnail endpoint for restricted files
            print(f"[warn] gdown failed ({e}), trying thumbnail fallback...")
            try:
                import requests
                import re
                drive_id_match = re.search(r'/d/([a-zA-Z0-9_-]+)', item['drive_url'])
                if drive_id_match:
                    drive_id = drive_id_match.group(1)
                else:
                    drive_id_match = re.search(r'id=([a-zA-Z0-9_-]+)', item['drive_url'])
                    drive_id = drive_id_match.group(1) if drive_id_match else None
                if not drive_id:
                    raise RuntimeError(f"Could not parse drive id from {item['drive_url']}")
                thumb_url = f"https://drive.google.com/thumbnail?id={drive_id}&sz=w5000-h5000"
                r = requests.get(thumb_url, stream=True)
                r.raise_for_status()
                if 'image' not in r.headers.get('Content-Type', ''):
                    raise RuntimeError("Thumbnail endpoint did not return an image")
                with open(dest, 'wb') as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                manifest_db.mark_downloaded(manifest_db_path, item["id"], dest)
                print(f"[ok-fallback] {item['paper_name']} / {item['issue_date']} -> {dest}")
            except Exception as e2:
                manifest_db.mark_failed(manifest_db_path, item["id"], f"gdown: {e}, fallback: {e2}")
                print(f"[fail] id={item['id']} {item['drive_url']}: gdown: {e}, fallback: {e2}")
                print("  (if you see repeated failures here, STOP - this usually means "
                      "Drive is throttling; wait a while before retrying, don't just re-run in a loop)")
            print("  (if you see repeated failures here, STOP - this usually means "
                  "Drive is throttling; wait a while before retrying, don't just re-run in a loop)")

        time.sleep(delay_seconds)


if __name__ == "__main__":
    import sys

    db = sys.argv[1] if len(sys.argv) > 1 else "manifest.db"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "downloads"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    download_pending(db, out_dir, limit=n)
