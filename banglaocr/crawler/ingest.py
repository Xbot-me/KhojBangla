"""
Glue between the crawler's manifest DB and the OCR pipeline built earlier.
Takes every downloaded-but-not-yet-OCR'd file and runs it through
pipeline.process_page(), so the whole chain becomes:

    crawl (metadata)  ->  download (Drive files)  ->  ingest (this script, OCR)

Safe to re-run: skips anything already present in the OCR database.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler import manifest_db
from pipeline import PipelineConfig, process_page


def ingest_pending(manifest_db_path: str, ocr_db_path: str, limit: int | None = None) -> None:
    pending = manifest_db.get_downloaded_not_ocred(manifest_db_path, ocr_db_path)
    if limit:
        pending = pending[:limit]

    if not pending:
        print("Nothing new to OCR.")
        return

    config = PipelineConfig(db_path=ocr_db_path)
    for item in pending:
        print(f"Processing {item['paper_name']} / {item['issue_date']} ({item['local_path']})...")
        try:
            process_page(item["local_path"], config, source_url=item["local_path"])
        except Exception as e:
            print(f"[fail] {item['local_path']}: {e}")


if __name__ == "__main__":
    manifest_path = sys.argv[1] if len(sys.argv) > 1 else "manifest.db"
    ocr_path = sys.argv[2] if len(sys.argv) > 2 else "ocr.db"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    ingest_pending(manifest_path, ocr_path, limit=limit)
