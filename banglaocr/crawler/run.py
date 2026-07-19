"""
Top-level orchestrator. Runs the full chain with deliberately SMALL default
limits - this is meant to be run repeatedly in small batches while you
watch what happens, not as a one-shot "grab everything" script.

Usage:
    python3 crawler/run.py songramer   --manifest-db manifest.db --max-pages 2
    python3 crawler/run.py liberationwar --manifest-db manifest.db --max-pages 1
    python3 crawler/run.py download    --manifest-db manifest.db --out downloads --limit 10
    python3 crawler/run.py ingest      --manifest-db manifest.db --ocr-db ocr.db --limit 10
    python3 crawler/run.py status      --manifest-db manifest.db
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler import manifest_db, drive_downloader, liberationwar_crawler, songramer_crawler
from crawler.ingest import ingest_pending


def main():
    parser = argparse.ArgumentParser(description="Bangla newspaper archive crawler/downloader/OCR chain")
    sub = parser.add_subparsers(dest="command", required=True)

    p_song = sub.add_parser("songramer", help="Crawl songramernotebook.com metadata")
    p_song.add_argument("--manifest-db", default="manifest.db")
    p_song.add_argument("--max-pages", type=int, default=3, help="month-level pages to visit (keep small while testing)")

    p_lib = sub.add_parser("liberationwar", help="Crawl liberationwarbangladesh.org metadata")
    p_lib.add_argument("--manifest-db", default="manifest.db")
    p_lib.add_argument("--max-pages", type=int, default=1, help="listing pages to visit (keep small while testing)")
    p_lib.add_argument("--max-posts", type=int, default=5, help="monthly posts to open for Drive links")

    p_dl = sub.add_parser("download", help="Download pending Drive files (slow, rate-limited)")
    p_dl.add_argument("--manifest-db", default="manifest.db")
    p_dl.add_argument("--out", default="downloads")
    p_dl.add_argument("--limit", type=int, default=10)
    p_dl.add_argument("--delay", type=float, default=5.0)

    p_ing = sub.add_parser("ingest", help="Run downloaded files through the OCR pipeline")
    p_ing.add_argument("--manifest-db", default="manifest.db")
    p_ing.add_argument("--ocr-db", default="ocr.db")
    p_ing.add_argument("--limit", type=int, default=None)

    p_status = sub.add_parser("status", help="Show manifest counts by status")
    p_status.add_argument("--manifest-db", default="manifest.db")

    args = parser.parse_args()

    if args.command == "songramer":
        songramer_crawler.crawl(args.manifest_db, max_month_pages=args.max_pages)
    elif args.command == "liberationwar":
        liberationwar_crawler.crawl(args.manifest_db, max_listing_pages=args.max_pages, max_posts=args.max_posts)
    elif args.command == "download":
        drive_downloader.download_pending(args.manifest_db, args.out, limit=args.limit, delay_seconds=args.delay)
    elif args.command == "ingest":
        ingest_pending(args.manifest_db, args.ocr_db, limit=args.limit)
    elif args.command == "status":
        print(manifest_db.counts_by_status(args.manifest_db))


if __name__ == "__main__":
    main()
