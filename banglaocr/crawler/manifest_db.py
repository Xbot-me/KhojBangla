"""
Manifest DB: tracks every (paper, date) issue discovered while crawling, its
source page URL, its Google Drive link (once found), and whether it's been
downloaded yet. This is what makes the crawl resumable - re-running the
crawler or downloader just skips anything already recorded/downloaded.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS manifest (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_site TEXT NOT NULL,        -- 'liberationwarbangladesh' or 'songramernotebook'
    paper_name TEXT,
    issue_date TEXT,                  -- as text, formats vary by source
    listing_page_url TEXT NOT NULL,   -- WP page this was discovered on
    detail_page_url TEXT,             -- day-level page, if the source has one
    drive_url TEXT,                   -- Google Drive share link, once extracted
    local_path TEXT,                  -- once downloaded
    status TEXT DEFAULT 'discovered', -- discovered | drive_link_found | downloaded | failed
    error TEXT,
    discovered_at TEXT DEFAULT (datetime('now')),
    downloaded_at TEXT,
    UNIQUE(source_site, detail_page_url)
);

CREATE INDEX IF NOT EXISTS idx_manifest_status ON manifest(status);
CREATE INDEX IF NOT EXISTS idx_manifest_site ON manifest(source_site);
"""


@contextmanager
def connect(db_path: str):
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def upsert_discovered(
    db_path: str,
    source_site: str,
    listing_page_url: str,
    detail_page_url: str,
    paper_name: str | None = None,
    issue_date: str | None = None,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO manifest
                 (source_site, paper_name, issue_date, listing_page_url, detail_page_url, status)
               VALUES (?, ?, ?, ?, ?, 'discovered')
               ON CONFLICT(source_site, detail_page_url) DO NOTHING""",
            (source_site, paper_name, issue_date, listing_page_url, detail_page_url),
        )


def set_drive_url(db_path: str, source_site: str, detail_page_url: str, drive_url: str) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """UPDATE manifest SET drive_url = ?, status = 'drive_link_found'
               WHERE source_site = ? AND detail_page_url = ?""",
            (drive_url, source_site, detail_page_url),
        )


def mark_downloaded(db_path: str, manifest_id: int, local_path: str) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """UPDATE manifest SET local_path = ?, status = 'downloaded',
               downloaded_at = datetime('now') WHERE id = ?""",
            (local_path, manifest_id),
        )


def mark_failed(db_path: str, manifest_id: int, error: str) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE manifest SET status = 'failed', error = ? WHERE id = ?",
            (error, manifest_id),
        )


def get_pending_drive_links(db_path: str, limit: int | None = None) -> list[dict]:
    """Entries with a Drive URL found but not yet downloaded."""
    with connect(db_path) as conn:
        query = "SELECT id, paper_name, issue_date, drive_url FROM manifest WHERE status = 'drive_link_found'"
        if limit:
            query += f" LIMIT {int(limit)}"
        rows = conn.execute(query).fetchall()
        return [
            {"id": r[0], "paper_name": r[1], "issue_date": r[2], "drive_url": r[3]}
            for r in rows
        ]


def get_downloaded_not_ocred(db_path: str, ocr_db_path: str) -> list[dict]:
    """
    Cross-reference: files downloaded here but not yet present as a `page`
    in the OCR pipeline's database, so batch ingestion knows what's left to do.
    """
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, local_path, paper_name, issue_date FROM manifest WHERE status = 'downloaded'"
        ).fetchall()

    done_paths = set()
    try:
        ocr_conn = sqlite3.connect(ocr_db_path)
        done_paths = {r[0] for r in ocr_conn.execute("SELECT source_path FROM pages")}
        ocr_conn.close()
    except sqlite3.OperationalError:
        pass  # OCR db not initialized yet - nothing's been processed

    return [
        {"id": r[0], "local_path": r[1], "paper_name": r[2], "issue_date": r[3]}
        for r in rows
        if r[1] not in done_paths
    ]


def counts_by_status(db_path: str) -> dict:
    with connect(db_path) as conn:
        rows = conn.execute("SELECT status, COUNT(*) FROM manifest GROUP BY status").fetchall()
        return dict(rows)
