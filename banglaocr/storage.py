"""
SQLite storage for the OCR pipeline. Chosen to match your existing stack
(same as PashuCare) and because it needs zero setup - just a file.

Schema:
  pages       - one row per source scanned image
  line_crops  - one row per segmented column/line crop, in reading order
  ocr_results - one row per (line_crop, engine) OCR attempt, so you keep
                every engine's output rather than overwriting
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass


SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path TEXT NOT NULL,
    source_url TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS line_crops (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id INTEGER NOT NULL REFERENCES pages(id),
    column_index INTEGER NOT NULL,
    line_index INTEGER NOT NULL,
    bbox_x INTEGER, bbox_y INTEGER, bbox_w INTEGER, bbox_h INTEGER,
    UNIQUE(page_id, column_index, line_index)
);

CREATE TABLE IF NOT EXISTS ocr_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    line_crop_id INTEGER NOT NULL REFERENCES line_crops(id),
    engine TEXT NOT NULL,
    text TEXT NOT NULL,
    confidence REAL NOT NULL,
    is_accepted INTEGER DEFAULT 0,   -- 1 if this became the "final" text for the line
    needs_review INTEGER DEFAULT 0,  -- 1 if engines disagreed / confidence was low
    corrected_text TEXT,             -- LLM corrected text
    is_reconstructed INTEGER DEFAULT 0, -- 1 if LLM significantly reconstructed text
    status TEXT DEFAULT 'pending_review', -- 'pending_review', 'verified'
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_line_crops_page ON line_crops(page_id);
CREATE INDEX IF NOT EXISTS idx_ocr_results_line ON ocr_results(line_crop_id);
CREATE INDEX IF NOT EXISTS idx_ocr_results_review ON ocr_results(needs_review);
CREATE INDEX IF NOT EXISTS idx_ocr_results_status ON ocr_results(status);
"""


@contextmanager
def connect(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def insert_page(db_path: str, source_path: str, source_url: str | None = None) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO pages (source_path, source_url) VALUES (?, ?)",
            (source_path, source_url),
        )
        return cur.lastrowid


def insert_line_crop(
    db_path: str, page_id: int, column_index: int, line_index: int, bbox
) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            """INSERT OR IGNORE INTO line_crops
               (page_id, column_index, line_index, bbox_x, bbox_y, bbox_w, bbox_h)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (page_id, column_index, line_index, bbox.x, bbox.y, bbox.w, bbox.h),
        )
        if cur.lastrowid:
            return cur.lastrowid
        # already existed - fetch its id
        row = conn.execute(
            """SELECT id FROM line_crops
               WHERE page_id=? AND column_index=? AND line_index=?""",
            (page_id, column_index, line_index),
        ).fetchone()
        return row[0]


def insert_ocr_result(
    db_path: str,
    line_crop_id: int,
    engine: str,
    text: str,
    confidence: float,
    is_accepted: bool = False,
    needs_review: bool = False,
    corrected_text: str | None = None,
    is_reconstructed: bool = False,
    status: str = 'pending_review'
) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO ocr_results
               (line_crop_id, engine, text, confidence, is_accepted, needs_review, corrected_text, is_reconstructed, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (line_crop_id, engine, text, confidence, int(is_accepted), int(needs_review), corrected_text, int(is_reconstructed), status),
        )
        return cur.lastrowid


def update_review_status(db_path: str, result_id: int, corrected_text: str, status: str = 'verified') -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE ocr_results SET corrected_text = ?, status = ?, needs_review = 0 WHERE id = ?",
            (corrected_text, status, result_id)
        )


def get_page_text(db_path: str, page_id: int, only_accepted: bool = True) -> list[dict]:
    """Reconstruct a page's text in correct reading order (column, then line)."""
    with connect(db_path) as conn:
        query = """
            SELECT lc.column_index, lc.line_index, orr.text, orr.confidence,
                   orr.engine, orr.needs_review, orr.corrected_text, orr.is_reconstructed, orr.status
            FROM line_crops lc
            JOIN ocr_results orr ON orr.line_crop_id = lc.id
            WHERE lc.page_id = ?
        """
        if only_accepted:
            query += " AND orr.is_accepted = 1"
        query += " ORDER BY lc.column_index, lc.line_index"

        rows = conn.execute(query, (page_id,)).fetchall()
        return [
            {
                "column_index": r[0],
                "line_index": r[1],
                "text": r[2],
                "confidence": r[3],
                "engine": r[4],
                "needs_review": bool(r[5]),
                "corrected_text": r[6],
                "is_reconstructed": bool(r[7]),
                "status": r[8],
            }
            for r in rows
        ]


def get_review_queue(db_path: str) -> list[dict]:
    """All OCR results flagged for human review, across all pages."""
    with connect(db_path) as conn:
        rows = conn.execute(
            """SELECT p.id, p.source_path, lc.column_index, lc.line_index,
                      orr.engine, orr.text, orr.confidence, orr.id, orr.corrected_text,
                      lc.bbox_x, lc.bbox_y, lc.bbox_w, lc.bbox_h
               FROM ocr_results orr
               JOIN line_crops lc ON lc.id = orr.line_crop_id
               JOIN pages p ON p.id = lc.page_id
               WHERE orr.needs_review = 1 AND orr.status = 'pending_review'
               ORDER BY p.id, lc.column_index, lc.line_index"""
        ).fetchall()
        return [
            {
                "page_id": r[0],
                "source_path": r[1],
                "column_index": r[2],
                "line_index": r[3],
                "engine": r[4],
                "text": r[5],
                "confidence": r[6],
                "result_id": r[7],
                "corrected_text": r[8],
                "bbox": [r[9], r[10], r[11], r[12]]
            }
            for r in rows
        ]
