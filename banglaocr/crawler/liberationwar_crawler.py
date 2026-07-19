"""
Crawler for liberationwarbangladesh.org's newspaper archive section.

Structure (confirmed by manual inspection):
  ?page_id=1033[&paged=N]  - paginated index of monthly posts (one post per
                              paper per month), ~293 pages total
  ?p=<id>                  - a single monthly post, containing one Google
                              Drive "Readable Link" per day of that month

Only extracts metadata + Drive links; never downloads or reproduces article
content. Respects robots.txt and rate-limits via PoliteSession.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup

from . import manifest_db
from .http_client import PoliteSession

BASE_URL = "https://liberationwarbangladesh.org/"
SOURCE_SITE = "liberationwarbangladesh"

DRIVE_LINK_RE = re.compile(r"https://drive\.google\.com/file/d/[\w-]+/view[^\s\"'<>]*")
MONTH_POST_RE = re.compile(r"\?p=(\d+)")


def discover_monthly_posts(session: PoliteSession, manifest_db_path: str, max_pages: int | None = None) -> None:
    """Walk the paginated archive index, recording every monthly post URL found."""
    page = 1
    while True:
        if max_pages and page > max_pages:
            break

        url = BASE_URL + ("?page_id=1033" if page == 1 else f"?page_id=1033&paged={page}")
        resp = session.get(url)
        if resp is None:
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        post_links = set()
        for a in soup.find_all("a", href=True):
            m = MONTH_POST_RE.search(a["href"])
            if m:
                post_links.add(a["href"].split("#")[0])

        if not post_links:
            print(f"[liberationwar] no more posts found at page {page}, stopping")
            break

        for post_url in post_links:
            manifest_db.upsert_discovered(
                manifest_db_path,
                source_site=SOURCE_SITE,
                listing_page_url=url,
                detail_page_url=post_url,
            )

        print(f"[liberationwar] page {page}: {len(post_links)} monthly posts recorded")
        page += 1


def extract_drive_links_from_post(session: PoliteSession, manifest_db_path: str, post_url: str, paper_name: str | None = None) -> int:
    """
    A monthly post contains multiple day-level Drive links inline (not separate
    pages). Each becomes its own manifest row keyed by the Drive URL itself,
    since there's no separate per-day detail page on this site.
    """
    resp = session.get(post_url)
    if resp is None:
        return 0

    soup = BeautifulSoup(resp.text, "html.parser")
    title = soup.title.string.strip() if soup.title else paper_name

    found = 0
    for a in soup.find_all("a", href=True):
        drive_match = DRIVE_LINK_RE.search(a["href"])
        if not drive_match:
            continue
        drive_url = drive_match.group(0)
        day_label = a.get_text(strip=True)  # e.g. "February 01, 2002 (Readable Link)"

        manifest_db.upsert_discovered(
            manifest_db_path,
            source_site=SOURCE_SITE,
            listing_page_url=post_url,
            detail_page_url=drive_url,  # use the drive URL itself as the unique key
            paper_name=title,
            issue_date=day_label,
        )
        manifest_db.set_drive_url(manifest_db_path, SOURCE_SITE, drive_url, drive_url)
        found += 1

    return found


def crawl(manifest_db_path: str, max_listing_pages: int | None = None, max_posts: int | None = None) -> None:
    manifest_db.init_db(manifest_db_path)
    session = PoliteSession(BASE_URL)

    discover_monthly_posts(session, manifest_db_path, max_pages=max_listing_pages)

    # Now visit each discovered monthly post to pull out its day-level Drive links
    import sqlite3

    conn = sqlite3.connect(manifest_db_path)
    rows = conn.execute(
        "SELECT DISTINCT listing_page_url, detail_page_url FROM manifest "
        "WHERE source_site = ? AND status = 'discovered' AND detail_page_url LIKE '%?p=%'",
        (SOURCE_SITE,),
    ).fetchall()
    conn.close()

    if max_posts:
        rows = rows[:max_posts]

    for _, post_url in rows:
        n = extract_drive_links_from_post(session, manifest_db_path, post_url)
        print(f"[liberationwar] {post_url}: {n} day-level Drive links found")


if __name__ == "__main__":
    import sys

    db = sys.argv[1] if len(sys.argv) > 1 else "manifest.db"
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 1  # default: just page 1, be safe
    crawl(db, max_listing_pages=max_pages, max_posts=5)
    print(manifest_db.counts_by_status(db))
