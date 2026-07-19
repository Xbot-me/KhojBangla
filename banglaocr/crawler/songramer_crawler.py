"""
Crawler for songramernotebook.com's newspaper archive.

Structure (confirmed by manual inspection):
  /newspapers            - one big index page, organized by year/month/paper
                            name, linking to /archives/<id> "month" pages
  /archives/<id>          - either:
                              (a) a month-level page listing one /archives/<id>
                                  link per day, OR
                              (b) a day-level page with a single Google Drive
                                  "view" link (the actual issue)
  Some cells in the index have no link at all (issue not available).

Only extracts metadata + Drive links; never downloads or reproduces article
content. Respects robots.txt and rate-limits via PoliteSession.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup

from . import manifest_db
from .http_client import PoliteSession

BASE_URL = "https://songramernotebook.com/"
SOURCE_SITE = "songramernotebook"

ARCHIVE_LINK_RE = re.compile(r"https://songramernotebook\.com/archives/(\d+)/?$")
DRIVE_LINK_RE = re.compile(r"https://drive\.google\.com/file/d/[\w-]+/view[^\s\"'<>]*")


def discover_archive_links(session: PoliteSession, manifest_db_path: str) -> list[str]:
    """Crawl /newspapers and record every /archives/<id> link found (these are
    'month' level pages - each will be visited later to find day-level pages)."""
    resp = session.get(BASE_URL + "newspapers")
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        if ARCHIVE_LINK_RE.match(a["href"]):
            paper_name = a.get_text(strip=True)
            manifest_db.upsert_discovered(
                manifest_db_path,
                source_site=SOURCE_SITE,
                listing_page_url=BASE_URL + "newspapers",
                detail_page_url=a["href"],
                paper_name=paper_name,
            )
            links.append(a["href"])

    print(f"[songramer] {len(links)} month-level archive links found on /newspapers")
    return links


def visit_archive_page(session: PoliteSession, manifest_db_path: str, url: str) -> None:
    """
    An /archives/<id> page is either a month index (links to more /archives/<id>
    day pages) or a day page (single Drive link). Handle both by inspecting
    what's actually on the page rather than assuming which type it is.
    """
    resp = session.get(url)
    if resp is None:
        return

    soup = BeautifulSoup(resp.text, "html.parser")
    title = soup.title.string.strip() if soup.title else None

    drive_links = [
        DRIVE_LINK_RE.search(a["href"]).group(0)
        for a in soup.find_all("a", href=True)
        if DRIVE_LINK_RE.search(a["href"])
    ]
    if drive_links:
        # Day-level page: record the Drive link directly against this manifest entry
        for drive_url in drive_links:
            manifest_db.upsert_discovered(
                manifest_db_path,
                source_site=SOURCE_SITE,
                listing_page_url=url,
                detail_page_url=drive_url,
                paper_name=title,
            )
            manifest_db.set_drive_url(manifest_db_path, SOURCE_SITE, drive_url, drive_url)
        return

    # Otherwise treat as a month index: find nested /archives/<id> day links
    day_links = [a["href"] for a in soup.find_all("a", href=True) if ARCHIVE_LINK_RE.match(a["href"])]
    for day_url in day_links:
        manifest_db.upsert_discovered(
            manifest_db_path,
            source_site=SOURCE_SITE,
            listing_page_url=url,
            detail_page_url=day_url,
            paper_name=title,
        )
    if day_links:
        print(f"[songramer] {url}: {len(day_links)} day-level pages found")


def crawl(manifest_db_path: str, max_month_pages: int | None = None) -> None:
    manifest_db.init_db(manifest_db_path)
    session = PoliteSession(BASE_URL)

    discover_archive_links(session, manifest_db_path)

    import sqlite3

    conn = sqlite3.connect(manifest_db_path)
    rows = conn.execute(
        "SELECT DISTINCT detail_page_url FROM manifest "
        "WHERE source_site = ? AND status = 'discovered' AND detail_page_url LIKE '%/archives/%'",
        (SOURCE_SITE,),
    ).fetchall()
    conn.close()

    month_urls = [r[0] for r in rows]
    if max_month_pages:
        month_urls = month_urls[:max_month_pages]

    for url in month_urls:
        visit_archive_page(session, manifest_db_path, url)

    # Second pass: visit any newly discovered day-level pages to extract Drive links
    conn = sqlite3.connect(manifest_db_path)
    rows = conn.execute(
        "SELECT DISTINCT detail_page_url FROM manifest "
        "WHERE source_site = ? AND status = 'discovered' AND detail_page_url LIKE '%/archives/%'",
        (SOURCE_SITE,),
    ).fetchall()
    conn.close()

    for (url,) in rows:
        visit_archive_page(session, manifest_db_path, url)


if __name__ == "__main__":
    import sys

    db = sys.argv[1] if len(sys.argv) > 1 else "manifest.db"
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 3  # default: small safe test batch
    crawl(db, max_month_pages=max_pages)
    print(manifest_db.counts_by_status(db))
