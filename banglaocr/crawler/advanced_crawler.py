"""
Advanced Crawler for songramernotebook.com
Targeted specifically to extract the oldest newspapers (Year 1940) for validation.
"""
import os
import sqlite3
import sys
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crawler import manifest_db, songramer_crawler
from crawler.http_client import PoliteSession

def discover_1940_links(session: PoliteSession, manifest_db_path: str) -> list[str]:
    resp = session.get(songramer_crawler.BASE_URL + "newspapers")
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    
    start_node = soup.find(string="1940")
    if not start_node:
        print("Could not find 1940 section!")
        return []
        
    print("Found 1940 section!")
    
    # Traverse DOM forwards from the "1940" text node
    curr = start_node.parent
    while curr:
        text = curr.get_text(strip=True)
        if text == "1941":
            print("Hit 1941 section, stopping.")
            break
            
        if curr.name:
            if curr.name == 'a' and curr.has_attr('href') and songramer_crawler.ARCHIVE_LINK_RE.match(curr['href']):
                paper_name = curr.get_text(strip=True)
                manifest_db.upsert_discovered(
                    manifest_db_path,
                    source_site=songramer_crawler.SOURCE_SITE,
                    listing_page_url=songramer_crawler.BASE_URL + "newspapers",
                    detail_page_url=curr["href"],
                    paper_name=paper_name + " (1940)",
                )
                links.append(curr["href"])
                
            for a in curr.find_all("a", href=True):
                if songramer_crawler.ARCHIVE_LINK_RE.match(a["href"]):
                    if a["href"] not in links:
                        paper_name = a.get_text(strip=True)
                        manifest_db.upsert_discovered(
                            manifest_db_path,
                            source_site=songramer_crawler.SOURCE_SITE,
                            listing_page_url=songramer_crawler.BASE_URL + "newspapers",
                            detail_page_url=a["href"],
                            paper_name=paper_name + " (1940)",
                        )
                        links.append(a["href"])
                    
        curr = curr.next_sibling
        
    print(f"Found {len(links)} links for 1940.")
    return links

def run_1940_crawl(manifest_db_path: str):
    manifest_db.init_db(manifest_db_path)
    session = PoliteSession(songramer_crawler.BASE_URL)

    month_urls = discover_1940_links(session, manifest_db_path)
    
    for url in month_urls:
        print(f"Visiting month page: {url}")
        songramer_crawler.visit_archive_page(session, manifest_db_path, url)

    conn = sqlite3.connect(manifest_db_path)
    rows = conn.execute(
        "SELECT DISTINCT detail_page_url FROM manifest "
        "WHERE source_site = ? AND status = 'discovered' AND detail_page_url LIKE '%/archives/%'",
        (songramer_crawler.SOURCE_SITE,),
    ).fetchall()
    conn.close()

    for (url,) in rows:
        songramer_crawler.visit_archive_page(session, manifest_db_path, url)

if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "manifest.db"
    run_1940_crawl(db)
    print(manifest_db.counts_by_status(db))
