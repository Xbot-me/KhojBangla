"""
Crawler for 1980 Jan 1 newspaper
"""
import os
import sys
import sqlite3
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crawler import manifest_db, songramer_crawler
from crawler.http_client import PoliteSession

def discover_1980_jan_link(session: PoliteSession, manifest_db_path: str) -> list[str]:
    resp = session.get(songramer_crawler.BASE_URL + "newspapers")
    if not resp: return []

    soup = BeautifulSoup(resp.text, "html.parser")
    jan_node = soup.find(string="January")
    if not jan_node:
        print("Could not find January node")
        return []
        
    # January is inside a span inside a p, we need to go to its parent and then next sibling
    curr = jan_node.parent
    if curr.name == 'span':
        curr = curr.parent # p

    links = []
    # traverse forward
    curr = curr.next_sibling
    while curr:
        text = curr.get_text(strip=True) if hasattr(curr, 'get_text') else str(curr).strip()
        if text.startswith("February"):
            break
            
        if hasattr(curr, 'find_all'):
            for a in curr.find_all("a", href=True):
                if songramer_crawler.ARCHIVE_LINK_RE.match(a["href"]):
                    paper_name = a.get_text(strip=True)
                    manifest_db.upsert_discovered(
                        manifest_db_path,
                        source_site=songramer_crawler.SOURCE_SITE,
                        listing_page_url=songramer_crawler.BASE_URL + "newspapers",
                        detail_page_url=a["href"],
                        paper_name=paper_name + " (1980 Jan)",
                    )
                    links.append(a["href"])
                
        curr = curr.next_sibling

    print(f"Found {len(links)} links for 1980 January.")
    return links

def run_1980_crawl(manifest_db_path: str):
    manifest_db.init_db(manifest_db_path)
    session = PoliteSession(songramer_crawler.BASE_URL)

    links = discover_1980_jan_link(session, manifest_db_path)
    if not links:
        print("No links found!")
        return

    # Just visit the first one (Jan 1)
    first_link = links[0]
    print(f"Visiting 1st page: {first_link}")
    songramer_crawler.visit_archive_page(session, manifest_db_path, first_link)

if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "manifest.db"
    run_1980_crawl(db)
    print(manifest_db.counts_by_status(db))
