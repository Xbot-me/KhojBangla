"""
Polite, rate-limited HTTP client for the metadata crawler. Deliberately slow
by default - this is crawling a small non-profit archive's WordPress site,
not something that needs to be fast. Respects robots.txt crawl-delay if set,
otherwise uses a conservative default.
"""
from __future__ import annotations

import time

import requests

from .robots import RobotsChecker, USER_AGENT


class PoliteSession:
    def __init__(self, base_url: str, min_delay_seconds: float = 2.0, timeout: float = 20.0):
        self.robots = RobotsChecker(base_url)
        self.robots.load()
        crawl_delay = self.robots.crawl_delay()
        self.delay = max(min_delay_seconds, crawl_delay or 0.0)
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})
        self._last_request_time = 0.0

    def get(self, url: str) -> requests.Response | None:
        if not self.robots.can_fetch(url):
            print(f"[skip] robots.txt disallows: {url}")
            return None

        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

        try:
            resp = self._session.get(url, timeout=self.timeout)
            self._last_request_time = time.monotonic()
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            print(f"[error] GET {url} failed: {e}")
            return None
