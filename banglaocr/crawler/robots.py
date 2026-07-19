"""
Checks robots.txt at runtime before crawling anything - never hardcode an
assumption about what's allowed. If robots.txt can't be fetched, fail closed
(treat as disallowed) rather than assuming permission, and let a human decide.
"""
from __future__ import annotations

import urllib.robotparser
from urllib.parse import urlparse

USER_AGENT = "BanglaArchiveResearchBot/0.1 (contact: set-your-email-here@example.com)"


class RobotsChecker:
    def __init__(self, base_url: str, user_agent: str = USER_AGENT):
        parsed = urlparse(base_url)
        self.robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        self.user_agent = user_agent
        self._parser = urllib.robotparser.RobotFileParser()
        self._loaded = False
        self._load_failed = False

    def load(self) -> None:
        try:
            self._parser.set_url(self.robots_url)
            self._parser.read()
            self._loaded = True
        except Exception as e:
            print(f"[robots] Could not fetch {self.robots_url}: {e}. "
                  f"Failing CLOSED - treating all URLs as disallowed until this is resolved.")
            self._load_failed = True

    def can_fetch(self, url: str) -> bool:
        if not self._loaded and not self._load_failed:
            self.load()
        if self._load_failed:
            return False
        return self._parser.can_fetch(self.user_agent, url)

    def crawl_delay(self) -> float | None:
        if not self._loaded:
            return None
        delay = self._parser.crawl_delay(self.user_agent)
        return float(delay) if delay else None
