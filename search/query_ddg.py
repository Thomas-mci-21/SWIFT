"""DuckDuckGo search wrapper as fallback when Serper is out of credits."""

import os
import time
from typing import Optional

from ddgs import DDGS

NO_RESULT_MSG = 'No good search result was found'


class DuckDuckGoSearch:
    """DuckDuckGo search wrapper matching SerperAPI interface."""

    def __init__(self, k: int = 3, **kwargs):
        self.k = k
        proxy = os.environ.get('http_proxy') or os.environ.get('https_proxy')
        self.ddgs = DDGS(proxy=proxy) if proxy else DDGS()

    def run(self, query: str, **kwargs) -> str:
        """Run query and return concatenated snippets."""
        try:
            results = self.ddgs.text(query, max_results=self.k)
            if not results:
                return NO_RESULT_MSG

            snippets = []
            for r in results:
                title = r.get('title', '')
                body = r.get('body', '')
                if body:
                    snippets.append(f"{title}: {body}")
                elif title:
                    snippets.append(title)

            return ' '.join(snippets) if snippets else NO_RESULT_MSG
        except Exception as e:
            return NO_RESULT_MSG
