"""HTTP layer, built on Scrapling.

Scrapling gives us three things worth having:

  1. Adaptive selectors -- `Selector.css(..., auto_save=True)` remembers where an
     element was and relocates it after the site's markup changes. Debate sites
     are volunteer-maintained and re-theme without notice; a scraper that dies
     on every redesign is a scraper nobody runs twice.
  2. Realistic request fingerprints, so we are not blocked as a crawler while
     doing something the site permits.
  3. A single API across static and browser-rendered pages.

Every request goes through `AccessPolicy.check` first (see policy.py). The
policy gate is deliberately above the fetcher rather than inside it: Scrapling's
stealth modes exist to defeat bot detection, and bot detection is not the same
thing as authorization. Making an unauthorized request harder to detect does not
make it permitted.

Scrapling is an optional dependency. Without it we fall back to urllib, which
handles the plain-file downloads (.docx, .zip, raw git content) that make up
most of the open corpora anyway.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Any

from .policy import AccessPolicy, AccessRefused  # noqa: F401  (re-exported)

USER_AGENT = (
    "cardgraph/0.1 (debate evidence indexer; contact: set CARDGRAPH_CONTACT)"
)


def _contact_ua() -> str:
    contact = os.environ.get("CARDGRAPH_CONTACT", "")
    return f"cardgraph/0.1 (+{contact})" if contact else USER_AGENT


@dataclass
class FetchResult:
    url: str
    status: int
    text: str = ""
    content: bytes = b""
    selector: Any = None  # scrapling Selector when available

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


class Fetcher:
    """Policy-gated fetcher.

    Parameters
    ----------
    policy:
        Required. There is no default-permissive mode.
    stealth:
        Use Scrapling's StealthyFetcher. Only meaningful for sites that block
        ordinary clients while still permitting the access -- e.g. a public
        archive behind an aggressive CDN. Never a workaround for a login.
    """

    def __init__(self, policy: AccessPolicy, stealth: bool = False, timeout: int = 30):
        self.policy = policy
        self.stealth = stealth
        self.timeout = timeout
        self._scrapling = None

    def _backend(self):
        if self._scrapling is not None:
            return self._scrapling
        try:
            if self.stealth:
                from scrapling.fetchers import StealthyFetcher as F
            else:
                from scrapling.fetchers import Fetcher as F
            self._scrapling = F
        except ImportError:
            self._scrapling = False
        return self._scrapling

    def get(self, url: str) -> FetchResult:
        self.policy.check(url)      # refuses before any network call
        self.policy.wait(url)       # per-host rate limit

        backend = self._backend()
        if backend:
            kwargs: dict[str, Any] = {"timeout": self.timeout}
            if not self.stealth:
                kwargs["headers"] = {"User-Agent": _contact_ua()}
            try:
                resp = backend.get(url, **kwargs)
                return FetchResult(
                    url=url,
                    status=getattr(resp, "status", 200),
                    text=getattr(resp, "body", "") or "",
                    content=getattr(resp, "content", b"") or b"",
                    selector=resp,
                )
            except TypeError:
                # signature drift between Scrapling versions -- retry bare
                resp = backend.get(url)
                return FetchResult(url=url, status=getattr(resp, "status", 200),
                                   text=getattr(resp, "body", "") or "",
                                   selector=resp)
            except Exception:
                pass  # fall through to urllib

        req = urllib.request.Request(url, headers={"User-Agent": _contact_ua()})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            raw = r.read()
        text = ""
        try:
            text = raw.decode("utf-8", "replace")
        except Exception:
            pass
        return FetchResult(url=url, status=200, text=text, content=raw)

    def download(self, url: str, dest: str) -> str:
        """Fetch a binary (.docx / .zip) to disk."""
        res = self.get(url)
        payload = res.content or res.text.encode("utf-8", "ignore")
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(payload)
        return dest
