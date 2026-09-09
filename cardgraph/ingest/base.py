"""Source adapters.

An adapter's only job is to produce local .docx paths plus a Source record.
Parsing, indexing and graph building are the same for every source, so adapters
stay small and new ones are cheap to add.
"""

from __future__ import annotations

import glob
import hashlib
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

from ..models import Source
from .fetch import Fetcher
from .policy import AccessPolicy


def _sid(*parts: str) -> str:
    return hashlib.sha1("||".join(parts).encode()).hexdigest()[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Acquired:
    path: str
    source: Source


class SourceAdapter(ABC):
    name: str = "base"
    license: str = "unknown"

    def __init__(self, workdir: str = "data/corpus", policy: AccessPolicy | None = None):
        self.workdir = workdir
        self.policy = policy or AccessPolicy()
        os.makedirs(workdir, exist_ok=True)

    @abstractmethod
    def acquire(self, limit: int | None = None) -> list[Acquired]:
        """Produce local files ready to parse."""


class LocalDirAdapter(SourceAdapter):
    """Files you already have. The path of least drama: point it at a folder of
    camp files or your own Verbatim documents and everything downstream works
    identically to a network source."""

    name = "local"
    license = "as supplied"

    def __init__(self, root: str, **kw):
        super().__init__(**kw)
        self.root = root

    # Both shapes of debate source: Verbatim card files, and archived caselist
    # wiki pages. Globbing only *.docx was a real bug -- the caselist-archive
    # adapter reported success and ingested zero cards, because that archive is
    # 40,000 .htm files.
    PATTERNS = ("*.docx", "*.docm", "*.htm", "*.html")

    def acquire(self, limit: int | None = None) -> list[Acquired]:
        found: list[str] = []
        for pat in self.PATTERNS:
            found += glob.glob(os.path.join(self.root, "**", pat), recursive=True)
        found = sorted(p for p in set(found)
                       if not os.path.basename(p).startswith("~$"))
        if limit:
            found = found[:limit]
        out = []
        for p in found:
            out.append(Acquired(
                path=p,
                source=Source(
                    source_id=_sid(self.name, os.path.abspath(p)),
                    path=p,
                    title=os.path.splitext(os.path.basename(p))[0],
                    origin=self.name,
                    license=self.license,
                    fetched_at=_now(),
                ),
            ))
        return out


class GitRepoAdapter(SourceAdapter):
    """Clone a published archive and index the .docx files in it.

    `git clone` rather than HTTP crawling: it is one request instead of
    thousands, it gets you the whole history, and it is unambiguously the
    access method the host intends.
    """

    name = "git"
    license = "per-repository"

    def __init__(self, repo_url: str, subdir: str = "", **kw):
        super().__init__(**kw)
        self.repo_url = repo_url
        self.subdir = subdir

    def acquire(self, limit: int | None = None) -> list[Acquired]:
        self.policy.check(self.repo_url)
        slug = self.repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        dest = os.path.join(self.workdir, "_git", slug)
        if not os.path.isdir(os.path.join(dest, ".git")):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", self.repo_url, dest],
                check=True, capture_output=True, text=True, timeout=1800,
            )
        else:
            subprocess.run(["git", "-C", dest, "pull", "--ff-only"],
                           capture_output=True, text=True, timeout=600)

        root = os.path.join(dest, self.subdir) if self.subdir else dest
        local = LocalDirAdapter(root, workdir=self.workdir, policy=self.policy)
        acquired = local.acquire(limit=limit)
        for a in acquired:
            a.source.origin = f"{self.name}:{slug}"
            a.source.license = self.license
            a.source.url = self.repo_url
        return acquired


class HttpIndexAdapter(SourceAdapter):
    """Walk an HTML index page and download the .docx / .zip files it links.

    This is the shape of the Open Evidence Project and most camp file releases:
    a directory page of download links. Selectors are declared as a list so a
    site redesign is a config change, not a code change -- and Scrapling's
    adaptive matching gives us a second chance when even that goes stale.
    """

    name = "http-index"

    LINK_SELECTORS = [
        'a[href$=".docx"]',
        'a[href$=".zip"]',
        ".file-list a",
        "table a",
    ]

    def __init__(self, index_url: str, license: str = "unknown",
                 stealth: bool = False, **kw):
        super().__init__(**kw)
        self.index_url = index_url
        self.license = license
        self.fetcher = Fetcher(self.policy, stealth=stealth)

    def discover(self) -> list[str]:
        res = self.fetcher.get(self.index_url)
        urls: list[str] = []
        sel = res.selector
        if sel is not None and hasattr(sel, "css"):
            for css in self.LINK_SELECTORS:
                try:
                    for el in sel.css(css):
                        href = el.attrib.get("href")
                        if href:
                            urls.append(self._absolutize(href))
                except Exception:
                    continue
        if not urls:
            import re
            for m in re.finditer(r'href=["\']([^"\']+\.(?:docx|zip))["\']',
                                 res.text, re.IGNORECASE):
                urls.append(self._absolutize(m.group(1)))
        seen, out = set(), []
        for u in urls:
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out

    def _absolutize(self, href: str) -> str:
        from urllib.parse import urljoin
        return urljoin(self.index_url, href)

    def acquire(self, limit: int | None = None) -> list[Acquired]:
        links = self.discover()
        if limit:
            links = links[:limit]
        out: list[Acquired] = []
        for url in links:
            if not url.lower().endswith(".docx"):
                continue  # zip expansion left to the caller for now
            fname = os.path.basename(url.split("?")[0])
            dest = os.path.join(self.workdir, self.name, fname)
            try:
                self.fetcher.download(url, dest)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! skip {url}: {exc}")
                continue
            out.append(Acquired(
                path=dest,
                source=Source(
                    source_id=_sid(self.name, url),
                    path=dest,
                    title=os.path.splitext(fname)[0],
                    origin=self.name,
                    license=self.license,
                    fetched_at=_now(),
                    url=url,
                ),
            ))
        return out


class OpenEvidenceAdapter(HttpIndexAdapter):
    """Open Evidence Project -- camp files published for free community use.

    The index URL changes shape between seasons; pass the current one in rather
    than hardcoding a guess that will rot.
    """

    name = "openev"

    def __init__(self, index_url: str = "https://openev.debatecoaches.org/", **kw):
        kw.setdefault("license", "published for open community use (verify terms)")
        super().__init__(index_url=index_url, **kw)


class CaselistArchiveAdapter(GitRepoAdapter):
    """Published archive of past caselist seasons."""

    name = "caselist-archive"

    def __init__(self, **kw):
        super().__init__(
            repo_url="https://github.com/ashtarcommunications/caselist-archive",
            **kw,
        )
        self.license = "see repository"
