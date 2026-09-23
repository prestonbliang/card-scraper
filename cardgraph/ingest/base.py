"""Source adapters.

An adapter's only job is to produce local .docx paths plus a Source record.
Parsing, indexing and graph building are the same for every source, so adapters
stay small and new ones are cheap to add.
"""

from __future__ import annotations

import glob
import hashlib
import os
import posixpath
import subprocess
import tempfile
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urljoin, urlsplit

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
    PATTERNS = ("*.docx", "*.docm", "*.htm", "*.html", "*.pdf")

    def acquire(self, limit: int | None = None) -> list[Acquired]:
        found: list[str] = []
        for pat in self.PATTERNS:
            found += glob.glob(os.path.join(self.root, "**", pat), recursive=True)
        found = sorted(p for p in set(found)
                       if not os.path.basename(p).startswith("~$"))
        if limit is not None:
            found = found[:limit]
        out = []
        for p in found:
            absolute = os.path.abspath(p)
            stat = os.stat(p)
            # Include file metadata so resume mode notices a document edited in
            # place instead of silently keeping the old cards for that path.
            version = f"{stat.st_size}:{stat.st_mtime_ns}"
            out.append(Acquired(
                path=p,
                source=Source(
                    source_id=_sid(self.name, absolute, version),
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
    """Walk an HTML index page and download the .docx / .pdf / .zip files it links.

    This is the shape of the Open Evidence Project and most camp file releases:
    a directory page of download links. Selectors are declared as a list so a
    site redesign is a config change, not a code change -- and Scrapling's
    adaptive matching gives us a second chance when even that goes stale.
    """

    name = "http-index"

    LINK_SELECTORS = [
        # Filter suffixes in Python: CSS `$=` misses ordinary links carrying
        # query strings such as `cases.docx?season=2026`.
        "a[href]",
        ".file-list a",
        "table a",
    ]

    def __init__(self, index_url: str, license: str = "unknown",
                 stealth: bool = False, **kw):
        super().__init__(**kw)
        self.index_url = index_url
        self.license = license
        self.fetcher = Fetcher(self.policy, stealth=stealth)

    @staticmethod
    def _is_supported_url(url: str) -> bool:
        return os.path.splitext(urlsplit(url).path)[1].lower() in {
            ".docx", ".docm", ".zip", ".htm", ".html", ".pdf",
        }

    def _allowed_link(self, url: str) -> bool:
        """Keep discovery policy-gated, not only download policy-gated."""
        try:
            self.policy.check(url)
        except Exception:
            return False
        return True

    def discover(self) -> list[str]:
        # Check the index itself even when it is a direct file URL. Otherwise a
        # gated or unknown host would fail later inside a per-file skip handler
        # and look like an ordinary empty source instead of a policy refusal.
        self.policy.check(self.index_url)

        # Accept a direct public file URL as well as an HTML directory page.
        # This makes `ingest online https://.../cases.zip` useful for small
        # releases whose host does not provide a separate index page.
        direct_ext = os.path.splitext(urlsplit(self.index_url).path)[1].lower()
        if direct_ext in {".docx", ".docm", ".zip", ".htm", ".html", ".pdf"}:
            return [self.index_url]

        res = self.fetcher.get(self.index_url)
        if not res.ok:
            raise RuntimeError(f"source index returned HTTP {res.status}: {self.index_url}")
        urls: list[str] = []
        sel = res.selector
        if sel is not None and hasattr(sel, "css"):
            for css in self.LINK_SELECTORS:
                try:
                    for el in sel.css(css):
                        href = el.attrib.get("href")
                        if href:
                            url = self._absolutize(href)
                            if self._is_supported_url(url) and self._allowed_link(url):
                                urls.append(url)
                except Exception:
                    continue
        if not urls:
            import re
            for m in re.finditer(
                    r'href=["\']([^"\']+[.](?:docx|docm|zip|pdf|htm|html)(?:[?#][^"\']*)?)["\']',
                    res.text, re.IGNORECASE):
                url = self._absolutize(m.group(1))
                if self._is_supported_url(url) and self._allowed_link(url):
                    urls.append(url)
        seen, out = set(), []
        for u in urls:
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out

    def _absolutize(self, href: str) -> str:
        return urljoin(self.index_url, href)

    @staticmethod
    def _extension(url: str) -> str:
        return os.path.splitext(urlsplit(url).path)[1].lower()

    @staticmethod
    def _safe_member(name: str) -> str | None:
        """Return a safe archive member path, or None for a traversal entry."""
        normalized = posixpath.normpath(name.replace("\\", "/"))
        if normalized in {"", "."} or normalized.startswith("/") \
                or normalized == ".." or normalized.startswith("../") \
                or normalized.split("/", 1)[0].endswith(":") \
                or "\x00" in normalized:
            return None
        return normalized

    def _source(self, path: str, url: str, title: str,
                source_id: str | None = None) -> Acquired:
        return Acquired(
            path=path,
            source=Source(
                source_id=source_id or _sid(self.name, url),
                path=path,
                title=title,
                origin=self.name,
                license=self.license,
                fetched_at=_now(),
                url=url,
            ),
        )

    MAX_MEMBER_BYTES = 50 * 1024 * 1024
    MAX_ARCHIVE_BYTES = 500 * 1024 * 1024

    def _acquire_zip(self, url: str, limit: int | None = None) -> list[Acquired]:
        """Download and safely unpack only debate-document members.

        ZIPs are common for camp releases. Never extract arbitrary members:
        besides wasting space, a crafted archive could write outside workdir.
        """
        archive_dir = os.path.join(self.workdir, self.name,
                                   hashlib.sha1(url.encode()).hexdigest()[:12])
        archive_path = os.path.join(archive_dir, "source.zip")
        self.fetcher.download(url, archive_path)
        if os.path.getsize(archive_path) > self.MAX_ARCHIVE_BYTES:
            raise ValueError(
                f"archive exceeds {self.MAX_ARCHIVE_BYTES // (1024 * 1024)} MiB limit")
        out: list[Acquired] = []
        seen_members: set[str] = set()
        total_bytes = 0
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                safe = self._safe_member(member.filename)
                if safe is None or member.is_dir() or safe in seen_members:
                    continue
                seen_members.add(safe)
                ext = os.path.splitext(safe)[1].lower()
                if ext not in {".docx", ".docm", ".htm", ".html", ".pdf"}:
                    continue
                if member.file_size > self.MAX_MEMBER_BYTES:
                    continue
                if total_bytes + member.file_size > self.MAX_ARCHIVE_BYTES:
                    break
                if limit is not None and len(out) >= limit:
                    break
                destination = os.path.join(archive_dir, "files", *safe.split("/"))
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                temporary = None
                try:
                    fd, temporary = tempfile.mkstemp(
                        prefix=".card-scraper-member-",
                        dir=os.path.dirname(destination),
                    )
                    with archive.open(member) as src, os.fdopen(fd, "wb") as dst:
                        written = 0
                        while True:
                            chunk = src.read(1024 * 1024)
                            if not chunk:
                                break
                            written += len(chunk)
                            if written > self.MAX_MEMBER_BYTES:
                                raise ValueError("archive member exceeds size limit")
                            dst.write(chunk)
                        dst.flush()
                        os.fsync(dst.fileno())
                    os.replace(temporary, destination)
                    temporary = None
                finally:
                    if temporary and os.path.exists(temporary):
                        os.remove(temporary)
                total_bytes += written
                member_url = f"{url}#{safe}"
                out.append(self._source(
                    destination, member_url,
                    os.path.splitext(os.path.basename(safe))[0],
                    source_id=_sid(self.name, member_url),
                ))
        return out

    def _download_path(self, url: str, extension: str) -> str:
        """Choose a deterministic, collision-free local path for a URL."""
        basename = os.path.basename(urlsplit(url).path) or f"download{extension}"
        digest = hashlib.sha1(url.encode()).hexdigest()[:12]
        return os.path.join(self.workdir, self.name, f"{digest}-{basename}")

    def acquire(self, limit: int | None = None) -> list[Acquired]:
        links = self.discover()
        out: list[Acquired] = []
        for url in links:
            if limit is not None and len(out) >= limit:
                break
            ext = self._extension(url)
            if ext == ".zip":
                try:
                    out.extend(self._acquire_zip(
                        url, None if limit is None else limit - len(out)))
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! skip {url}: {exc}")
                continue
            if ext not in {".docx", ".docm", ".htm", ".html", ".pdf"}:
                continue
            fname = os.path.basename(urlsplit(url).path) or f"download{ext}"
            dest = self._download_path(url, ext)
            try:
                self.fetcher.download(url, dest)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! skip {url}: {exc}")
                continue
            out.append(self._source(dest, url, os.path.splitext(fname)[0]))
        return out


class OnlineEvidenceAdapter(HttpIndexAdapter):
    """Public online speech-and-debate evidence release.

    This is deliberately source-agnostic: point it at a public HTML index or
    direct .docx/.zip/.htm URL after checking the source's terms. The access
    policy still decides which hosts may be fetched.
    """

    name = "online"


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
