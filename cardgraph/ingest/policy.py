"""Access policy for the fetch layer.

Read this before adding a source.

cardgraph fetches from an explicit allowlist. Everything else is refused at the
library level, not by convention. The reason is specific rather than
decorative: the debate evidence ecosystem runs on a disclosure norm, and the
main caselist wiki gates accounts to enforce reciprocity -- you post your cases,
you get to read others'. A tool that launders around that gate does not just
risk a terms-of-service claim; it defects on the norm that produces the data in
the first place, in a community small enough to notice within a week.

Scrapling (the fetcher underneath) is very good at defeating bot detection.
That capability is orthogonal to authorization: it can make an unauthorized
request *succeed*, which is exactly why the allowlist lives above it rather
than below. `AccessPolicy.check` runs before any fetcher is constructed.

Three tiers:

  OPEN        published for reuse; fetch freely, honor robots.txt and rate limits
  ATTRIBUTED  fetchable, but records must carry the source's attribution
  GATED       requires authorization we do not have -> refused, with the
              contact path for getting it legitimately

If you want caselist data, the supported route is `authorized_session`: you
supply your own credentials for an account you personally hold, the fetch runs
at human pace, and the index stays on your machine. That is a different act
from operating a public scraper, and the code makes you say so explicitly.
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.robotparser
from dataclasses import dataclass, field
from enum import Enum


class Tier(str, Enum):
    OPEN = "open"
    ATTRIBUTED = "attributed"
    GATED = "gated"


@dataclass(frozen=True)
class SourceRule:
    host: str
    tier: Tier
    license: str
    note: str = ""
    contact: str = ""
    min_interval_s: float = 2.0


# The allowlist. Add entries deliberately, with a license you have actually
# read. `license` here is a claim about what the operator published, not legal
# advice -- verify before you redistribute anything.
RULES: dict[str, SourceRule] = {
    "openev.debatecoaches.org": SourceRule(
        host="openev.debatecoaches.org",
        tier=Tier.OPEN,
        license="published for open community use (verify current terms)",
        note="Open Evidence Project. Camp files released for free use. "
             "The primary legitimate full-text corpus.",
        min_interval_s=2.0,
    ),
    "github.com": SourceRule(
        host="github.com",
        tier=Tier.ATTRIBUTED,
        license="per-repository",
        note="Used for ashtarcommunications/caselist-archive and similar "
             "published archives. Prefer `git clone` over HTTP scraping.",
        min_interval_s=1.0,
    ),
    "raw.githubusercontent.com": SourceRule(
        host="raw.githubusercontent.com",
        tier=Tier.ATTRIBUTED,
        license="per-repository",
        min_interval_s=1.0,
    ),
    "web.archive.org": SourceRule(
        host="web.archive.org",
        tier=Tier.ATTRIBUTED,
        license="archived copies; original terms still apply",
        note="Useful for dead wiki seasons. Slow it down; they are a nonprofit.",
        min_interval_s=5.0,
    ),
    "huggingface.co": SourceRule(
        host="huggingface.co",
        tier=Tier.ATTRIBUTED,
        license="per-dataset; OpenDebateEvidence is a research release",
        note="Hosts OpenDebateEvidence (Yusuf5/OpenCaselist and the "
             "deduplicated mirror) -- the caselist corpus published as a "
             "dataset with the OpenCaseList project's blessing. This is the "
             "front door to the data that opencaselist.com gates.",
        contact="Cite arXiv:2406.14657 when you use it.",
        min_interval_s=1.0,
    ),
    "opencaselist.com": SourceRule(
        host="opencaselist.com",
        tier=Tier.GATED,
        license="account required; disclosure-reciprocity norm",
        note="Login-gated on purpose. Not scraped by this tool. For the same "
             "underlying evidence through a published channel, use the "
             "opendebateevidence adapter instead -- it is the front door.",
        contact="Ask the maintainers (Ashtar Communications / Paperless Debate) "
                "for API access, or use authorized_session with your own account.",
    ),
    "opencaselist.paperlessdebate.com": SourceRule(
        host="opencaselist.paperlessdebate.com",
        tier=Tier.GATED,
        license="account required; disclosure-reciprocity norm",
        contact="Ask the maintainers for API access, or use authorized_session.",
    ),
}


class AccessRefused(RuntimeError):
    """Raised when a URL is outside the allowlist or in a gated tier."""


@dataclass
class AccessPolicy:
    """Gate + rate limiter. One instance per ingest run."""

    respect_robots: bool = True
    # Set only when the operator is using credentials for an account they
    # personally hold, for personal use, at human pace. Not a bypass for the
    # allowlist -- it re-permits GATED hosts and nothing else.
    authorized_session: bool = False
    authorized_hosts: frozenset[str] = frozenset()

    _last_hit: dict[str, float] = field(default_factory=dict, repr=False)
    _robots: dict[str, urllib.robotparser.RobotFileParser] = field(
        default_factory=dict, repr=False
    )

    def rule_for(self, url: str) -> SourceRule | None:
        host = urllib.parse.urlparse(url).netloc.lower().split(":")[0]
        if host in RULES:
            return RULES[host]
        # allow subdomains of allowlisted hosts
        for known, rule in RULES.items():
            if host.endswith("." + known):
                return rule
        return None

    def check(self, url: str) -> SourceRule:
        """Raise AccessRefused unless this URL may be fetched. Call before
        constructing any fetcher."""
        rule = self.rule_for(url)
        if rule is None:
            raise AccessRefused(
                f"{url!r} is not on the allowlist. Add a SourceRule in "
                f"cardgraph/ingest/policy.py after reading that site's terms. "
                f"Refusing by default is intentional."
            )
        if rule.tier is Tier.GATED:
            host = urllib.parse.urlparse(url).netloc.lower().split(":")[0]
            if not (self.authorized_session and host in self.authorized_hosts):
                raise AccessRefused(
                    f"{rule.host} is gated: {rule.license}. {rule.contact}\n"
                    f"If you hold an account there yourself, construct "
                    f"AccessPolicy(authorized_session=True, "
                    f"authorized_hosts=frozenset({{'{host}'}})) and supply your "
                    f"own credentials. Do not run that mode as a service for "
                    f"other people."
                )
        if self.respect_robots and not self._robots_ok(url):
            raise AccessRefused(f"robots.txt disallows {url!r}")
        return rule

    def _robots_ok(self, url: str) -> bool:
        parts = urllib.parse.urlparse(url)
        base = f"{parts.scheme}://{parts.netloc}"
        rp = self._robots.get(base)
        if rp is None:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(base + "/robots.txt")
            try:
                rp.read()
            except Exception:
                # Unreachable robots.txt is not consent, but it is also not a
                # prohibition. Allowlisted hosts get the benefit of the doubt.
                self._robots[base] = rp
                return True
            self._robots[base] = rp
        try:
            return rp.can_fetch("cardgraph", url)
        except Exception:
            return True

    def wait(self, url: str) -> None:
        """Block until this host's minimum interval has elapsed."""
        rule = self.rule_for(url)
        interval = rule.min_interval_s if rule else 3.0
        host = urllib.parse.urlparse(url).netloc.lower()
        last = self._last_hit.get(host, 0.0)
        delta = time.monotonic() - last
        if delta < interval:
            time.sleep(interval - delta)
        self._last_hit[host] = time.monotonic()


def explain_allowlist() -> str:
    lines = ["cardgraph fetch allowlist:", ""]
    for rule in RULES.values():
        lines.append(f"  {rule.host}")
        lines.append(f"      tier    : {rule.tier.value}")
        lines.append(f"      license : {rule.license}")
        if rule.note:
            lines.append(f"      note    : {rule.note}")
        if rule.contact:
            lines.append(f"      contact : {rule.contact}")
        lines.append("")
    lines.append("Anything not listed is refused. That is the point.")
    return "\n".join(lines)
