# Changelog

## Unreleased

- Recover crashed or interrupted background imports as durable failed jobs.
- Make source downloads, ZIP extraction, and analysis/LLM caches atomic and self-healing.
- Add `/api/health` readiness diagnostics and coalesce duplicate source refreshes.
- Preserve last-known-good evidence when refreshes fail or return no supported files.

## 0.2.0 — Public beta

Card Scraper is now a local-first debate research application rather than a
plain document scraper.

- Search the read text of debate cards with exact, hybrid, and Smart retrieval.
- Preserve provenance, citation health, evidence traceability, and source freshness.
- Import public releases in the browser with background progress and cancellation.
- Refresh, remove, and monitor indexed sources without losing the last-known-good evidence.
- Share searches, cards, and contention/block detail links.
- Pin cards to a local research board with ordering, private notes, and cited Markdown export.
- Export and restore hash-verified portable workspaces from the CLI or browser, including source files and browser state.
- Preflight workspace bundles before restore so replacement is explicit and reviewable; stale board pins are reported and removed safely.
- Include the MIT license in source distributions and built wheels.
- Run the complete test, lint, compilation, and packaging gate on Python 3.10–3.12.

This release remains intentionally local-first. It is not a hosted multi-user
service and should not be exposed beyond localhost without authentication and
careful review of the indexed evidence licenses.
