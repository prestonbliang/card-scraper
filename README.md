# Card Scraper

Search, browse and **stress-test** debate evidence.

Card Scraper (the Python package remains `cardgraph` for compatibility) parses Verbatim-style `.docx` into structured cards, indexes them on the text
that actually gets read aloud, links `AT:` blocks to what they answer across
every file you own, and then reads your positions the way a prepared opponent
would: reconstructing each internal link chain, marking the steps no card
supports, and naming the answers you have no block for.

```bash
pip install -r requirements.txt

python -m cardgraph.cli ingest local ./my-files        # your own .docx/.html
python -m cardgraph.cli ingest online \
    https://openev.debatecoaches.org/  # public index or direct file
python -m cardgraph.cli ingest-opendebate --query "data center" --limit 3000
python -m cardgraph.cli graph                          # build answer edges
python -m cardgraph.cli analyze --owner Greenhill --top 5   # find what breaks
python -m cardgraph.cli serve                          # http://127.0.0.1:8000
```

---

## The four ideas this is built on

### 1. Index the read text, not the document

A card's body is the full quoted passage. The *read text* is the underlined
subset the debater actually says. These are very different documents, and almost
every naive evidence search indexes the wrong one — so a search for "ratepayer
transfer" returns a card whose underlined portion is about something else, because
the phrase appeared in unread context.

Extracting it is harder than it sounds and **the failure is silent**. Verbatim
applies underlining through character styles rather than direct run formatting,
so a parser that trusts `run.font.underline` reads roughly zero percent of a real
file as marked. It still emits cards. It still prints a count. It is just useless.

`parse/styles.py` resolves marks through the full inheritance chain — direct
properties, then the character style and its `basedOn` ancestors, then the
paragraph style — and every ingest reports read-text **coverage**, so a silent
miss shows up as a number rather than as bad search results three weeks later.

Underlined spans are also non-contiguous: the debater cuts *through* the
paragraph. Concatenating them welds clauses into sentences the author never
wrote, so fragments are stitched with an explicit elision marker and the seams
stay visible.

### 2. Block names are a graph

`AT: Ratepayer Harm` is not a heading. It is an edge pointing at every block
named `Ratepayer Harm` anywhere in your corpus. Once those edges exist, "show me
every answer to my ratepayer contention" is a query instead of a memory test.

Matching is fuzzy and confidence-scored, because `AT: Ratepayer Harm`,
`AT Ratepayer DA` and `A/T: Rate Payer` all mean the same thing and none of them
string-match. The UI shows the confidence rather than pretending the graph is
clean. It is not clean. Nothing built on volunteer-authored headings ever is.

### 3. The analysis must not be able to lie

The analysis layer finds weaknesses — some computed, some read out by a model.
Model-backed criticism of a debate position is *fluent and plausible whether or
not it is true*, which makes an invented citation more dangerous than no analysis
at all: a debater acts on it and walks into a round holding an argument that is
not in their file.

So every model claim is grounded and verified. Cards are presented as `[C1]`…
`[Cn]` rather than as hex ids (models copy short labels reliably; a mis-copied
hex id is indistinguishable from a fabrication). Every citation is resolved
against the presented range. Every quote is checked as a normalized substring of
the card it is attributed to — including the case where the words exist in the
corpus but not in the cited card. Findings that cannot be backed are **demoted
and marked, not silently dropped**, and every run reports a grounding rate.

Full detail, including two bugs where the tool confidently answered a question it
could not actually answer: **[ANALYSIS.md](ANALYSIS.md)**.

### 4. Access is a policy decision, not an accident

`ingest/policy.py` holds an explicit allowlist; everything else is refused at the
library level, before any fetcher is constructed. The fetch layer is built on
[Scrapling](https://github.com/D4Vinci/Scrapling), which is very good at
defeating bot detection — and bot detection is not authorization, so the gate
sits *above* the fetcher rather than inside it.

OpenCaseList is marked `GATED` and is not scraped. Its login enforces a
disclosure-reciprocity norm, and a tool that launders around it defects on the
norm that produces the data. But the data is available through the front door
anyway — see below.

---

## Where the evidence comes from

| source | what it is | command |
| --- | --- | --- |
| **OpenDebateEvidence** | ~3.5M cards from OpenCaseList, published as a research dataset *with the project's blessing*, PII-anonymized ([arXiv:2406.14657](https://arxiv.org/abs/2406.14657)) | `ingest-opendebate` |
| Open Evidence Project | camp files released for free community use | `ingest openev [url]` or `ingest online <public index-or-file-url>` |
| public case release | an allowlisted HTML index or direct `.docx`, `.docm`, `.htm`, `.html`, or `.pdf` URL | `ingest online <url>` |
| `caselist-archive` | ~40,000 archived caselist wiki pages, 2013 onward | `ingest caselist-archive` |
| your own files | no network, no questions | `ingest local <dir>` |

`ingest online` accepts a public HTML index or a direct supported file URL. ZIP
releases are unpacked safely, and PDFs are imported conservatively one page at a
time because PDF layout does not reliably preserve card boundaries or read
marking. Only supported debate documents are retained, and source URL/license
metadata is stored with every imported file. Links to unknown or login-gated
hosts are refused by the allowlist before a request is made.
Use `--license` to record the source's attribution note and `--stealth` only when
the public source permits access but its CDN blocks ordinary clients; stealth is
not an authorization bypass.

```bash
python -m cardgraph.cli ingest online https://openev.debatecoaches.org/releases/ \
    --limit 500 --license "Open Evidence Project terms; verify before redistribution"
python -m cardgraph.cli search "households pay for transmission" \
    --source openev --side aff
python -m cardgraph.cli search --smart "Please find cards about household costs"
python -m cardgraph.cli catalog  # reviewed sources and their access boundaries
```

Search also has a **Smart** mode in the web UI and at `/api/smart-search`. It
locally removes conversational framing and tries a few focused debate synonyms,
then fuses only cards actually present in your index. It is intentionally not a
chatbot: no model key is required, no evidence is generated, and the API returns
the exact variants used so every result remains explainable and citable. Each hit
also reports whether it came from exact lexical terms, semantic ranking, or both.
Each result carries a small confidence badge: **High confidence** means exact
terms were found and lexical retrieval ranked the card near the top. Matching
terms are highlighted in the read text so a debater can scan why a result
matched before opening it. **Strong
match** means lexical and semantic signals support it; **Explore match** means
it was found semantically and should be verified against the card before use.
These are retrieval signals, not truth scores. Search URLs are shareable: the
query, Smart toggle, precision mode, side/source/block filters, and current view
are encoded in the address bar. Use **Copy link** to send a teammate the exact
search state; browser Back/Forward restores it without losing the result. Cards
and contention/block details have their own **Copy link** button too, so a
shared URL can open the exact evidence or argument node directly. The browser also keeps the last six searches locally for quick reuse; `/` focuses search and
`Esc` clears it without sending anything to a server. Use **Pin** on any result to
send it to the local **Board**, where affirmative, negative, and unclassified
cards are compared side by side. **Export cited brief** downloads a Markdown
review packet with each card's citation, provenance, and read text; it does not
upload your evidence anywhere.


The API exposes the same provenance through `GET /api/sources`; `/api/search`
returns `source_title`, `source_origin`, and `source_url` for each hit.

```bash
python -m cardgraph.cli ingest-opendebate \
    --query "data center" --year-min 2021 --event cx --min-duplicates 3 --limit 5000
```

`--min-duplicates` is a quality filter: `duplicateCount` is how many teams
independently thought a card was worth reading.

> Needs network access to huggingface.co. Some sandboxed and corporate networks
> block it; the failure is a clean connection error rather than a silent empty
> ingest. The row→Card mapping is tested against a schema-faithful fixture
> (`tests/fixtures/opendebate_rows.jsonl`) so it is verified without network.

`python -m cardgraph.cli policy` prints the fetch allowlist and why each entry is there. `python -m cardgraph.cli catalog` prints reviewed source profiles, including public/attributed/gated boundaries; a catalog entry is not a blanket redistribution license.

---

## What `analyze` gives you

```
positions analyzed : 12   (model pass on 5)
findings           : 34   critical 3 / major 18
model              : claude-cli   [14 call(s), 2 cached, 41,220 in / 9,113 out]
grounding          : 16/16 model findings cite a real card (100%)

[aff] Contention Two -- Fossil Lock-In
     4 cards · 3 sources · 2029-2031 · support 0.68 · exposure 0.71
     chain:
       ok   1. Large-load contracts are firmed by existing gas units  [05a207a3]
       WEAK 2. This pattern is systemic rather than anecdotal
            "every contract we reviewed" gestures at generality but gives no
            sample size or scope
       GAP  3. Deferred retirements produce emissions inside the transition window
            No card connects deferral to any downstream impact.
 !! Contention Two: step 3 is missing — emissions inside the transition window
     fix: Cut a card for: deferred retirements produce emissions inside the window

ANSWERS YOU DO NOT HAVE
 [critical] vs Contention Two -- Fossil Lock-In
   AT: Causal Direction Reversed
   The card shows contracted plants and deferred retirements co-occur; it never
   shows the contract caused the deferral rather than utilities retaining a unit
   for reliability and then selling firm capacity off it.
   -> NOT IN YOUR FILES
      go cut: "gas plant retirement deferral decision timeline before large load contract"
```

Runs without a model too — `--no-llm` keeps the deterministic checks, which are
complete, free, offline, and still find real problems. The report says which
analyses were skipped rather than quietly returning less.

**Providers**, auto-detected in order: `ANTHROPIC_API_KEY` (SDK, forced tool
calls), then the `claude` CLI (no API key needed — reuses an existing Claude Code
login). Override with `CARDGRAPH_LLM=anthropic|claude-cli|none`.

---

## Validated against the real archive

Everything above was built against a fixture I wrote myself, which proves
nothing. So it was run against the published caselist archive — first 992 files,
then all five seasons: **22,221 files, 11,643 ingested, 166,816 cards, 2.5GB.**

At 2,000 cards, five things broke, all of them silently.

| what broke | why it mattered |
| --- | --- |
| The archive is 40,000 `.htm` files, not `.docx` | The `caselist-archive` adapter globbed `*.docx`, found nothing, and reported success. A documented source ingested zero cards. |
| Caselist pages aren't card files | They disclose the **first and last lines** of a card with `AND` marking the omitted middle. No body, no underlining. Needed its own parser and a `disclosed_only` flag, or every archive card gets a true and useless "nothing is underlined" finding. |
| "The" and "I" were the top two authors | The HTML parser took line 0 as the cite for any multi-line paragraph, eating the first line of every analytic ("I affirm."). 58 cards attributed to pronouns, quietly corrupting every source-concentration finding. |
| `self_contradiction` flagged half the canon | It assumed one owner's files. Across 789 schools it reported Bostrom, Baudrillard and 40 others as "cited on both sides" — true of the community, meaningless as advice. Now scoped by team: 40+ → 16, each real. |
| Read-text coverage was 13% on real `.docx` | Not a parser bug, and proving that took an independent OOXML audit: 22 of 34 real uploads contain **no highlighting at all** (open-source disclosures are often plain speech docs), 10 are marked and extracted correctly, 2 are marked at 4–8% density and tracked closely. The warning now distinguishes "no highlighting here" from "we may have missed this file's convention" — opposite problems that look identical in a coverage number. |

What worked unchanged: the HTML parser hit 100% read-text and 82% author
coverage; duplicate detection found real evidence recut across schools (Woller
97 tagged differently by Apple Valley and Woodlands College Park; Fai 2001 by
Flower Mound and Katy Taylor); the model pass ran over 406 positions with
**22/22 findings citing a real card** at $0.78.

At 166,000 cards, five more did — none of which the small run could have shown.

| what broke | why it mattered |
| --- | --- |
| A `.docx` took **4.4 seconds** to parse | 8.6 hours for the corpus. `resolve_marks` was 93% of it: python-docx re-resolves the same styles once per run (11,137 lookups, 631,000 attribute reads for one file). The style table is invariant per document, so `StyleIndex` now flattens it once. **4,368ms → 429ms.** Verified by diffing output against the old parser over 3,618 real cards — the first attempt differed on 2%, because it applied style-*name* implication to paragraph styles where the original applied it only to character styles. Now byte-for-byte identical. |
| The search index refit on **every process start** | ~4 minutes, paid by every `search` and every `serve`. Now cached and fingerprinted on card ids: **warm start 6s.** The fingerprint is exact rather than a count, because deleting one card and adding another leaves a count unchanged — and a stale index is search that silently cannot see your newest evidence. |
| Indexing a **small** file crashed | `max_df=0.85` prunes terms in most documents; on six similar cards it prunes *everything* and sklearn raises. A debater indexing their own file — the most likely first run — got a traceback instead of a search index. |
| The sidebar rendered **11,643 root nodes** | One per ingested file, all into the DOM at once. Now paginated, sorted by card count, with a filter — and the filter searches the whole outline, because the root level is filenames and scoping a search for "Framework" to it returns nothing while the corpus holds 612 matching blocks. |
| "7,110 unparseable" on a healthy run | 7,094 were pre-2011 ndtceda exports: unstructured `<br>`-separated text with no headers or cite markup, where parsing by guesswork would invent tag and body boundaries. `UnsupportedFormat` now separates a clean skip from a real failure, so genuine failures are not buried in the count. |

Ingest is resumable and reports rate and ETA, because a 58-minute run with no
output is indistinguishable from a hang and an interruption at file 20,000 meant
starting over.

Regression tests for every one of these live in
`tests/test_caselist_html.py::TestRealDataRegressions`,
`tests/test_authors.py`, and `TestIndexPersistence` /
`TestSmallCorpusRobustness` in `tests/test_analysis.py`.

**Scope your own files.** Every check asks about *your* evidence. Unscoped over
the archive that is 46,712 positions across 11,643 teams and 129,410 findings
about other people — all true, none actionable. `--owner Greenhill` gives 752
positions and 2,315 findings in 1.5s instead of 99s. See
[ANALYSIS.md §11](ANALYSIS.md) for why the first implementation of that filter
silently leaked other schools' data into your report.

**At full scale:** ingest 11,643 files in 58 min · graph build 2m34s (40,518
answer edges, 31,011 duplicate edges) · index fit 3m52s once, 6s warm · search
1.8s · UI first paint 0.5s · 23,214 distinct authors.

## Install

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/ -q          # offline regression suite
```

Optional fetch extras include Scrapling (`scrapling curl_cffi patchright browserforge`) for the
stealth fetch path — without them, fetching falls back to `urllib`, which handles
the plain `.docx`/`.zip`/`.pdf` downloads that make up most open corpora. `pypdf`
is installed for conservative page-level PDF extraction. `datasets` supports the
OpenDebateEvidence adapter. `anthropic` enables the SDK provider.

Embeddings default to TF-IDF + SVD (scikit-learn): installs in seconds, runs on
CPU, and on a single-topic corpus — which every debate corpus is — the gap
against a transformer is small. A judgment call, not laziness; swap in
`sentence-transformers` in `index/search.py` when your corpus gets diverse
enough to need it.

## Seed data

`seed/generate_synthetic.py` builds a Verbatim-shaped fixture whose every author,
outlet and quotation is invented, with a banner saying so. Fabricating quotes and
attributing them to real authors produces documents indistinguishable from real
evidence the moment they leave the repo. What the fixture *is* faithful to is the
structure — which is the part the parser has to get right, and the part you
cannot test without it.

`seed/moratorium_outline.yaml` is an argument skeleton for the hyperscale data
center moratorium resolution: contentions, blocks and `AT:` blocks for both
sides, with no card bodies. Build it into an empty Verbatim file and paste your
real evidence in:

```bash
python -m seed.build_outline seed/moratorium_outline.yaml data/corpus/moratorium.docx
```

`analyze` detects unfilled outlines and excludes them from position scoring —
otherwise a skeleton of 33 empty tags buries every real finding under
true-but-useless "this card has no read text".

## Layout

```
cardgraph/
  models.py            Card, OutlineNode, side inference, AT: prefix grammar
  parse/styles.py      run-mark resolution through the style inheritance chain
  parse/docx_card.py   the .docx state machine + fallback parser
  parse/caselist_html.py  archived caselist wiki pages (XWiki HTML)
  parse/cite.py        cite field extraction
  parse/authors.py     canonical author keys (one person, many spellings)
  ingest/policy.py     allowlist, tiers, robots, rate limiting
  ingest/fetch.py      Scrapling wrapper, urllib fallback
  ingest/base.py       local / git / http-index / online / openev adapters
  ingest/opendebate.py OpenDebateEvidence streaming adapter
  index/store.py       SQLite schema, FTS5 over read_text
  index/search.py      BM25 + TF-IDF, RRF fusion, coverage check, cached index
  graph/relate.py      answer edges, duplicate detection, warrant flags
  llm/provider.py      providers, structured output, cache, ledger, salvage
  analysis/schema.py       Finding / ChainLink / GeneratedBlock, kind taxonomy
  analysis/grounding.py    citation resolution + quote verification
  analysis/deterministic.py computable weaknesses
  analysis/llm_analyzers.py chains, warrant critique, block generation
  analysis/engine.py       orchestration, scoring, rendering
  api/main.py          FastAPI, localhost by default
web/index.html         single-file UI: search, browse, analysis
seed/                  synthetic fixture + moratorium skeleton
tests/                 regression suite; the grounding and ingest suites are the important ones
```

## Notes on running it as a service

Everything above assumes localhost. If you expose this: put auth in front of it,
re-read the license of every source in your index first ("I built a search engine
over camp files" and "I republished camp files" are different acts), and do not
run `authorized_session` on anyone's behalf but your own.

## License

MIT for this code. It says nothing about the licenses of documents you index.
