# cardgraph

Search, browse and **stress-test** debate evidence.

It parses Verbatim-style `.docx` into structured cards, indexes them on the text
that actually gets read aloud, links `AT:` blocks to what they answer across
every file you own, and then reads your positions the way a prepared opponent
would: reconstructing each internal link chain, marking the steps no card
supports, and naming the answers you have no block for.

```bash
pip install -r requirements.txt

python -m cardgraph.cli ingest local ./my-files        # your own .docx
python -m cardgraph.cli ingest-opendebate --query "data center" --limit 3000
python -m cardgraph.cli graph                          # build answer edges
python -m cardgraph.cli analyze --top 5                # find what breaks
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
| Open Evidence Project | camp files released for free community use | `ingest openev <url>` |
| `caselist-archive` | published archives of past seasons | `ingest caselist-archive` |
| your own files | no network, no questions | `ingest local <dir>` |

OpenDebateEvidence is the important one. It is the same evidence the gated wiki
holds, released deliberately as a dataset — and its schema includes a `spoken`
field that is *already* the underlined portion, so the hardest step in the whole
pipeline arrives pre-solved. Filter before you pull:

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

`python -m cardgraph.cli policy` prints the allowlist and why each entry is there.

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

## Install

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # 78 tests, offline, ~2s
```

Optional: Scrapling (`scrapling curl_cffi patchright browserforge`) for the
stealth fetch path — without it, fetching falls back to `urllib`, which handles
the plain `.docx`/`.zip` downloads that make up most open corpora. `datasets` for
the OpenDebateEvidence adapter. `anthropic` for the SDK provider.

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
  parse/cite.py        cite field extraction
  ingest/policy.py     allowlist, tiers, robots, rate limiting
  ingest/fetch.py      Scrapling wrapper, urllib fallback
  ingest/base.py       local / git / http-index / openev adapters
  ingest/opendebate.py OpenDebateEvidence streaming adapter
  index/store.py       SQLite schema, FTS5 over read_text
  index/search.py      BM25 + TF-IDF, RRF fusion, calibrated coverage check
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
tests/                 78 tests; the grounding suite is the important one
```

## Notes on running it as a service

Everything above assumes localhost. If you expose this: put auth in front of it,
re-read the license of every source in your index first ("I built a search engine
over camp files" and "I republished camp files" are different acts), and do not
run `authorized_session` on anyone's behalf but your own.

## License

MIT for this code. It says nothing about the licenses of documents you index.
