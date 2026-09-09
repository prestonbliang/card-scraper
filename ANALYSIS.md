# The analysis layer

`cardgraph analyze` reads your files the way a prepared opponent would and tells
you what breaks. This document is why it is built the way it is, including the
parts that were wrong first.

```
cardgraph analyze --top 5              # deterministic + model pass
cardgraph analyze --no-llm             # offline, free, still useful
cardgraph analyze --json report.json   # machine-readable
```

---

## 1. Two analyzers, and why the order matters

**Deterministic checks run first, always, offline.** They compute things that
are true whether or not a model notices them: how many distinct sources carry a
position, whether a tag makes a causal claim the underlined text hedges,
whether an answer block exists that nothing in your files responds to.

**Model-backed checks run second, on a budget**, and are handed the
deterministic findings with an instruction not to repeat them.

That ordering does real work. A model asked to critique a position cold spends
most of its attention rediscovering "this only has two sources" — which is
arithmetic, and which the cheap layer already knows exactly. Told what is
already found, it goes after the things only reading can find: whether the
internal link chain actually closes, what unstated premise the argument needs,
which turn the position is structurally open to.

It also means the tool degrades honestly. With no API key and no `claude` CLI,
`analyze` still runs, still finds real problems, and says plainly in its
warnings which analyses were skipped. It does not quietly return less and let
you assume that was everything.

## 2. The weakness taxonomy

`kind` is a closed vocabulary (`analysis/schema.py:KINDS`). A model allowed to
invent categories produces twenty near-synonyms — "weak warrant", "warrant
issue", "insufficient warrant" — and an ungroupable UI. The enum is in the
schema the model is given, and unknown kinds are dropped on the way back.

Computed:

| kind | fires when |
| --- | --- |
| `source_concentration` | ≤2 distinct authors across ≥4 cards — one indict takes the position |
| `single_card_contention` | a whole contention rests on one card |
| `power_tagging` | tag makes a strong causal claim; read text hedges, or is under 15% of the body, or is empty |
| `qualification_gap` | no credential or outlet parsed on load-bearing cards |
| `recency_decay` | position's newest card lags the corpus by 3+ years |
| `no_read_text` / `thin_read` | nothing underlined, or almost nothing |
| `answer_gap` | blocks in the index attack this and nothing answers them back |
| `self_contradiction` | the same author is cited on both sides of your own files |
| `duplicate_bloat` | the same cut appears repeatedly under different tags |

Model-backed:

| kind | fires when |
| --- | --- |
| `chain_break` | a step the argument needs has no card at all |
| `warrant_mismatch` | the card's reasoning does not establish what the tag asserts |
| `hidden_assumption` | the argument needs an unstated premise |
| `impact_gap` | the chain does not reach the claimed impact |
| `turn_exposure` | the position is structurally open to a specific turn |
| `definitional_weakness` | a key term is doing undefined work |

`self_contradiction` deserves a note: it is cheap to compute and genuinely
round-losing. If your aff cites an author for one claim and your neg cites the
same author for the opposite, a good opponent reads your own evidence back at
you.

## 3. Internal link chains

The single most useful output. The model reconstructs what the position needs
to be true, step by step, and marks each step `supported`, `weak`, or
`missing` — with the cards that carry it.

```
  OK   1. Large-load contracts are firmed by existing gas units   [C1]
  WEAK 2. This pattern is systemic, not anecdotal                 [C1]
       "every contract we reviewed" gestures at generality but gives no
       sample size or scope
  GAP  3. Deferred retirements produce emissions inside the transition window
       No card connects deferral to any downstream impact; the chain stops
       at the fact of deferral.
```

The prompt explicitly asks for steps the position *needs* even where no card
covers them, because a chain listing only the steps you have cards for is a
description of your file rather than an analysis of your argument. The gap is
the finding.

One guard worth calling out: **a step marked `supported` that cites no card is
silently downgraded to `missing`.** A model doing that is contradicting itself,
and the citation is the more trustworthy half of the contradiction.

Missing *internal* links are `critical`; missing first or last steps are
`major`. Ends get conceded in rounds. Internal links never do.

## 4. Grounding: the part that must not be wrong

A model asked to critique a debate position will produce fluent, plausible,
entirely invented criticism if you let it — citing a card that does not exist,
or quoting a sentence the card does not contain. That output is worse than no
output, because it is indistinguishable from the real thing, and a debater who
acts on it walks into a round holding an argument that is not in their file.

Three mechanisms, in `analysis/grounding.py`:

**Citation by index.** Cards are presented as `[C1]`…`[Cn]`, not as 16-hex ids.
Models copy short integer labels reliably and long hex strings unreliably, and a
mis-copied hex id is indistinguishable from a fabricated one. An index outside
the presented range is detectable *exactly* — `resolve(["C99"])` over a two-card
set returns a specific error, not a silent miss.

**Quote verification.** Every quote is checked as a normalized substring of the
cited card. Normalization is deliberately aggressive (NFKD, smart quotes, dashes,
ellipses, punctuation, whitespace) because the model's rendering of a card
legitimately differs from the `.docx`. A token-overlap fallback at 85% tolerates
a dropped stop word. Below that, it is an invention and gets dropped.

Note the second test in that suite: a quote that exists *in the corpus* but not
in the *cited* card is also rejected. Right words, wrong attribution, still
fabrication.

**Demote, do not delete.** A finding that survives with no valid citation is
kept, marked `grounded: false`, capped at 0.3 confidence, and demoted to
`minor`. Deleting it silently would hide a prompt regression; keeping it
unmarked would launder a hallucination. The UI shows it with an `ungrounded`
badge.

Every run reports a grounding rate:

```
grounding : 16/16 model findings cite a real card (100%), 0 quote(s) dropped
```

A sudden drop there is how you notice a prompt regression before it reaches
anyone's speech doc.

## 5. Two bugs worth documenting

Both shipped, both were caught in review, and both were the same class of error:
**reporting a confident answer to a question the system could not actually
answer.**

### Ranked search cannot answer "do I have this?"

Generated blocks are checked against your corpus so the output can distinguish
"you own a card for this, it is just filed elsewhere" from "go cut this". The
first implementation used `SearchEngine.search`.

Ranked retrieval always returns its top k. So every proposed block came back
with four confident-looking hits and the label **"you already have cards for
this"** — including blocks nothing in the corpus supported. The tool was telling
a debater they were covered when they were not, which is the most damaging thing
it could say.

Two corrections, both in `SearchEngine.covers`:

- **A calibrated floor, not a rank.** Raw cosine is comparable across queries —
  an off-topic query scores ~0 — while the fused RRF score is not and cannot
  carry a threshold.
- **Exclude the position's own cards.** They were matching themselves. A card
  inside a position is not a candidate answer *to* that position.

`tests/test_analysis.py::TestCoverage::test_ranked_search_would_have_lied` pins
the difference: for "zebra grazing patterns in the Serengeti", `search` returns
hits and `covers` returns nothing.

### A boolean cannot carry a continuous signal

After the floor was added, matches at 0.41 cosine were still reported as "you
have this". The threshold was doing work a threshold cannot do.

`coverage` is now three-valued — `have` (≥0.60), `partial`, `none` — with the
actual score shown next to every match, and below `SMALL_CORPUS_CARDS` (200) no
result is reported as covered at all, because on a small index the SVD space is
degenerate and every cosine inflates. The report says so in its warnings rather
than quietly reporting inflated confidence.

## 6. Scoring

Two numbers per position, both in [0,1]:

- **support** — card count with diminishing returns (35%), source diversity
  (30%), read-text health (25%), highlighting depth (10%).
- **exposure** — severity-weighted findings, `1 - 1/(1+raw)`. It saturates, so
  twenty minor findings never outrank one critical one, which matches how rounds
  actually go.

**priority** = `0.65·exposure + 0.35·weight`, where weight is card count capped
at six. A broken position nobody runs is not urgent. A broken position carrying
eight cards is. Priority also selects which positions get the paid model pass,
so the budget goes where the stakes are.

## 7. Cost control

A corpus is thousands of positions; calling a model on each would be absurd.

- `--top N` caps the model pass at the N highest-priority positions.
- Every model call is content-addressed and cached on disk. Re-running after a
  prompt edit is a cache miss; re-running unchanged is free.
- The ledger reports tokens and cost per run, printed in the report header.
- The API never triggers a run implicitly. `GET /api/analysis` serves the last
  saved report; spending money requires `POST /api/analysis/run`.

Token counts always print. Dollar figures print only when `CARDGRAPH_PRICE_IN`
and `CARDGRAPH_PRICE_OUT` are set, because hardcoded prices go stale silently
and a wrong cost figure is worse than none.

## 8. Providers

| provider | when | structured output |
| --- | --- | --- |
| `anthropic` | `ANTHROPIC_API_KEY` set | forced tool call — the API validates the shape |
| `claude-cli` | `claude` on PATH | JSON requested in-prompt, extracted and repaired |
| `stub` | tests | replays cassettes keyed on the prompt |

Auto-detected in that order; override with `CARDGRAPH_LLM`.

The forced tool call matters. "Please return JSON" is a request; a forced tool
call is validated against your schema by the API before it comes back. Where
that is unavailable (the CLI path), output goes through `extract_json`, which
handles prose wrapping, markdown fences and trailing commas, then a local
validator, then one repair retry with the specific errors named — telling a
model exactly which field was wrong fixes it far more often than resampling.

**Partial salvage.** `salvage_key` lets one invalid element be dropped from a
list rather than discarding the whole response. Without it, one bad enum value
in a list of eight findings throws away seven good ones you already paid for and
spends another call regenerating them. Used for findings and blocks; *not* used
for chains, where a hole in the middle is not a partial result.

## 9. Testing model-dependent code

Every test runs offline, with no API key, in under two seconds.

`ScriptedProvider` returns queued payloads and records the prompts it saw, so
tests assert on real behavior: that an unknown `kind` is dropped, that prior
findings actually reach the prompt, that `--top 1` produces exactly three calls.

`StubProvider` handles cassette record/replay, keyed on a hash of
`(system, user, schema)`. That key is the point — an edited prompt *misses*
loudly instead of replaying a stale answer to a question you no longer ask.
`test_stub_misses_loudly_on_a_prompt_change` pins that.

The suite's centre of gravity is `TestGrounding`. Those are the tests that check
the system refuses to lie.

## 10. What this does not do

- It does not judge whether an argument is *true* — only whether your cards
  establish what your tags claim.
- It does not write cards. Generated blocks are arguments plus a search query;
  cutting is yours.
- `hedged_body`, `power_tagging` and friends are heuristics with false
  positives. They are prompts to go read the card, and the UI labels them that
  way.
- Model findings are a strong opponent's opinion, not a verdict. They cite the
  cards; go look.
