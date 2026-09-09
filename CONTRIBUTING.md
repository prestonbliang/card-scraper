# Contributing

## Run the suite

```bash
pip install -r requirements.txt
pytest tests/ -q          # 79 tests, offline, no API key, ~2.5s
ruff check .
```

Every test runs without network and without a model provider. Keep it that way.
Model-dependent code is tested against `ScriptedProvider` (fixed payloads) and
`StubProvider` (recorded cassettes) — see `tests/test_analysis.py`.

## Adding a fetch source

Add a `SourceRule` to `cardgraph/ingest/policy.py` **after reading that site's
terms**, then an adapter in `cardgraph/ingest/`. The allowlist refuses unknown
hosts by default and that is deliberate; do not work around it. If a source is
gated, it stays `Tier.GATED`.

## Adding a weakness check

Deterministic checks go in `analysis/deterministic.py` and must return
`Finding`s with real `card_ids`. Add the `kind` to `analysis/schema.py:KINDS`
first — it is a closed vocabulary, and the UI, the tests and the model prompt
all read from it.

Model-backed checks go in `analysis/llm_analyzers.py` and must pass their output
through `ground_finding`. A finding the model cannot tie to a real card gets
demoted, not published as fact. If you add a prompt that skips grounding, the
reviewer's job is to reject it.

## The one rule

Do not let the tool report confidence it has not earned. Two bugs in this
repo's history were both that mistake (see ANALYSIS.md §5): ranked search
answering "do I have this?" and a boolean carrying a continuous similarity. If
you find yourself thresholding a score into a yes/no that a user will act on,
show the score.
