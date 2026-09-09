# Getting cardgraph onto GitHub

The repo is committed with full history. I could not create the GitHub repo from
this session — its token is bound to pre-configured repositories and the API
refuses `POST /user/repos` (403). So the last step is yours; it is one command.

## Option A — from the zip (has the `.git` directory already)

```bash
unzip cardgraph.zip && cd cardgraph
gh repo create cardgraph --private --source=. --push
```

No `gh`? Create an empty repo named `cardgraph` on github.com, then:

```bash
git remote add origin git@github.com:crisliangct-source/cardgraph.git
git push -u origin main
```

## Option B — from the bundle

```bash
git clone cardgraph.bundle cardgraph && cd cardgraph
git remote remove origin
gh repo create cardgraph --private --source=. --push
```

## Before you make it public

- Nothing in the repo is secret: no keys, no tokens, no personal files. The
  `.gitignore` excludes `data/`, so no database, no cached model responses, and
  no ingested evidence is committed.
- The only content is code, docs, the synthetic fixture (all invented authors)
  and the moratorium outline (structure only, no card bodies).
- CI runs on push: pytest across Python 3.10/3.11/3.12, ruff, and an end-to-end
  smoke test that runs the documented quickstart from an empty checkout. No
  secrets required — the suite passes with no API key and no network.

## Verified before shipping

The bundle was cloned into a clean directory and, from that clone alone:
79 tests pass in 2.4s, and the quickstart (`generate_synthetic` → `ingest` →
`analyze`) runs end to end.
