The current checkout contains the source and the uncommitted release changes.
This session does not publish or push it automatically. Review the diff, run the
checks below, then make the repository public or private according to the
licenses of the evidence sources you plan to index.

## Create the repository

```bash
gh repo create cardgraph --private --source=. --remote=origin --push
```

Or create an empty repository on GitHub and push explicitly:

```bash
git remote add origin git@github.com:YOUR-ACCOUNT/cardgraph.git
git push -u origin main
```

Do not commit downloaded evidence, databases, vector caches, API keys, or
personal credentials. The `.gitignore` excludes the normal local data outputs;
review `git status` before staging.

## Before publishing publicly

- Verify every source's license and terms. The code's MIT license does not grant
  rights to redistribute documents indexed by it.
- Keep `opencaselist.com` and other login-gated sources gated; do not add a
  scraper workaround.
- Confirm that the allowlist contains only sources you have permission to fetch.
- Review the generated README examples and remove any claim that is not backed by
  a reproducible test or current source documentation.
- Add repository metadata such as a description, topics, and a security policy
  if the project will accept public issue reports.

## Verification

```bash
python -m compileall -q cardgraph
ruff check --select E9,F63,F7,F82,F401,F811,F841 cardgraph tests
python -m pytest tests/ -q
```

The full test suite requires the dependencies in `pyproject.toml`; in a minimal
checkout, install them first with `python -m pip install -e ".[dev]"` or
`python -m pip install -r requirements.txt`.
