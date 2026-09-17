"""Release metadata and documentation checks."""

from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_mit_license_is_present_and_attributed_to_the_project():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert license_text.startswith("MIT License")
    assert "Card Scraper contributors" in license_text
    assert "WITHOUT WARRANTY" in license_text


def test_readme_describes_the_local_web_application():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "local-first web application" in readme
    assert "not a hosted public website" in readme
    assert "cardgraph serve" in readme
    assert "[MIT License](LICENSE)" in readme
