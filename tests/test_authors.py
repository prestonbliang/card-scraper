"""Tests for author canonicalization.

The bug this module exists to fix is a *false negative*, which is the worse
kind here. `source_concentration` fires when a position leans on too few
authors. Counting "Nick Bostrom", "Bostrom" and "Bostrum, Nick. University" as
three people makes a single-source position look diverse, and the check that
was supposed to catch it stays silent. Nothing tells you that happened.

So these tests are weighted toward *not merging* things that differ. An
over-merge is invisible and dangerous; an under-merge just leaves the old
behavior.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cardgraph.parse.authors import (author_key, distinct_authors,
                                     group_authors)


class TestAuthorKey:

    @pytest.mark.parametrize("raw,expected", [
        ("Bostrom", "bostrom"),
        ("Nick Bostrom", "bostrom"),
        ("Bostrom 02", "bostrom"),          # trailing year is not a name
        ("Bostrum, Nick. University", "bostrum"),   # "Last, First" -> Last
        ("Dr. Tommy J. Curry", "curry"),    # honorific and initial dropped
        ("Woller, Gary BYU Prof.", "woller"),
        ("Mumia Abu-Jamal", "abu-jamal"),   # hyphens are part of the name
        ("  ", ""),
        ("et al", ""),                      # nothing left after stopwords
        ("J. A.", ""),                      # bare initials are not a name
    ])
    def test_key(self, raw, expected):
        assert author_key(raw) == expected

    def test_accents_fold(self):
        assert author_key("Žižek") == author_key("Zizek")

    def test_none_and_empty(self):
        assert author_key(None) == ""
        assert author_key("") == ""


class TestGrouping:

    def test_merges_real_variants_of_one_person(self):
        names = ["Nick Bostrom", "Bostrom", "Bostrum, Nick. University"]
        assert len(distinct_authors(names)) == 1

    @pytest.mark.parametrize("a,b", [
        ("Tickell", "Mitchell"),      # different people, similar tails
        ("Tonn", "Tong"),             # too short to risk merging
        ("Harris", "Morris"),
        ("Curry", "Currie"),
        ("Wise", "Wiser"),
        ("Leonardo", "Bernardo"),
    ])
    def test_refuses_to_merge_different_people(self, a, b):
        assert len(distinct_authors([a, b])) == 2

    def test_digit_differences_are_never_typos(self):
        """A digit difference is a deliberate distinction, not a misspelling.

        Caught by a test using synthetic Author0..Author4 names: the fuzzy
        matcher happily merged all five, because they differ by one character.
        Real surnames do not disambiguate themselves with digits.
        """
        names = [f"Author{i}" for i in range(5)]
        assert len(distinct_authors(names)) == 5

    def test_group_authors_maps_every_input(self):
        names = ["Nick Bostrom", "Bostrom", "Tickell"]
        groups = group_authors(names)
        assert set(groups) == set(names)
        assert groups["Nick Bostrom"] == groups["Bostrom"]
        assert groups["Tickell"] != groups["Bostrom"]

    def test_unparseable_names_do_not_collapse_together(self):
        """Two cards with unusable author strings are not evidence of one
        source -- unknown is not sameness."""
        assert distinct_authors(["et al", "J. A."]) == set()


class TestConcentrationUsesCanonicalAuthors:

    def _cards(self, authors):
        return [{"card_id": f"c{i}", "cite_author": a, "cite_raw": f"{a} 20",
                 "cite_pub": "Outlet", "tag": "t", "read_text": "r",
                 "body": "b", "read_ratio": 0.5}
                for i, a in enumerate(authors)]

    def test_one_person_under_three_names_is_flagged(self):
        """The whole point: this used to look like four distinct sources."""
        from cardgraph.analysis.deterministic import check_source_concentration
        cards = self._cards(["Nick Bostrom", "Bostrom",
                             "Bostrum, Nick. University", "Bostrom"])
        found = check_source_concentration("P", cards)
        assert found, "four cards by one author must fire"
        assert found[0].evidence["distinct_sources"] == 1

    def test_genuinely_diverse_position_stays_quiet(self):
        from cardgraph.analysis.deterministic import check_source_concentration
        cards = self._cards(["Bostrom", "Tickell", "Curry", "Leonardo",
                             "Hallberg"])
        assert check_source_concentration("P", cards) == []

    def test_missing_authors_count_separately(self):
        """Cards with no parseable author must not all collapse into one
        source -- that would over-report concentration."""
        from cardgraph.analysis.deterministic import _distinct_sources
        cards = self._cards([None, None, None])
        assert len(_distinct_sources(cards)) == 3
