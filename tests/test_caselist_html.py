"""Tests for the archived-caselist HTML parser.

Written after running the .docx pipeline against the published archive and
discovering two things at once: the archive is 40,000 `.htm` files rather than
`.docx` (so the `caselist-archive` adapter had been silently ingesting nothing),
and caselist pages encode a genuinely different object -- a *disclosure*, which
is the first and last lines of a card with the middle omitted, not a card with
underlining.

The fixture is structurally faithful to a real archived XWiki page and
semantically invented, same discipline as the .docx fixture: real markup, fake
evidence.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cardgraph.parse.caselist_html import is_caselist_page, parse_caselist_html
from cardgraph.parse.docx_card import ELISION, parse_any

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fixtures", "caselist_page.htm")


@pytest.fixture(scope="module")
def parsed():
    return parse_caselist_html(FIXTURE)


class TestDetection:

    def test_recognizes_a_caselist_page(self):
        assert is_caselist_page(FIXTURE)

    def test_rejects_unrelated_html(self, tmp_path):
        p = tmp_path / "blog.html"
        p.write_text("<html><body><h1>hi</h1><p>not a caselist</p></body></html>")
        assert not is_caselist_page(str(p))

    def test_parse_any_dispatches_on_extension(self):
        cards, _, _ = parse_any(FIXTURE)
        assert cards

    def test_parse_any_skips_unsupported_html_as_a_distinct_type(self, tmp_path):
        """The exception type is the contract, not the message.

        An ingest that counts "we do not handle this format" as a parse failure
        reports 7,110 unparseable files on the real archive and looks
        catastrophically broken when nothing is wrong -- and buries the handful
        of genuine failures inside that number.
        """
        from cardgraph.parse.docx_card import UnsupportedFormat
        p = tmp_path / "blog.html"
        p.write_text("<html><body><p>nope</p></body></html>")
        with pytest.raises(UnsupportedFormat) as exc:
            parse_any(str(p))
        assert isinstance(exc.value, ValueError)   # stays catchable as before
        assert exc.value.reason                    # carries a stated reason

    def test_unknown_extension_is_also_unsupported_not_a_failure(self, tmp_path):
        from cardgraph.parse.docx_card import UnsupportedFormat
        p = tmp_path / "notes.pdf"
        p.write_text("x")
        with pytest.raises(UnsupportedFormat):
            parse_any(str(p))


class TestExtraction:

    def test_card_count_excludes_analytics_and_stubs(self, parsed):
        """A bare <p> with no header above it is analytic, not a card, and a
        header whose text is too short to be evidence is not one either."""
        cards, _, _ = parsed
        assert len(cards) == 3

    def test_full_coverage_on_a_well_formed_page(self, parsed):
        _, _, report = parsed
        assert report.read_text_coverage == 1.0
        assert report.cards_without_cite == 0

    def test_and_becomes_the_elision_marker(self, parsed):
        """The caselist 'AND' convention and Verbatim's non-contiguous
        underlining mean the same thing, so they must render identically."""
        cards, _, _ = parsed
        elided = [c for c in cards if ELISION.strip() in c.read_text]
        assert len(elided) == 2
        # and the marker must not leave the literal word behind
        for c in elided:
            assert " AND " not in f" {c.read_text} "

    def test_no_and_marker_when_nothing_was_omitted(self, parsed):
        cards, _, _ = parsed
        single = next(c for c in cards if "does not repeal" in c.read_text)
        assert ELISION.strip() not in single.read_text

    def test_cites_parse(self, parsed):
        cards, _, _ = parsed
        by_author = {c.cite.author: c for c in cards}
        assert set(by_author) == {"Hallberg", "Nakashima", "Bergstrom"}
        assert by_author["Hallberg"].cite.year == 2030
        assert by_author["Bergstrom"].cite.year == 2031

    def test_tildes_are_stripped(self, parsed):
        """XWiki substitutes ~ for the brackets around cite detail."""
        cards, _, _ = parsed
        assert not any("~" in c.cite.raw for c in cards)

    def test_entries_become_outline_nodes(self, parsed):
        cards, root, _ = parsed
        titles = []

        def walk(n):
            titles.append(n.title)
            for ch in n.children:
                walk(ch)

        walk(root)
        assert "Grid Cost AC" in titles
        assert "AT: Grid Cost DA" in titles

    def test_round_context_is_captured_and_drops_NA(self, parsed):
        cards, _, _ = parsed
        ctx = cards[0].round_context
        assert "Tournament: Fictional Invitational" in ctx
        assert "Round: 3" in ctx
        assert "NA" not in ctx     # empty opponent/judge fields are dropped

    def test_side_inferred_from_answer_block(self, parsed):
        cards, _, _ = parsed
        at_card = next(c for c in cards if "does not repeal" in c.read_text)
        assert at_card.side.value == "neg"


class TestDisclosureSemantics:
    """The flag that stops an archive ingest drowning the analysis layer."""

    def test_every_card_is_marked_disclosure_only(self, parsed):
        cards, _, _ = parsed
        assert all(c.disclosed_only for c in cards)

    def test_body_equals_read_text_because_no_more_exists(self, parsed):
        """Not a truncation we chose -- the caselist never published more."""
        cards, _, _ = parsed
        assert all(c.body == c.read_text for c in cards)

    def test_read_health_checks_skip_disclosure_cards(self):
        """Without this, ingesting the 40k-page archive emits one true and
        useless 'nothing is underlined' finding per card, burying everything."""
        from cardgraph.analysis.deterministic import check_read_text_health
        disclosed = [{"card_id": "a", "read_text": "", "read_ratio": 1.0,
                      "disclosed_only": True}]
        assert check_read_text_health(disclosed) == []
        normal = [{"card_id": "b", "read_text": "", "read_ratio": 0.0,
                   "disclosed_only": False}]
        assert check_read_text_health(normal)

    def test_power_tagging_ignores_read_ratio_for_disclosure_cards(self):
        """read_ratio is 1.0 by construction there, so it carries no signal."""
        from cardgraph.analysis.deterministic import check_power_tagging
        card = {"card_id": "a", "tag": "This collapses the economy",
                "read_text": "", "body": "", "read_ratio": 0.01,
                "disclosed_only": True}
        assert check_power_tagging([card]) == []
        card["disclosed_only"] = False
        assert check_power_tagging([card])


class TestUnmarkedSourceDetection:
    """Real archive .docx uploads are frequently plain speech documents with no
    highlighting at all. 'We found nothing' and 'there is nothing to find' look
    identical in a coverage number and need opposite responses."""

    def _report(self, marked, body):
        from cardgraph.parse.docx_card import ParseReport
        r = ParseReport(path="x.docx", cards=10, cards_without_read_text=10)
        r.marked_runs, r.body_runs = marked, body
        return r

    def test_no_marking_is_reported_as_a_source_property(self):
        r = self._report(marked=0, body=80)
        assert r.unmarked_source
        assert r.marking_density == 0.0
        assert "no highlighting" in r.summary()

    def test_marking_present_but_unextracted_is_a_parser_warning(self):
        r = self._report(marked=30, body=80)
        assert not r.unmarked_source
        assert r.marking_density > 0.3

    def test_too_few_runs_to_judge(self):
        """A three-line document tells you nothing either way."""
        r = self._report(marked=0, body=5)
        assert not r.unmarked_source


class TestRealDataRegressions:
    """Bugs that only appeared when the pipeline met the published archive.

    Both were silent: the code ran, produced output, and the output was wrong in
    a way no synthetic fixture would have shown.
    """

    def test_analytic_first_line_is_not_taken_as_a_cite(self):
        """Taking line 0 as the cite for any multi-line paragraph ate the first
        line of every analytic block. On the real archive that made "The" the
        most-cited author in the corpus and "I" the second."""
        from cardgraph.parse.caselist_html import _split_cite_and_body
        cite, read = _split_cite_and_body(
            ["I affirm and value morality.",
             "Only a constitutivist account provides agents with reasons."])
        assert cite == ""
        assert read.startswith("I affirm")

    def test_bolded_lead_is_taken_as_the_cite(self):
        from cardgraph.parse.caselist_html import _split_cite_and_body
        cite, read = _split_cite_and_body(
            ["Woller 97 (Gary Woller, BYU Prof.)", "Moreover, virtually all"],
            strong="Woller 97 (Gary Woller, BYU Prof.)")
        assert cite.startswith("Woller 97")
        assert read == "Moreover, virtually all"

    @pytest.mark.parametrize("raw", [
        "I affirm.",
        "I contend that truth seeking is most consistent with the system",
        "A framework centered around plans is the best form of debate",
        "The modern age has culminated in a structural revolution",
        "This card says something about policy",
    ])
    def test_sentence_openers_are_not_authors(self, raw):
        from cardgraph.parse.cite import parse_cite
        assert parse_cite(raw).author is None

    @pytest.mark.parametrize("raw,author", [
        ("Woller 97 (Woller, Gary BYU Prof.)", "Woller"),
        ("Bostrom 02", "Bostrom"),
        ("Harvard Law Review 2011", "Harvard Law Review"),
        ("Hallberg 30 (Inge Hallberg, Consumer Utility Board)", "Hallberg"),
    ])
    def test_real_authors_still_parse(self, raw, author):
        from cardgraph.parse.cite import parse_cite
        assert parse_cite(raw).author == author

    def test_self_contradiction_is_scoped_to_one_owner(self, tmp_path):
        """Unscoped, this check reports that half the canon is "cited on both
        sides" of an 800-school archive -- true of the community, useless as
        advice, because the aff card is one school's and the neg card another's.
        """
        from cardgraph.analysis.deterministic import check_self_contradiction
        from cardgraph.index.store import Store
        from cardgraph.models import Card, Cite, Source

        st = Store(str(tmp_path / "t.db"))
        st.add_source(Source(source_id="s", path="p", title="t"))

        def card(school, side, tag, ordinal):
            c = Card(tag=tag, cite=Cite(raw="Bostrom 02", author="Bostrom",
                                        year=2002),
                     body="b" * 90, read_text=f"read {ordinal}",
                     source_id="s", source_path=f"/files/{school}/round.docx",
                     ordinal=ordinal)
            c.side = __import__("cardgraph.models", fromlist=["Side"]).Side(side)
            return c

        # two different schools, opposite sides -> not a contradiction
        st.add_cards([card("Flower Mound", "aff", "extinction good", 1),
                      card("Lexington", "neg", "extinction bad", 2)])
        assert check_self_contradiction(st) == []

        # one school, opposite sides -> a real contradiction
        st.add_cards([card("Greenhill", "aff", "extinction good", 3),
                      card("Greenhill", "neg", "extinction bad", 4)])
        found = check_self_contradiction(st)
        assert len(found) == 1
        assert found[0].evidence["owner"] == "Greenhill"
        assert "Greenhill" in found[0].title
