"""Generate a synthetic Verbatim-shaped card file.

Why synthetic and not "a few real cards": fabricating quotations and attributing
them to real authors produces documents that are indistinguishable from real
evidence once they leave this repo. Debate already has a cut-evidence-integrity
problem. So every author, publication and quotation below is invented, the file
carries a SYNTHETIC banner, and the cite years are deliberately implausible.

What this file *is* faithful to is the **structure**: Heading 1-4 outline,
Cite paragraph style, and -- the part that matters -- body paragraphs where only
a subset of runs are underlined/highlighted. That is what the parser must get
right, and it is what you cannot test without a fixture like this.
"""

from __future__ import annotations

import os

from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.shared import Pt

BANNER = (
    "SYNTHETIC TEST FIXTURE -- every author, outlet and quotation in this file "
    "is invented for software testing. Not evidence. Do not read in a round."
)

# (pocket, hat, block, [(tag, cite, [(text, mark)]), ...])
# mark: "" plain | "u" underlined (read) | "h" highlighted | "ue" underline+bold
CORPUS = [
    (
        "Grid Cost DA",
        "1NC",
        "Uniqueness",
        [
            (
                "Regional transmission planners currently absorb new industrial load without rate shock",
                'Almeida 31 (Renata Almeida, Sr. Fellow, Institute for Load Studies, "Interconnection Queues and Cost Allocation," Journal of Fictional Energy Policy, 3/2/31, https://example.invalid/almeida31)',
                [
                    ("Across the six planning regions we surveyed, ", ""),
                    ("large-load interconnection agreements signed since 2029 have shifted upgrade costs onto the requesting customer rather than the general ratepayer base", "u"),
                    (", a reversal of the allocation practice that prevailed a decade earlier. ", ""),
                    ("The result is that ", ""),
                    ("residential rate impacts attributable to industrial interconnection have stayed within two percent", "ue"),
                    (" in every region except one, where a legacy settlement predates the current tariff.", ""),
                ],
            ),
            (
                "Moratorium collapses the cost-allocation reform now underway",
                'Restrepo & Vance 30 (Dolores Restrepo and P. Vance, economists, Center for Applied Tariff Design, "Freeze Effects," 11/14/30)',
                [
                    ("A construction freeze does not pause the regulatory process; it removes the pressure that produced the process. ", ""),
                    ("Cost-allocation reform advanced precisely because utilities faced imminent large-load requests they could not otherwise serve", "u"),
                    (". Remove the requests and ", ""),
                    ("the reform loses its constituency and reverts to the socialized default", "ue"),
                    (", which is the outcome moratorium advocates say they want to prevent.", ""),
                ],
            ),
        ],
    ),
    (
        "Grid Cost DA",
        "1NC",
        "AT: Ratepayer Harm",
        [
            (
                "Ratepayer harm studies conflate correlation with allocation",
                'Okonkwo 30 (T. Okonkwo, Prof. of Regulatory Economics, Fictional State University, "Reading Rate Cases Correctly," 8/1/30)',
                [
                    ("The three studies most often cited for consumer harm compare average residential rates in counties with and without hyperscale facilities. ", ""),
                    ("Neither controls for the fuel-cost pass-through that drove rate increases in the same period", "u"),
                    (", and ", ""),
                    ("when we re-estimate with a fuel-adjusted baseline the facility coefficient is statistically indistinguishable from zero", "ue"),
                    (". The harm finding is an artifact of the specification, not of the load.", ""),
                ],
            ),
        ],
    ),
    (
        "Ratepayer Advantage",
        "1AC",
        "Contention One -- Ratepayer Harm",
        [
            (
                "Large-load buildout is shifting transmission costs onto households",
                'Hallberg 30 (Inge Hallberg, Consumer Utility Board of the Fictional Republic, "Who Pays for the Wires," 6/19/30, https://example.invalid/hallberg)',
                [
                    ("Our review of forty-one rate cases finds a consistent pattern. ", ""),
                    ("Transmission upgrades justified by a single large customer are entered into the general rate base and recovered from all classes", "u"),
                    (", because the interconnection agreement governs only the facility's direct connection, not the network reinforcement behind it. ", ""),
                    ("Households therefore finance capacity they will never use", "ue"),
                    (", a transfer that averages a meaningful share of the residential bill in the affected territories.", ""),
                ],
            ),
            (
                "The transfer is regressive and compounds",
                'Nakashima 31 (Y. Nakashima, PhD, Fictional Institute for Household Energy, "Regressive Recovery," 1/8/31)',
                [
                    ("Because residential rate design is largely volumetric and low-income households spend a larger share of income on electricity, ", ""),
                    ("any socialized transmission cost lands hardest on the households least able to absorb it", "u"),
                    (". ", ""),
                    ("The effect compounds across rate cycles because each approved upgrade raises the base on which the next return is calculated", "ue"),
                    (".", ""),
                ],
            ),
        ],
    ),
    (
        "Ratepayer Advantage",
        "1AC",
        "Contention Two -- Fossil Lock-In",
        [
            (
                "Firm-power procurement for large loads is extending gas plant lifetimes",
                'Iversen 30 (M. Iversen, Fictional Grid Transition Observatory, "Firming the Unfirmable," 9/30/30)',
                [
                    ("Operators contracting for round-the-clock supply cannot meet that profile from variable renewables alone at current storage costs. ", ""),
                    ("Every large-load contract we reviewed was firmed by an existing gas unit whose scheduled retirement was subsequently deferred", "u"),
                    (". ", ""),
                    ("The deferrals average most of a decade", "ue"),
                    (", which places the emissions squarely inside the window the transition plans assume will be clean.", ""),
                ],
            ),
        ],
    ),
    (
        "Ratepayer Advantage",
        "2AC",
        "AT: Grid Cost DA",
        [
            (
                "Cost-allocation reform is not reversible by demand alone -- it is codified",
                'Bergström 31 (A. Bergström, Fictional Association of Utility Commissioners, "Tariff Durability," 2/22/31)',
                [
                    ("The tariffs at issue were adopted through notice-and-comment and carry sunset provisions measured in years, not months. ", ""),
                    ("A pause in new requests does not repeal an adopted tariff", "u"),
                    ("; it merely reduces the number of customers to whom it applies. ", ""),
                    ("The reform survives the pause", "ue"),
                    (" and is available the moment construction resumes.", ""),
                ],
            ),
        ],
    ),
]


def build(out_path: str) -> str:
    doc = Document()
    doc.styles["Normal"].font.size = Pt(11)

    banner = doc.add_paragraph()
    r = banner.add_run(BANNER)
    r.bold = True
    r.font.highlight_color = WD_COLOR_INDEX.RED

    last = (None, None, None)
    for pocket, hat, block, cards in CORPUS:
        if pocket != last[0]:
            doc.add_heading(pocket, level=1)
        if (pocket, hat) != (last[0], last[1]):
            doc.add_heading(hat, level=2)
        if (pocket, hat, block) != last:
            doc.add_heading(block, level=3)
        last = (pocket, hat, block)

        for tag, cite, runs in cards:
            doc.add_heading(tag, level=4)

            cp = doc.add_paragraph()
            cr = cp.add_run(cite)
            cr.bold = True

            bp = doc.add_paragraph()
            for text, mark in runs:
                run = bp.add_run(text)
                if mark == "u":
                    run.underline = True
                elif mark == "ue":
                    run.underline = True
                    run.bold = True
                elif mark == "h":
                    run.font.highlight_color = WD_COLOR_INDEX.YELLOW

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc.save(out_path)
    return out_path


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "data/corpus/synthetic_grid_cost.docx"
    print("wrote", build(target))
