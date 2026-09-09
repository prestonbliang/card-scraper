"""Turn an argument-skeleton YAML into an empty Verbatim-shaped .docx.

    python -m seed.build_outline seed/moratorium_outline.yaml data/corpus/moratorium.docx

The output has the Heading 1-4 structure already correct and a placeholder line
under each tag. Paste your real cite and card body over the placeholder in Word,
keep the underlining you already use, re-run ingest, and the outline, the AT:
edges and the side inference all come out right without any further work.

The placeholders are short and obviously not evidence, so a file you have not
finished filling in cannot be mistaken for one you have.
"""

from __future__ import annotations

import os
import sys

from docx import Document
from docx.shared import Pt

PLACEHOLDER = "[paste cite here]"
BODY_PLACEHOLDER = "[paste card body here — keep your underlining]"


def _load(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        print("pip install pyyaml", file=sys.stderr)
        raise
    with open(path) as fh:
        return yaml.safe_load(fh)


def build(spec: dict, out_path: str) -> str:
    doc = Document()
    doc.styles["Normal"].font.size = Pt(11)

    res = (spec.get("resolution") or "").strip()
    if res:
        p = doc.add_paragraph()
        r = p.add_run(f"Resolution: {res}")
        r.italic = True

    for side_name, side in (spec.get("sides") or {}).items():
        for pocket in side.get("pockets", []):
            doc.add_heading(pocket["title"], level=1)
            for hat in pocket.get("hats", []):
                doc.add_heading(hat["title"], level=2)
                for block in hat.get("blocks", []):
                    doc.add_heading(block["title"], level=3)
                    srcs = block.get("sources") or []
                    if srcs:
                        n = doc.add_paragraph()
                        run = n.add_run(f"go cut: {', '.join(srcs)}")
                        run.italic = True
                    for tag in block.get("tags", []):
                        doc.add_heading(tag, level=4)
                        cp = doc.add_paragraph()
                        cp.add_run(PLACEHOLDER).bold = True
                        doc.add_paragraph(BODY_PLACEHOLDER)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    doc.save(out_path)
    return out_path


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    src, dst = argv[0], argv[1]
    print("wrote", build(_load(src), dst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
