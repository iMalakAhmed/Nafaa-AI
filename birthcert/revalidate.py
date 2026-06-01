"""Re-run validation over saved raw model outputs (no GPU needed).

Because every run saves the raw model text, we can re-parse and re-validate it
locally whenever the validation logic improves — far cheaper than re-running the
vision model. Use this after changing schema/validate.py.

Usage (from repo root):
  python -m birthcert.revalidate --raw outputs/birthcert/raw --out outputs/birthcert/records
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .extract import empty_record
from .jsonparse import extract_first_json_object
from .validate import validate_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-validate saved raw model outputs.")
    parser.add_argument("--raw", type=str, default="outputs/birthcert/raw")
    parser.add_argument("--out", type=str, default="outputs/birthcert/records")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(raw_dir.glob("*.txt"))
    if not raw_files:
        print(f"No raw .txt files in {raw_dir}")
        return

    ok = recovered = failed = 0
    for f in raw_files:
        doc_id = f.stem
        source = f"data/raw_images/DataSet/Birth Certificate/{doc_id}.jpeg"
        text = f.read_text(encoding="utf-8")
        try:
            raw_obj = extract_first_json_object(text)
            record = validate_record(raw_obj, document_id=doc_id, source_files=[source])
            ok += 1
            if record["personal_and_other"]["child"]["name"] or any(
                record["ids"][k] for k in ("child_national_id", "father_national_id", "mother_national_id")
            ):
                recovered += 1
        except ValueError:
            record = empty_record(document_id=doc_id, source_files=[source])
            record["review_required"] = True
            record["review_notes"] = ["model output was not valid JSON"]
            failed += 1
        (out_dir / f"{doc_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"Re-validated {ok + failed} record(s) -> {out_dir}")
    print(f"  parsed OK: {ok}   with name/id: {recovered}   unparseable: {failed}")


if __name__ == "__main__":
    main()
