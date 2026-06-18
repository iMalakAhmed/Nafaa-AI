from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import flatten_json, normalize_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark predicted JSON against reviewed gold JSON.")
    parser.add_argument("--predictions-dir", type=Path, required=True, help="Directory with predicted JSON files.")
    parser.add_argument("--reviewed-dir", type=Path, required=True, help="Directory with reviewed gold JSON files.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("document_parsing/outputs/benchmark/summary.json"),
        help="Where to write the summary report.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_flat_map(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in flatten_json(payload)}


def score_document(predicted: dict[str, Any], reviewed: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    pred_flat = build_flat_map(predicted)
    reviewed_flat = build_flat_map(reviewed)

    field_counts: dict[str, dict[str, int]] = {}
    mismatches: list[dict[str, Any]] = []

    all_keys = sorted(set(pred_flat) | set(reviewed_flat))
    correct = 0
    total = 0

    for key in all_keys:
        pred_value = normalize_value(pred_flat.get(key))
        gold_value = normalize_value(reviewed_flat.get(key))
        is_match = pred_value == gold_value

        total += 1
        if is_match:
            correct += 1

        field_counts[key] = {
            "correct": int(is_match),
            "total": 1,
        }

        if not is_match:
            mismatches.append(
                {
                    "field": key,
                    "predicted": pred_value,
                    "reviewed": gold_value,
                }
            )

    summary = {
        "correct_fields": correct,
        "total_fields": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
    }
    details = {"mismatches": mismatches}
    return summary, {"field_counts": field_counts, "details": details}


def main() -> None:
    args = parse_args()
    prediction_files = {path.stem: path for path in args.predictions_dir.glob("*.json")}
    reviewed_files = {path.stem: path for path in args.reviewed_dir.glob("*.json")}
    shared_ids = sorted(set(prediction_files) & set(reviewed_files))

    if not shared_ids:
        raise SystemExit("No matching filenames found between predictions and reviewed directories.")

    field_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    document_results: dict[str, Any] = {}
    overall_correct = 0
    overall_total = 0

    for document_id in shared_ids:
        predicted = load_json(prediction_files[document_id])
        reviewed = load_json(reviewed_files[document_id])
        summary, extras = score_document(predicted, reviewed)

        overall_correct += summary["correct_fields"]
        overall_total += summary["total_fields"]

        for field, counts in extras["field_counts"].items():
            field_totals[field]["correct"] += counts["correct"]
            field_totals[field]["total"] += counts["total"]

        document_results[document_id] = {
            **summary,
            **extras["details"],
        }

    field_accuracy = {
        field: {
            "correct": counts["correct"],
            "total": counts["total"],
            "accuracy": round(counts["correct"] / counts["total"], 4) if counts["total"] else 0.0,
        }
        for field, counts in sorted(field_totals.items())
    }

    output = {
        "documents_scored": len(shared_ids),
        "overall_accuracy": round(overall_correct / overall_total, 4) if overall_total else 0.0,
        "overall_correct_fields": overall_correct,
        "overall_total_fields": overall_total,
        "field_accuracy": field_accuracy,
        "documents": document_results,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Documents scored: {output['documents_scored']}")
    print(f"Overall accuracy: {output['overall_accuracy']:.4f}")
    print(f"Summary written to: {args.output_path}")


if __name__ == "__main__":
    main()
