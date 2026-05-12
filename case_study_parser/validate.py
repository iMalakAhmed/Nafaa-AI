from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import validate_birth_certificate_payload, validate_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate JSON files against a schema.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing JSON files.")
    parser.add_argument(
        "--birth-certificate",
        action="store_true",
        help="Validate against schemas/birth_certificate.schema.json instead of case study.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_files = sorted(args.input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {args.input_dir}")
        return

    invalid_count = 0
    for json_file in json_files:
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        errors = (
            validate_birth_certificate_payload(payload)
            if args.birth_certificate
            else validate_payload(payload)
        )
        if errors:
            invalid_count += 1
            print(f"{json_file.name}: INVALID")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"{json_file.name}: OK")

    print()
    print(f"Validated {len(json_files)} file(s)")
    print(f"Invalid file(s): {invalid_count}")


if __name__ == "__main__":
    main()
