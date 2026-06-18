from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator

ROOT_DIR = Path(__file__).resolve().parents[3]
SCHEMA_PATH = ROOT_DIR / "document_parsing" / "casestudy" / "schemas" / "case_study.schema.json"
PROMPT_PATH = ROOT_DIR / "document_parsing" / "casestudy" / "prompts" / "extract_case_study_prompt.txt"
BIRTH_CERT_SCHEMA_PATH = (
    ROOT_DIR / "document_parsing" / "birthcert" / "schemas" / "birth_certificate.schema.json"
)
BIRTH_CERT_PROMPT_PATH = (
    ROOT_DIR / "document_parsing" / "birthcert" / "prompts" / "extract_birth_certificate_prompt.txt"
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


@dataclass(slots=True)
class DocumentSpec:
    document_id: str
    image_paths: list[Path]
    document_type: str = "default"


def load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def load_birth_certificate_prompt() -> str:
    return BIRTH_CERT_PROMPT_PATH.read_text(encoding="utf-8").strip()


def load_birth_certificate_schema() -> dict[str, Any]:
    return json.loads(BIRTH_CERT_SCHEMA_PATH.read_text(encoding="utf-8"))


def build_validator() -> Draft202012Validator:
    return Draft202012Validator(load_schema())


def build_birth_certificate_validator() -> Draft202012Validator:
    return Draft202012Validator(load_birth_certificate_schema())


def list_image_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    return sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def list_image_files_recursive(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    return sorted(
        path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _dir_has_image_files(directory: Path) -> bool:
    try:
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                return True
    except OSError:
        return False
    return False


def find_birth_certificate_image_dir(project_root: Path) -> Path | None:
    """Search under ``project_root`` for birth-certificate scans (folder layout or ``BC_*`` files)."""
    project_root = project_root.resolve()
    for parts in (
        ("data", "raw_images", "DataSet", "Birth Certificate"),
        ("data", "raw_images", "dataset", "Birth Certificate"),
    ):
        candidate = project_root.joinpath(*parts)
        if candidate.is_dir() and _dir_has_image_files(candidate):
            return candidate
    for pattern in ("BC_*.jpeg", "BC_*.jpg", "BC_*.JPEG", "BC_*.JPG"):
        found = next(project_root.rglob(pattern), None)
        if found is not None:
            return found.parent
    for directory in project_root.rglob("*"):
        if (
            directory.is_dir()
            and directory.name.lower() == "birth certificate"
            and _dir_has_image_files(directory)
        ):
            return directory
    return None


def load_documents(input_dir: Path | None = None, manifest_path: Path | None = None) -> list[DocumentSpec]:
    if manifest_path:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "pipeline_manifest" in raw:
            manifest = raw["pipeline_manifest"]
        elif isinstance(raw, list):
            manifest = raw
        else:
            raise ValueError(
                "Manifest must be a JSON array of document objects, or an object with key "
                "'pipeline_manifest' (see document_parsing/data/birth_certificate_bundle.json)."
            )
        if not isinstance(manifest, list):
            raise ValueError("Manifest must be a JSON array of document objects.")

        documents: list[DocumentSpec] = []
        for item in manifest:
            if not isinstance(item, dict):
                raise ValueError("Each manifest item must be an object.")
            document_id = str(item["document_id"])
            document_type = str(item.get("document_type", "default")).strip() or "default"
            images = item.get("images", [])
            if not images:
                raise ValueError(f"Manifest entry {document_id!r} has no images.")

            resolved_images: list[Path] = []
            for image in images:
                image_path = Path(image)
                if not image_path.is_absolute():
                    candidates = [
                        (ROOT_DIR / image_path).resolve(),
                        (manifest_path.parent / image_path).resolve(),
                    ]
                    image_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
                resolved_images.append(image_path)

            documents.append(
                DocumentSpec(
                    document_id=document_id,
                    image_paths=resolved_images,
                    document_type=document_type,
                )
            )

        return documents

    if input_dir is None:
        raise ValueError("Either input_dir or manifest_path must be provided.")

    image_files = list_image_files(input_dir)
    return [
        DocumentSpec(document_id=image_path.stem, image_paths=[image_path.resolve()], document_type="default")
        for image_path in image_files
    ]


def ensure_output_dirs(base_output_dir: Path) -> tuple[Path, Path]:
    raw_dir = base_output_dir / "raw_model_outputs"
    pred_dir = base_output_dir / "predictions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, pred_dir


def extract_first_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = strip_code_fences(candidate)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(candidate)):
        char = candidate[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(candidate[start : index + 1])

    raise ValueError("Could not parse a complete JSON object from model output.")


def strip_code_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.DOTALL)


def empty_payload(document: DocumentSpec, note: str) -> dict[str, Any]:
    return {
        "document_id": document.document_id,
        "source_files": [str(path) for path in document.image_paths],
        "page_count": len(document.image_paths),
        "case_office": None,
        "social_unit": None,
        "research_date": None,
        "researcher_name": None,
        "research_purpose": None,
        "head_of_household_name": None,
        "head_of_household_age": None,
        "national_id": None,
        "insurance_id": None,
        "phone": None,
        "address_center": None,
        "address_governorate": None,
        "current_job": None,
        "family_members": [],
        "housing_type": None,
        "housing_area": None,
        "rooms_count": None,
        "water_available": None,
        "water_source": None,
        "furniture_status": None,
        "rent_amount": None,
        "housing_independence": None,
        "assets": [],
        "social_status_summary": None,
        "health_status_summary": None,
        "economic_status_summary": None,
        "family_needs": None,
        "previous_aid": None,
        "attachments": [],
        "uncertain_fields": ["model_output"],
        "review_required": True,
        "review_notes": [note],
    }


def empty_birth_certificate_payload(document: DocumentSpec, note: str) -> dict[str, Any]:
    """Schema-shaped fallback when model output is not valid JSON (birth_certificate document type)."""
    return {
        "document_id": document.document_id,
        "source_files": [str(path) for path in document.image_paths],
        "page_count": len(document.image_paths),
        "birth_certificate": {
            "document_title_ar": None,
            "serial_number": None,
            "registration_number": None,
            "registration_date": None,
            "issue_date": None,
            "health_office": None,
            "civil_registry": None,
            "issuing_authority": None,
            "governorate_or_administration": None,
            "form_reference": None,
            "issuance_system_version": None,
            "barcode_or_machine_notes": None,
        },
        "ids": {
            "child_national_id": None,
            "father_national_id": None,
            "mother_national_id": None,
            "other_ids": [],
        },
        "personal_and_other": {
            "child": {
                "name": None,
                "gender": None,
                "religion": None,
                "nationality": None,
                "place_of_birth": None,
                "date_of_birth": None,
            },
            "father": {"name": None, "religion": None, "nationality": None},
            "mother": {"name": None, "religion": None, "nationality": None},
            "misc_notes": None,
        },
        "uncertain_fields": ["model_output"],
        "review_required": True,
        "review_notes": [note],
    }


def apply_birth_certificate_defaults(payload: dict[str, Any], document: DocumentSpec) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["document_id"] = normalized.get("document_id") or document.document_id
    normalized["source_files"] = normalized.get("source_files") or [str(path) for path in document.image_paths]
    normalized["page_count"] = int(normalized.get("page_count") or len(document.image_paths))

    bc_keys = (
        "document_title_ar",
        "serial_number",
        "registration_number",
        "registration_date",
        "issue_date",
        "health_office",
        "civil_registry",
        "issuing_authority",
        "governorate_or_administration",
        "form_reference",
        "issuance_system_version",
        "barcode_or_machine_notes",
    )
    bc = dict(normalized.get("birth_certificate") or {})
    for key in bc_keys:
        bc.setdefault(key, None)
    normalized["birth_certificate"] = bc

    ids = dict(normalized.get("ids") or {})
    ids.setdefault("child_national_id", None)
    ids.setdefault("father_national_id", None)
    ids.setdefault("mother_national_id", None)
    raw_other = ids.get("other_ids")
    if not isinstance(raw_other, list):
        raw_other = []
    cleaned_other: list[dict[str, Any]] = []
    for item in raw_other:
        if isinstance(item, dict):
            cleaned_other.append(
                {
                    "label_ar": item.get("label_ar"),
                    "value": item.get("value"),
                }
            )
    ids["other_ids"] = cleaned_other
    normalized["ids"] = ids

    po = dict(normalized.get("personal_and_other") or {})
    child = dict(po.get("child") or {})
    for key in ("name", "gender", "religion", "nationality", "place_of_birth", "date_of_birth"):
        child.setdefault(key, None)
    po["child"] = child
    father = dict(po.get("father") or {})
    for key in ("name", "religion", "nationality"):
        father.setdefault(key, None)
    po["father"] = father
    mother = dict(po.get("mother") or {})
    for key in ("name", "religion", "nationality"):
        mother.setdefault(key, None)
    po["mother"] = mother
    po.setdefault("misc_notes", None)
    normalized["personal_and_other"] = po

    normalized["uncertain_fields"] = normalized.get("uncertain_fields") or []
    normalized["review_notes"] = normalized.get("review_notes") or []
    normalized["review_required"] = bool(normalized.get("review_required", False))
    return normalized


def validate_birth_certificate_payload(payload: dict[str, Any]) -> list[str]:
    validator = build_birth_certificate_validator()
    errors = []
    for error in validator.iter_errors(payload):
        path = ".".join(str(part) for part in error.absolute_path) or "<root>"
        errors.append(f"{path}: {error.message}")
    return sorted(errors)


def apply_defaults(payload: dict[str, Any], document: DocumentSpec) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["document_id"] = normalized.get("document_id") or document.document_id
    normalized["source_files"] = normalized.get("source_files") or [str(path) for path in document.image_paths]
    normalized["page_count"] = normalized.get("page_count") or len(document.image_paths)
    normalized["family_members"] = normalized.get("family_members") or []
    normalized["assets"] = normalized.get("assets") or []
    normalized["attachments"] = normalized.get("attachments") or []
    normalized["uncertain_fields"] = normalized.get("uncertain_fields") or []
    normalized["review_notes"] = normalized.get("review_notes") or []
    normalized["review_required"] = bool(normalized.get("review_required", False))
    return normalized


def validate_payload(payload: dict[str, Any]) -> list[str]:
    validator = build_validator()
    errors = []
    for error in validator.iter_errors(payload):
        path = ".".join(str(part) for part in error.absolute_path) or "<root>"
        errors.append(f"{path}: {error.message}")
    return sorted(errors)


def normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_value(val) for key, val in sorted(value.items())}
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value

    text = str(value).translate(ARABIC_DIGITS).translate(PERSIAN_DIGITS)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def flatten_json(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key in sorted(value):
            next_prefix = f"{prefix}.{key}" if prefix else key
            yield from flatten_json(value[key], next_prefix)
        return

    if isinstance(value, list):
        if not value:
            yield prefix, []
            return
        for index, item in enumerate(value):
            next_prefix = f"{prefix}[{index}]"
            yield from flatten_json(item, next_prefix)
        return

    yield prefix, normalize_value(value)
