from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .common import (
    DocumentSpec,
    apply_defaults,
    empty_payload,
    ensure_output_dirs,
    extract_first_json_object,
    list_image_files_recursive,
    load_documents,
    load_prompt,
    validate_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract structured JSON from charity case-study images.")
    parser.add_argument("--input-dir", type=Path, help="Directory containing image files.")
    parser.add_argument("--manifest", type=Path, help="Optional JSON manifest for multi-page documents.")
    parser.add_argument(
        "--typed-input-dir",
        action="append",
        default=[],
        metavar="TYPE=DIR",
        help=(
            "Load single-image documents from a directory and assign a document type. "
            "Example: --typed-input-dir birth_certificate=data/raw_images/DataSet/Birth Certificate"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory.")
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Default Hugging Face model name or local path. Used when no type-specific model is provided.",
    )
    parser.add_argument(
        "--typed-model",
        action="append",
        default=[],
        metavar="TYPE=MODEL",
        help=(
            "Route documents by type to different models. "
            "Example: --typed-model handwritten=Qwen/Qwen2.5-VL-7B-Instruct "
            "--typed-model id_card=google/gemma-3-12b-it"
        ),
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Torch dtype for model loading.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum generated tokens.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code.")
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def process_vision_info(messages: list[dict[str, Any]]) -> tuple[list[Image.Image] | None, list[Any] | None]:
    image_inputs: list[Image.Image] = []
    video_inputs: list[Any] = []

    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue

        for item in content:
            item_type = item.get("type")
            if item_type == "image":
                image = item["image"]
                if isinstance(image, (str, Path)):
                    image = Image.open(image).convert("RGB")
                elif isinstance(image, Image.Image):
                    image = image.convert("RGB")
                else:
                    raise ValueError(f"Unsupported image type: {type(image)}")
                image_inputs.append(image)
            elif item_type == "video":
                video_inputs.append(item["video"])

    return (image_inputs or None, video_inputs or None)


def build_messages(document: DocumentSpec, instruction: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image_path in document.image_paths:
        content.append({"type": "image", "image": str(image_path)})
    content.append({"type": "text", "text": instruction})
    return [{"role": "user", "content": content}]


def parse_typed_models(values: list[str]) -> dict[str, str]:
    typed_models: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --typed-model value {value!r}. Expected TYPE=MODEL.")
        doc_type, model_name = value.split("=", 1)
        doc_type = doc_type.strip()
        model_name = model_name.strip()
        if not doc_type or not model_name:
            raise ValueError(f"Invalid --typed-model value {value!r}. Expected TYPE=MODEL.")
        typed_models[doc_type] = model_name
    return typed_models


def parse_typed_dirs(values: list[str]) -> dict[str, Path]:
    typed_dirs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --typed-input-dir value {value!r}. Expected TYPE=DIR.")
        doc_type, directory = value.split("=", 1)
        doc_type = doc_type.strip()
        directory = directory.strip()
        if not doc_type or not directory:
            raise ValueError(f"Invalid --typed-input-dir value {value!r}. Expected TYPE=DIR.")
        typed_dirs[doc_type] = Path(directory)
    return typed_dirs


def generate_payload(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    document: DocumentSpec,
    instruction: str,
    max_new_tokens: int,
) -> tuple[dict[str, Any], str]:
    messages = build_messages(document, instruction)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    if video_inputs:
        raise NotImplementedError("Video inputs are not supported in this pipeline.")

    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=processor.tokenizer.eos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    input_len = inputs.input_ids.shape[1]
    raw_output = processor.batch_decode(
        generated_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()

    try:
        payload = extract_first_json_object(raw_output)
        payload = apply_defaults(payload, document)
        validation_errors = validate_payload(payload)
        if validation_errors:
            payload["review_required"] = True
            payload["review_notes"] = list(payload.get("review_notes", [])) + validation_errors
            payload["uncertain_fields"] = sorted(
                set(payload.get("uncertain_fields", [])) | {"schema_validation"}
            )
        return payload, raw_output
    except Exception as exc:  # noqa: BLE001
        fallback = empty_payload(document, f"Could not parse valid JSON from model output: {exc}")
        return fallback, raw_output


def main() -> None:
    args = parse_args()
    typed_dirs = parse_typed_dirs(args.typed_input_dir)
    if typed_dirs:
        if args.input_dir or args.manifest:
            raise ValueError("Use --typed-input-dir by itself, or use --input-dir/--manifest. Do not mix.")
        documents: list[DocumentSpec] = []
        multi_type = len(typed_dirs) > 1
        for doc_type, directory in typed_dirs.items():
            images = list_image_files_recursive(directory)
            for image_path in images:
                stem = image_path.stem
                document_id = f"{doc_type}_{stem}" if multi_type else stem
                documents.append(
                    DocumentSpec(
                        document_id=document_id,
                        image_paths=[image_path.resolve()],
                        document_type=doc_type,
                    )
                )
        documents.sort(key=lambda doc: doc.document_id)
    else:
        documents = load_documents(input_dir=args.input_dir, manifest_path=args.manifest)
    typed_models = parse_typed_models(args.typed_model)
    instruction = load_prompt()
    raw_dir, pred_dir = ensure_output_dirs(args.output_dir)
    model_cache: dict[str, tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]] = {}

    print(f"Loaded {len(documents)} document(s)")
    print(f"Writing raw outputs to: {raw_dir}")
    print(f"Writing predictions to: {pred_dir}")
    if typed_models:
        print("Enabled document-type routing:")
        for doc_type, model_name in sorted(typed_models.items()):
            print(f"  - {doc_type}: {model_name}")
        print(f"  - default: {args.model_name}")

    def get_model_and_processor(model_name: str):
        cached = model_cache.get(model_name)
        if cached is not None:
            return cached

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=resolve_dtype(args.torch_dtype),
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=args.trust_remote_code,
        )
        model_cache[model_name] = (model, processor)
        return model, processor

    for index, document in enumerate(documents, start=1):
        selected_model = typed_models.get(document.document_type, args.model_name)
        print(
            f"[{index}/{len(documents)}] Processing {document.document_id} "
            f"(type={document.document_type}, model={selected_model})"
        )
        model, processor = get_model_and_processor(selected_model)
        payload, raw_output = generate_payload(
            model=model,
            processor=processor,
            document=document,
            instruction=instruction,
            max_new_tokens=args.max_new_tokens,
        )

        (raw_dir / f"{document.document_id}.txt").write_text(raw_output, encoding="utf-8")
        (pred_dir / f"{document.document_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("Extraction complete.")


if __name__ == "__main__":
    main()
