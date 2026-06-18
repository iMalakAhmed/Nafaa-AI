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
    apply_birth_certificate_defaults,
    apply_defaults,
    empty_birth_certificate_payload,
    empty_payload,
    ensure_output_dirs,
    extract_first_json_object,
    list_image_files_recursive,
    load_birth_certificate_prompt,
    load_documents,
    load_prompt,
    validate_birth_certificate_payload,
    validate_payload,
)

# Persists across repeated `run_extract` calls in the same process (e.g. Jupyter re-runs a cell).
# Key: (model_name, torch_dtype, attn_implementation) so dtype/attention fixes are not masked by cache.
_MODEL_CACHE: dict[tuple[str, str, str], tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]] = {}


def clear_model_cache() -> None:
    """Drop cached models/processors (frees GPU RAM). Next `run_extract` will load again."""
    global _MODEL_CACHE
    for model, _proc in _MODEL_CACHE.values():
        del model
    _MODEL_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
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
            "Example: --typed-input-dir birth_certificate=document_parsing/data/raw_images/DataSet/Birth Certificate"
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
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        metavar="N",
        help="Process at most the first N documents after sorting (smoke test / partial runs).",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code.")
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="eager",
        help=(
            "Attention backend for the vision-language model. "
            "'eager' is slower but avoids SDPA shape bugs on some CPU/GPU + dtype setups; "
            "'sdpa' can be faster on CUDA when stable."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


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


def model_inference_device(model: Qwen2_5_VLForConditionalGeneration) -> torch.device:
    return next(model.parameters()).device


def model_inference_dtype(model: Qwen2_5_VLForConditionalGeneration) -> torch.dtype:
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.float32


def align_processor_batch_to_model(
    batch: Any, model: Qwen2_5_VLForConditionalGeneration
) -> Any:
    """Move tensors to the module device and match floating dtype (avoids VL merge / norm bugs)."""
    device = model_inference_device(model)
    dtype = model_inference_dtype(model)
    batch = batch.to(device)
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            batch[key] = value.to(device=device, dtype=dtype)
    return batch


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
    )
    inputs = align_processor_batch_to_model(inputs, model)

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
        if document.document_type == "birth_certificate":
            payload = apply_birth_certificate_defaults(payload, document)
            validation_errors = validate_birth_certificate_payload(payload)
        else:
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
        if document.document_type == "birth_certificate":
            fallback = empty_birth_certificate_payload(
                document, f"Could not parse valid JSON from model output: {exc}"
            )
        else:
            fallback = empty_payload(document, f"Could not parse valid JSON from model output: {exc}")
        return fallback, raw_output


def run_extract(args: argparse.Namespace) -> None:
    """Run extraction for ``args``. Reuses loaded models in-process via ``_MODEL_CACHE``."""
    print(f"[extract] torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[extract] GPU device 0: {torch.cuda.get_device_name(0)}")
    else:
        print("[extract] WARNING: CUDA not visible — inference runs on CPU (orders of magnitude slower).")

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
    if args.max_documents is not None:
        if args.max_documents < 1:
            raise ValueError("--max-documents must be >= 1")
        documents = documents[: args.max_documents]
        print(f"Capped at {len(documents)} document(s) (--max-documents)")
    typed_models = parse_typed_models(args.typed_model)
    instruction_case_study = load_prompt()
    instruction_birth_certificate = load_birth_certificate_prompt()
    raw_dir, pred_dir = ensure_output_dirs(args.output_dir)

    if not torch.cuda.is_available() and args.torch_dtype in ("float16", "bfloat16"):
        print(
            "[extract] WARNING: float16/bfloat16 on CPU is unreliable for Qwen2.5-VL; "
            "using float32. On GPU, keep --torch-dtype float16 or auto."
        )
        args.torch_dtype = "float32"

    print(f"Loaded {len(documents)} document(s)")
    print(f"Writing raw outputs to: {raw_dir}")
    print(f"Writing predictions to: {pred_dir}")
    if typed_models:
        print("Enabled document-type routing:")
        for doc_type, model_name in sorted(typed_models.items()):
            print(f"  - {doc_type}: {model_name}")
        print(f"  - default: {args.model_name}")

    def get_model_and_processor(model_name: str):
        eff_dtype = args.torch_dtype
        if not torch.cuda.is_available() and eff_dtype in ("float16", "bfloat16"):
            eff_dtype = "float32"
        eff_attn = args.attn_implementation
        if eff_attn == "sdpa" and not torch.cuda.is_available():
            eff_attn = "eager"
        cache_key = (model_name, eff_dtype, eff_attn)
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            print(f"[extract] Reusing in-memory model: {model_name!r} ({eff_dtype}, {eff_attn})")
            return cached

        print(
            f"[extract] Loading model & processor: {model_name!r} ({eff_dtype}, {eff_attn})\n"
            "          (First run downloads weights from Hugging Face — multi‑GB; progress depends on disk/network.)"
        )
        load_kw: dict[str, Any] = {
            "torch_dtype": resolve_dtype(eff_dtype),
            "device_map": "auto",
            "trust_remote_code": args.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        attn = eff_attn
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                **load_kw,
                attn_implementation=attn,
            )
        except TypeError:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **load_kw)
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=args.trust_remote_code,
        )
        _MODEL_CACHE[cache_key] = (model, processor)
        return model, processor

    for index, document in enumerate(documents, start=1):
        selected_model = typed_models.get(document.document_type, args.model_name)
        print(
            f"[{index}/{len(documents)}] Processing {document.document_id} "
            f"(type={document.document_type}, model={selected_model})"
        )
        model, processor = get_model_and_processor(selected_model)
        instruction = (
            instruction_birth_certificate
            if document.document_type == "birth_certificate"
            else instruction_case_study
        )
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


def main() -> None:
    run_extract(parse_args())


if __name__ == "__main__":
    main()
