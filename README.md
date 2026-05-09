# Case Study Parsing

This repo is a practical V1 pipeline for extracting structured JSON from Arabic charity case-study images.

It is built for your current situation:

- about 20 documents
- scanned image files, not clean digital PDFs
- handwritten and printed Arabic mixed on the same pages
- need for human review before data goes into your system

Do not fine-tune with 20 documents. Use these 20 files as your first benchmark and prompt-design set.

## V1 Workflow

1. Put raw images in [`data/raw_images`](./data/raw_images) or create a multi-page manifest.
2. Run the extraction script with one model, or route document types to different models.
3. Review and correct the predicted JSON.
4. Save corrected JSON files in [`data/reviewed_json`](./data/reviewed_json).
5. Run the benchmark script to measure accuracy.
6. Only after that decide whether to add another model or fine-tune.

## Recommended First Model

Start with one model only:

- `Qwen/Qwen2.5-VL-7B-Instruct`

The extraction script in this repo is wired for Qwen2.5-VL style models through `transformers`.

## Project Layout

```text
case_study_parser/
  __init__.py
  benchmark.py
  common.py
  extract.py
  validate.py
data/
  raw_images/
  reviewed_json/
  documents.template.json
examples/
  reviewed_case_template.json
outputs/
  benchmark/
  predictions/
  raw_model_outputs/
prompts/
  extract_case_study_prompt.txt
schemas/
  case_study.schema.json
requirements.txt
```

## Setup

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Step 1: Add Your Images

If each document is a single image, copy them into [`data/raw_images`](./data/raw_images).

If each document has multiple page images, create `data/documents.json` using [`data/documents.template.json`](./data/documents.template.json).

Example:

```json
[
  {
    "document_id": "case_0001",
    "images": [
      "data/raw_images/case_0001_page_1.jpg",
      "data/raw_images/case_0001_page_2.jpg",
      "data/raw_images/case_0001_page_3.jpg"
    ]
  },
  {
    "document_id": "case_0002",
    "images": [
      "data/raw_images/case_0002_page_1.jpg"
    ]
  }
]
```

## Step 2: Run Extraction

Single-image-per-document mode:

```powershell
python -m case_study_parser.extract `
  --input-dir data/raw_images `
  --output-dir outputs `
  --model-name Qwen/Qwen2.5-VL-7B-Instruct
```

Typed-directory mode (auto-assign document type by folder):

```powershell
python -m case_study_parser.extract `
  --typed-input-dir birth_certificate="data/raw_images/DataSet/Birth Certificate" `
  --output-dir outputs `
  --model-name Qwen/Qwen2.5-VL-7B-Instruct `
  --typed-model birth_certificate=Qwen/Qwen2.5-VL-7B-Instruct
```

Two folders in one run (handwritten case forms plus birth certificates):

```powershell
python -m case_study_parser.extract `
  --typed-input-dir handwritten="path/to/handwritten_images" `
  --typed-input-dir birth_certificate="data/raw_images/DataSet/Birth Certificate" `
  --output-dir outputs `
  --model-name Qwen/Qwen2.5-VL-7B-Instruct `
  --typed-model handwritten=Qwen/Qwen2.5-VL-7B-Instruct `
  --typed-model birth_certificate=Qwen/Qwen2.5-VL-7B-Instruct
```

Use this when you already grouped files by category in folders.

If you pass more than one `--typed-input-dir`, output filenames are prefixed with the type (for example `birth_certificate_BC_001`) so two folders cannot overwrite each other when stems match.

Manifest mode for multi-page documents:

```powershell
python -m case_study_parser.extract `
  --manifest data/documents.json `
  --output-dir outputs `
  --model-name Qwen/Qwen2.5-VL-7B-Instruct
```

Type-routed mode (recommended when you want a model for IDs/birth certificates and another model for handwritten pages):

1. Add `document_type` in each manifest item (for example: `handwritten`, `id_card`, `birth_certificate`).
2. Pass one or more `--typed-model TYPE=MODEL` flags.
3. Keep `--model-name` as fallback for any type not explicitly mapped.

Example:

```powershell
python -m case_study_parser.extract `
  --manifest data/documents.json `
  --output-dir outputs `
  --model-name Qwen/Qwen2.5-VL-7B-Instruct `
  --typed-model handwritten=Qwen/Qwen2.5-VL-7B-Instruct `
  --typed-model id_card=Qwen/Qwen2.5-VL-7B-Instruct `
  --typed-model birth_certificate=Qwen/Qwen2.5-VL-7B-Instruct
```

This lets you run different model checkpoints for different document categories while keeping one output format.

This produces:

- [`outputs/predictions`](./outputs/predictions): parsed JSON predictions
- [`outputs/raw_model_outputs`](./outputs/raw_model_outputs): raw model text for debugging

## Step 3: Review Predictions

Review each predicted JSON file and correct it manually.

Save the corrected version in [`data/reviewed_json`](./data/reviewed_json) with the same filename as the prediction.

If the prediction is `outputs/predictions/case_0001.json`, the reviewed gold file must be:

`data/reviewed_json/case_0001.json`

Use [`examples/reviewed_case_template.json`](./examples/reviewed_case_template.json) as a guide.

## Step 4: Validate JSON Files

Validate your reviewed files against the schema:

```powershell
python -m case_study_parser.validate --input-dir data/reviewed_json
```

You can also validate predictions:

```powershell
python -m case_study_parser.validate --input-dir outputs/predictions
```

## Step 5: Run Benchmark

```powershell
python -m case_study_parser.benchmark `
  --predictions-dir outputs/predictions `
  --reviewed-dir data/reviewed_json `
  --output-path outputs/benchmark/summary.json
```

The benchmark will tell you:

- overall field accuracy
- per-field accuracy
- which documents have the most mismatches

## What Benchmark Means Here

Your benchmark is:

- the 20 real documents
- the corrected JSON for each one
- the scoring script that compares model output to the corrected answer

Without that benchmark, you cannot tell if prompt changes or model changes are actually improving the system.

## What Prompt Design Means Here

Prompt design is how you instruct the model so it returns stable, valid JSON instead of vague text.

This repo already includes a first extraction prompt in [`prompts/extract_case_study_prompt.txt`](./prompts/extract_case_study_prompt.txt).

When you see repeated mistakes, update the prompt and re-run the benchmark.

## Suggested Workflow For Your 20 Documents

1. Run extraction on all 20 documents.
2. Manually correct all 20 JSON outputs.
3. Benchmark the current prompt and model.
4. Improve the prompt.
5. Re-run benchmark.
6. Only add another model if a specific failure pattern remains.

## Common Failure Patterns

- Names partially missing
- National ID digits wrong
- Family member rows missed
- Housing fields confused
- Attachment pages mixed with form pages
- Hallucinated values when handwriting is unclear

These are normal. V1 should prioritize:

- valid JSON
- low hallucination rate
- clear `uncertain_fields`
- reliable human review

## Notes

- The schema stores many extracted values as strings on purpose. This avoids losing ambiguous handwritten values during OCR.
- You can normalize IDs, phone numbers, and dates later after review.
- With only 20 documents, manual review is not optional.
