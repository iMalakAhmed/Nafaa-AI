# RunPod Document Extraction Endpoint

This project exposes one RunPod Serverless handler that can extract:

- birth certificates with Qwen2.5-VL + `bc_lora_v4` LoRA adapter
- case-study forms with Gemini
- Egyptian national IDs with `ebrahimabdelghfar/National-ID-Reader` YOLO weights + EasyOCR

## Build Image

```bash
docker build -f Dockerfile.runpod -t <dockerhub-user>/doc-extraction:latest .
docker push <dockerhub-user>/doc-extraction:latest
```

## RunPod Endpoint Settings

Create a RunPod Serverless endpoint with:

- Container image: `<dockerhub-user>/doc-extraction:latest`
- GPU: L4 or better
- Container start command: leave default from Dockerfile
- Environment variables:

```text
GEMINI_API_KEY=your_google_ai_studio_key
ADAPTER_BC_PATH=/app/deploy/adapters/bc_lora_v4
```

Optional:

```text
GEMINI_CASESTUDY_MODEL=gemini-2.5-flash
GEMINI_CLASSIFIER_MODEL=gemini-2.5-flash
BIRTHCERT_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
BIRTHCERT_TORCH_DTYPE=bfloat16
BIRTHCERT_ATTN=sdpa
```

The Dockerfile defaults `ADAPTER_BC_PATH` to `/app/deploy/adapters/bc_lora_v4`.
Before building the final production image, copy the trained adapter folder into:

```text
deploy/adapters/bc_lora_v4
```

or mount it in RunPod and set `ADAPTER_BC_PATH` to that mounted path.

## Request Shape

RunPod Serverless expects JSON. Send the image as base64:

```json
{
  "input": {
    "document_type": "birthcert",
    "filename": "BC_00001.jpeg",
    "image": "<base64 image bytes>"
  }
}
```

Supported `document_type` values:

```text
birthcert
casestudy
national_id
auto
```

If `document_type` is omitted or set to `auto`, Gemini classifies the image as one of:

```text
birthcert
casestudy
national_id
```

## Example Responses

Birth certificate and case study:

```json
{
  "output": {
    "document_type": "birthcert",
    "record": {},
    "raw_text": "only returned when include_raw=true"
  }
}
```

National ID:

```json
{
  "output": {
    "document_type": "national_id",
    "record": {
      "document_type": "national_id",
      "full_name": "...",
      "city": "...",
      "governorate": "...",
      "national_id": "...",
      "review_required": false,
      "review_notes": []
    }
  }
}
```

## Backend Curl Example

```bash
IMAGE_B64=$(base64 -w 0 BC_00001.jpeg)

curl -X POST "https://api.runpod.ai/v2/<endpoint-id>/runsync" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": {
      \"document_type\": \"birthcert\",
      \"filename\": \"BC_00001.jpeg\",
      \"image\": \"$IMAGE_B64\"
    }
  }"
```

For case study, you can disable region crops:

```json
{
  "input": {
    "document_type": "casestudy",
    "use_regions": false,
    "filename": "CS_002.jpeg",
    "image": "<base64>"
  }
}
```

For auto-routing:

```json
{
  "input": {
    "document_type": "auto",
    "filename": "upload.jpg",
    "image": "<base64>"
  }
}
```
