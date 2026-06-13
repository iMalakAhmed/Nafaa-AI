# RunPod YOLO + Arabic OCR Job

This runs the birth-certificate YOLO detector training on a GPU pod, then runs
OCR on the detected field crops and evaluates against `data/birth_cert_labels`.

## Pod Setup

Use a GPU pod with this repo uploaded or mounted. From the repo root:

```bash
pip install -r requirements-yolo-ocr.txt
```

If you use the Dockerfile:

```bash
docker build -f Dockerfile.runpod -t birthcert-yolo-ocr:latest .
```

The Dockerfile includes the YOLO/OCR requirements and project source. For a
normal interactive pod, you still need the `data/` folder mounted or copied into
the container because training needs the images and labels.

## Train + Evaluate With EasyOCR

```bash
python tools/run_birthcert_yolo_pod_job.py --epochs 100 --imgsz 960 --batch 8
```

Outputs:

- YOLO weights: `outputs/birthcert_yolo/field_detector/weights/best.pt`
- OCR JSON records: `outputs/birthcert_yolo_ocr/records`
- OCR raw crop reads: `outputs/birthcert_yolo_ocr/raw`
- Field crops: `outputs/birthcert_yolo_ocr/crops`

## Train + Evaluate With QARI/Baseer/Arabic-GLM Style OCR

Use the HuggingFace model ID for the OCR/VLM model:

```bash
python tools/run_birthcert_yolo_pod_job.py \
  --epochs 100 \
  --imgsz 960 \
  --batch 8 \
  --ocr-backend hf-vlm \
  --ocr-model MODEL_ID_HERE
```

Replace `MODEL_ID_HERE` with the actual HuggingFace model ID.

## Evaluation Only

If training already finished:

```bash
python tools/run_birthcert_yolo_pod_job.py --skip-train
```

Or run manually:

```bash
python -m birthcert.yolo_ocr \
  --weights outputs/birthcert_yolo/field_detector/weights/best.pt \
  --images data/birthcert_yolo/images/val \
  --ocr-backend easyocr

python -m birthcert.evaluate \
  --pred outputs/birthcert_yolo_ocr/records \
  --labels data/birth_cert_labels
```
