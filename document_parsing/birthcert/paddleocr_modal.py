"""Modal batch function — PaddleOCR evaluation on birth-certificate images.

Modal containers have internet access, so PaddleOCR can download its models.

Deploy & run:
    modal run document_parsing/birthcert/paddleocr_modal.py
"""

from __future__ import annotations

import modal

PROJECT_DIR  = "/root/project"
HF_CACHE_VOL = "case-study-hf-cache"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libgomp1")
    .pip_install(
        "paddlepaddle==3.0.0",
        "paddleocr>=3.0.0",
        "Pillow",
        "numpy<2",
        "opencv-python-headless",
    )
    .add_local_dir(
        ".",
        remote_path=PROJECT_DIR,
        ignore=[
            ".venv", ".git", "__pycache__", ".pytest_cache",
            "outputs", "notebooks", ".modal", "agent-tools",
        ],
    )
)

paddle_cache = modal.Volume.from_name("paddle-model-cache", create_if_missing=True)
app = modal.App("birthcert-paddleocr-eval")


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=600,
    volumes={"/root/.paddlex": paddle_cache},
)
def evaluate_paddleocr():
    """Run PaddleOCR over all birth-certificate images and score vs labels."""
    import os, sys, json, re, unicodedata
    from pathlib import Path
    from difflib import SequenceMatcher

    os.chdir(PROJECT_DIR)
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)

    from document_parsing.birthcert.schema import SCALAR_FIELD_PATHS, get_path
    from document_parsing.birthcert.validate import normalize_digits

    _TASHKEEL = re.compile(r"[ؗ-ًؚ-ْٰـ]")

    def _norm(value):
        if value is None:
            return ""
        text = normalize_digits(str(value))
        text = unicodedata.normalize("NFKC", text)
        text = _TASHKEEL.sub("", text)
        text = (text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
                    .replace("ى", "ي").replace("ة", "ه"))
        return re.sub(r"\s+", " ", text).strip().lower()

    def _best_match(target_n, texts):
        best = 0.0
        for t in texts:
            s = SequenceMatcher(None, target_n, _norm(t)).ratio()
            if s > best:
                best = s
        for i in range(len(texts) - 1):
            s = SequenceMatcher(None, target_n, _norm(texts[i] + " " + texts[i+1])).ratio()
            if s > best:
                best = s
        return best

    from paddleocr import PaddleOCR
    print("[paddle-eval] Loading PaddleOCR (lang=ar) …", flush=True)
    ocr = PaddleOCR(
        lang="ar",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    images_dir = Path(f"{PROJECT_DIR}/document_parsing/data/raw_images/DataSet/Birth Certificate")
    labels_dir = Path(f"{PROJECT_DIR}/document_parsing/data/birth_cert_labels")

    tp = fp = fn = tn = 0
    per_field_correct = {p: 0 for p in SCALAR_FIELD_PATHS}
    per_field_present = {p: 0 for p in SCALAR_FIELD_PATHS}
    scored = []

    for label_file in sorted(labels_dir.glob("*.json")):
        doc_id = label_file.stem
        image_path = None
        for ext in (".jpeg", ".jpg", ".png"):
            c = images_dir / f"{doc_id}{ext}"
            if c.exists():
                image_path = c
                break
        if image_path is None:
            continue

        label = json.loads(label_file.read_text(encoding="utf-8"))
        print(f"[paddle-eval] {doc_id} …", end=" ", flush=True)

        texts = []
        for res in ocr.predict(str(image_path)):
            texts.extend(res.get("rec_texts", []))

        doc_tp = doc_fp = doc_fn = doc_tn = 0
        for field_path in SCALAR_FIELD_PATHS:
            gt_val = get_path(label, field_path)
            gt_n = _norm(gt_val)
            if not gt_n:
                tn += 1; doc_tn += 1
                continue
            per_field_present[field_path] += 1
            score = _best_match(gt_n, texts)
            if score >= 0.85:
                tp += 1; doc_tp += 1
                per_field_correct[field_path] += 1
            elif score >= 0.50:
                fp += 1; doc_fp += 1
            else:
                fn += 1; doc_fn += 1

        scored.append(doc_id)
        print(f"tp={doc_tp} fp={doc_fp} fn={doc_fn} tn={doc_tn}")

    produced = tp + fp
    present  = tp + fn
    precision = tp / produced * 100 if produced else 0
    recall    = tp / present  * 100 if present  else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    result = {
        "documents_scored": len(scored),
        "precision": round(precision, 1),
        "recall": round(recall, 1),
        "f1": round(f1, 1),
        "true_negatives": tn,
        "per_field": {
            p: {"correct": per_field_correct[p], "present": per_field_present[p]}
            for p in SCALAR_FIELD_PATHS if per_field_present[p] > 0
        }
    }

    print(f"\nDocuments scored: {len(scored)}")
    print("=" * 50)
    print(f"Precision: {precision:.1f}%  [{tp}/{produced}]")
    print(f"Recall:    {recall:.1f}%  [{tp}/{present}]")
    print(f"F1:        {f1:.1f}%")
    print("=" * 50)
    print("Per-field accuracy:")
    for p in SCALAR_FIELD_PATHS:
        n = per_field_present[p]
        if n == 0:
            continue
        c = per_field_correct[p]
        print(f"  {p:<52} {c}/{n}  ({100*c//n}%)")

    return result


@app.local_entrypoint()
def run():
    result = evaluate_paddleocr.remote()
    print("\n[local] Final result:", result)
