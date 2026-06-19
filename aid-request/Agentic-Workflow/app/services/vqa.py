from typing import List, Dict, Any, Optional
import json
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import os
import base64

# ============================================================================
# GLOBAL MODEL LOADING (SINGLETON)
# ============================================================================

_model = None
_processor = None


def get_vqa_model():
    global _model, _processor

    if _model is None:
        print("[VQA] Loading Qwen2-VL model...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "Qwen/Qwen2-VL-2B-Instruct"

        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else {"": "cpu"},
            torch_dtype=torch.float16 if device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        _model.eval()

        _processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        print("[VQA] Model loaded successfully.")

    return _model, _processor


# ============================================================================
# CORE VQA INFERENCE
# ============================================================================

def run_vqa(image_path: str, question: str) -> str:
    image = Image.open(image_path).convert("RGB")

    model, processor = get_vqa_model()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    device = next(model.parameters()).device

    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True
    )

    # safer device move
    inputs = {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=128)

    generated = output_ids[0][inputs["input_ids"].shape[1]:]

    answer = processor.decode(
        generated,
        skip_special_tokens=True
    )

    return answer.strip()


# ============================================================================
# QUESTION PIPELINE
# ============================================================================

def answer_questions(
    images: List[str],
    category: str,
    vqa_questions: Dict[str, List[str]],
    ocr_texts: Optional[List[str]] = None
) -> Dict[str, Any]:

    selected_questions = select_category_questions(category, vqa_questions)

    results = []
    all_confidences = []

    for idx, image_path in enumerate(images):
        image_results = []

        ocr_text = ocr_texts[idx] if ocr_texts and idx < len(ocr_texts) else ""

        for question in selected_questions:
            qa = answer_single_question(
                image_path=image_path,
                question=question,
                ocr_text=ocr_text,
                context=f"Category: {category}"
            )
            image_results.append(qa)
            all_confidences.append(qa["confidence"])

        consistency_prompt = (
            "النص المذكور:\n"
            f"{ocr_text}\n\n"
            "هل الصورة تتفق مع هذا النص؟\n"
            "أجب بنعم أو لا أو غير واضح مع توضيح مختصر."
        )

        consistency_answer = answer_single_question(
            image_path=image_path,
            question=consistency_prompt,
            ocr_text=ocr_text,
            context=consistency_prompt
        )

        image_results.append({
            **consistency_answer,
            "question_type": "image_text_consistency"
        })

        results.append({
            "image_id": f"IMG-{idx+1:03d}",
            "image_path": image_path,
            "vqa_results": image_results
        })

    overall_confidence = sum(all_confidences) / max(len(all_confidences), 1)

    return {
        "category": category,
        "images": results,
        "overall_confidence": round(overall_confidence, 3),
        "processing_notes": [
            "Processed using Qwen2-VL",
            "Image-text consistency check enabled",
            f"Category: {category}",
            f"Total images: {len(images)}"
        ]
    }


def answer_single_question(
    image_path: str,
    question: str,
    ocr_text: str = "",
    context: str = ""
) -> Dict[str, Any]:

    full_question = create_vqa_prompt(
        question=question,
        ocr_text=ocr_text,
        context=context
    )

    try:
        raw_answer = run_vqa(image_path, full_question)
        parsed = parse_vqa_response(raw_answer, question)

    except Exception as e:
        return {
            "question": question,
            "answer": "",
            "confidence": 0.0,
            "reasoning": f"VQA error: {str(e)}"
        }

    return {
        "question": question,
        "answer": parsed["answer"],
        "confidence": parsed["confidence"],
        "reasoning": parsed["reasoning"]
    }


# ============================================================================
# BATCH MODE
# ============================================================================

def answer_three_questions_batch(
    image_paths: List[str],
    ocr_texts: List[str],
    description: str,
    questions: List[str]
) -> List[Dict[str, Any]]:

    batch_results = []

    for idx, image_path in enumerate(image_paths):
        ocr_text = ocr_texts[idx] if idx < len(ocr_texts) else ""
        results = []

        for q in questions:

            context = description

            if "تناقض" in q:
                context += f"\n\nOCR:\n{ocr_text}"
            elif "يتفق" in q:
                context += f"\n\nOCR:\n{ocr_text}"

            prompt = create_vqa_prompt(
                question=q,
                ocr_text=ocr_text,
                context=context
            )

            try:
                answer = run_vqa(image_path, prompt)
                parsed = parse_vqa_response(answer, q)

            except Exception as e:
                parsed = {
                    "answer": "",
                    "confidence": 0.0,
                    "reasoning": f"VQA error: {str(e)}"
                }

            if any(x in parsed["answer"] for x in ["نعم", "لا", "غير"]):
                parsed["confidence"] = 0.95

            results.append({
                "question": q,
                "answer": parsed["answer"],
                "confidence": parsed["confidence"],
                "reasoning": parsed["reasoning"]
            })

        batch_results.append({
            "image_path": image_path,
            "results": results
        })

    return batch_results


# ============================================================================
# HELPERS
# ============================================================================

def select_category_questions(category: str, all_questions: Dict[str, List[str]], num_questions: int = 5):
    if category not in all_questions:
        raise ValueError(f"Unknown category: {category}")

    return all_questions[category][:num_questions]


def create_vqa_prompt(question: str, ocr_text: str = "", context: str = "", category: str = "") -> str:

    parts = []

    if category:
        parts.append(f"التصنيف: {category}")

    if context:
        parts.append(f"السياق:\n{context}")

    if ocr_text:
        parts.append(f"OCR:\n{ocr_text}")

    parts.append(f"السؤال:\n{question}")

    parts.append(
        """
        IMPORTANT RULES:
        - لا تخمن أي معلومة غير موجودة في الصورة
        - إذا لم تجد إجابة واضحة اكتب: "غير واضح"
        - لا تكتب كلمات مثل: أجرب / محاولة / تخمين
        - أجب فقط بمعلومة واحدة قصيرة
        """
        )

    return "\n\n".join(parts)


def parse_vqa_response(response: str, question: str) -> Dict[str, Any]:

    response = response.strip()

    if not response:
        return {
            "answer": "",
            "confidence": 0.0,
            "reasoning": "Empty response"
        }

    low_conf = ["غير واضح", "لا يمكن", "غير معروف", "لا أستطيع"]

    confidence = 0.9

    if any(x in response for x in low_conf):
        confidence = 0.4

    if len(response.split()) <= 3:
        confidence = min(confidence + 0.05, 0.95)

    return {
        "answer": response,
        "confidence": round(confidence, 2),
        "reasoning": "Parsed Qwen response"
    }


# ============================================================================
# OPTIONAL UTILITIES
# ============================================================================

def load_image(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    return Image.open(image_path).convert("RGB")


def encode_image_for_api(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============================================================================
# TEST ENTRYPOINT
# ============================================================================

if __name__ == '__main__':

    QUESTIONS3_JSON = "data/vqa_3questions.json"

    with open(QUESTIONS3_JSON, "r", encoding="utf-8") as f:
        questions3 = json.load(f)

    TEST_IMAGE_PATHS = ["ta7lel.jpg"]
    TEST_OCR_TEXTS = ["Male 20 year kidney functions , lipids profile"]
    description = "تحليل وظائف كبد"

    print(f"Loaded {len(questions3)} questions")

    result = answer_three_questions_batch(
        image_paths=TEST_IMAGE_PATHS,
        ocr_texts=TEST_OCR_TEXTS,
        description=description,
        questions=questions3
    )

    with open("vqa_three_questions.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("DONE")