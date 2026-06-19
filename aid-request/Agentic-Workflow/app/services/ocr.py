from app.services.vqa import run_vqa  # or your vqa wrapper

def extract_text_from_image(image_path: str) -> str:
    """
    OCR replacement using VQA model.
    Instead of traditional OCR, we ask the VQA model to read the image.
    """

    try:
        question = "استخرج النص المكتوب في الصورة بدقة كما هو بدون شرح"

        answer = run_vqa(image_path, question)

        if not answer:
            return ""

        return answer.strip()

    except Exception as e:
        return f"OCR_ERROR: {str(e)}"