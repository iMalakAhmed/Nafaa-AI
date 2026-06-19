"""
================================================================================
ENHANCED VOICE TO TEXT MODULE FOR CHARITY AID REQUESTS
================================================================================
This version focuses on transcribing and post-processing Egyptian Arabic aid requests, 
extracting key insights for charity organizations.
"""

import os
import re
import logging
import json
from transformers import pipeline
from typing import Dict, List, Any
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Hugging Face Pipeline for Transcription
try:
    logger.info("Loading Egyptian Arabic Wav2Vec2 model...")
    transcribe_pipeline = pipeline(
        "automatic-speech-recognition", 
        model="IbrahimAmin/egyptian-arabic-wav2vec2-xlsr-53"
    )
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    transcribe_pipeline = None

try:
    from pydub.utils import mediainfo
    from pydub import AudioSegment, effects
except ImportError:
    logger.warning("pydub not available. Install with: pip install pydub")

try:
    from word2number import w2n
except ImportError:
    logger.warning("word2number not available. Install with: pip install word2number")


def get_audio_duration(audio_path: str) -> float:
    """
    Get the duration of an audio file in seconds.
    """
    try:
        info = mediainfo(audio_path)
        duration = float(info['duration'])
        logger.info(f"Audio duration: {duration:.2f} seconds.")
        return duration
    except Exception as e:
        logger.error(f"Failed to retrieve audio duration: {e}")
        return 0.0


def simple_audio_enhancement(audio_path: str, output_path: str = "enhanced_audio.wav") -> str:
    """
    Simple audio preprocessing without noisereduce dependency.
    Applies normalization and frequency filtering.
    """
    try:
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        
        # Apply simple preprocessing
        # 1. Normalize volume
        normalized = effects.normalize(audio)
        
        # 2. Remove very low frequencies (rumble)
        filtered = normalized.high_pass_filter(80)
        
        # 3. Reduce high frequencies (hiss)
        filtered = filtered.low_pass_filter(8000)
        
        # 4. Compress dynamic range
        filtered = filtered.compress_dynamic_range()
        
        # Export enhanced audio
        filtered.export(output_path, format="wav")
        logger.info(f"Audio enhanced and saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Audio enhancement failed: {e}")
        # Return original path if enhancement fails
        return audio_path


# Define Arabic number conversion
EGYPTIAN_TO_ENGLISH_NUMS = {
    "واحد": "one", "اثنين": "two", "اتنين": "two",
    "تلاتة": "three", "ثلاثة": "three", "أربعة": "four", "اربعة": "four",
    "خمسة": "five", "ستة": "six", "سبعة": "seven", 
    "ثمانية": "eight", "تمانية": "eight", "تسعة": "nine", 
    "عشرة": "ten", "عشر": "ten", 
    "مئة": "hundred", "مائة": "hundred", "ميت": "hundred",
    "ألف": "thousand", "الف": "thousand", 
    "ألفين": "two thousand", "الفين": "two thousand",
    "مليون": "million"
}


def arabic_to_english_numbers(text: str) -> str:
    """
    Convert Arabic number words to English for word2number compatibility.
    """
    for ar, en in EGYPTIAN_TO_ENGLISH_NUMS.items():
        text = re.sub(r'\b' + re.escape(ar) + r'\b', en, text)
    return text


def transcribe(audio_path: str) -> Dict[str, Any]:
    """
    Transcribe Egyptian Arabic audio and return structured results.
    """
    logger.info(f"Starting transcription for: {audio_path}")

    # Check if pipeline is available
    if not transcribe_pipeline:
        logger.error("Transcription pipeline is not available.")
        return {"error": "Pipeline not available"}

    # Validate audio file exists
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Transcribe audio
    try:
        logger.info("Enhancing and transcribing audio...")
        
        # Enhance audio first
        enhanced_path = simple_audio_enhancement(audio_path)
        
        # Transcribe enhanced audio
        transcription = transcribe_pipeline(enhanced_path)
        transcribed_text = transcription.get('text', '').strip()
        
        logger.info("Transcription completed successfully.")
        duration = get_audio_duration(audio_path)
        
        # Clean up temporary enhanced file
        if enhanced_path != audio_path and os.path.exists(enhanced_path):
            try:
                os.remove(enhanced_path)
            except:
                pass

        return {
            "transcribed_text": transcribed_text,
            "language": "Arabic",
            "duration_seconds": duration,
            "processing_notes": [
                f"File: {Path(audio_path).name}", 
                "Audio enhanced before transcription"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return {"error": str(e)}


# ============================================================================
# POST-PROCESSING FOR CHARITY AID REQUESTS
# ============================================================================

# Configure constants for aid context
URGENT_KEYWORDS = ['طارئ', 'عاجل', 'مستعجل', 'دلوقتي', 'ضروري', 'بأسرع وقت', 'محتاج حالاً']
FINANCIAL_WORDS = ['فلوس', 'مال', 'مبلغ', 'دين', 'مساعدة مادية']
MEDICAL_WORDS = ['علاج', 'عملية', 'دواء', 'أدوية', 'مستشفى', 'مرض']
FOOD_WORDS = ['تموين', 'طعام', 'غذاء', 'أكل']
PEOPLE_WORDS = ['أطفال', 'أسرة', 'أم', 'أب', 'عائلة', 'طفل']
THANK_WORDS = ['شكراً', 'ممتن', 'جزاكم الله خير', 'متشكر']

EGYPTIAN_LOCATIONS = [
    'القاهرة', 'الإسكندرية', 'الجيزة', 'الشرقية', 
    'سوهاج', 'المنصورة', 'طنطا', 'بنها', 'الفيوم'
]


def normalize_egyptian_arabic(text: str) -> str:
    """Normalize Egyptian Arabic text to standard Arabic."""
    replacements = {
        'عايز': 'أحتاج',
        'محتاج': 'أحتاج',
        'مش': 'ليس',
        'بتاع': 'لـ',
        'دلوقتي': 'الآن',
        'بكره': 'غداً',
        'النهاردة': 'اليوم',
        'حأ': 'سوف',
    }
    for src, tgt in replacements.items():
        text = text.replace(src, tgt)
    return text.strip()


def detect_sentiment(text: str) -> str:
    """
    Detect sentiment for aid requests.
    """
    if any(word in text for word in URGENT_KEYWORDS):
        return "urgent"
    elif any(word in text for word in FINANCIAL_WORDS + MEDICAL_WORDS + FOOD_WORDS):
        return "request"
    elif any(word in text for word in THANK_WORDS):
        return "grateful"
    else:
        return "neutral"


def extract_keywords(text: str, num_keywords: int = 5) -> List[str]:
    """
    Extract important keywords for charity aid requests.
    """
    stop_words = {'في', 'من', 'على', 'عن', 'إلى', 'و', 'أن', 'هذا', 'الذي', 'التي'}
    words = [word.strip(".,!?") for word in text.split() if len(word) > 2 and word not in stop_words]
    unique_words = []
    for word in words:
        if word not in unique_words:
            unique_words.append(word)
    return unique_words[:num_keywords]


def extract_numbers(text: str) -> List[int]:
    """
    Extract numbers from text:
      - Digits
      - Arabic number words converted to integers
    """
    numbers = []

    # 1. Extract digit numbers
    numbers += [int(n) for n in re.findall(r'\d+', text)]

    # 2. Extract Arabic number words
    try:
        # First convert Arabic number words to English
        text_eng = arabic_to_english_numbers(text)
        
        # Try to extract numbers from the converted text
        words = text_eng.split()
        for i, word in enumerate(words):
            # Try to convert single words
            try:
                num = w2n.word_to_num(word)
                numbers.append(num)
            except ValueError:
                # Try multi-word numbers
                for j in range(1, 4):  # Try up to 3-word numbers
                    if i + j <= len(words):
                        phrase = ' '.join(words[i:i+j])
                        try:
                            num = w2n.word_to_num(phrase)
                            numbers.append(num)
                        except ValueError:
                            continue
    except Exception as e:
        logger.error(f"Error extracting numbers: {e}")

    return list(set(numbers))  # remove duplicates


def extract_locations(text: str) -> List[str]:
    """
    Extract mentions of Egyptian locations.
    """
    return [loc for loc in EGYPTIAN_LOCATIONS if loc in text]


def post_process_request(text: str) -> Dict[str, Any]:
    """
    Post-process transcription for charity insights.
    """
    cleaned_text = " ".join(text.split())
    normalized_text = normalize_egyptian_arabic(cleaned_text)
    sentiment = detect_sentiment(normalized_text)
    keywords = extract_keywords(normalized_text)
    numbers = extract_numbers(normalized_text)
    locations = extract_locations(normalized_text)

    return {
        "cleaned_text": cleaned_text,
        "normalized_text": normalized_text,
        "sentiment": sentiment,
        "keywords": keywords,
        "numbers": numbers,
        "locations": locations
    }


def save_result_to_json(result: Dict[str, Any], output_path: str) -> None:
    """
    Save transcription and post-processing result to a JSON file.
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    # Example usage: process an audio file
    audio_file = "test_audio.wav"  # Replace with actual audio file path

    if os.path.exists(audio_file):
        try:
            logger.info("Processing audio file for transcription and insights...")

            # Transcription
            transcription = transcribe(audio_file)

            if "transcribed_text" in transcription:
                logger.info("Post-processing transcription...")
                request_insights = post_process_request(transcription["transcribed_text"])
                full_result = {**transcription, **request_insights}

                audio_name = Path(audio_file).stem
                output_json_path = f"{audio_name}_result.json"

                save_result_to_json(full_result, output_json_path)

                print("\n" + "="*60)
                print("TRANSCRIPTION AND ANALYSIS RESULTS")
                print("="*60)
                print(json.dumps(full_result, indent=2, ensure_ascii=False))
                print("="*60)
            else:
                logger.error("No transcription text found. Review transcription result.")

        except Exception as e:
            logger.error(f"An error occurred: {e}")
    else:
        logger.info(f"Audio file not found at: {audio_file}")
        logger.info("Please provide a valid audio file path.")
