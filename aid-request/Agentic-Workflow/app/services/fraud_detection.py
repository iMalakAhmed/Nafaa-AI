# -----------------------------
# Fraud Detection Module
# Install dependencies: pip install transformers pillow
# -----------------------------

import json
import os
from PIL import Image
from transformers import pipeline
import numpy as np
import importlib.util
import sys

# Import reverse image search functionality using importlib (folder has space)
spec = importlib.util.spec_from_file_location("reverse_image", os.path.join(os.path.dirname(__file__), "reverse_image.py"))
reverse_image = importlib.util.module_from_spec(spec)
sys.modules["reverse_image"] = reverse_image
spec.loader.exec_module(reverse_image)

# -----------------------------
# 1. AI-generated / manipulated image detection
# -----------------------------
AI_DETECTOR_MODEL = "capcheck/ai-image-detection"

# Initialize the image classification pipeline
ai_detector = pipeline("image-classification", model=AI_DETECTOR_MODEL)

def ai_generated_probability(image_path):
    """
    Returns probability that the image is AI-generated / manipulated.
    Uses capcheck/ai-image-detection model.
    
    Output format:
    [{'label': 'Fake', 'score': 0.95}, {'label': 'Real', 'score': 0.05}]
    
    Raises:
        ValueError: If model output does not contain 'Fake' label
    """
    results = ai_detector(image_path)
    
    # Find the score for 'Fake' label
    for result in results:
        if result['label'].lower() == 'fake':
            return float(result['score'])
    
    # If 'Fake' label not found, raise error (no fallback)
    raise ValueError(f"Model did not return 'Fake' label in output: {results}")

# -----------------------------
# 2. Get image embedding (placeholder for compatibility)
# -----------------------------
def get_clip_embedding(image_path):
    """
    Get image embedding for an image (simplified version).
    
    Args:
        image_path: Path to the image file
        
    Returns:
        np.ndarray: Image feature vector
    """
    image = Image.open(image_path).convert("RGB")
    # Resize to standard size for consistency
    image = image.resize((224, 224))
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    return img_array.flatten()

# 3. Main function for fraud detection
# -----------------------------
def process_image(user_id, image_path, sim_threshold, ai_threshold):
    """
    Process an image for fraud detection.
    
    Args:
        user_id: User ID submitting the image
        image_path: Path to the image file
        sim_threshold: Similarity threshold for duplicate detection (required)
        ai_threshold: Probability threshold for AI-generated detection (required)
        
    Returns:
        dict: Fraud detection results
    """
    # AI-generated check
    ai_prob = ai_generated_probability(image_path)
    is_ai = ai_prob >= ai_threshold
    
    # Duplicate check using reverse image search
    dup_result = reverse_image.find_duplicates(image_path, user_id, sim_threshold)
    is_duplicate_same_user = dup_result['duplicate_same_user']
    is_duplicate_different_user = dup_result['duplicate_different_user']
    similarity_same_user = dup_result['similarity_same_user']
    similarity_different_user = dup_result['similarity_different_user']
    matches_same_user = dup_result['matches_same_user']
    matches_different_user = dup_result['matches_different_user']
    
    # Overall duplicate flag
    is_duplicate = is_duplicate_same_user or is_duplicate_different_user
    
    # Store embedding if accepted (not duplicate and not AI)
    if not is_duplicate and not is_ai:
        reverse_image.add_image_to_index(image_path, user_id)
    
    return {
        "image_path": image_path,
        "user_id": user_id,
        "ai_probability": ai_prob,
        "is_ai": is_ai,
        "is_duplicate_same_user": is_duplicate_same_user,
        "is_duplicate_different_user": is_duplicate_different_user,
        "similarity_same_user": similarity_same_user,
        "similarity_different_user": similarity_different_user,
        "matches_same_user": matches_same_user,
        "matches_different_user": matches_different_user,
        "is_duplicate": is_duplicate,
        "passed": not is_ai and not is_duplicate,
    }

# 4. Process JSON input and output
# ============================================================================
def main(input_json, output_json, sim_threshold=0.85, ai_threshold=0.7):
    """
    Process multiple images from JSON input with fraud detection.
    
    Reads a JSON file with image paths and user IDs, processes each image
    for fraud detection (AI-generated check and duplicate detection), and
    writes results to output JSON file.
    
    Args:
        input_json: Path to input JSON file with image data (required)
        output_json: Path to output JSON file for results (required)
        sim_threshold: Similarity threshold for duplicate detection (default: 0.85)
        ai_threshold: Probability threshold for AI-generated detection (default: 0.7)
        
    Example input JSON:
        {
          "images": [
            {"user_id": "user_001", "image_path": "path/to/image.jpg"},
            {"user_id": "user_002", "image_path": "path/to/image2.jpg"}
          ]
        }
    
    Example output JSON:
        {
          "module": "fraud_detection",
          "summary": {
            "total_processed": 2,
            "passed": 1,
            "ai_generated_detected": 0,
            "duplicates_by_same_user": 0,
            "duplicates_by_different_user": 1,
            "duplicates_total": 1,
            "errors": 0,
            "pass_rate": "50.0%"
          },
          "results": [
            {
              "id": "test_001",
              "user_id": "user_001",
              "image_path": "...",
              "ai_probability": 0.15,
              "is_ai": false,
              "is_duplicate_same_user": false,
              "is_duplicate_different_user": true,
              "similarity_same_user": 0.0,
              "similarity_different_user": 0.87,
              "is_duplicate": true,
              "passed": false,
              "timestamp": "..."
            }
          ]
        }
    """
    from datetime import datetime
    
    print("=" * 70)
    print("FRAUD DETECTION MODULE")
    print("=" * 70)
    print(f"Input file: {input_json}")
    print(f"Output file: {output_json}")
    
    # Load input data
    try:
        with open(input_json, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_json}' not found!")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in '{input_json}'")
        return
    
    # Get images from input JSON
    images = data.get("images", [])
    if not images:
        print("⚠️  Warning: No images found in input JSON")
        return
    
    print(f"\n📊 Processing {len(images)} images...\n")
    
    results = []
    for idx, entry in enumerate(images, 1):
        user_id = entry.get("user_id", "unknown")
        image_path = entry.get("image_path", "")
        image_id = entry.get("id", f"image_{idx}")
        
        if not image_path or not os.path.exists(image_path):
            print(f"[{idx}/{len(images)}] ❌ {image_id}: File not found - {image_path}")
            continue
        
        try:
            result = process_image(user_id, image_path, sim_threshold, ai_threshold)
            result["id"] = image_id
            result["timestamp"] = datetime.now().isoformat()
            
            # Determine status emoji
            if result["is_ai"]:
                status = "🚨 AI-GENERATED"
            elif result["is_duplicate"]:
                status = "⚠️  DUPLICATE"
            elif result["passed"]:
                status = "✅ PASSED"
            else:
                status = "❌ FAILED"
            
            print(f"[{idx}/{len(images)}] {status} | {image_id}")
            results.append(result)
        except Exception as e:
            print(f"[{idx}/{len(images)}] ❌ Error processing {image_id}: {str(e)}")
            results.append({
                "id": image_id,
                "user_id": user_id,
                "image_path": image_path,
                "error": str(e),
                "passed": False,
                "timestamp": datetime.now().isoformat()
            })
    
    # Calculate summary statistics
    total = len(results)
    passed = sum(1 for r in results if r.get('passed', False))
    ai_detected = sum(1 for r in results if r.get('is_ai', False))
    duplicates_same_user = sum(1 for r in results if r.get('is_duplicate_same_user', False))
    duplicates_different_user = sum(1 for r in results if r.get('is_duplicate_different_user', False))
    duplicates_total = sum(1 for r in results if r.get('is_duplicate', False))
    errors = sum(1 for r in results if 'error' in r)
    
    summary = {
        "total_processed": total,
        "passed": passed,
        "ai_generated_detected": ai_detected,
        "duplicates_by_same_user": duplicates_same_user,
        "duplicates_by_different_user": duplicates_different_user,
        "duplicates_total": duplicates_total,
        "errors": errors,
        "pass_rate": f"{(passed/total*100):.1f}%" if total > 0 else "0%",
        "timestamp": datetime.now().isoformat()
    }
    
    # Save results to JSON
    output_data = {
        "module": "fraud_detection",
        "summary": summary,
        "results": results
    }
    
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Persist embeddings from reverse image module
    reverse_image.save_index()
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"Total processed: {total}")
    print(f"  ✅ Passed:                    {passed} ({summary['pass_rate']})")
    print(f"  🚨 AI-generated:              {ai_detected}")
    print(f"  ⚠️  Duplicates (same user):    {duplicates_same_user}")
    print(f"  ⚠️  Duplicates (other users):  {duplicates_different_user}")
    print(f"  ⚠️  Duplicates (total):        {duplicates_total}")
    print(f"  ❌ Errors:                    {errors}")
    print(f"\n💾 Results saved to: {output_json}")
    print("=" * 70 + "\n")
    
    return output_data


# ============================================================================
# Example usage
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fraud Detection: AI-generated and duplicate image detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fraud_detection.py test_data.json fraud_results.json
  python fraud_detection.py --input test_data.json --output my_results.json
        """
    )
    parser.add_argument("--input", "-i", required=True,
                       help="Input JSON file with images (required)")
    parser.add_argument("--output", "-o", required=True,
                       help="Output JSON file for results (required)")
    parser.add_argument("--sim-threshold", type=float, default=0.85,
                       help="Similarity threshold for duplicate detection (default: 0.85)")
    parser.add_argument("--ai-threshold", type=float, default=0.7,
                       help="AI probability threshold (default: 0.7)")
    
    args = parser.parse_args()
    main(args.input, args.output, args.sim_threshold, args.ai_threshold)


# ===========================================
# Wrapper function for main.py compatibility
# ===========================================
def detect_fraud(image_path):
    """
    Detect AI-generated/manipulated images only.
    
    Args:
        image_path: Path to image file
    
    Returns:
        dict: AI fraud detection results
    """
    try:
        ai_prob = ai_generated_probability(image_path)
        is_ai = ai_prob > 0.7
        
        return {
            'ai_manipulated_probability': ai_prob,
            'is_ai_generated': is_ai,
            'fraud_risk': 'High' if is_ai else 'Low'
        }
    except Exception as e:
        print(f"Error in detect_fraud: {e}")
        return {
            'ai_manipulated_probability': 0.0,
            'is_ai_generated': False,
            'fraud_risk': 'Low',
            'error': str(e)
        }

