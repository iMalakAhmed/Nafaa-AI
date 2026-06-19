# -----------------------------
# Reverse Image Search Module
# Install dependencies: pip install torch torchvision transformers pillow faiss-cpu
# -----------------------------

import os
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import sys
import importlib.util

# Import embeddings_db from same directory
spec = importlib.util.spec_from_file_location("embeddings_db", os.path.join(os.path.dirname(__file__), "embeddings_db.py"))
embeddings_db = importlib.util.module_from_spec(spec)
spec.loader.exec_module(embeddings_db)

# -----------------------------
# CLIP Embeddings for reverse search
# -----------------------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

def get_clip_embedding(image_path):
    """
    Extract CLIP embedding for an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        np.array: Normalized CLIP embedding vector
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path} (absolute path: {os.path.abspath(image_path)})")
    
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    # Extract pooled embeddings from BaseModelOutputWithPooling
    emb = outputs if isinstance(outputs, torch.Tensor) else outputs.pooler_output
    # Normalize the embedding
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.squeeze().numpy()

# -----------------------------
# FAISS index for similarity search
# -----------------------------
embedding_dim = 512  # CLIP embedding size
index = faiss.IndexFlatIP(embedding_dim)  # cosine similarity
all_embeddings = []  # list of np.array embeddings
all_metadata = []    # parallel list for metadata (user_id, image_path, etc.)

def load_index():
    """Load embeddings and metadata from database."""
    global all_embeddings, all_metadata
    try:
        embeddings, metadata = embeddings_db.load_all_embeddings()
        all_embeddings = embeddings
        all_metadata = metadata
        print(f"Loaded {len(all_embeddings)} embeddings from database")
    except Exception as e:
        print(f"Warning: Failed to load embeddings on startup: {e}")
        all_embeddings = []
        all_metadata = []

def save_index():
    """Save embeddings - handled automatically by database on insert."""
    try:
        stats = embeddings_db.get_stats()
        print(f"Database stats: {stats.get('total_embeddings', 'N/A')} embeddings, {stats.get('unique_users', 'N/A')} users")
    except Exception as e:
        print(f"[WARN] Could not fetch stats (non-critical): {e}")

def add_image_to_index(image_path, user_id, request_id=None, additional_metadata=None):
    """
    Add an image to the reverse search index and save to database.
    
    Args:
        image_path: Path to the image file
        user_id: User ID associated with the image
        request_id: Optional request ID (GUID string)
        additional_metadata: Optional dict with additional metadata
        
    Returns:
        np.array: The embedding that was added
    """
    emb = get_clip_embedding(image_path)
    all_embeddings.append(emb.astype('float32'))
    
    metadata = {
        "user_id_hash": embeddings_db.hash_user_id(user_id),
        "image_path": image_path,
        **(additional_metadata or {})
    }
    all_metadata.append(metadata)
    
    # Save to database immediately with named parameters
    embeddings_db.save_embedding(user_id, image_path, emb, request_id=request_id, metadata=additional_metadata)
    
    return emb

def search_similar_images(image_path, top_k=5, similarity_threshold=0.0):
    """
    Search for similar images in the index.
    
    Args:
        image_path: Path to the query image
        top_k: Number of similar images to return
        similarity_threshold: Minimum similarity score (0-1)
        
    Returns:
        list: List of dicts with 'similarity', 'metadata' keys
    """
    if not all_embeddings:
        return []
    
    emb = get_clip_embedding(image_path)
    emb_array = np.vstack(all_embeddings).astype('float32')
    
    # Rebuild FAISS index
    index.reset()
    index.add(emb_array)
    
    # Search
    D, I = index.search(np.array([emb], dtype='float32'), k=min(top_k, len(all_embeddings)))
    
    results = []
    for sim, idx in zip(D[0], I[0]):
        if sim >= similarity_threshold:
            results.append({
                "similarity": float(sim),
                "metadata": all_metadata[idx]
            })
    
    return results

def find_duplicates(image_path, user_id, similarity_threshold=0.85):
    """
    Check if an image is a duplicate for the same user AND across all users.
    
    Args:
        image_path: Path to the image to check
        user_id: User ID to check for duplicates
        similarity_threshold: Threshold for considering images as duplicates
        
    Returns:
        dict: {
            'duplicate_same_user': bool,
            'duplicate_different_user': bool,
            'similarity_same_user': float,
            'similarity_different_user': float,
            'matches_same_user': list,
            'matches_different_user': list
        }
    """
    similar = search_similar_images(image_path, top_k=10, similarity_threshold=similarity_threshold)
    
    user_hash = embeddings_db.hash_user_id(user_id)
    
    # Filter for same user (by hashed user_id) 
    same_user_matches = [s for s in similar 
                        if s['metadata'].get('user_id_hash') == user_hash 
                ]   
    
    # Filter for different users 
    different_user_matches = [s for s in similar 
                             if s['metadata'].get('user_id_hash') != user_hash
                           ]   
    
    duplicate_same_user = len(same_user_matches) > 0
    duplicate_different_user = len(different_user_matches) > 0
    
    similarity_same_user = max([s['similarity'] for s in same_user_matches], default=0.0)
    similarity_different_user = max([s['similarity'] for s in different_user_matches], default=0.0)
    
    return {
        "duplicate_same_user": duplicate_same_user,
        "duplicate_different_user": duplicate_different_user,
        "similarity_same_user": similarity_same_user,
        "similarity_different_user": similarity_different_user,
        "matches_same_user": same_user_matches,
        "matches_different_user": different_user_matches
    }

# Initialize index on module import
load_index()

# ============================================================================
# JSON Input/Output Support
# ============================================================================

def process_json_input(input_json, output_json):
    """
    Process images from JSON input file for reverse image search.
    
    Searches for similar/duplicate images in a batch of images.
    
    Args:
        input_json: Path to input JSON file with image paths
        output_json: Path to output JSON file for results
        
    Returns:
        dict: Summary of processing results
        
    Example input JSON:
        {
          "images": [
            {"id": "test_001", "user_id": "user_001", "image_path": "path/to/image1.jpg"},
            {"id": "test_002", "user_id": "user_001", "image_path": "path/to/image2.jpg"}
          ]
        }
    
    Example output JSON:
        {
          "module": "reverse_image",
          "summary": {...},
          "results": [
            {
              "id": "test_001",
              "user_id": "user_001",
              "image_path": "...",
              "matches_found": 2,
              "similar_images": [...],
              "timestamp": "..."
            }
          ]
        }
    """
    import json
    from datetime import datetime
    from pathlib import Path
    
    print("=" * 70)
    print("REVERSE IMAGE SEARCH MODULE")
    print("=" * 70)
    print(f"Input file: {input_json}")
    print(f"Output file: {output_json}")
    print(f"Index size: {len(all_embeddings)} embeddings")
    
    # Load input data
    try:
        with open(input_json, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_json}' not found!")
        return None
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in '{input_json}'")
        return None
    
    # Get images from input JSON
    images = data.get("images", [])
    if not images:
        print("⚠️  Warning: No images found in input JSON")
        return None
    
    print(f"\n📊 Processing {len(images)} images for reverse search...\n")
    
    results = []
    for idx, entry in enumerate(images, 1):
        image_path = entry.get("image_path", "")
        image_id = entry.get("id", f"image_{idx}")
        user_id = entry.get("user_id", "unknown")
        
        if not image_path or not Path(image_path).exists():
            print(f"[{idx}/{len(images)}] ❌ {image_id}: File not found - {image_path}")
            results.append({
                "id": image_id,
                "user_id": user_id,
                "image_path": image_path,
                "error": "File not found",
                "matches_found": 0,
                "timestamp": datetime.now().isoformat()
            })
            continue
        
        try:
            # Search for similar images
            similar = search_similar_images(image_path, top_k=10, similarity_threshold=0.7)
            
            # Check for duplicates (high similarity within same user)
            dup_result = find_duplicates(image_path, user_id, similarity_threshold=0.85)
            
            matches_found = len(similar)
            is_duplicate = dup_result['is_duplicate']
            
            # Determine status
            if is_duplicate:
                status = "⚠️  DUPLICATE"
            elif matches_found > 0:
                status = "ℹ️  SIMILAR"
            else:
                status = "✅ UNIQUE"
            
            print(f"[{idx}/{len(images)}] {status} | {image_id} | Matches: {matches_found}")
            
            results.append({
                "id": image_id,
                "user_id": user_id,
                "image_path": image_path,
                "matches_found": matches_found,
                "is_duplicate": is_duplicate,
                "max_similarity": float(dup_result['similarity']),
                "similar_images": [
                    {
                        "similarity": float(s['similarity']),
                        "user_id": s['metadata']['user_id'],
                        "image_path": s['metadata']['image_path']
                    }
                    for s in similar
                ],
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            print(f"[{idx}/{len(images)}] ❌ Error processing {image_id}: {str(e)}")
            results.append({
                "id": image_id,
                "user_id": user_id,
                "image_path": image_path,
                "error": str(e),
                "matches_found": 0,
                "timestamp": datetime.now().isoformat()
            })
    
    # Calculate summary statistics
    total = len(results)
    unique = sum(1 for r in results if r.get('matches_found', 0) == 0)
    duplicates = sum(1 for r in results if r.get('is_duplicate', False))
    errors = sum(1 for r in results if 'error' in r)
    
    summary = {
        "total_processed": total,
        "unique_images": unique,
        "duplicates_detected": duplicates,
        "errors": errors,
        "index_size": len(all_embeddings),
        "timestamp": datetime.now().isoformat()
    }
    
    # Save results to JSON
    output_data = {
        "module": "reverse_image",
        "summary": summary,
        "results": results
    }
    
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"Total processed: {total}")
    print(f"  ✅ Unique images:       {unique}")
    print(f"  ⚠️  Duplicates:          {duplicates}")
    print(f"  ❌ Errors:              {errors}")
    print(f"  📊 Index size:          {summary['index_size']}")
    print(f"\n💾 Results saved to: {output_json}")
    print("=" * 70 + "\n")
    
    return output_data


# ============================================================================
# Example usage
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Reverse Image Search: Find similar and duplicate images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reverse_image.py --input test_data.json --output reverse_results.json
  python reverse_image.py -i batch_images.json -o search_results.json
        """
    )
    parser.add_argument("--input", "-i", default="test_data.json",
                       help="Input JSON file with images (default: test_data.json)")
    parser.add_argument("--output", "-o", default="reverse_image_results.json",
                       help="Output JSON file for results (default: reverse_image_results.json)")
    
    args = parser.parse_args()
    process_json_input(args.input, args.output)


# ===========================================
# Wrapper function for main.py compatibility
# ===========================================
def correct_image(image_path):
    """Wrapper to verify/correct image (returns True if valid)"""
    try:
        from PIL import Image
        img = Image.open(image_path)
        img.verify()
        return True
    except Exception as e:
        return False

