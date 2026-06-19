# -----------------------------
# Embeddings Database Module
# API Integration for Azure SQL Server
# Install dependencies: pip install requests numpy
# -----------------------------

import requests
import hashlib
import json
import numpy as np
from datetime import datetime
import os
import base64
import uuid

# API Configuration
API_BASE_URL = os.environ.get('EMBEDDINGS_API_URL', 'https://nafaa-frfve0gyfyatgzh0.uaenorth-01.azurewebsites.net/api/embeddings')
API_TIMEOUT = 120  # Increased from 30 to allow slow API responses


def hash_user_id(user_id):
    """
    Hash user ID for privacy using SHA-256.
    
    Args:
        user_id: Original user ID string
        
    Returns:
        str: SHA-256 hash of user ID
    """
    return hashlib.sha256(user_id.encode('utf-8')).hexdigest()


def hash_image(image_path):
    """
    Calculate SHA-256 hash of image file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        str: SHA-256 hash of image file
    """
    sha256_hash = hashlib.sha256()
    with open(image_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_file_size(image_path):
    """
    Get file size in bytes.
    
    Args:
        image_path: Path to image file
        
    Returns:
        int: File size in bytes
    """
    return os.path.getsize(image_path)


def save_embedding(user_id, image_path, embedding, request_id=None, metadata=None):
    """
    Save an embedding via API to Azure SQL database.
    
    Args:
        user_id: Original user ID (will be hashed)
        image_path: Path to the image
        embedding: numpy array of the embedding
        request_id: Optional request GUID (string or uuid.UUID)
        metadata: Optional dict with additional metadata
        
    Returns:
        str: EmbeddingId (GUID) from API response
        
    Raises:
        Exception: If API request fails
    """
    # Guard against accidental arg swap (dict passed as request_id)
    if isinstance(request_id, dict) and metadata is None:
        print("[WARN] Dict passed as request_id; swapping to metadata parameter")
        metadata = request_id
        request_id = None
    
    # Generate request ID if not provided
    if request_id is None:
        request_id = str(uuid.uuid4())
    elif isinstance(request_id, uuid.UUID):
        request_id = str(request_id)

    # Hash and calculate metadata
    user_hash = hash_user_id(user_id)
    image_hash = hash_image(image_path)
    file_size = get_file_size(image_path)
    
    # Convert numpy array to base64 string
    embedding_bytes = embedding.astype(np.float32).tobytes()
    embedding_base64 = base64.b64encode(embedding_bytes).decode('utf-8')
    
    # Prepare payload
    payload = {
        "requestId": request_id,
        "userIdHash": user_hash,
        "imagePath": image_path,
        "imageHash": image_hash,
        "fileSizeBytes": file_size,
        "embedding": embedding_base64,
        "metadata": metadata or {}
    }
    
    # Debug logging
    print(f"[DEBUG] Storing embedding:")
    print(f"  - RequestId: {request_id}")
    print(f"  - UserIdHash: {user_hash[:16]}...")
    print(f"  - ImagePath: {image_path}")
    print(f"  - ImageHash: {image_hash[:16]}...")
    print(f"  - FileSizeBytes: {file_size}")
    print(f"  - Embedding length: {len(embedding_base64)} chars")
    print(f"  - Metadata: {metadata}")
    
    # POST to API
    response = requests.post(
        f"{API_BASE_URL}/store",
        json=payload,
        timeout=API_TIMEOUT
    )
    
    if response.status_code not in [200, 201]:
        print(f"[ERROR] API Response Status: {response.status_code}")
        print(f"[ERROR] API Response Body: {response.text}")
        print(f"[ERROR] Full payload was: {json.dumps({k: v if k != 'embedding' else f'{v[:50]}...' for k, v in payload.items()}, indent=2)}")
        raise Exception(f"Failed to save embedding. Status: {response.status_code}, Response: {response.text}")
    
    result = response.json()
    return result.get('embeddingId', result.get('id'))


def load_all_embeddings():
    """
    Load all embeddings from the database via API.
    
    Returns:
        tuple: (embeddings_list, metadata_list)
            - embeddings_list: List of numpy arrays
            - metadata_list: List of dicts with metadata for each embedding
            
    Raises:
        Exception: If API request fails
    """
    # Note: This may need pagination for large datasets
    # For now, fetching all - consider adding pagination if needed
    
    response = requests.get(
        f"{API_BASE_URL}/all",
        timeout=API_TIMEOUT
    )
    
    if response.status_code != 200:
        raise Exception(f"Failed to load embeddings. Status: {response.status_code}, Response: {response.text}")
    
    data = response.json()
    embeddings = []
    metadata_list = []
    
    # Handle both list and dict responses
    if isinstance(data, list):
        items = data
    else:
        items = data.get('embeddings', data.get('data', []))
    
    for item in items:
        # Decode base64 embedding back to numpy array
        embedding_base64 = item.get('embedding')
        if embedding_base64:
            embedding_bytes = base64.b64decode(embedding_base64)
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            embeddings.append(embedding)
            
            # Build metadata dict
            metadata_dict = item.get('metadata', {})
            if isinstance(metadata_dict, str):
                metadata_dict = json.loads(metadata_dict)
            
            # Add core fields to metadata
            metadata_dict.update({
                'embedding_id': item.get('embeddingId', item.get('id')),
                'user_id_hash': item.get('userIdHash'),
                'image_path': item.get('imagePath'),
                'timestamp': item.get('timestamp')
            })
            
            metadata_list.append(metadata_dict)
    
    return embeddings, metadata_list


def load_user_embeddings(user_id):
    """
    Load embeddings for a specific user via API.
    
    Args:
        user_id: User ID (will be hashed)
        
    Returns:
        tuple: (embeddings_list, metadata_list)
        
    Raises:
        Exception: If API request fails
    """
    user_hash = hash_user_id(user_id)
    
    response = requests.get(
        f"{API_BASE_URL}/user/{user_hash}",
        timeout=API_TIMEOUT
    )
    
    if response.status_code != 200:
        raise Exception(f"Failed to load user embeddings. Status: {response.status_code}, Response: {response.text}")
    
    data = response.json()
    embeddings = []
    metadata_list = []
    
    # Handle both list and dict responses
    if isinstance(data, list):
        items = data
    else:
        items = data.get('embeddings', data.get('data', []))
    
    for item in items:
        # Decode base64 embedding back to numpy array
        embedding_base64 = item.get('embedding')
        if embedding_base64:
            embedding_bytes = base64.b64decode(embedding_base64)
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            embeddings.append(embedding)
            
            # Build metadata dict
            metadata_dict = item.get('metadata', {})
            if isinstance(metadata_dict, str):
                metadata_dict = json.loads(metadata_dict)
            
            metadata_dict.update({
                'embedding_id': item.get('embeddingId', item.get('id')),
                'user_id_hash': user_hash,
                'image_path': item.get('imagePath'),
                'timestamp': item.get('timestamp')
            })
            
            metadata_list.append(metadata_dict)
    
    return embeddings, metadata_list


def get_stats():
    """
    Get statistics about stored embeddings via API.
    
    Returns:
        dict: Statistics including total embeddings and unique users
        
    Raises:
        Exception: If API request fails
    """
    # Try stats endpoint (handle if API routing treats 'stats' as ID)
    response = requests.get(
        f"{API_BASE_URL}/stats",
        timeout=API_TIMEOUT
    )
    
    if response.status_code != 200:
        # If stats endpoint fails, return empty stats instead of blocking pipeline
        print(f"[WARN] Stats endpoint unavailable (Status {response.status_code})")
        return {'total_embeddings': 0, 'unique_users': 0, 'available': False}
    
    return response.json()


def check_duplicate_by_hash(image_hash, user_id_hash=None):
    """
    Check if an image with the same hash exists via API.
    
    Args:
        image_hash: SHA-256 hash of the image
        user_id_hash: Optional hashed user ID to filter by user
        
    Returns:
        dict or None: Existing record if found, None otherwise
        
    Raises:
        Exception: If API request fails
    """
    response = requests.get(
        f"{API_BASE_URL}/hash/{image_hash}",
        timeout=API_TIMEOUT
    )
    
    if response.status_code == 404:
        return None
    
    if response.status_code != 200:
        raise Exception(f"Failed to check duplicate. Status: {response.status_code}, Response: {response.text}")
    
    data = response.json()
    
    # If user_id_hash provided, filter results
    if user_id_hash and data.get('userIdHash') != user_id_hash:
        return None
    
    return {
        'embedding_id': data.get('embeddingId', data.get('id')),
        'image_path': data.get('imagePath'),
        'timestamp': data.get('timestamp'),
        'user_id_hash': data.get('userIdHash')
    }


def get_embedding_by_id(embedding_id):
    """
    Get a specific embedding by ID via API.
    
    Args:
        embedding_id: Embedding GUID
        
    Returns:
        dict: Embedding data
        
    Raises:
        Exception: If API request fails
    """
    response = requests.get(
        f"{API_BASE_URL}/{embedding_id}",
        timeout=API_TIMEOUT
    )
    
    if response.status_code == 404:
        return None
    
    if response.status_code != 200:
        raise Exception(f"Failed to get embedding. Status: {response.status_code}, Response: {response.text}")
    
    return response.json()


def delete_embedding(embedding_id):
    """
    Delete an embedding by ID via API.
    
    Args:
        embedding_id: Embedding GUID
        
    Returns:
        bool: True if deleted successfully
        
    Raises:
        Exception: If API request fails
    """
    response = requests.delete(
        f"{API_BASE_URL}/{embedding_id}",
        timeout=API_TIMEOUT
    )
    
    if response.status_code not in [200, 204]:
        raise Exception(f"Failed to delete embedding. Status: {response.status_code}, Response: {response.text}")
    
    return True


def save_index():
    """
    Save index - no-op for API-backed storage.
    API/Database automatically persists all changes.
    """
    pass


def init_csv():
    """
    Initialize - no-op for API storage.
    """
    pass
