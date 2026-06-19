# import os
# from transformers import (
#     pipeline, 
#     CLIPModel, 
#     CLIPProcessor, 
#     Qwen2VLForConditionalGeneration, 
#     AutoProcessor
# )
# from huggingface_hub import snapshot_download

# # Define where you want the models stored inside the container
# BASE_CACHE_DIR = "/app/models"
# os.makedirs(BASE_CACHE_DIR, exist_ok=True)

# # def download_hf_model(repo_id, local_name):
# #     print(f"--- Downloading {repo_id} to {local_name} ---")
# #     snapshot_download(repo_id=repo_id, local_dir=os.path.join(BASE_CACHE_DIR, local_name))

# def main():
#     # # 1. AI Fraud Detection Model
#     # # Used in: fraud_detection.py
#     # download_hf_model("capcheck/ai-image-detection", "fraud_detection")

#     # # 2. CLIP Model (Reverse Image Search)
#     # # Used in: reverse_image.py
#     # download_hf_model("openai/clip-vit-base-patch32", "clip_vit")

#     # # 3. STT Model (Egyptian Arabic Wav2Vec2)
#     # # Used in: stt.py
#     # download_hf_model("IbrahimAmin/egyptian-arabic-wav2vec2-xlsr-53", "stt_model")

#     # # 4. Qwen2-VL Model (Vision Question Answering)
#     # # Used in: vqa.py
#     # # Note: This is large; expect the build process to take a few minutes.
#     # download_hf_model("Qwen/Qwen2-VL-2B-Instruct", "qwen_vl")

#     # print("✅ All models successfully cached locally.")

# if __name__ == "__main__":
#     main()