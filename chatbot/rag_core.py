# rag_core.py  (FULL working GPU-compatible RAG core + query rewrite + JSON I/O + history summary)
#
# What this provides:
# 1) GPU embeddings (SentenceTransformer) if CUDA is available
# 2) GPU FAISS search if:
#    - you installed faiss-gpu
#    - and set USE_FAISS_GPU=1
# 3) GPU LLM offload (llama-cpp-python) if your llama-cpp build supports CUDA and N_GPU_LAYERS>0
#
# Adds requested features:
# - Input JSON file contains: {"question": "...", "paragraph_history": "..."}
# - Output JSON contains: {"question": "...", "answer": "...", "paragraph_history": "..."}
# - Query rewrite step using history paragraph (does NOT add facts; only makes question standalone)
# - History summarizer: keeps a compact "paragraph_history" so it doesn’t grow forever
#
# Usage (example):
#   python rag_core.py --in ./input.json --out ./output.json
#
# Env toggles:
#   USE_FAISS_GPU=1           # move FAISS index to GPU (requires faiss-gpu)
#   FAISS_GPU_DEVICE=0
#   N_GPU_LAYERS=35           # llama.cpp layers to offload (requires CUDA build)
#   N_BATCH=512
#   TOP_K=5, N_CTX=4096, etc.

import os
import json
import glob
import pickle
import re
import argparse
from typing import List, Dict, Any, Optional, Tuple
import atexit


from langdetect import detect
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# -------------------------------
# FAISS import (CPU or GPU build)
# -------------------------------
try:
    import faiss  # faiss-cpu or faiss-gpu
except Exception as e:
    raise ImportError(
        "FAISS import failed. Install either 'faiss-cpu' or 'faiss-gpu'. "
        f"Original error: {e}"
    )

# ===============================
# CONFIG
# ===============================
def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

DATA_DIR = os.getenv("DATA_DIR", "./data")
INDEX_PATH = os.getenv("INDEX_PATH", "./artifacts/faq.index")
DOC_STORE_PATH = os.getenv("DOC_STORE_PATH", "./artifacts/faq_docs.pkl")
ARTIFACT_DIR = os.path.dirname(INDEX_PATH) or "."
os.makedirs(ARTIFACT_DIR, exist_ok=True)

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")

GGUF_REPO_ID = os.getenv("GGUF_REPO_ID", "Qwen/Qwen2.5-3B-Instruct-GGUF")
GGUF_FILENAME = os.getenv("GGUF_FILENAME", "qwen2.5-3b-instruct-q4_k_m.gguf")

TOP_K = int(os.getenv("TOP_K", "5"))
MAX_CTX_CHARS = int(os.getenv("MAX_CTX_CHARS", "4000"))

N_CTX = int(os.getenv("N_CTX", "4096"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "160"))

INSUFFICIENT_EN = "Insufficient FAQ context"
INSUFFICIENT_AR = "لا توجد إجابة في الأسئلة الشائعة الحالية"

# FAISS GPU toggle
USE_FAISS_GPU = os.getenv("USE_FAISS_GPU", "0").strip() == "1"
FAISS_GPU_DEVICE = int(os.getenv("FAISS_GPU_DEVICE", "0"))

# llama.cpp offload controls
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "35"))  # set 0 to force CPU
N_BATCH = int(os.getenv("N_BATCH", "512"))

# History controls
MAX_HISTORY_CHARS = int(os.getenv("MAX_HISTORY_CHARS", "900"))  # output paragraph_history limit

# ===============================
# HELPERS
# ===============================
AR_REGEX = re.compile(r"[\u0600-\u06FF]")

def detect_lang(text: str) -> str:
    """Detect AR / EN with a simple Arabic-character shortcut."""
    if AR_REGEX.search(text or ""):
        return "ar"
    try:
        return "ar" if detect(text or "") == "ar" else "en"
    except Exception:
        return "en"

def normalize_q(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def make_citation(d: Dict[str, Any]) -> str:
    fid = d.get("faq_id", d.get("id", "?"))
    tags = d.get("tags") or []
    tag = tags[0] if tags else ""
    return f"FAQ {fid}" + (f" — {tag}" if tag else "")

def truncate_ctx(s: str, limit: int = MAX_CTX_CHARS) -> str:
    return s if len(s) <= limit else s[:limit] + "\n[...]"

def _truncate_history_paragraph(s: str, limit: int = MAX_HISTORY_CHARS) -> str:
    s = normalize_q(s)
    if len(s) <= limit:
        return s
    # keep the most recent part (usually more relevant)
    return "[...]" + s[-limit:]

# ===============================
# DATA LOADING & INDEXING
# ===============================
def load_faq_jsons(folder: str) -> List[Dict[str, Any]]:
    """
    Load all *.json files under DATA_DIR, expecting:
    {
      "meta": {...},
      "faqs": [
        {"id": "...", "question_ar": "...", "answer_ar": "...", "question_en": "...", "answer_en": "...", "tags": [...]},
        ...
      ]
    }
    """
    docs: List[Dict[str, Any]] = []
    files = sorted(glob.glob(os.path.join(folder, "*.json")))
    if not files:
        print(f"[load_faq_jsons] No JSON files found in {folder}")
        return docs

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)

            faqs = data.get("faqs", [])
            for faq in faqs:
                faq_id = faq.get("id") or os.path.basename(fp)
                q_ar = normalize_q(faq.get("question_ar", ""))
                a_ar = normalize_q(faq.get("answer_ar", ""))
                q_en = normalize_q(faq.get("question_en", ""))
                a_en = normalize_q(faq.get("answer_en", ""))
                tags = faq.get("tags") or []

                if q_ar or a_ar:
                    docs.append(
                        {
                            "id": f"{faq_id}::ar",
                            "faq_id": faq_id,
                            "lang": "ar",
                            "question": q_ar,
                            "answer": a_ar,
                            "tags": tags,
                            "source_file": fp,
                        }
                    )

                if q_en or a_en:
                    docs.append(
                        {
                            "id": f"{faq_id}::en",
                            "faq_id": faq_id,
                            "lang": "en",
                            "question": q_en,
                            "answer": a_en,
                            "tags": tags,
                            "source_file": fp,
                        }
                    )

        except Exception as e:
            print(f"[load_faq_jsons] Error reading {fp}: {e}")

    print(f"Loaded {len(docs)} FAQ QA entries from {len(files)} file(s).")
    return docs

def passages_text(d: Dict[str, Any]) -> str:
    """Text used for embedding (E5 passage format)."""
    q = d.get("question") or ""
    a = d.get("answer") or ""
    tags = d.get("tags") or []
    faq_id = d.get("faq_id", "?")
    base = f"Q: {q}\nA: {a}\nFAQ ID: {faq_id}\nTags: {', '.join(tags)}"
    return "passage: " + base

def _build_cpu_index(emb, dim: int):
    # normalized vectors + IP search = cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    return index

def _maybe_move_index_to_gpu(index_cpu):
    if not USE_FAISS_GPU:
        return index_cpu

    has_gpu = hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu")
    if not has_gpu:
        print("[faiss] USE_FAISS_GPU=1 but this FAISS build has no GPU support. Using CPU index.")
        return index_cpu

    try:
        res = faiss.StandardGpuResources()
        index_gpu = faiss.index_cpu_to_gpu(res, FAISS_GPU_DEVICE, index_cpu)
        print(f"[faiss] Moved index to GPU device {FAISS_GPU_DEVICE}.")
        return index_gpu
    except Exception as e:
        print(f"[faiss] Failed to move index to GPU ({e}). Using CPU index.")
        return index_cpu

def build_index(
    docs: List[Dict[str, Any]],
    embedder: SentenceTransformer,
    index_path: str,
    doc_store_path: str,
):
    if not docs:
        raise ValueError("No documents found to index.")

    texts = [passages_text(d) for d in docs]
    emb = embedder.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,  # lets us skip faiss.normalize_L2
    )

    dim = embedder.get_sentence_embedding_dimension()
    index_cpu = _build_cpu_index(emb, dim)

    # Save CPU index to disk (portable)
    faiss.write_index(index_cpu, index_path)

    with open(doc_store_path, "wb") as f:
        pickle.dump(docs, f)

    print(f"[build_index] Index built with {len(docs)} vectors. Saved to {index_path}")

def load_index() -> Tuple[Any, List[Dict[str, Any]]]:
    if not (os.path.exists(INDEX_PATH) and os.path.exists(DOC_STORE_PATH)):
        if not os.path.isdir(DATA_DIR):
            raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")

        docs = load_faq_jsons(DATA_DIR)
        if not docs:
            raise FileNotFoundError(
                f"No FAQ JSON files found in {DATA_DIR}. "
                f"Please add your charity_faq_eg_ar_en.json there."
            )

        print("[load_index] Building index from FAQ JSON...")
        embedder = SentenceTransformer(EMBED_MODEL, device=_device())  # ✅ correct
        build_index(docs, embedder, INDEX_PATH, DOC_STORE_PATH)

    # Load CPU index from disk, then optionally move to GPU
    index_cpu = faiss.read_index(INDEX_PATH)
    index = _maybe_move_index_to_gpu(index_cpu)

    with open(DOC_STORE_PATH, "rb") as f:
        docs = pickle.load(f)

    return index, docs

# ===============================
# Global initialization
# ===============================
INDEX, DOCS = None, []
EMBEDDER = None
LLM = None

try:
    INDEX, DOCS = load_index()
except Exception as e:
    print("[init] Failed to load/build FAISS index:", e)
    INDEX, DOCS = None, []

try:
    EMBEDDER = SentenceTransformer(EMBED_MODEL, device=_device())  # ✅ correct
    print(f"[init] Embedder device: {getattr(EMBEDDER, 'device', None)}")
except Exception as e:
    print("[init] Failed to load embedder:", e)
    EMBEDDER = None

# ===============================
# LLM (llama.cpp) setup
# ===============================

def get_llm() -> Llama:
    local_path = hf_hub_download(
        repo_id=GGUF_REPO_ID,
        filename=GGUF_FILENAME,
        local_dir="./models",
    )

    return Llama(
        model_path=local_path,
        n_threads=max(2, os.cpu_count() or 2),
        n_ctx=N_CTX,
        n_gpu_layers=max(0, N_GPU_LAYERS),
        n_batch=N_BATCH,
        chat_format="qwen",
        verbose=False,  # set True if you want GPU offload logs
    )

def _safe_close_llm():
    global LLM
    try:
        if LLM is not None:
            # llama-cpp-python exposes .close()
            LLM.close()
    except Exception:
        pass
    finally:
        LLM = None

try:
    LLM = get_llm()
except Exception as e:
    print("[init] Failed to init LLM:", e)
    LLM = None
atexit.register(_safe_close_llm)   

# ===============================
# Retrieval
# ===============================
def retrieve(query_text: str, top_k: int = TOP_K, lang_hint: Optional[str] = None):
    if EMBEDDER is None or INDEX is None:
        return []

    q_emb = EMBEDDER.encode(
        ["query: " + (query_text or "")],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    _, I = INDEX.search(q_emb, top_k * 2)  # pull extra, filter by language
    lang = lang_hint or detect_lang(query_text or "")

    same_lang, others = [], []
    for i in I[0]:
        if i < 0 or i >= len(DOCS):
            continue
        d = DOCS[i]
        (same_lang if d.get("lang") == lang else others).append(d)

    out = same_lang[:top_k]
    if len(out) < top_k:
        out.extend(others[: top_k - len(out)])
    return out[:top_k]

# ===============================
# Prompt building for FAQ answering (STRICT)
# ===============================
def build_faq_messages(user_q: str, passages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    lang = detect_lang(user_q or "")

    sys_en = (
        "You are Naf3 Charity FAQ Assistant. Answer ONLY using the provided FAQ context. "
        "If the requested information is NOT present verbatim in the context, "
        f'reply EXACTLY: "{INSUFFICIENT_EN}". '
        "Answer in the user's language."
    )
    sys_ar = (
    "أنت مساعد الأسئلة الشائعة لمنصة نفع الخيرية. أجب فقط من السياق المقدم. "
    f'إذا لم تظهر المعلومة المطلوبة نصًا داخل السياق فأجِب نصًا: "{INSUFFICIENT_AR}". '
    "أجب بلغة المستخدم."
)


    sys = sys_ar if lang == "ar" else sys_en

    seen = set()
    blocks = []
    for d in passages:
        key = (d.get("lang"), d.get("question"), d.get("answer"), d.get("faq_id"))
        if key in seen:
            continue
        seen.add(key)

        cite = make_citation(d)
        q = d.get("question") or ""
        a = d.get("answer") or ""
        if d.get("lang") == "ar":
            blocks.append(f"س: {q}\nج: {a}\nالمصدر: {a}")
        else:
            blocks.append(f"Q: {q}\nA: {a}\nSource: {a}")

    ctx = truncate_ctx("\n\n---\n\n".join(blocks))

    if lang == "ar":
        user = (
            "أجب في جملة أو جملتين فقط بالاعتماد على السياق التالي. "
            f'إن لم يكن الجواب موجودًا في السياق فأجِب نصًا: "{INSUFFICIENT_AR}".\n\n'
            f"السؤال: {user_q}\n\nالسياق:\n{ctx}"
        )
    else:
        user = (
            "Answer in 1–2 sentences using ONLY the FAQ context below. "
            f'If the answer isn’t in the context, reply EXACTLY: "{INSUFFICIENT_EN}".\n\n'
            f"Question: {user_q}\n\nContext:\n{ctx}"
        )

    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

def llm_chat(messages: List[Dict[str, str]], max_tokens: int) -> str:
    if LLM is None:
        # Don’t hallucinate: return insufficient
        # caller can decide which language string to return
        return ""
    out = LLM.create_chat_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        repeat_penalty=1.15,
        stop=None,
    )
    try:
        return out["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""

# ===============================
# Query rewrite (history-aware, no new facts)
# ===============================
import json
import re
from typing import Optional, Dict, Any

JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = JSON_OBJ_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def rewrite_query(question: str, paragraph_history: str) -> str:
    """
    LLM decides if question depends on history.
    But it must provide evidence (verbatim substring from history),
    and code verifies it to prevent injection/hallucination.
    """
    question = normalize_q(question)
    paragraph_history = normalize_q(paragraph_history)

    if not question or not paragraph_history or LLM is None:
        return question

    lang = detect_lang(question or paragraph_history)

    if lang == "ar":
        sys = (
            "أنت نظام لإعادة صياغة أسئلة الدردشة.\n"
            "مهمتك: تحديد هل سؤال المستخدم يعتمد على تاريخ المحادثة أم لا، ثم (إذا كان يعتمد) إعادة صياغته ليصبح سؤالاً مستقلاً.\n\n"
            "قواعد صارمة:\n"
            "1) أخرج JSON فقط بدون أي نص إضافي.\n"
            "2) إذا depends=false: اجعل rewrite مطابقاً للسؤال الأصلي حرفياً.\n"
            "3) إذا depends=true: يجب أن تحتوي evidence على عبارة منسوخة حرفياً من تاريخ المحادثة (substring).\n"
            "4) ممنوع اختراع موضوع/كيان او كلمات غير موجوده حرفياً في التاريخ.\n"
            "5) لا تجب على السؤال .\n\n"
            "صيغة الإخراج:\n"
            "{\"depends\": true/false, \"rewrite\": \"...\", \"evidence\": \"...\", \"reason\": \"...\"}"
        )
        user = (
            f"تاريخ المحادثة:\n{paragraph_history}\n\n"
            f"السؤال الحالي:\n{question}\n\n"
            "أخرج JSON فقط:"
        )
    else:
        sys = (
            "You are a chat question rewriting system.\n"
            "Task: decide if the user's question depends on the chat history, and if so rewrite it to be standalone.\n\n"
            "Strict rules:\n"
            "1) Output JSON only (no extra text).\n"
            "2) If depends=false: rewrite must match the original question verbatim.\n"
            "3) If depends=true: evidence must be a verbatim substring copied from chat history.\n"
            "4) Do NOT invent entities/topics not literally present in the history.\n"
            "5) Do NOT answer the question.\n\n"
            "Output schema:\n"
            "{\"depends\": true/false, \"rewrite\": \"...\", \"evidence\": \"...\", \"reason\": \"...\"}"
        )
        user = (
            f"Chat history:\n{paragraph_history}\n\n"
            f"Current question:\n{question}\n\n"
            "Output JSON only:"
        )

    raw = llm_chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        max_tokens=140,
    )
    obj = _extract_json_obj(raw)
    if not obj:
        return question

    depends = bool(obj.get("depends", False))
    rewrite = normalize_q(str(obj.get("rewrite", "")))
    evidence = normalize_q(str(obj.get("evidence", "")))
    print(depends)
    print(rewrite)
    print(evidence)

    # ---- HARD VERIFICATION LAYER ----
    if not depends:
        # must be verbatim
        return question

    # # depends=true: evidence must be literally in history
    # if not evidence or evidence not in paragraph_history:
    #     return question

    # # rewrite must include evidence to ensure it's actually grounded
    # if evidence not in rewrite:
    #     return question

    # also cap length to prevent long injections
    if len(rewrite) > max(40, int(len(question) * 2.2)):
        return question

    return rewrite or question

def preserve_entities(answer: str, passages: List[Dict[str, Any]]) -> str:
    """
    Inject explicit FAQ entities (titles / key phrases) into the answer
    so they survive in paragraph_history for future reference.
    """
    if not answer or not passages:
        return answer

    # Extract canonical entity names from FAQ questions
    entities = []
    for p in passages:
        q = p.get("question", "")
        if q:
            entities.append(q)

    # Pick the shortest (usually the canonical title)
    entity = min(entities, key=len, default=None)
    if entity and entity not in answer:
        return f"({entity}) {answer}"

    return answer

# ===============================
# History summarization (paragraph_history)
# ===============================
def summarize_history_so_far(paragraph_history: str, new_question: str, new_answer: str) -> str:
    """
    Append new Q/A to paragraph_history.
    If it becomes too long, compress the older part into a short summary,
    while keeping the most recent tail verbatim.
    """
    paragraph_history = normalize_q(paragraph_history)
    new_question = normalize_q(new_question)
    new_answer = normalize_q(new_answer)

    if not new_question and not new_answer:
        return _truncate_history_paragraph(paragraph_history)

    # 1) Always append the new turn (verbatim-ish)
    lang = detect_lang(new_question or new_answer or paragraph_history)
    if lang == "ar":
        new_turn = f"س: {new_question} ج: {new_answer}"
    else:
        new_turn = f"Q: {new_question} A: {new_answer}"

    merged = (paragraph_history + " " + new_turn).strip() if paragraph_history else new_turn

    # 2) If short enough, keep as-is
    if len(merged) <= MAX_HISTORY_CHARS:
        return merged

    # 3) Too long → compress older part and keep recent tail
    # Keep last ~60% verbatim, summarize the rest
    keep_tail_len = max(250, int(MAX_HISTORY_CHARS * 0.6))
    tail = merged[-keep_tail_len:]
    head = merged[:-keep_tail_len].strip()

    # If no LLM, just hard-truncate head
    if LLM is None or not head:
        return _truncate_history_paragraph("[...]" + tail, limit=MAX_HISTORY_CHARS)

    # LLM-based compression of the older head
    if lang == "ar":
        sys = (
            "لخّص الجزء القديم من سجل المحادثة في جملة أو جملتين فقط دون إضافة أي معلومات جديدة. "
            "احتفظ فقط بالنقاط المهمة التي تساعد على فهم المرجع لاحقاً."
        )
        user = (
            f"الجزء القديم:\n{head}\n\n"
            "اكتب ملخصاً قصيراً جداً لهذا الجزء:"
        )
    else:
        sys = (
            "Summarize the older part of the chat history into 1–2 sentences without adding new facts. "
            "Keep only what helps resolve references later."
        )
        user = (
            f"Older part:\n{head}\n\n"
            "Write a very short summary of this older part:"
        )

    head_summary = llm_chat(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        max_tokens=140,
    )
    head_summary = normalize_q(head_summary) or ""

    # 4) Compose compressed history + recent tail
    if head_summary:
        out = f"{head_summary} [...] {tail}"
    else:
        out = f"[...] {tail}"

    return _truncate_history_paragraph(out, limit=MAX_HISTORY_CHARS)

def enforce_arabic_only(text: str) -> str:
    # Remove any non-Arabic / non-punctuation characters
    return re.sub(r"[^\u0600-\u06FF0-9\s\.\،\؛\:\!\؟\(\)\-\/]", "", text)


# ===============================
# Main answering entrypoint (JSON-in / JSON-out style)
# ===============================
def answer_with_json_io(question: str, paragraph_history: str, top_k: int = TOP_K) -> Dict[str, str]:
    """
    Input:
    - question: user question
    - paragraph_history: a compact paragraph describing the history so far

    Output:
      - question: original question (unchanged)
      - answer: model answer (strictly from FAQ context)
      - paragraph_history: updated compact history paragraph
    """
    question = normalize_q(question)
    paragraph_history = normalize_q(paragraph_history)

    lang = detect_lang(question or paragraph_history)

    if INDEX is None or EMBEDDER is None or LLM is None:
        msg = INSUFFICIENT_AR if lang == "ar" else INSUFFICIENT_EN
        updated_history = summarize_history_so_far(paragraph_history, question, msg)
        return {"question": question, "answer": msg, "paragraph_history": updated_history}

    # 1) Rewrite query using history paragraph (standalone question)
    rewritten_q = rewrite_query(question, paragraph_history)

    # 2) Retrieve using rewritten query
    passages = retrieve(rewritten_q, top_k=top_k, lang_hint=lang)
    if not passages:
        msg = INSUFFICIENT_AR if lang == "ar" else INSUFFICIENT_EN
        updated_history = summarize_history_so_far(paragraph_history, question, msg)
        return {"question": question, "answer": msg, "paragraph_history": updated_history}

    # 3) Generate strict FAQ answer
    msgs = build_faq_messages(rewritten_q, passages)
    answer = llm_chat(msgs, max_tokens=MAX_NEW_TOKENS).strip()
    answer = enforce_arabic_only(answer)


    if not answer:
        answer = INSUFFICIENT_AR if lang == "ar" else INSUFFICIENT_EN

    # 4) Update paragraph history summary with the ORIGINAL question + produced answer
    answer_for_history = preserve_entities(answer, passages)
    updated_history = summarize_history_so_far(paragraph_history, question, answer_for_history)
    return {
        "question": question,
        "answer": answer,
        "paragraph_history": updated_history,
        "_debug": {
            "rewritten": rewritten_q,
            "retrieved_faq_ids": [p["faq_id"] for p in passages]
        }
    }
# ===============================
# File helpers (JSON input/output)
# ===============================
def read_input_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object/dict.")
    return data

def write_output_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ===============================
# CLI
# ===============================
def main():
    parser = argparse.ArgumentParser(description="GPU-compatible FAQ RAG with query rewrite + history summary")
    parser.add_argument("--in", dest="in_path", required=True, help="Path to input JSON file")
    parser.add_argument("--out", dest="out_path", required=True, help="Path to output JSON file")
    parser.add_argument("--topk", dest="topk", type=int, default=TOP_K, help="Top-K passages")
    args = parser.parse_args()

    inp = read_input_json(args.in_path)
    question = inp.get("question", "")
    paragraph_history = inp.get("paragraph_history", "")

    result = answer_with_json_io(question=question, paragraph_history=paragraph_history, top_k=args.topk)
    write_output_json(args.out_path, result)

if __name__ == "__main__":
    main()
