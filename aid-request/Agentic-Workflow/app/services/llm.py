from langchain_ollama import ChatOllama

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL = "qwen2.5:7b"

# إعدادات الـ LLM
llm_model = ChatOllama(
    model=MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    num_ctx=8192,
    num_predict=1024
)

def llm(prompt: str) -> str:
    try:
        response = llm_model.invoke(prompt)
        return response.content
    except Exception as e:
        print("❌ Ollama Error:", e)
        return ""
