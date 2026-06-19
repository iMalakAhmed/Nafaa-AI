import json
from langchain_core.tools import tool
from rapidfuzz import fuzz, process
from tavily import TavilyClient

tavily_client = TavilyClient(api_key="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")

def load_medical_db():
    try:
        with open("data/medical_products_full.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

MEDICAL_DATABASE = load_medical_db()

@tool
def medical_search(query: str) -> str:
    """
    البحث في قاعدة بيانات الأدوية المحلية.
    استخدم هذه الأداة دائمًا كخطوة أولى عند ذكر اسم دواء لمعرفة تفاصيله ودواعي استعماله، أو عند ذكر حالة مرضية/أعراض للبحث عن الأدوية المتعلقة بها محلياً قبل الانتقال للبحث العام.
    """
    if not MEDICAL_DATABASE or not query:
        return "قاعدة بيانات الأدوية غير متوفرة أو فارغة."
        
    query = query.lower().strip()
    best_score = 0
    best_match = None

    for drug in MEDICAL_DATABASE:
        # Check against names and potential matching properties like usage keys if they exist
        candidates = [drug.get("enName", ""), drug.get("arName", ""), drug.get("key", ""), drug.get("usage", "")]
        candidates = [c.lower() for c in candidates if c.strip()]
        if not candidates:
            continue

        _, score, _ = process.extractOne(query, candidates, scorer=fuzz.token_set_ratio)
        if score > best_score:
            best_score = score
            best_match = drug

    if best_score >= 75 and best_match:
        return f"تم العثور على تطابق في قاعدة البيانات المحلية: {json.dumps(best_match, ensure_ascii=False)}"
    
    return f"لم يتم العثور على نتائج مباشرة لـ '{query}' في قاعدة البيانات المحلية. إذا كان هذا اسم دواء، ابحث عن بدائله وأسعاره عبر tavily_search. وإذا كانت هذه حالة مرضية، استخدم tavily_search لمعرفة الأدوية المناسبة لها أولاً."


@tool
def tavily_search(query: str) -> str:
    """
    البحث على الإنترنت (شامل لجميع الحالات الثلاث):
    1. جلب تعريف وشرح المصطلحات المعرفية والتقنية (Domain-specific knowledge).
    2. معرفة أسعار أو تكاليف استئجار/شراء أي سلعة، جهاز، أو خدمة (Pricing / Renting).
    3. البحث عن الأدوية المناسبة لحالة مرضية معينة، أو البحث عن أسعار الأدوية وبدائلها خارج قاعدة البيانات المحلية.
    """
    if not query:
        return "استعلام البحث فارغ."
    try:
        # Keeping it localized to Egypt for general relevance unless specified
        if "مصر" not in query:
            query += " في مصر"
        res = tavily_client.search(query=query, search_depth="basic", max_results=3)
        results = res.get("results", [])
        
        context = ""
        for idx, r in enumerate(results):
            context += f"المصدر {idx+1}: {r.get('title')} | الرابط: {r.get('url')}\nالملخص: {r.get('content', r.get('snippet'))}\n\n"
        return context if context else "لم يتم العثور على نتائج للبحث."
    except Exception as e:
        return f"فشل الاتصال بالإنترنت: {str(e)}"