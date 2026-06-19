from langchain_core.messages import HumanMessage
from app.state import CaseState
from app.services.llm import llm_model
import json
import re

def safe_parse_json(text: str):
    text = re.sub(r"```(?:json)?", "", text)
    text = text.strip()

    # extract first valid JSON object only
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return {}

    candidate = text[start:end+1]

    # fix trailing commas
    candidate = re.sub(r",\s*([\]}])", r"\1", candidate)

    try:
        return json.loads(candidate)
    except:
        return {}

def report_node(state: CaseState) -> dict:
    evidence = state.get("evidence", {})
    reasoning = state.get("reasoning", {})
    text = state.get("text", "")
    inquiry_history = state.get("inquiry_history", [])

    prompt = f"""
أنت نظام تحليل نهائي متقدم لمؤسسة خيرية إنسانية.

مهمتك: بناء فهم موحد للحالة من جميع الأدلة (نص + صور + بحث + سجل عمليات) ثم إصدار قرار إنساني دقيق.

━━━━━━━━━━━━━━━━━━━━━━
📌 INPUT
━━━━━━━━━━━━━━━━━━━━━━
النص الأصلي: {text}
نتائج الاستدلال: {reasoning}
الأدلة المجمعة: {evidence}
سجل العمليات: {inquiry_history}

━━━━━━━━━━━━━━━━━━━━━━
🚨 قواعد صارمة (مهمة جداً)
━━━━━━━━━━━━━━━━━━━━━━
- ممنوع استخدام أي أمثلة أو سيناريوهات خارج هذه الحالة
- كل الاستنتاجات يجب أن تعتمد فقط على المدخلات الحالية
- إذا كانت المعلومة غير موجودة → اذكر "غير متوفر"
- لا تخترع معلومات طبية أو مالية أو اجتماعية

━━━━━━━━━━━━━━━━━━━━━━
🧠 الهدف الحقيقي
━━━━━━━━━━━━━━━━━━━━━━
أنت لا تلخص.
أنت "تدمج الأدلة" لتكوين صورة واحدة للحالة.

━━━━━━━━━━━━━━━━━━━━━━
🔗 طريقة التفكير
━━━━━━━━━━━━━━━━━━━━━━

1) ما الذي يقوله الشخص فعلياً (Claim)؟
2) ماذا تقول الصور إن وجدت؟
3) ماذا تقول نتائج البحث إن وجدت؟
4) هل توجد تناقضات أو فجوات؟
5) ما الاستنتاج الموحد النهائي؟

━━━━━━━━━━━━━━━━━━━━━━
🧩 تحليل الربط
━━━━━━━━━━━━━━━━━━━━━━
- اربط أي مصطلح غير مفهوم بنتائج البحث فقط إن وجدت
- اربط الصور فقط بما يظهر فعلياً في البيانات
- اربط التكاليف أو المعلومات الخارجية فقط إذا كانت موجودة
- لا تستنتج شيء بدون دليل مباشر

━━━━━━━━━━━━━━━━━━━━━━
🧠 منظور المؤسسة الخيرية
━━━━━━━━━━━━━━━━━━━━━━
فكر كموظف ميداني:
- هل هناك حاجة حقيقية للمساعدة؟
- هل الحالة طارئة أم يمكن تأجيلها؟
- ما أسوأ خطأ ممكن (مساعدة خاطئة أو رفض خاطئ)؟
- هل البيانات كافية لاتخاذ قرار؟

━━━━━━━━━━━━━━━━━━━━━━
🟩 أمثلة (بسيطة جداً)
━━━━━━━━━━━━━━━━━━━━━━
نص: "House burned"
→ طارئ → دعم إسكان

نص: "medical clot"
→ يحتاج تحقق → غير مؤكد/مرجح حسب الأدلة

نص + صورة متناقضة
→ طلب معلومات إضافية

━━━━━━━━━━━━━━━━━━━━━━
📤 OUTPUT (JSON only)
━━━━━━━━━━━━━━━━━━━━━━

{{
  "case_synthesis": "وصف موحد للحالة بناءً على الأدلة البصريه فقط",
  "claim_interpretation": " ما الذي يحتاجه الشخص فعلياً مع توضيح مصدر المعلومه في كلمه",
  "evidence_map": {{
    "text": ["ملخص مختصر للغايه للأدلة النصية"],
    "images": ["ملخص مختصر للغايه للأدلة البصرية"],
    "search": ["نتائج البحث باختصار شديد إن وجدت"]
  }},
  "consistency_check": {{
    "consistent": true/false,
    "conflicts": ["أي تناقضات إن وجدت"],
    "missing_gaps": ["أي معلومات ناقصة"]
  }},
  "severity_assessment": "طارئ جداً | متوسط | عادي",
  "final_decision": "صرف دعم / رفض / طلب معلومات إضافية / تحويل",
  "reasoning": "شرح مختصر يوضح كيف تم ربط الأدلة لاتخاذ القرار"
}}
"""

    response = llm_model.invoke([
        HumanMessage(content=prompt)
    ]).content
    parsed = safe_parse_json(response)
    return {
    "final_output": parsed if parsed else {
        "error": "failed_to_parse_llm_output",
        "raw": response
    }
    }