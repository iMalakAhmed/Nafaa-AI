import json
import re
from typing import Any, Dict
from langchain_core.messages import HumanMessage
from app.state import CaseState
from app.services.llm import llm_model

def _safe_parse_json(text: str) -> Dict[str, Any]:
    """Safely extract and parse JSON from LLM response."""
    text = re.sub(r"```(?:json)?", "", text)
    text = re.sub(r"\x0c", "", text).strip()
    match = re.search(r"(\{[\s\S]*\})", text)
    if not match:
        return {}
    
    candidate = match.group(1)
    candidate = re.sub(r",\s*([\]}])", r"\1", candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {}

def reasoning_node(state: CaseState) -> dict:
    """
    Decision node: determines whether to use VQA, Search, or Report.
    """
    # 1. استخراج وتحويل البيانات إلى نصوص بسيطة لتجنب خطأ التمرير
    case = state.get("normalized_case", {})
    evidence = state.get("evidence", {})
    images = state.get("images", [])
    text = state.get("text", "")
    
    extracted_text = str(case.get('extracted_text', text))
    risk_level = str(evidence.get('cold_acquisition', {}).get('overall_risk_level', 'unknown'))
    image_count = len(images)
    history = state.get("inquiry_history", [])

    history_summary = ""

    if history:
        history_summary = "\n".join(
            f"- {item.get('type')}: {item.get('target','')}"
            for item in history
            if isinstance(item, dict)
        )
    else:
        history_summary = "لا يوجد"
        # 2. بناء نص الـ Prompt
    prompt_text = f"""
    أنت نظام تخطيط جمع معلومات لمؤسسة خيرية إنسانية.

    مهمتك: تحديد أفضل خطوة لجمع المعلومات، مع تحديد "ماذا يجب أن نعرف بالضبط" قبل اتخاذ القرار.

    ━━━━━━━━━━━━━━━━━━━━━━
    📌 المدخلات
    ━━━━━━━━━━━━━━━━━━━━━━
    النص: {extracted_text}
    مستوى الخطر: {risk_level}
    عدد الصور: {image_count}
    سجل العمليات: {history_summary}

    ━━━━━━━━━━━━━━━━━━━━━━
    🧠 الفكرة الأساسية
    ━━━━━━━━━━━━━━━━━━━━━━
    لا تختار أداة فقط.
    بل حدد:
    - ما الذي نحتاج فهمه؟
    - ما الذي إذا عرفناه سيتغير القرار؟
    - ما الفجوة في الفهم؟

    ━━━━━━━━━━━━━━━━━━━━━━
    🧭 متى نستخدم VQA
    ━━━━━━━━━━━━━━━━━━━━━━
    إذا كانت الصورة تحتوي على:
    - دليل بصري مهم (إصابة / حريق / مستند / تقرير)
    - أو عنصر غير واضح قد يغيّر القرار

    → الهدف: التحقق فقط من وجود عناصر محددة داخل الصورة
    (هل الشيءاو النص موجود أم لا، بدون تفسير أو استنتاج)
    ━━━━━━━━━━━━━━━━━━━━━━
    🧭 متى نستخدم SEARCH (مهم جداً)
    ━━━━━━━━━━━━━━━━━━━━━━
    استخدم SEARCH عندما يظهر في INPUT:

    1) احتياج مادي مباشر
    (كرسي متحرك / طعام / دواء / سكن / دفع إيجار)

    2) حالة صحية أو إعاقة أو مرض
    (MS, paralysis, disability, chronic illness)

    3) طلب دعم مالي أو معيشي

    إذا استخدمت SEARCH، يجب أن تفكر بهذه الطريقة:

    أنت لا تبحث عن كلمة واحدة،
    أنت تبني "خريطة معرفة" تشمل:

    1) تعريف الشيء  
    2) خطورته أو معناه  
    3) تكلفته أو تأثيره  
    4) البدائل  
    5) السياق الواقعي  

    ━━━━━━━━━━━━━━━━━━━━━━
    📌 يجب أن تنتج SEARCH target كالتالي:
    ━━━━━━━━━━━━━━━━━━━━━━
    بدلاً من:
    "منزل محترق"

    اكتب:
    - ما نتائج احتراق المنزل علي الاسره؟
    - تكلفة الحل أو الإصلاح
    - كيق يتم تقدير الضرر
    - لمساعده اسره بيتها احترق خيارات بديلة

    ━━━━━━━━━━━━━━━━━━━━━━
    🧭 متى نستخدم REPORT
    ━━━━━━━━━━━━━━━━━━━━━━
    فقط إذا:
    - لدينا فهم كافٍ لكل العناصر المهمة
    - لا توجد مصطلحات أو أشياء غير مفهومة
    - ولا توجد فجوات تؤثر على القرار

    ━━━━━━━━━━━━━━━━━━━━━━
    🚨 قاعدة ذهبية
    ━━━━━━━━━━━━━━━━━━━━━━
    إذا كان هناك:
    - مصطلح غير مفهوم
    - أو عنصر قد يغير التقييم
    → يجب SEARCH و VQA قبل التقرير

    ━━━━━━━━━━━━━━━━━━━━━━
    🟩 أمثلة

    مثال 1:
    "أعيش في منزل احترق بالكامل ولم يعد صالحًا للسكن"
    → SEARCH:
    - ما مدى الأضرار المعتادة بعد حرائق المنازل الكاملة
    - متوسط تكلفة إعادة بناء منزل في مصر
    - خيارات السكن المؤقت للأسر التي فقدت منزلها
    - أنواع الدعم الإنساني المتوفر في حالات فقدان السكن بسبب الحريق

    مثال 2:
    "تقرير طبي مكتوب فيه وجود جلطة دموية في الساق"
    → SEARCH:
    - ما هي خطورة الجلطة الدموية في الساق عادة
    - متى تعتبر الجلطة حالة طبية طارئة
    - ما خيارات العلاج المتوفرة عادة لهذه الحالة
    - ما مدى تكلفة وتوفر العلاج في الحالات المشابهة

    مثال 3:
    "صورة يظهر فيها جرح عميق في الذراع"
    → VQA

    مثال 4:
    "شخص يطلب مساعدة مالية ويذكر أنه بلا دخل منذ 3 أشهر ولديه أطفال ولا يستطيع دفع الإيجار"
    → REPORT
    ━━━━━━━━━━━━━━━━━━━━━━
    📤 JSON فقط
    ━━━━━━━━━━━━━━━━━━━━━━

    {{
    "next_step": "vqa | search | report",
    "action_details": {{
    "target": [
        "استعلامات دقيقة تغطي تعريف الحالة، مستوى الخطورة، التكلفة، والخيارات المتاحة"
    ],
    "reasoning": "سبب الحاجة لتوسيع المعرفة هو وجود نقص في المعلومات يمنع اتخاذ قرار دقيق أو تقييم صحيح للحالة"
    }}
    }}
    """
    
    # 3. إرسال الرسالة بشكل صحيح باستخدام HumanMessage
    response = llm_model.invoke([HumanMessage(content=prompt_text)]).content
    
    # 4. معالجة القرار
    decision = _safe_parse_json(response)
    
    next_step = str(decision.get("next_step", "report")).strip().lower()
    action_details = decision.get("action_details", {})
    
    # ضمان أن action_details قاموس قبل استخراج القيم
    if not isinstance(action_details, dict):
        action_details = {}
        
    target = action_details.get("target", "")
    action_reasoning = action_details.get("reasoning", "")
    
    if next_step not in {"vqa", "search", "report"}:
        next_step = "report"
    
    current_loop = state.get("loop_count", 0) + 1
    context_pack = {
    "text": extracted_text,
    "risk_level": risk_level,
    "image_count": image_count,
    "history": history_summary,
    "raw_images": images
}
    return {
        "reasoning": {
            "next_step": next_step,
            "question_or_query": target,
            "reasoning": action_reasoning,
            "context": context_pack if next_step == "search" else None
        },
        "loop_count": current_loop
    }