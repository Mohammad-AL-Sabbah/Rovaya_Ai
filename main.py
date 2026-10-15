import os
import json
import time
import pandas as pd
from groq import Groq
from flask import Flask, request, jsonify
from flask_cors import CORS

# ═══════════════════════════════════════════════════════════
# 1. إعداد التطبيق
# ═══════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app)

# إعداد Groq API
API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=API_KEY)

# ═══════════════════════════════════════════════════════════
# 2. قراءة ملف CSV
# ═══════════════════════════════════════════════════════════
try:
    df = pd.read_csv('palestine_tourism_data.csv')
    print(f"✅ تم تحميل {len(df)} مكان سياحي من CSV")
except Exception as e:
    print(f"❌ خطأ في قراءة CSV: {e}")
    df = pd.DataFrame()

def prepare_places_for_ai():
    places_text = ""
    for index, row in df.iterrows():
        places_text += f"""
--- مكان رقم {row['id']} ---
الاسم: {row['name_ar']} ({row['name_en']})
المنطقة: {row['region']}
المدينة: {row['city']}
التصنيفات: {row['categories']}
الوصف: {row['description_ar']}
التكلفة: {row['avg_cost_usd']}$
المدة: {row['visit_duration_min']} دقيقة
"""
    return places_text

places_context = prepare_places_for_ai()

# ═══════════════════════════════════════════════════════════
# 3. دالة بناء الجملة النصية من الخيارات
# ═══════════════════════════════════════════════════════════
def build_user_request_from_options(options):
    cities = ", ".join(options.get('cities', []))
    days = options.get('number_of_days', 1)
    people = options.get('number_of_people', 1)
    budget = options.get('budget', 100)
    interests = ", ".join(options.get('interests', []))
    start_date = options.get('start_date', '2026-10-15')
    activity_level = options.get('activity_level', 'moderate')
    
    level_map = {
        'relaxed': 'هادئ مع فترات راحة طويلة',
        'moderate': 'متوازن بين الزيارة والراحة',
        'intensive': 'مكثف مع زيارة أكبر عدد من الأماكن'
    }
    level_ar = level_map.get(activity_level, 'متوازن')
    
    return f"""
أريد رحلة لمدة {days} يوم في {cities}.
عدد الأشخاص: {people} أشخاص.
الميزانية الإجمالية: {budget} دولار.
الاهتمامات: {interests}.
تاريخ البدء: {start_date}.
مستوى النشاط: {level_ar}.
"""

# ═══════════════════════════════════════════════════════════
# 4. دالة توليد الخطة مع نظام Retry ذكي
# ═══════════════════════════════════════════════════════════
def generate_trip_plan(user_request):
    prompt = f"""
أنت خبير سياحي فلسطيني محترف. بناء خطة رحلة واقعية بناءً على:
طلب المستخدم: {user_request}

الأماكن المتاحة فقط (استخدم هذه القائمة ولا تخترع أماكن):
{places_context}

قواعد صارمة:
1. لا تخلط مناطق مختلفة في نفس اليوم (غزة مع الضفة).
2. ابدأ من 09:00، اترك وقت للانتقال والغداء (13:00-14:30).
3. التزم بالميزانية المحددة بدقة.
4. أعد النتيجة بصيغة JSON فقط بدون أي نص قبلها أو بعدها.

هيكل JSON المطلوب:
{{
  "trip_title": "عنوان جذاب",
  "summary": "ملخص قصير",
  "total_estimated_cost": 0,
  "days": [{{
    "day_number": 1,
    "date": "2026-10-15",
    "city": "المدينة",
    "theme": "موضوع اليوم",
    "activities": [{{
      "time": "09:00",
      "end_time": "11:00",
      "place_id": 1,
      "place_name": "اسم المكان",
      "activity_type": "visit",
      "description": "وصف النشاط",
      "cost_per_person": 5,
      "tips": "نصيحة مفيدة"
    }}],
    "daily_cost": 50
  }}]
}}
"""
    
    # قائمة النماذج النشطة والمدعومة رسمياً ومجانياً على Groq حالياً
    models_to_try = [
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
    
    for model_name in models_to_try:
        try:
            print(f"🔄 محاولة استخدام نموذج Groq: {model_name}")
            
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "أنت مساعد ذكي يجب أن ترد بصيغة JSON فقط دون أي نص إضافي."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            print(f"✅ نجح مع: {model_name}")
            return json.loads(completion.choices[0].message.content)
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ فشل مع {model_name}: {error_msg}")
            
            # تجاهل أخطاء النماذج الملغاة أو غير الموجودة والمحاولة مع التالي
            if "model_not_found" in error_msg or "decommissioned" in error_msg or "404" in error_msg:
                continue
            elif "503" in error_msg or "UNAVAILABLE" in error_msg:
                print("⏳ انتظار 3 ثوانٍ قبل المحاولة التالية...")
                time.sleep(3)
                continue
            else:
                return {"error": error_msg}
    
    return {"error": "فشلت جميع المحاولات مع نماذج Groq"}

# ═══════════════════════════════════════════════════════════
# 5. مسارات الـ API (Endpoints)
# ═══════════════════════════════════════════════════════════
@app.route('/api/generate-trip', methods=['POST'])
def api_generate_trip():
    data = request.json
    if not data:
        return jsonify({"success": False, "message": "البيانات مطلوبة"}), 400
    
    if 'cities' not in data or 'number_of_days' not in data:
        return jsonify({"success": False, "message": "المدن وعدد الأيام مطلوبان"}), 400
    
    user_request = build_user_request_from_options(data)
    plan = generate_trip_plan(user_request)
    
    if "error" in plan:
        return jsonify({"success": False, "message": plan["error"]}), 500
    
    return jsonify({"success": True, "plan": plan}), 200

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "places_loaded": len(df)}), 200

# ═══════════════════════════════════════════════════════════
# 6. تشغيل التطبيق
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
