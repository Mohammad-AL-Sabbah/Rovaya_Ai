import os
import json
import pandas as pd
from google import genai
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# إعداد Gemini API
API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=API_KEY)

# ═══════════════════════════════════════════════════════════
# قراءة ملف CSV
# ═══════════════════════════════════════════════════════════
try:
    df = pd.read_csv('palestine_tourism_data.csv')
    print(f"✅ تم تحميل {len(df)} مكان سياحي من CSV")
except Exception as e:
    print(f" خطأ في قراءة CSV: {e}")
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
# دالة بناء الجملة النصية من الخيارات
# ══════════════════════════════════════════════════════════
def build_user_request_from_options(options):
    """تحويل الخيارات المختارة إلى جملة نصية يفهمها الـ AI"""
    
    cities = ", ".join(options.get('cities', []))
    days = options.get('number_of_days', 1)
    people = options.get('number_of_people', 1)
    budget = options.get('budget', 100)
    interests = ", ".join(options.get('interests', []))
    start_date = options.get('start_date', '2026-10-15')
    activity_level = options.get('activity_level', 'moderate')
    
    # ترجمة مستوى النشاط
    level_map = {
        'relaxed': 'هادئ مع فترات راحة طويلة',
        'moderate': 'متوازن بين الزيارة والراحة',
        'intensive': 'مكثف مع زيارة أكبر عدد من الأماكن'
    }
    level_ar = level_map.get(activity_level, 'متوازن')
    
    # بناء الجملة
    request_text = f"""
أريد رحلة لمدة {days} يوم في {cities}.
عدد الأشخاص: {people} أشخاص.
الميزانية الإجمالية: {budget} دولار.
الاهتمامات: {interests}.
تاريخ البدء: {start_date}.
مستوى النشاط: {level_ar}.
"""
    return request_text

# ══════════════════════════════════════════════════════════
# دالة توليد الخطة
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
    try:
        response = client.models.generate_content(
         model="gemini-3.8-flash",,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": 0.7
            }
        )
        return json.loads(response.text)
    except Exception as e:
        error_msg = str(e)
        print(f"❌ خطأ في توليد الخطة: {error_msg}")
        return {"error": error_msg}

# ═══════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════
@app.route('/api/generate-trip', methods=['POST'])
def api_generate_trip():
    """
    يستقبل خيارات منظمة من التطبيق ويبني منها خطة الرحلة
    
    مثال على الطلب:
    {
        "cities": ["نابلس", "القدس"],
        "number_of_days": 2,
        "number_of_people": 2,
        "budget": 100,
        "interests": ["historical", "food"],
        "start_date": "2026-10-15",
        "activity_level": "moderate"
    }
    """
    data = request.json
    
    if not data:
        return jsonify({"success": False, "message": "البيانات مطلوبة"}), 400
    
    # التحقق من الحقول الأساسية
    if 'cities' not in data or 'number_of_days' not in data:
        return jsonify({"success": False, "message": "المدن وعدد الأيام مطلوبان"}), 400
    
    # بناء الجملة النصية من الخيارات
    user_request = build_user_request_from_options(data)
    
    # توليد الخطة
    plan = generate_trip_plan(user_request)
    
    if "error" in plan:
        return jsonify({"success": False, "message": plan["error"]}), 500
    
    return jsonify({"success": True, "plan": plan}), 200

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "places_loaded": len(df)}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
