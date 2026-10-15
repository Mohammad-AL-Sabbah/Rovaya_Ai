import os
import json
import pandas as pd
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS

# ═══════════════════════════════════════════════════════════
# 1. إعداد التطبيق
# ═══════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app)

# إعداد Gemini API
API_KEY = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=API_KEY)

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
        places_text += f"مكان {row['id']}: {row['name_ar']} في {row['city']} ({row['region']}). التصنيف: {row['categories']}. الوصف: {row['description_ar']}. التكلفة: {row['avg_cost_usd']}$.\n"
    return places_text

places_context = prepare_places_for_ai()

# ═══════════════════════════════════════════════════════════
# 3. دالة بناء الجملة النصية
# ═══════════════════════════════════════════════════════════
def build_user_request_from_options(options):
    cities = ", ".join(options.get('cities', []))
    days = options.get('number_of_days', 1)
    people = options.get('number_of_people', 1)
    budget = options.get('budget', 100)
    interests = ", ".join(options.get('interests', []))
    start_date = options.get('start_date', '2026-10-15')
    
    return f"أريد رحلة لمدة {days} يوم في {cities}. عدد الأشخاص: {people}. الميزانية: {budget} دولار. الاهتمامات: {interests}. تاريخ البدء: {start_date}."

# ═══════════════════════════════════════════════════════════
# 4. دالة توليد الخطة (مبسطة ومضمونة)
# ═══════════════════════════════════════════════════════════
def generate_trip_plan(user_request):
    prompt = f"""
أنت خبير سياحي فلسطيني. بناء خطة رحلة واقعية بناءً على: {user_request}
الأماكن المتاحة فقط: {places_context}
قواعد: 1. لا تخلط مناطق مختلفة. 2. ابدأ من 09:00. 3. التزم بالميزانية. 4. أعد النتيجة بصيغة JSON فقط.
هيكل JSON المطلوب:
{{
  "trip_title": "عنوان",
  "summary": "ملخص",
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
      "tips": "نصيحة"
    }}],
    "daily_cost": 50
  }}]
}}
"""
    try:
        # استخدام نموذج واحد مستقر ومجاني بحدود عالية
        model = genai.GenerativeModel(
            'gemini-3.5-flash-lite'
           
            
            ,
            generation_config={"response_mime_type": "application/json"}
        )
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        return {"error": str(e)}

# ═══════════════════════════════════════════════════════════
# 5. مسارات الـ API
# ═══════════════════════════════════════════════════════════
@app.route('/api/generate-trip', methods=['POST'])
def api_generate_trip():
    data = request.json
    if not data or 'cities' not in data or 'number_of_days' not in data:
        return jsonify({"success": False, "message": "بيانات غير مكتملة"}), 400
    
    user_request = build_user_request_from_options(data)
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
