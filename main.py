import os
import json
import urllib.request
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
        places_text += f"مكان {row['id']}: {row['name_ar']} في {row['city']} ({row['region']}). التصنيف: {row['categories']}. الوصف: {row['description_ar']}. التكلفة: {row['avg_cost_usd']}$. المدة: {row['visit_duration_min']} دقيقة.\n"
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
# 4. دالة توليد الخطة
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
        model = genai.GenerativeModel(
            'gemini-3.5-flash-lite',
            generation_config={"response_mime_type": "application/json"}
        )
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        return {"error": str(e)}

# ═══════════════════════════════════════════════════════════
# 5. دالة جلب الصور من ويكيميديا (محدثة وذكية!)
# ═══════════════════════════════════════════════════════════
def get_wikimedia_image(place_name_en, city_name=""):
    """جلب صورة ديناميكية من ويكيميديا باستخدام بحث ذكي متعدد الصيغ"""
    try:
        # تحضير كلمات البحث المختلفة لزيادة فرص العثور على صورة
        search_terms = [
            f"{place_name_en}",
            f"{place_name_en}, {city_name}",
            f"Tel {place_name_en}",
            f"{place_name_en} archaeological site"
        ]
        
        for search_term in search_terms:
            try:
                # تنظيف الاسم ليتناسب مع روابط الويب
                clean_term = search_term.replace(" ", "_")
                
                # استخدام API البحث في ويكيميديا كومنز (namespace=6 يعني البحث في الملفات/الصور فقط)
                url = f"https://commons.wikimedia.org/w/api.php?action=query&list=search&srsearch={clean_term}&srnamespace=6&srprop=size&srwhat=text&format=json&srlimit=1"
                
                req = urllib.request.Request(url, headers={'User-Agent': 'RovayaApp/1.0'})
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    
                    if data.get('query') and data['query'].get('search'):
                        # الحصول على عنوان الصفحة التي تحتوي على الصورة
                        title = data['query']['search'][0]['title']
                        
                        # الآن جلب الصورة الفعلية من عنوان الصفحة
                        img_url = f"https://commons.wikimedia.org/w/api.php?action=query&titles={title}&prop=pageimages&format=json&pithumbsize=800"
                        
                        req2 = urllib.request.Request(img_url, headers={'User-Agent': 'RovayaApp/1.0'})
                        with urllib.request.urlopen(req2, timeout=5) as response2:
                            data2 = json.loads(response2.read().decode())
                            pages = data2['query']['pages']
                            for page_id in pages:
                                if 'thumbnail' in pages[page_id]:
                                    print(f"✅ تم العثور على صورة لـ '{search_term}': {pages[page_id]['thumbnail']['source']}")
                                    return pages[page_id]['thumbnail']['source']
            except Exception as e:
                # إذا فشل بحث واحد، ينتقل للتجربة التالية تلقائياً
                continue
        
        return None
    except Exception as e:
        print(f"❌ خطأ عام في جلب الصورة من ويكيميديا: {e}")
        return None

# ═══════════════════════════════════════════════════════════
# 6. مسارات الـ API
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

@app.route('/api/place-details', methods=['POST'])
def api_place_details():
    """توليد وصف تفصيلي لمكان معين مع صورة ديناميكية"""
    data = request.json
    if not data or 'place_name' not in data:
        return jsonify({"success": False, "message": "اسم المكان مطلوب"}), 400
    
    place_name = data['place_name']
    
    # البحث عن المكان في CSV
    place_info = None
    for index, row in df.iterrows():
        if place_name in str(row['name_ar']) or place_name in str(row['name_en']):
            place_info = row
            break
    
    if place_info is None:
        return jsonify({"success": False, "message": "المكان غير موجود في قاعدة البيانات"}), 404
    
    # بناء الـ Prompt
    prompt = f"""أنت مرشد سياحي فلسطيني خبير. 
اكتب وصفاً تاريخياً وثقافياً غنياً وممتعاً في 3 فقرات قصيرة عن المكان التالي:

الاسم: {place_info['name_ar']} ({place_info['name_en']})
المدينة: {place_info['city']}
المنطقة: {place_info['region']}
التصنيفات: {place_info['categories']}
الوصف الأساسي: {place_info['description_ar']}

اذكر معلومة غريبة أو مميزة عنه، وقدم نصيحة ذهبية للزوار.
اكتب باللغة العربية بأسلوب جذاب وسلس.
"""
    
    try:
        # 1. توليد النص بالذكاء الاصطناعي
        model = genai.GenerativeModel('gemini-3.5-flash-lite')
        response = model.generate_content(prompt)
        description_text = response.text
        
        # 2. جلب الصورة ديناميكياً من ويكيميديا (مع تمرير اسم المدينة لتحسين البحث)
        print(f"🖼️ جاري البحث عن صورة لـ: {place_info['name_en']} في {place_info['city']}")
        image_url = get_wikimedia_image(place_info['name_en'], place_info['city'])
        
        if image_url:
            print(f"✅ تم العثور على صورة: {image_url}")
        else:
            print(f"⚠️ لم يتم العثور على صورة، سيتم استخدام صورة بديلة")
            image_url = "https://via.placeholder.com/800x400?text=No+Image+Available"
        
        # 3. إرجاع النص والصورة معاً
        return jsonify({
            "success": True, 
            "details": description_text,
            "image_url": image_url,
            "place_name_ar": place_info['name_ar'],
            "place_name_en": place_info['name_en'],
            "city": place_info['city']
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
