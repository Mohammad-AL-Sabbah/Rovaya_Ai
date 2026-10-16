import os
import json
import urllib.request
import urllib.parse
import pandas as pd
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS

# ══════════════════════════════════════════════════════════
# 1. إعداد التطبيق
# ═══════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    print("️ تحذير: متغير البيئة GEMINI_API_KEY غير مضبوط.")


# ═══════════════════════════════════════════════════════════
# 2. قراءة ملف CSV
# ═══════════════════════════════════════════════════════════
CSV_PATH = os.environ.get("TOURISM_CSV_PATH", "palestine_tourism_data.csv")

try:
    df = pd.read_csv(CSV_PATH)
    print(f"✅ تم تحميل {len(df)} مكان سياحي من CSV: {CSV_PATH}")
except Exception as e:
    print(f"❌ خطأ في قراءة CSV ({CSV_PATH}): {e}")
    df = pd.DataFrame()

REQUIRED_COLUMNS = [
    "id",
    "name_ar",
    "name_en",
    "city",
    "region",
    "categories",
    "description_ar",
    "avg_cost_usd",
    "visit_duration_min",
]

if not df.empty:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        print(f"❌ أعمدة مفقودة في CSV: {', '.join(missing_columns)}")


def safe_value(row, column, default="غير متوفر"):
    """قراءة قيمة من صف CSV مع التعامل مع القيم الفارغة."""
    try:
        value = row.get(column, default)
        if pd.isna(value):
            return default
        return value
    except Exception:
        return default


def prepare_places_for_ai():
    """تجهيز الأماكن المتاحة حتى يستخدم النموذج البيانات الموجودة فقط."""
    if df.empty:
        return "لا توجد أماكن سياحية محملة حاليًا."

    places = []

    for _, row in df.iterrows():
        places.append(
            f"مكان {safe_value(row, 'id')}: "
            f"{safe_value(row, 'name_ar')} "
            f"({safe_value(row, 'name_en')}) في "
            f"{safe_value(row, 'city')} "
            f"({safe_value(row, 'region')}). "
            f"التصنيف: {safe_value(row, 'categories')}. "
            f"الوصف: {safe_value(row, 'description_ar')}. "
            f"التكلفة: {safe_value(row, 'avg_cost_usd', 0)} دولار. "
            f"المدة: {safe_value(row, 'visit_duration_min', 60)} دقيقة."
        )

    return "\n".join(places)


places_context = prepare_places_for_ai()


# ═══════════════════════════════════════════════════════════
# 3. بناء طلب المستخدم
# ═══════════════════════════════════════════════════════════
def build_user_request_from_options(options):
    cities_value = options.get("cities", [])
    interests_value = options.get("interests", [])

    if isinstance(cities_value, str):
        cities = cities_value
    else:
        cities = ", ".join(str(city) for city in cities_value)

    if isinstance(interests_value, str):
        interests = interests_value
    else:
        interests = ", ".join(str(item) for item in interests_value)

    days = options.get("number_of_days", 1)
    people = options.get("number_of_people", 1)
    budget = options.get("budget", 100)
    start_date = options.get("start_date", "2026-10-15")

    return (
        f"أريد رحلة لمدة {days} يوم في {cities}. "
        f"عدد الأشخاص: {people}. "
        f"الميزانية الإجمالية: {budget} دولار. "
        f"الاهتمامات: {interests}. "
        f"تاريخ البدء: {start_date}."
    )


# ═══════════════════════════════════════════════════════════
# 4. توليد خطة الرحلة باستخدام Gemini
# ═══════════════════════════════════════════════════════════
def generate_trip_plan(user_request):
    if not API_KEY:
        return {
            "error": "مفتاح Gemini غير مضبوط. أضف GEMINI_API_KEY إلى متغيرات البيئة."
        }

    prompt = f"""
أنت خبير سياحي فلسطيني محترف. أنشئ خطة رحلة واقعية بناءً على طلب المستخدم.

طلب المستخدم:
{user_request}

الأماكن المتاحة فقط:
{places_context}

قواعد إلزامية:
1. استخدم الأماكن الموجودة في القائمة فقط، ولا تخترع مكانًا أو place_id.
2. لا تخلط قطاع غزة والضفة الغربية والداخل الفلسطيني في اليوم نفسه.
3. ابدأ الأنشطة الساعة 09:00 صباحًا، واجعل الجدول واقعيًا مع وقت للتنقل والراحة.
4. التزم بالميزانية الإجمالية وعدد الأشخاص.
5. استخدم التواريخ والمدن المطلوبة قدر الإمكان.
6. اجعل place_id مطابقًا تمامًا لمعرف المكان في القائمة.
7. أعد JSON صالحًا فقط، دون Markdown أو أي شرح خارجه.
8. اجعل التكاليف بالدولار، وميّز بين تكلفة الشخص والتكلفة اليومية الإجمالية.
9. إذا كانت المعلومات غير كافية، لا تخترع بيانات مؤكدة؛ استخدم تقديرات معقولة.

هيكل JSON المطلوب:
{{
  "trip_title": "عنوان الرحلة",
  "summary": "ملخص الرحلة",
  "total_estimated_cost": 0,
  "days": [
    {{
      "day_number": 1,
      "date": "2026-10-15",
      "city": "المدينة",
      "theme": "موضوع اليوم",
      "activities": [
        {{
          "time": "09:00",
          "end_time": "11:00",
          "place_id": 1,
          "place_name": "اسم المكان",
          "activity_type": "visit",
          "description": "وصف النشاط",
          "cost_per_person": 5,
          "tips": "نصيحة للزائر"
        }}
      ],
      "daily_cost": 50
    }}
  ]
}}
"""

    try:
        model = genai.GenerativeModel(
            "gemini-3.5-flash-lite",
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.4,
            },
        )

        response = model.generate_content(prompt)

        if not getattr(response, "text", None):
            return {"error": "أعاد Gemini استجابة فارغة."}

        return json.loads(response.text)

    except json.JSONDecodeError as e:
        print(f"❌ Gemini أعاد JSON غير صالح: {e}")
        return {"error": "تعذر قراءة استجابة الذكاء الاصطناعي بصيغة JSON."}
    except Exception as e:
        print(f"❌ خطأ في توليد خطة الرحلة: {e}")
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════
# 5. جلب صور متعددة من Wikimedia Commons و Wikipedia
# ═══════════════════════════════════════════════════════════
def get_place_images(place_name_en, city_name="", region_name=""):
    """
    جلب 5 صور على الأقل للمكان من Wikimedia Commons و Wikipedia
    يركز على الصور المحددة للمكان وليس صور عشوائية
    """
    if not place_name_en:
        return []

    place_name_en = str(place_name_en).strip()
    city_name = str(city_name or "").strip()
    region_name = str(region_name or "").strip()

    all_images = []
    headers = {
        "User-Agent": "ROVAYA-Tourism-App/1.0 (tourism application)"
    }

    # ══════════════════════════════════════════════════════
    # الطريقة 1: جلب الصور من صفحة Wikipedia (الأكثر دقة)
    # ═══════════════════════════════════════════════════════
    search_titles = [
        f"Sebastia, Nablus",
        f"Sebastia",
        f"{place_name_en}, Palestine",
        f"{place_name_en}, {city_name}" if city_name else f"{place_name_en}",
    ]

    for title in search_titles:
        if len(all_images) >= 5:
            break
            
        try:
            wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
            req = urllib.request.Request(wiki_url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
                
                # الصورة الرئيسية من Wikipedia
                thumbnail = data.get("thumbnail")
                if thumbnail and thumbnail.get("source"):
                    image_url = thumbnail["source"]
                    if image_url.startswith("http://"):
                        image_url = image_url.replace("http://", "https://", 1)
                    
                    if image_url not in all_images:
                        all_images.append(image_url)
                        print(f"✅ صورة رئيسية من Wikipedia: {image_url}")
                
                # جلب صور إضافية من صفحة Wikipedia
                page_title = data.get("title", "").replace(" ", "_")
                if page_title:
                    # الحصول على جميع الصور من المقالة
                    images_url = f"https://en.wikipedia.org/w/api.php?action=parse&page={page_title}&prop=images&format=json"
                    req2 = urllib.request.Request(images_url, headers=headers)
                    
                    with urllib.request.urlopen(req2, timeout=10) as response2:
                        images_data = json.loads(response2.read().decode("utf-8"))
                        
                        image_titles = images_data.get("parse", {}).get("images", [])
                        
                        # جلب معلومات كل صورة
                        for img_title in image_titles[:15]:  # نفحص أول 15 صورة
                            if len(all_images) >= 5:
                                break
                                
                            try:
                                img_info_url = (
                                    "https://commons.wikimedia.org/w/api.php?"
                                    + urllib.parse.urlencode({
                                        "action": "query",
                                        "titles": f"File:{img_title}",
                                        "prop": "imageinfo",
                                        "iiprop": "url|extmetadata",
                                        "iiurlwidth": 800,
                                        "format": "json",
                                    })
                                )
                                
                                req3 = urllib.request.Request(img_info_url, headers=headers)
                                
                                with urllib.request.urlopen(req3, timeout=8) as response3:
                                    img_data = json.loads(response3.read().decode("utf-8"))
                                    
                                    pages = img_data.get("query", {}).get("pages", {})
                                    for page in pages.values():
                                        imageinfo = page.get("imageinfo", [])
                                        if imageinfo:
                                            img_url = imageinfo[0].get("url")
                                            if img_url and img_url.startswith("https://"):
                                                if img_url not in all_images:
                                                    all_images.append(img_url)
                                                    print(f"✅ صورة إضافية من Wikipedia: {img_title}")
                                                    
                                                    if len(all_images) >= 5:
                                                        return all_images
                                                        
                            except Exception as e:
                                continue
                                
        except Exception as e:
            continue

    # ══════════════════════════════════════════════════════
    # الطريقة 2: البحث المباشر في Wikimedia Commons
    # ═══════════════════════════════════════════════════════
    search_queries = [
        f'"{place_name_en}" "{city_name}" Palestine',
        f'"{place_name_en}" archaeological site',
        f'"{place_name_en}" ruins',
        f'"{place_name_en}" heritage',
    ]

    for query in search_queries:
        if len(all_images) >= 5:
            break
            
        try:
            search_url = (
                "https://commons.wikimedia.org/w/api.php?"
                + urllib.parse.urlencode({
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srnamespace": 6,  # Namespace للصور فقط
                    "srlimit": 10,
                    "srprop": "title",
                    "format": "json",
                })
            )
            
            req = urllib.request.Request(search_url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
                
                results = data.get("query", {}).get("search", [])
                
                for result in results:
                    if len(all_images) >= 5:
                        break
                        
                    title = result.get("title", "")
                    
                    # تخطي ملفات SVG
                    if title.lower().endswith(".svg"):
                        continue
                    
                    # جلب معلومات الصورة
                    img_info_url = (
                        "https://commons.wikimedia.org/w/api.php?"
                        + urllib.parse.urlencode({
                            "action": "query",
                            "titles": title,
                            "prop": "imageinfo",
                            "iiprop": "url",
                            "iiurlwidth": 800,
                            "format": "json",
                        })
                    )
                    
                    req2 = urllib.request.Request(img_info_url, headers=headers)
                    
                    with urllib.request.urlopen(req2, timeout=8) as response2:
                        img_data = json.loads(response2.read().decode("utf-8"))
                        
                        pages = img_data.get("query", {}).get("pages", {})
                        for page in pages.values():
                            imageinfo = page.get("imageinfo", [])
                            if imageinfo:
                                img_url = imageinfo[0].get("url")
                                if img_url and img_url.startswith("https://"):
                                    if img_url not in all_images:
                                        all_images.append(img_url)
                                        print(f"✅ صورة من Commons: {title}")
                                        
        except Exception as e:
            print(f"⚠️ خطأ في البحث عن '{query}': {e}")
            continue

    if len(all_images) < 5:
        print(f"⚠️ تم العثور على {len(all_images)} صور فقط (أقل من 5)")
    
    return all_images[:10]  # نرجع maximum 10 صور


# ═══════════════════════════════════════════════════════════
# 6. مسارات API
# ═══════════════════════════════════════════════════════════
@app.route("/api/generate-trip", methods=["POST"])
def api_generate_trip():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "success": False,
            "message": "يجب إرسال جسم الطلب بصيغة JSON."
        }), 400

    if "cities" not in data or "number_of_days" not in data:
        return jsonify({
            "success": False,
            "message": "بيانات غير مكتملة: cities و number_of_days مطلوبان."
        }), 400

    if df.empty:
        return jsonify({
            "success": False,
            "message": "ملف بيانات الأماكن السياحية غير محمل أو فارغ."
        }), 503

    user_request = build_user_request_from_options(data)
    plan = generate_trip_plan(user_request)

    if "error" in plan:
        return jsonify({
            "success": False,
            "message": plan["error"]
        }), 500

    return jsonify({
        "success": True,
        "plan": plan
    }), 200


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "places_loaded": int(len(df)),
        "gemini_api_key_configured": bool(API_KEY),
    }), 200


@app.route("/api/place-details", methods=["POST"])
def api_place_details():
    """إرجاع وصف تفصيلي للمكان مع 5 صور على الأقل."""

    data = request.get_json(silent=True)

    if not isinstance(data, dict) or not data.get("place_name"):
        return jsonify({
            "success": False,
            "message": "اسم المكان مطلوب في place_name."
        }), 400

    if df.empty:
        return jsonify({
            "success": False,
            "message": "ملف بيانات الأماكن السياحية غير محمل."
        }), 503

    place_name = str(data["place_name"]).strip()

    # البحث بالاسم العربي أو الإنجليزي
    place_info = None
    search_name = place_name.casefold()

    for _, row in df.iterrows():
        name_ar = str(safe_value(row, "name_ar", "")).strip()
        name_en = str(safe_value(row, "name_en", "")).strip()

        if (
            search_name == name_ar.casefold()
            or search_name == name_en.casefold()
            or search_name in name_ar.casefold()
            or search_name in name_en.casefold()
        ):
            place_info = row
            break

    if place_info is None:
        return jsonify({
            "success": False,
            "message": "المكان غير موجود في قاعدة البيانات."
        }), 404

    name_ar = str(safe_value(place_info, "name_ar"))
    name_en = str(safe_value(place_info, "name_en"))
    city = str(safe_value(place_info, "city"))
    region = str(safe_value(place_info, "region"))
    categories = str(safe_value(place_info, "categories"))
    base_description = str(safe_value(place_info, "description_ar"))

    # توليد الوصف بالذكاء الاصطناعي
    if not API_KEY:
        return jsonify({
            "success": False,
            "message": "مفتاح Gemini غير مضبوط على الخادم."
        }), 500

    prompt = f"""
أنت مرشد سياحي فلسطيني خبير.

اكتب وصفًا تاريخيًا وثقافيًا غنيًا وممتعًا باللغة العربية في ثلاث فقرات قصيرة عن المكان التالي.

الاسم العربي: {name_ar}
الاسم الإنجليزي: {name_en}
المدينة: {city}
المنطقة: {region}
التصنيفات: {categories}
الوصف الأساسي من قاعدة البيانات: {base_description}

اكتب بأسلوب جذاب وسلس. لا تختلق حقائق تاريخية غير مؤكدة.
اذكر معلومة مميزة موثوقة قدر الإمكان، ثم قدم نصيحة عملية للزوار.
"""

    try:
        model = genai.GenerativeModel(
            "gemini-3.5-flash-lite"
        )
        response = model.generate_content(prompt)
        description_text = getattr(response, "text", None)

        if not description_text:
            raise ValueError("أعاد Gemini وصفًا فارغًا.")

    except Exception as e:
        print(f"❌ خطأ في توليد وصف المكان: {e}")
        return jsonify({
            "success": False,
            "message": f"تعذر توليد وصف المكان: {str(e)}"
        }), 500

    # جلب صور متعددة (5 على الأقل)
    print(f"🖼️ جاري البحث عن صور لـ {name_en} في {city}")
    images = get_place_images(name_en, city, region)

    # الصورة الرئيسية هي أول صورة في القائمة
    main_image = images[0] if images else ""

    return jsonify({
        "success": True,
        "details": description_text,
        "images": images,  # ✅ قائمة صور (5 على الأقل)
        "image_url": main_image,  # ✅ الصورة الرئيسية (للتوافق مع الـ DTO القديم)
        "image_count": len(images),  # ✅ عدد الصور
        "place_name_ar": name_ar,
        "place_name_en": name_en,
        "city": city,
        "region": region,
    }), 200


# ═══════════════════════════════════════════════════════════
# 7. تشغيل التطبيق
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
