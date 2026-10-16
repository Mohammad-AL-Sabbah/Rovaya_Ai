import os
import json
import urllib.request
import urllib.parse
import pandas as pd
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS

# ═══════════════════════════════════════════════════════════
# 1. إعداد التطبيق
# ═══════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    print("⚠️ تحذير: متغير البيئة GEMINI_API_KEY غير مضبوط.")


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
    "id", "name_ar", "name_en", "city", "region",
    "categories", "description_ar", "avg_cost_usd", "visit_duration_min",
]

if not df.empty:
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        print(f"❌ أعمدة مفقودة في CSV: {', '.join(missing_columns)}")


def safe_value(row, column, default="غير متوفر"):
    try:
        value = row.get(column, default)
        if pd.isna(value):
            return default
        return value
    except Exception:
        return default


def prepare_places_for_ai():
    if df.empty:
        return "لا توجد أماكن سياحية محملة حاليًا."
    places = []
    for _, row in df.iterrows():
        places.append(
            f"مكان {safe_value(row, 'id')}: "
            f"{safe_value(row, 'name_ar')} ({safe_value(row, 'name_en')}) في "
            f"{safe_value(row, 'city')} ({safe_value(row, 'region')}). "
            f"التصنيف: {safe_value(row, 'categories')}. "
            f"الوصف: {safe_value(row, 'description_ar')}. "
            f"التكلفة: {safe_value(row, 'avg_cost_usd', 0)}$. "
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

    cities = cities_value if isinstance(cities_value, str) else ", ".join(str(c) for c in cities_value)
    interests = interests_value if isinstance(interests_value, str) else ", ".join(str(i) for i in interests_value)

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
# 4. توليد خطة الرحلة
# ═══════════════════════════════════════════════════════════
def generate_trip_plan(user_request):
    if not API_KEY:
        return {"error": "مفتاح Gemini غير مضبوط."}

    prompt = f"""
أنت خبير سياحي فلسطيني محترف. أنشئ خطة رحلة واقعية بناءً على طلب المستخدم.

طلب المستخدم:
{user_request}

الأماكن المتاحة فقط:
{places_context}

قواعد إلزامية:
1. استخدم الأماكن الموجودة في القائمة فقط، ولا تخترع مكانًا أو place_id.
2. لا تخلط قطاع غزة والضفة الغربية والداخل الفلسطيني في اليوم نفسه.
3. ابدأ الأنشطة الساعة 09:00 صباحًا.
4. التزم بالميزانية.
5. أعد JSON صالحًا فقط، دون Markdown.

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
            generation_config={"response_mime_type": "application/json", "temperature": 0.4},
        )
        response = model.generate_content(prompt)
        if not getattr(response, "text", None):
            return {"error": "أعاد Gemini استجابة فارغة."}
        return json.loads(response.text)
    except json.JSONDecodeError as e:
        print(f"❌ Gemini أعاد JSON غير صالح: {e}")
        return {"error": "تعذر قراءة استجابة الذكاء الاصطناعي."}
    except Exception as e:
        print(f"❌ خطأ في توليد خطة الرحلة: {e}")
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════
# 5. جلب صور ديناميكية حقيقية لكل مكان (بدون hardcoding)
# ═══════════════════════════════════════════════════════════
def search_wikipedia_page(query, headers):
    """
    البحث في Wikipedia وإرجاع عنوان الصفحة الأفضل مطابقة
    """
    try:
        search_url = (
            "https://en.wikipedia.org/w/api.php?"
            + urllib.parse.urlencode({
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 5,
                "srprop": "snippet|wordcount",
                "format": "json",
            })
        )
        
        req = urllib.request.Request(search_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
        
        results = data.get("query", {}).get("search", [])
        
        if results:
            # نرجع أفضل نتيجة (الأولى عادة هي الأدق)
            return results[0].get("title")
        return None
    except Exception as e:
        print(f"️ خطأ في البحث عن '{query}': {e}")
        return None


def get_images_from_wikipedia_page(page_title, headers, max_images=10):
    """
    جلب الصور من صفحة Wikipedia المحددة
    """
    images = []
    try:
        # 1. جلب قائمة الصور من المقالة
        parse_url = (
            "https://en.wikipedia.org/w/api.php?"
            + urllib.parse.urlencode({
                "action": "parse",
                "page": page_title,
                "prop": "images",
                "format": "json",
            })
        )
        
        req = urllib.request.Request(parse_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
        
        image_titles = data.get("parse", {}).get("images", [])
        
        # 2. جلب معلومات كل صورة (نتجاوز الأيقونات والشعارات)
        skip_patterns = [
            "icon", "logo", "flag", "coat_of_arms", "symbol",
            "common", "wikidata", "wikimedia", "disambig",
            "question_book", "ambox", "edit-clear", "portal",
            "commons", "wiki", "ocr", "svg",
        ]
        
        for img_title in image_titles:
            if len(images) >= max_images:
                break
            
            # تخطي الأيقونات والشعارات
            title_lower = img_title.lower()
            if any(pattern in title_lower for pattern in skip_patterns):
                continue
            
            try:
                img_info_url = (
                    "https://commons.wikimedia.org/w/api.php?"
                    + urllib.parse.urlencode({
                        "action": "query",
                        "titles": f"File:{img_title}",
                        "prop": "imageinfo",
                        "iiprop": "url|mime",
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
                        mime = imageinfo[0].get("mime", "")
                        # نتجاوز الصور غير الصالحة (SVG, PDF)
                        if mime in ("image/svg+xml", "application/pdf"):
                            continue
                        
                        img_url = imageinfo[0].get("thumburl") or imageinfo[0].get("url")
                        if img_url and img_url.startswith("https://") and img_url not in images:
                            images.append(img_url)
                            print(f"  📸 {img_title}")
            except Exception:
                continue
                
    except Exception as e:
        print(f"⚠️ خطأ في جلب صور صفحة '{page_title}': {e}")
    
    return images


def get_place_images(place_name_en, city_name="", region_name=""):
    """
    جلب صور ديناميكية حقيقية لكل مكان باستخدام Wikipedia Search
    بدون أي hardcoding - يعمل مع أي مكان في فلسطين أو العالم
    """
    if not place_name_en:
        return []

    place_name_en = str(place_name_en).strip()
    city_name = str(city_name or "").strip()
    region_name = str(region_name or "").strip()
    
    all_images = []
    headers = {"User-Agent": "ROVAYA-Tourism-App/1.0 (tourism application)"}

    # ═══════════════════════════════════════════════════════
    # الخطوة 1: بناء استعلامات بحث ذكية وديناميكية
    # ═══════════════════════════════════════════════════════
    search_queries = []
    
    # الأولوية: اسم المكان + المدينة (الأدق)
    if city_name:
        search_queries.append(f"{place_name_en} {city_name} Palestine")
        search_queries.append(f"{place_name_en}, {city_name}")
    
    # ثم: اسم المكان + فلسطين
    search_queries.append(f"{place_name_en} Palestine")
    
    # ثم: اسم المكان فقط
    search_queries.append(place_name_en)
    
    # أخيراً: اسم المكان + المنطقة
    if region_name:
        search_queries.append(f"{place_name_en} {region_name}")

    print(f"🔍 البحث عن صور لـ: {place_name_en} (المدينة: {city_name})")

    # ═══════════════════════════════════════════════════════
    # الخطوة 2: البحث في Wikipedia عن أفضل صفحة مطابقة
    # ═══════════════════════════════════════════════════════
    best_page_title = None
    
    for query in search_queries:
        print(f"  🔎 بحث: '{query}'")
        page_title = search_wikipedia_page(query, headers)
        
        if page_title:
            print(f"  ✅ وجد صفحة: {page_title}")
            best_page_title = page_title
            break
    
    # ═══════════════════════════════════════════════════════
    # الخطوة 3: جلب الصور من الصفحة
    # ═══════════════════════════════════════════════════════
    if best_page_title:
        # جلب الصورة الرئيسية أولاً
        try:
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(best_page_title)}"
            req = urllib.request.Request(summary_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                summary_data = json.loads(response.read().decode("utf-8"))
            
            thumbnail = summary_data.get("thumbnail")
            if thumbnail and thumbnail.get("source"):
                main_img = thumbnail["source"]
                if main_img.startswith("http://"):
                    main_img = main_img.replace("http://", "https://", 1)
                all_images.append(main_img)
                print(f"  🖼️ الصورة الرئيسية: {main_img}")
        except Exception as e:
            print(f"  ⚠️ تعذر جلب الصورة الرئيسية: {e}")
        
        # جلب باقي الصور من المقالة
        page_images = get_images_from_wikipedia_page(best_page_title, headers, max_images=10)
        for img in page_images:
            if img not in all_images:
                all_images.append(img)

    # ═══════════════════════════════════════════════════════
    # الخطوة 4: إذا لم نجد صور كافية، نبحث في Wikimedia Commons
    # ═══════════════════════════════════════════════════════
    if len(all_images) < 5:
        print(f"  🔄 نحتاج مزيد من الصور ({len(all_images)} فقط)، نبحث في Commons...")
        
        commons_queries = [
            f'"{place_name_en}" "{city_name}"' if city_name else f'"{place_name_en}"',
            f'"{place_name_en}" Palestine',
        ]
        
        for query in commons_queries:
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
                    if title.lower().endswith(".svg"):
                        continue
                    
                    img_info_url = (
                        "https://commons.wikimedia.org/w/api.php?"
                        + urllib.parse.urlencode({
                            "action": "query",
                            "titles": title,
                            "prop": "imageinfo",
                            "iiprop": "url|mime",
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
                            mime = imageinfo[0].get("mime", "")
                            if mime in ("image/svg+xml", "application/pdf"):
                                continue
                            img_url = imageinfo[0].get("url")
                            if img_url and img_url.startswith("https://") and img_url not in all_images:
                                all_images.append(img_url)
                                print(f"  📸 من Commons: {title}")
            except Exception as e:
                print(f"  ⚠️ خطأ في Commons: {e}")

    if len(all_images) < 5:
        print(f"  ⚠️ تم العثور على {len(all_images)} صور فقط")
    
    return all_images[:10]


# ═══════════════════════════════════════════════════════════
# 6. مسارات API
# ═══════════════════════════════════════════════════════════
@app.route("/api/generate-trip", methods=["POST"])
def api_generate_trip():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"success": False, "message": "يجب إرسال JSON."}), 400
    if "cities" not in data or "number_of_days" not in data:
        return jsonify({"success": False, "message": "بيانات غير مكتملة."}), 400
    if df.empty:
        return jsonify({"success": False, "message": "ملف CSV غير محمل."}), 503

    user_request = build_user_request_from_options(data)
    plan = generate_trip_plan(user_request)

    if "error" in plan:
        return jsonify({"success": False, "message": plan["error"]}), 500

    return jsonify({"success": True, "plan": plan}), 200


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "places_loaded": int(len(df)),
        "gemini_api_key_configured": bool(API_KEY),
    }), 200


@app.route("/api/place-details", methods=["POST"])
def api_place_details():
    """إرجاع وصف تفصيلي للمكان مع صور ديناميكية حقيقية."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not data.get("place_name"):
        return jsonify({"success": False, "message": "اسم المكان مطلوب."}), 400
    if df.empty:
        return jsonify({"success": False, "message": "ملف CSV غير محمل."}), 503

    place_name = str(data["place_name"]).strip()

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
        return jsonify({"success": False, "message": "المكان غير موجود."}), 404

    name_ar = str(safe_value(place_info, "name_ar"))
    name_en = str(safe_value(place_info, "name_en"))
    city = str(safe_value(place_info, "city"))
    region = str(safe_value(place_info, "region"))
    categories = str(safe_value(place_info, "categories"))
    base_description = str(safe_value(place_info, "description_ar"))

    if not API_KEY:
        return jsonify({"success": False, "message": "مفتاح Gemini غير مضبوط."}), 500

    prompt = f"""
أنت مرشد سياحي فلسطيني خبير.
اكتب وصفًا تاريخيًا وثقافيًا غنيًا وممتعًا باللغة العربية في ثلاث فقرات قصيرة عن المكان التالي.

الاسم العربي: {name_ar}
الاسم الإنجليزي: {name_en}
المدينة: {city}
المنطقة: {region}
التصنيفات: {categories}
الوصف الأساسي: {base_description}

اكتب بأسلوب جذاب وسلس. لا تختلق حقائق. اذكر معلومة مميزة، ثم نصيحة للزوار.
"""

    try:
        model = genai.GenerativeModel("gemini-3.5-flash-lite")
        response = model.generate_content(prompt)
        description_text = getattr(response, "text", None)
        if not description_text:
            raise ValueError("أعاد Gemini وصفًا فارغًا.")
    except Exception as e:
        print(f"❌ خطأ في توليد وصف المكان: {e}")
        return jsonify({"success": False, "message": f"تعذر توليد الوصف: {str(e)}"}), 500

    # جلب صور ديناميكية حقيقية
    print(f"\n🖼️ ═══════════════════════════════════════════════")
    print(f"🖼️ جاري البحث عن صور لـ: {name_en} ({name_ar})")
    print(f" المدينة: {city} | المنطقة: {region}")
    print(f"═══════════════════════════════════════════════")
    images = get_place_images(name_en, city, region)
    print(f"✅ إجمالي الصور: {len(images)}")
    print(f"═══════════════════════════════════════════════\n")

    main_image = images[0] if images else ""

    return jsonify({
        "success": True,
        "details": description_text,
        "images": images,
        "image_url": main_image,
        "image_count": len(images),
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
