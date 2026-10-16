import os
import json
import urllib.request
import urllib.parse

import pandas as pd
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS


# ============================================================
# 1. إعداد التطبيق
# ============================================================
app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    print("⚠️ تحذير: متغير البيئة GEMINI_API_KEY غير مضبوط.")


# ============================================================
# 2. قراءة ملف CSV
# ============================================================
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


# ============================================================
# 3. بناء طلب المستخدم
# ============================================================
def build_user_request_from_options(options):
