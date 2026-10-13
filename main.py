def generate_trip_plan(user_request):
    """توليد خطة رحلة مع نظام Retry ونماذج بديلة"""
    
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
    
    # قائمة النماذج البديلة (بالترتيب)
    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-flash-latest",
        "gemini-pro-latest"
    ]
    
    for model_name in models_to_try:
        try:
            print(f"🔄 محاولة استخدام: {model_name}")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.7
                }
            )
            print(f"✅ نجح مع: {model_name}")
            return json.loads(response.text)
        except Exception as e:
            error_msg = str(e)
            print(f"❌ فشل مع {model_name}: {error_msg}")
            
            # إذا كان الخطأ 404 (نموذج غير موجود)، انتقل للتالي مباشرة
            if "404" in error_msg or "NOT_FOUND" in error_msg:
                continue
            # إذا كان 503 (ضغط عالٍ)، انتظر 3 ثوانٍ وحاول مرة أخرى
            elif "503" in error_msg or "UNAVAILABLE" in error_msg:
                print("⏳ انتظار 3 ثوانٍ قبل المحاولة التالية...")
                import time
                time.sleep(3)
                continue
            else:
                # خطأ آخر، ارجع الخطأ
                return {"error": error_msg}
    
    # إذا فشلت جميع المحاولات
    return {"error": "فشلت جميع المحاولات مع النماذج المتاحة"}
