import streamlit as st
import numpy as np
import joblib
from PIL import Image

# إعداد واجهة التطبيق
st.set_page_config(page_title="تصنيف صور الأشعة", layout="centered")
st.title("🩻قراءة و تحليل صور الأشعة بواسطة الذكاء الاصطناعي")

# تحميل النماذج المدربة مسبقًا
pca = joblib.load("pca_model2.joblib")
clf = joblib.load("classifier2_model.joblib")

# خريطة الفئات (حسب المجلدات التي دربنا عليها)
class_map = {
    0: " covid",
    1: "⚠️ normal",
    2: "virus"
}

# رفع صورة جديدة
uploaded_file = st.file_uploader("📂  ارفع صورة الأشعة هنا رجاء", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # معالجة الصورة بنفس خطوات التدريب
    img = Image.open(uploaded_file).convert("L")   # تحويل إلى Grayscale
    img = img.resize((128, 128))                   # نفس الحجم المستخدم في التدريب
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_flatten = img_array.flatten().reshape(1, -1)

    # تطبيق PCA ثم التنبؤ
    img_pca = pca.transform(img_flatten)
    prediction = clf.predict(img_pca)[0]

    # عرض الصورة والنتيجة
    st.image(img, caption="الصورة المرفوعة", use_column_width=True)
    st.subheader(f"📌 التشخيص المتوقع: {class_map[prediction]}")