import streamlit as st
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
import torch
import datetime
import pytz

# API Keys
HF_TOKEN = st.secrets["hf_token"]
GOOGLE_API_KEY = st.secrets["google_api_key"]
CSE_ID = st.secrets["cse_id"]


# Greeting 
def get_time_greeting():
    tz = pytz.timezone("Asia/Bangkok")
    now = datetime.datetime.now(tz).time()
    if datetime.time(5, 0) <= now < datetime.time(12, 0):
        return "☀️ สวัสดีตอนเช้า! วันนี้อยากกินอะไรดี?"
    elif datetime.time(12, 0) <= now < datetime.time(17, 0):
        return "🌤️ สวัสดีตอนบ่าย! หิวหรือยัง อยากกินอะไรดี?"
    elif datetime.time(17, 0) <= now < datetime.time(22, 0):
        return "🌇 สวัสดีตอนเย็น! หาอะไรกินกันเถอะ"
    else:
        return "🌙 ดึกแล้วน้า~ แต่ถ้าหิวก็มาหาเมนูกินกัน!"

# Load models and data
@st.cache_resource
def load_search_model():
    return SentenceTransformer("Chanisorn/thai-food-mpnet-new-v8", use_auth_token=HF_TOKEN)

@st.cache_data
def load_data():
    return pd.read_csv("thai_food_data2.csv")

@st.cache_resource
def embed_corpus(_model, texts):
    return _model.encode(texts, convert_to_tensor=True)

# Google image search
def google_image_search(query, num_images=1):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "cx": CSE_ID,
        "key": GOOGLE_API_KEY,
        "searchType": "image",
        "num": num_images
    }
    res = requests.get(url, params=params)
    data = res.json()
    if "items" in data:
        return [item["link"] for item in data["items"]]
    return []

# ปรับขนาดรูปภาพเหลือ 80% และจัดกลาง
def display_images(image_urls):
    for url in image_urls:
        try:
            st.markdown(
                f"""
                <div style='text-align: center;'>
                    <img src="{url}" style="width:80%; border-radius: 12px; margin-top: 10px;" />
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception:
            st.write("❌ ไม่สามารถโหลดรูปภาพได้")

# Load all resources
search_model = load_search_model()
df = load_data()
corpus_texts = (df["วัตถุดิบ_ไม่มีปริมาณ"].fillna("") + " " + df["query1 อยากกินอาหารครบ"].fillna("")).tolist()
corpus_embeddings = embed_corpus(search_model, corpus_texts)

# UI styling 
st.markdown("""
    <style>
        .greeting {
            font-size: 28px;
            color: var(--text-color);  /* 👈 Match current theme */
            text-align: center;
            animation: fadeIn 2s ease-in-out;
            margin-top: 50px;
            margin-bottom: 30px;
        }
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(-10px); }
            100% { opacity: 1; transform: translateY(0); }
        }
    </style>
""", unsafe_allow_html=True)

# 👋 Greeting section
if "query_sent" not in st.session_state:
    st.session_state.query_sent = False

if not st.session_state.query_sent:
    greeting = get_time_greeting()
    st.markdown(f"<div class='greeting'>{greeting}</div>", unsafe_allow_html=True)

# 📥 Input section
query = st.text_input("กรอกชื่อเมนูหรือส่วนผสมที่อยากกิน 👇")

if query:
    st.session_state.query_sent = True

    # 🔍 Find similar dishes
    query_embedding = search_model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_k = min(3, len(df))
    results = torch.topk(scores, k=top_k)

    st.subheader("📌 เมนูที่แนะนำ:")

    for idx in results.indices:
        row = df.iloc[idx.item()]
        dish_name = row["ชื่ออาหาร"]
        st.markdown(f"### 🍽️ {dish_name}")
        st.markdown(f"- 🧂 **ส่วนผสม:** {row.get('วัตถุดิบ_ไม่มีปริมาณ', 'ไม่ระบุ')}")
        image_urls = google_image_search(dish_name, num_images=1)
        display_images(image_urls)
