import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# Load model with caching
@st.cache_resource
def load_model():
    hf_token = "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"
    return SentenceTransformer("Chanisorn/thai-food-mpnet-tuned", use_auth_token=hf_token)

# Load data with caching
@st.cache_data
def load_data():
    return pd.read_csv("Copy of Copy of 500 lable แล้ว ไม่มีปริมาณ superclean + มีของคาวของหวาน + query - Sheet1.csv")

# Embed corpus with caching — FIXED: renamed model to _model
@st.cache_resource
def embed_corpus(_model, texts):
    return _model.encode(texts, convert_to_tensor=True)

# Load resources
model = load_model()
df = load_data()

# Prepare corpus text for embedding
corpus_texts = (df["วัตถุดิบ_ไม่มีปริมาณ"].fillna("") + " " + df["query1 อยากกินอาหารครบ"].fillna("")).tolist()
corpus_embeddings = embed_corpus(model, corpus_texts)

# Streamlit UI
st.title("🍜 Thai Food Search")
query = st.text_input("กรอกคำค้นหาเมนู หรือส่วนผสมที่อยากกิน:")

if query:
    # Embed the query
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

    # Get top 3 matches
    top_k = min(3, len(df))
    results = torch.topk(scores, k=top_k)

    st.subheader("📌 เมนูแนะนำ:")

    for idx in results.indices:
        row = df.iloc[idx.item()]  # Convert tensor to int
        st.markdown(f"""
        ### 🍽️ {row['ชื่ออาหาร']}
        - 🧂 **ส่วนผสม:** {row.get('วัตถุดิบ_ไม่มีปริมาณ', 'ไม่ระบุ')}
        """)
