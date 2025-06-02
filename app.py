import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# Load model with caching
@st.cache_resource
def load_model():
    hf_token = "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"
    model = SentenceTransformer("Chanisorn/thai-food-mpnet-tuned", use_auth_token=hf_token)
    return SentenceTransformer('Chanisorn/thai-food-mpnet-tuned')

# Load CSV data with caching
@st.cache_data
def load_data():
    df = pd.read_csv("Copy of Copy of 500 lable แล้ว ไม่มีปริมาณ superclean + มีของคาวของหวาน + query - Sheet1.csv")
    return df

# Compute embeddings once
@st.cache_resource
def embed_corpus(sentences):
    return model.encode(sentences, convert_to_tensor=True)

# Load model and data
model = load_model()
df = load_data()
corpus = df[["วัตถุดิบ_ไม่มีปริมาณ", "query1 อยากกินอาหารครบ"].tolist()]
corpus_embeddings = embed_corpus(corpus)

# Streamlit UI
st.title("🍜 Thai Food Search ")
query = st.text_input("กรอกคำค้นหาเมนู หรือส่วนผสมที่อยากกิน:")

if query:
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

    top_k = min(10, len(corpus))
    results = torch.topk(scores, k=top_k)

    st.subheader("📌 เมนูแนะนำ:")
    for idx in results.indices:
        row = df.iloc[idx]
        st.markdown(f"""
        - **{row['th_name']}**
        - 🧂 **ส่วนผสม:** {row.get('ingredients', 'ไม่ระบุ')}
        - 🏷️ **หมวดหมู่:** {row.get('category', 'ไม่ระบุ')}
        """)
