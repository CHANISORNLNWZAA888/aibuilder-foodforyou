# aibuilder-foodforyou

AI แนะนำเมนูอาหารจากวัตถุดิบที่มีอยู่ ด้วยการรวบรวมข้อมูลเมนูจากแหล่งต่าง ๆ ทั้งใน Hugging Face และ Kaggle แล้วนำมาประมวลผลด้วยโมเดลภาษา Sentence Transformerในการวิเคราะห์ความคล้ายของ input จากนั้นเรา fine-tune โมเดลกับข้อมูลอาหารไทยแล้วใช้ Cosine Similarity เพื่อเปรียบเทียบความคล้าย เพื่อแนะนำเมนูที่เหมาะสมที่สุด

ลองเล่น deploy https://aibuilder-foodforyou-zf3xejfktrfstrjezanotx.streamlit.app
ดู model https://huggingface.co/Chanisorn/thai-food-mpnet-new-v8
