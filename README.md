## aibuilder-foodforyou

AI แนะนำเมนูอาหารจากวัตถุดิบที่มีอยู่ ด้วยการรวบรวมข้อมูลเมนูจากแหล่งต่าง ๆ ทั้งใน Hugging Face และ Kaggle แล้วนำมาประมวลผลด้วยโมเดลภาษา Sentence Transformerในการวิเคราะห์ความคล้ายของ input จากนั้นเรา fine-tune โมเดลกับข้อมูลอาหารไทยแล้วใช้ Cosine Similarity เปรียบเทียบความคล้าย เพื่อแนะนำเมนูที่เหมาะสมที่สุด

แนะนำไฟล์

folder data 

Recipe_to_nutrition_ใช้จริง.ipynb, recipe_to_nutrition_ภาคต่อ.ipynb โดยทั้งสองอันนี้ใช้ไฟล์ csv = ingredientdnewnew.csv, เมนูไม่มีจุดแล้ว.csv เพื่อเติมสารอาหาร

notebook

ล่าสุดใช้อันนี้ รันได้โมเดลเลย.ipynb = โมเดลที่ใช้ โดยใช้ข้อมูลจาก Copy of 500 lable แล้ว ไม่มีปริมาณ superclean + มีของคาวของหวาน + queryครบ - Sheet1.csv

โดยโมเดลนี้จะได้ https://huggingface.co/Chanisorn/thai-food-mpnet-new-v8
notebook อื่นๆ คือ notebook ที่ลองจนกว่าจะเป็นตัว thai-food-mpnet-new-v8

ลองเล่น deploy https://aibuilder-foodforyou-zf3xejfktrfstrjezanotx.streamlit.app
