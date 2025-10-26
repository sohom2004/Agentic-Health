from pdf2image import convert_from_path
import numpy as np
import easyocr
import os
import ast

# convert pdf files to images for performing OCR
def convert_to_jpg(resp):
    pages = convert_from_path(resp)
    pg_arr = []
    for i, page in enumerate(pages):
        path = f"temp_page_{i}.png"
        page.save(path, "PNG")
        pg_arr.append(path)
    print("pages extracted from pdf")
    return pg_arr

# perform OCR
def get_ocr(imgs):
    print(imgs)
    if isinstance(imgs, str):
        try:
            imgs = ast.literal_eval(imgs)
        except Exception:
            imgs = [imgs]
    
    if not isinstance(imgs, list):
        imgs = [imgs]
    reader = easyocr.Reader(['en'], gpu=True)
    all_text = []
    all_score = []
    for img_path in imgs:
        result = reader.readtext(img_path, detail=1)
        for (_, text, score) in result:
            all_text.append(text)
            all_score.append(score)
    page_content = "\n".join(all_text)
    if all_score:
        avg_confidence = sum(all_score) / len(all_score)
    else: avg_confidence = 0.0
    return {"content": page_content, "confidence": avg_confidence}

# detects file type
def get_file_type(doc_path):
    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"File Not Fount: {doc_path}")
    if doc_path.lower().endswith(".pdf"):
        return "pdf"
    elif doc_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        return "image"
    else: raise ValueError("Unsupported File Type. Must be pdf or image.")