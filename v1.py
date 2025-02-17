# -*- coding: utf-8 -*-  # エンコーディング宣言

import base64
import io
import openai
import streamlit as st
from PIL import Image
import cv2
import numpy as np
from pyzbar.pyzbar import decode

st.title("OCR & Barcode Reader App")

openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    st.stop()

def extract_text_from_image(image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    client = openai.OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o",  # モデル名を変更
        messages=[
            {"role": "system", "content": "Extract text from the given image."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please extract text from this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            }
        ],
        max_tokens=500
    )

    return response.choices[0].message.content

def read_barcode(image):
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    decoded_objects = decode(image_cv)

    barcodes = []
    for obj in decoded_objects:
        barcode_data = obj.data.decode("utf-8")  # デコード時にUTF-8を指定
        barcode_type = obj.type
        barcodes.append(f"{barcode_type}: {barcode_data}")

    return barcodes if barcodes else ["No barcodes detected"]

image_file = st.camera_input("Take a picture with your camera")

if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption="Captured Image", use_container_width=True)

    with st.spinner("Analyzing text and barcodes..."):
        try:
            extracted_text = extract_text_from_image(image)
            barcodes = read_barcode(image)

            st.subheader("OCR (Text Extraction) Results:")
            st.text_area("Extracted Text", extracted_text, height=200)

            st.subheader("Barcode Reading Results:")
            for barcode in barcodes:
                st.write(barcode)
        except Exception as e:
            st.error(f"An error occurred during processing:{e}")
            st.stop()
