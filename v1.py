import streamlit as st
from PIL import Image
import openai
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import io

st.title("OCR & バーコード読み取りアプリ")

# OpenAI APIキーの入力を促す
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please enter your OpenAI API key to proceed.", icon="🗝️")
    st.stop()

# OpenAI APIの設定
def extract_text_from_image(image):
    """
    OpenAI APIを使用して画像内のテキストを抽出する
    """
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()

    client = openai.OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role": "system", "content": "Extract text from the given image."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please extract text from this image."},
                    {"type": "image", "image": image_bytes}
                ]
            }
        ],
        max_tokens=500
    )

    return response.choices[0].message.content

# バーコード読み取り関数
def read_barcode(image):
    """
    画像内のバーコードを検出し、デコードする
    """
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    decoded_objects = decode(image_cv)
    
    barcodes = []
    for obj in decoded_objects:
        barcode_data = obj.data.decode("utf-8")
        barcode_type = obj.type
        barcodes.append(f"{barcode_type}: {barcode_data}")
    
    return barcodes if barcodes else ["バーコードが検出されませんでした"]

# カメラで画像を撮影
image_file = st.camera_input("カメラで画像を撮影")

if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption="撮影した画像", use_column_width=True)

    # OCRとバーコードの解析
    with st.spinner("テキストとバーコードを解析中..."):
        extracted_text = extract_text_from_image(image)
        barcodes = read_barcode(image)
    
    # 結果表示
    st.subheader("OCR（テキスト抽出）結果:")
    st.text_area("抽出されたテキスト", extracted_text, height=200)

    st.subheader("バーコード読み取り結果:")
    for barcode in barcodes:
        st.write(barcode)
