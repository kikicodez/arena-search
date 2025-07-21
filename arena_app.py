import streamlit as st
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import base64
import time

CLIP_API_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"
HUGGINGFACE_API_TOKEN = st.secrets["HUGGINGFACE_API_TOKEN"]

headers_hf = {
    "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
    "Content-Type": "application/json"
}

def get_clip_score(image_bytes, prompt, retries=2, delay=1):
    payload = {
        "inputs": {
            "image": base64.b64encode(image_bytes).decode("utf-8"),
            "text": prompt
        }
    }
    for _ in range(retries):
        try:
            resp = requests.post(CLIP_API_URL, headers=headers_hf, json=payload, timeout=30)
            if resp.status_code == 200 and isinstance(resp.json(), list):
                return resp.json()[0].get("score", 0.0)
            else:
                st.warning(f"CLIP error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.warning(f"CLIP exception: {e}")
        time.sleep(delay)
    return 0.0

def search_arena_channels(keyword):
    resp = requests.get(f"https://api.are.na/v2/search/channels?q={keyword}", headers={"User-Agent":"Mozilla/5.0"})
    resp.raise_for_status()
    return resp.json().get("channels", [])[:5]

def get_blocks_from_channel(slug):
    resp = requests.get(f"https://api.are.na/v2/channels/{slug}/contents", headers={"User-Agent":"Mozilla/5.0"})
    resp.raise_for_status()
    return resp.json().get("contents", [])[:20]

# UI
st.set_page_config(page_title="Are.na Visual Search (CLIP)", layout="wide")
st.title("🔍 Are.na Visual Search")
keyword = st.text_input("Keyword")
threshold = st.slider("Min visual match score", 0.1, 1.0, 0.3, 0.05)

with st.expander("🧪 Test CLIP"):
    if st.button("Run test with watermelon"):
        resp = requests.get("https://upload.wikimedia.org/.../Watermelon_cross_BNC.jpg")
        score = get_clip_score(resp.content, "watermelon")
        try:
            img = Image.open(BytesIO(resp.content))
            st.image(img, caption=f"Score: {score:.2f}")
        except UnidentifiedImageError:
            st.warning("⚠️ Can't display test image")

if st.button("Search Are.na"):
    if not keyword:
        st.warning("Enter a keyword")
    else:
        cols = st.columns(5)
        idx = 0
        matches = 0
        for ch in search_arena_channels(keyword):
            for block in get_blocks_from_channel(ch["slug"]):
                if block.get("class")
