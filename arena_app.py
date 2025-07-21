import streamlit as st
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import base64
import time

# ✅ CLIP model endpoint (official, stable, authenticated)
CLIP_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/clip-ViT-B-32"
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

def search_arena_channels(keyword, max_channels=5):
    resp = requests.get(f"https://api.are.na/v2/search/channels?q={keyword}", headers={"User-Agent":"Mozilla/5.0"})
    resp.raise_for_status()
    return resp.json().get("channels", [])[:max_channels]

def get_blocks_from_channel(slug, max_blocks=20):
    resp = requests.get(f"https://api.are.na/v2/channels/{slug}/contents", headers={"User-Agent":"Mozilla/5.0"})
    resp.raise_for_status()
    return resp.json().get("contents", [])[:max_blocks]

# --- UI ---
st.set_page_config(page_title="Are.na Visual Search (CLIP)", layout="wide")
st.title("🔍 Are.na Visual Search (CLIP-powered)")
st.write("Type a keyword, test CLIP, then see only visually matching images!")

keyword = st.text_input("Keyword (e.g., 'watermelon', 'poster', 'zine')")
threshold = st.slider("Min visual match score", min_value=0.1, max_value=1.0, value=0.3, step=0.01)

# --- Test Button ---
with st.expander("🧪 Test CLIP with known image"):
    if st.button("Run test with watermelon"):
        resp = requests.get("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Watermelon_cross_BNC.jpg/640px-Watermelon_cross_BNC.jpg")
        score = get_clip_score(resp.content, "watermelon")
        try:
            img = Image.open(BytesIO(resp.content))
            st.image(img, caption=f"CLIP Score: {score:.2f}")
        except UnidentifiedImageError:
            st.warning("⚠️ Couldn't decode test image.")

# --- Main Search ---
if st.button("Search Are.na"):
    if not keyword:
        st.warning("Please enter a keyword.")
    else:
        st.info(f"Searching for images visually similar to: **{keyword}**")
        cols = st.columns(5)
        idx = 0
        matches = 0

        for ch in search_arena_channels(keyword):
            for block in get_blocks_from_channel(ch["slug"]):
                if block.get("class") != "Image":
                    continue
                try:
                    img_url = block["image"]["original"]["url"]
                    resp = requests.get(img_url, headers={"User-Agent":"Mozilla/5.0"}, timeout=15)
                    if not resp.headers.get("Content-Type","").startswith("image/"):
                        continue
                    img_bytes = resp.content
                    score = get_clip_score(img_bytes, keyword)
                    if score >= threshold:
                        img = Image.open(BytesIO(img_bytes))
                        title = block.get("title","")
                        caption = title + f"\nScore: {score:.2f}" if title else f"Score: {score:.2f}"
                        cols[idx % 5].image(img, caption=caption, use_column_width=True)
                        idx += 1
                        matches += 1
                except Exception:
                    continue

        if matches == 0:
            st.warning("No matches found. Try a broader keyword or lower threshold.")
