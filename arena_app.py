import streamlit as st
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import base64
import time

# ✅ Custom deployed CLIP API
CLIP_API_URL = "https://hf.space/embed/chatgpt-openai/clip-score/+/api/predict"
HUGGINGFACE_API_TOKEN = st.secrets["HUGGINGFACE_API_TOKEN"]

headers_hf = {
    "Content-Type": "application/json"
}


def get_clip_score(image_bytes, prompt, retries=3, delay=2):
    for attempt in range(retries):
        try:
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")
            payload = {
                "data": [encoded_image, prompt]
            }
            response = requests.post(CLIP_API_URL, headers=headers_hf, json=payload)
            if response.status_code == 200:
                result = response.json()
                score = result.get("data", [0.0])[0]
                return score
            else:
                st.warning(f"CLIP error (status {response.status_code}): {response.text}")
        except Exception as e:
            st.warning(f"CLIP exception: {e}")
        time.sleep(delay)
    return 0.0

# --- Are.na API ---
def search_arena_channels(keyword, max_channels=5):
    url = f"https://api.are.na/v2/search/channels?q={keyword}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()['channels'][:max_channels]

def get_blocks_from_channel(slug, max_blocks=20):
    url = f"https://api.are.na/v2/channels/{slug}/contents"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()['contents'][:max_blocks]

# --- UI ---
st.set_page_config(page_title="Are.na CLIP Search", layout="wide")
st.title("🔍 Are.na Visual Search (CLIP-powered)")
st.markdown("Find images on Are.na that visually match your concept using CLIP.")

keyword = st.text_input("Enter a visual concept (e.g. 'poster', 'fruit', 'zine')")
threshold = st.slider("Minimum CLIP match score", 0.1, 0.5, 0.28, step=0.01)

# 🧪 Test mode
with st.expander("🧪 Test CLIP with watermelon image"):
    if st.button("Run test match"):
        test_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Watermelon_cross_BNC.jpg/640px-Watermelon_cross_BNC.jpg"
        img_response = requests.get(test_url)
        score = get_clip_score(img_response.content, "watermelon")
        try:
            img = Image.open(BytesIO(img_response.content))
            st.image(img, caption=f"CLIP Score: {score:.2f}")
        except UnidentifiedImageError:
            st.warning("⚠️ Failed to display test image. Format not recognized.")

# 🔍 Search mode
if st.button("Search Are.na"):
    if not keyword:
        st.warning("Please enter a keyword.")
    else:
        st.info(f"Searching visually for: **{keyword}** (CLIP ≥ {threshold:.2f})")
        try:
            channels = search_arena_channels(keyword)
            cols = st.columns(5)
            col_idx = 0
            match_count = 0

            for channel in channels:
                blocks = get_blocks_from_channel(channel["slug"])
                for block in blocks:
                    if block.get("class") == "Image":
                        try:
                            img_url = block["image"]["original"]["url"]
                            img_response = requests.get(img_url, headers={"User-Agent": "Mozilla/5.0"})

                            # ✅ Skip non-image content
                            if not img_response.headers.get("Content-Type", "").startswith("image/"):
                                continue

                            img_bytes = img_response.content
                            score = get_clip_score(img_bytes, keyword)

                            if score >= threshold:
                                try:
                                    img = Image.open(BytesIO(img_bytes))
                                    title = block.get("title", "")
                                    caption = f"{title}\nScore: {score:.2f}" if title else f"Score: {score:.2f}"
                                    cols[col_idx].image(img, caption=caption, use_column_width=True)
                                    col_idx = (col_idx + 1) % 5
                                    match_count += 1
                                except Exception as e:
                                    st.warning(f"⚠️ Image skipped (decode error): {e}")
                                    continue

                        except Exception as e:
                            st.warning(f"⚠️ Failed to load image: {e}")
                            continue

            if match_count == 0:
                st.warning("No visually matching images found. Try a broader keyword or lower threshold.")

        except Exception as e:
            st.error(f"❌ Search failed: {e}")
