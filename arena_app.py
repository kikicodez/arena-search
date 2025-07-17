import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import os
import csv

SAVE_DIR = "arena_images"
CSV_PATH = os.path.join(SAVE_DIR, "metadata.csv")

# Utilities
def search_arena_channels(keyword, max_channels=5):
    url = f"https://api.are.na/v2/search/channels?q={keyword}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()['channels'][:max_channels]

def get_blocks_from_channel(slug, max_blocks=10):
    url = f"https://api.are.na/v2/channels/{slug}/contents"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()['contents'][:max_blocks]

def save_image(img_url, title, note, metadata_rows):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    img_data = requests.get(img_url).content
    img_name = img_url.split("/")[-1]
    img_path = os.path.join(SAVE_DIR, img_name)
    with open(img_path, "wb") as f:
        f.write(img_data)
    metadata_rows.append([img_name, title, note, img_url])
    return img_data, img_name

def save_csv(metadata_rows):
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name", "Title", "Note", "Image URL"])
        writer.writerows(metadata_rows)

# UI
st.title("📡 Are.na Image Fetcher")

keyword = st.text_input("Enter keyword to search Are.na channels")

if st.button("Search and Download"):
    if not keyword:
        st.warning("Please enter a keyword.")
    else:
        st.info(f"Searching channels for: **{keyword}**")
        metadata_rows = []

        try:
            channels = search_arena_channels(keyword)
            for channel in channels:
                st.subheader(f"📁 {channel['title']}")
                blocks = get_blocks_from_channel(channel['slug'])
                cols = st.columns(5)  # grid layout
                col_idx = 0

                for block in blocks:
                    if block['class'] == 'Image':
                        img_url = block['image']['original']['url']
                        title = block.get('title', '[No Title]')
                        note = block.get('description', '[No Description]')

                        img_data, img_name = save_image(img_url, title, note, metadata_rows)
                        img = Image.open(BytesIO(img_data))
                        cols[col_idx].image(img, caption=title, use_column_width=True)
                        col_idx = (col_idx + 1) % 5

            save_csv(metadata_rows)
            st.success("✅ Done! Metadata saved.")
            with open(CSV_PATH, "rb") as f:
                st.download_button("📄 Download Metadata CSV", f, file_name="arena_metadata.csv")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
