import streamlit as st
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology
import numpy as np
import re
from collections import defaultdict
import pandas as pd

# --- Session state for clear functionality ---
if "clear_uploads" not in st.session_state:
    st.session_state.clear_uploads = False

# --- Page config and style ---
st.set_page_config(page_title="Batch EGFP & DAPI Analysis", page_icon="🐱", layout="wide")
st.markdown(
    """
    <style>
        body {
            background-image: url('https://pbs.twimg.com/media/GmuT894XoAAGm5a?format=jpg&name=4096x4096');
            background-size: cover;
            color: white;
        }
        .stApp {
            background: rgba(255, 182, 193, 0.8);
            padding: 20px;
            border-radius: 15px;
        }
        h1, h2, h3 {
            color: #ff69b4;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Batch EGFP & DAPI Cell Analysis Web App 🐱")

# --- Clear all files button ---
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("🗑️ Clear All Files"):
        st.session_state.clear_uploads = True
        st.experimental_rerun()

# --- Upload files ---
uploaded_files = st.file_uploader(
    "Upload EGFP and DAPI TIFF files",
    type=["tif"],
    accept_multiple_files=True,
    key=None if not st.session_state.clear_uploads else str(np.random.rand())
)
st.session_state.clear_uploads = False  # reset after use

# --- Limit number of uploaded files ---
if uploaded_files and len(uploaded_files) > 20:
    st.error("⚠️ Please upload a maximum of 10 EGFP+DAPI image pairs (20 files total).")
    uploaded_files = uploaded_files[:20]

# --- Threshold control ---
egfp_threshold_multiplier = st.slider(
    "Adjust EGFP Threshold Multiplier",
    min_value=1.0,
    max_value=5.0,
    value=3.0,
    step=0.1
)

# --- Sample key extractor ---
def extract_sample_key(filename):
    parts = filename.replace(".tif", "").split("_")
    if len(parts) >= 3:
        return "_".join(parts[-3:])  # A01f00..._Grid_Condition
    return None

# --- Group files by sample key ---
file_dict = defaultdict(dict)
for file in uploaded_files:
    fname = file.name
    key = extract_sample_key(fname)
    if not key:
        st.warning(f"⚠️ Could not extract key from: {fname}")
    else:
        if "EGFP" in fname.upper():
            file_dict[key]["EGFP"] = file
        elif "DAPI" in fname.upper():
            file_dict[key]["DAPI"] = file

if file_dict:
    st.info(f"🔍 Found {len(file_dict)} matched sample(s).")

results = []

# --- Process each matched pair ---
for sample, files in file_dict.items():
    if "EGFP" in files and "DAPI" in files:
        egfp_image = tiff.imread(files["EGFP"])
        dapi_image = tiff.imread(files["DAPI"])

        # --- EGFP processing ---
        egfp_denoised = filters.gaussian(egfp_image, sigma=1)
        egfp_thresh = filters.threshold_otsu(egfp_denoised) * egfp_threshold_multiplier
        egfp_mask = morphology.remove_small_objects(egfp_denoised > egfp_thresh, min_size=10)
        egfp_labels = measure.label(egfp_mask)
        egfp_props = measure.regionprops(egfp_labels)
        egfp_count = len(egfp_props)

        # --- DAPI processing ---
        dapi_mask = morphology.remove_small_objects(dapi_image > filters.threshold_otsu(dapi_image), min_size=10)
        dapi_labels = measure.label(dapi_mask)
        dapi_count = len(measure.regionprops(dapi_labels))

        # --- Calculate percentage ---
        percentage = (egfp_count / dapi_count) * 100 if dapi_count > 0 else 0
        results.append({
            "Sample": sample,
            "DAPI+ Cells": dapi_count,
            "EGFP+ Cells": egfp_count,
            "EGFP+ %": f"{percentage:.2f}%"
        })

        # --- Visualization ---
        with st.expander(f"🔬 Results for {sample}"):
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes[0, 0].imshow(egfp_image, cmap='gray')
            axes[0, 0].set_title("Raw EGFP")

            axes[0, 1].imshow(egfp_mask, cmap='gray')
            axes[0, 1].set_title("Thresholded EGFP")

            axes[0, 2].imshow(egfp_image, cmap='gray')
            centroids = np.array([prop.centroid for prop in egfp_props])
            if centroids.size > 0:
                axes[0, 2].scatter(centroids[:, 1], centroids[:, 0], c='red', marker='x')
            axes[0, 2].set_title("EGFP+ Cells")

            axes[1, 0].imshow(dapi_image, cmap='gray')
            axes[1, 0].set_title("Raw DAPI")

            overlay_raw = np.dstack((np.zeros_like(dapi_image), egfp_image / egfp_image.max(), dapi_image / dapi_image.max()))
            axes[1, 1].imshow(overlay_raw)
            axes[1, 1].set_title("EGFP + DAPI Overlay")

            overlay_thresh = np.dstack((np.zeros_like(dapi_mask), egfp_mask.astype(float), dapi_mask.astype(float)))
            axes[1, 2].imshow(overlay_thresh)
            if centroids.size > 0:
                axes[1, 2].scatter(centroids[:, 1], centroids[:, 0], c='red', marker='x')
            axes[1, 2].set_title("Thresholded EGFP + DAPI")

            plt.tight_layout()
            st.pyplot(fig)

# --- Summary Table ---
if results:
    st.subheader("📊 Summary Table of Cell Counts")
    df_results = pd.DataFrame(results)
    st.dataframe(df_results)

    # CSV download
    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Summary CSV", data=csv, file_name="egfp_dapi_summary.csv", mime="text/csv")
else:
    if uploaded_files:
        st.warning("⚠️ No valid EGFP + DAPI pairs found among the uploaded files.")
    else:
        st.info("Please upload TIFF files to get started.")
