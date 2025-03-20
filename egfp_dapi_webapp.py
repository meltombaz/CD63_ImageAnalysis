import streamlit as st
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology, io
import numpy as np
import io as pyio

# Set custom page config with pink theme
st.set_page_config(page_title="EGFP & DAPI Cell Analysis", page_icon="🐱", layout="wide")

# Apply custom CSS for pink theme and kitten background
st.markdown(
    """
    <style>
        body {
            background-image: url('https://wallpapercat.com/w/full/e/9/c/24514-1920x1200-desktop-hd-kitten-background-photo.jpg');
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
        .stSlider > div > div > div > div {
            background: #ff69b4 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("EGFP & DAPI Cell Analysis Web App 🐱")

# Upload images
egfp_file = st.file_uploader("Upload EGFP Image (TIF) - Filename must include 'EGFP'", type=["tif"])
dapi_file = st.file_uploader("Upload DAPI Image (TIF) - Filename must include 'DAPI'", type=["tif"])

# Check if the filenames are valid
if egfp_file and "EGFP" not in egfp_file.name:
    st.error("Error: The EGFP image filename must contain 'EGFP'. Please check the file.")
    egfp_file = None

if dapi_file and "DAPI" not in dapi_file.name:
    st.error("Error: The DAPI image filename must contain 'DAPI'. Please check the file.")
    dapi_file = None

# Threshold multiplier input
egfp_threshold_multiplier = st.slider("Adjust EGFP Threshold Multiplier", min_value=1.0, max_value=5.0, value=3.0, step=0.1)

if egfp_file and dapi_file:
    egfp_image = tiff.imread(egfp_file)
    dapi_image = tiff.imread(dapi_file)
    
    ### --- Step 1: Detect EGFP-positive cells --- ###
    egfp_denoised = filters.gaussian(egfp_image, sigma=1)
    egfp_threshold = filters.threshold_otsu(egfp_denoised) * egfp_threshold_multiplier
    egfp_mask = egfp_denoised > egfp_threshold
    egfp_mask = morphology.remove_small_objects(egfp_mask, min_size=10)
    egfp_labels = measure.label(egfp_mask)
    egfp_props = measure.regionprops(egfp_labels)
    egfp_cell_count = len(egfp_props)
    
    ### --- Step 2: Count Total Cells from DAPI --- ###
    dapi_mask = dapi_image > filters.threshold_otsu(dapi_image)
    dapi_mask = morphology.remove_small_objects(dapi_mask, min_size=10)
    dapi_labels = measure.label(dapi_mask)
    dapi_cell_count = len(measure.regionprops(dapi_labels))
    
    ### --- Step 3: Calculate Percentage of EGFP+ Cells --- ###
    egfp_percentage = (egfp_cell_count / dapi_cell_count) * 100 if dapi_cell_count > 0 else 0
    
    st.write(f"**Total DAPI+ Cells:** {dapi_cell_count}")
    st.write(f"**EGFP+ Cells (above threshold):** {egfp_cell_count}")
    st.write(f"**Percentage of EGFP+ Cells:** {egfp_percentage:.2f}%")
    
    ### --- Step 4: Visualizations --- ###
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(egfp_image, cmap='gray')
    axes[0, 0].set_title("Raw EGFP Image")
    
    axes[0, 1].imshow(egfp_mask, cmap='gray')
    axes[0, 1].set_title("EGFP Thresholded Cells")
    
    axes[0, 2].imshow(egfp_image, cmap='gray')
    centroids = np.array([prop.centroid for prop in egfp_props])
    if centroids.size > 0:
        axes[0, 2].scatter(centroids[:, 1], centroids[:, 0], c='red', marker='x', label='EGFP+ Cells')
    axes[0, 2].set_title("EGFP+ Cells Marked")
    axes[0, 2].legend()
    
    axes[1, 0].imshow(dapi_image, cmap='gray')
    axes[1, 0].set_title("Raw DAPI Image")
    
    overlay_raw = np.dstack((np.zeros_like(dapi_image), egfp_image / egfp_image.max(), dapi_image / dapi_image.max()))
    axes[1, 1].imshow(overlay_raw)
    axes[1, 1].set_title("EGFP + DAPI Overlay (Raw)")
    
    overlay_thresholded = np.dstack((np.zeros_like(dapi_mask), egfp_mask.astype(float), dapi_mask.astype(float)))
    axes[1, 2].imshow(overlay_thresholded)
    if centroids.size > 0:
        axes[1, 2].scatter(centroids[:, 1], centroids[:, 0], c='red', marker='x', label='EGFP+ Cells')
    axes[1, 2].set_title("EGFP Thresholded Cells + DAPI")
    axes[1, 2].legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Save the processed image for download
    buf = pyio.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    st.download_button("Download Processed Image", buf, file_name="processed_image.png", mime="image/png")
