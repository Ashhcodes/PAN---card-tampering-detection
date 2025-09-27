# main_app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import imutils
import io
import requests
import os

st.set_page_config(page_title="PAN Card Tamper Detection", layout="wide")

# Resolve base directory (folder containing this script).
# This makes loading "pan_template.jpg" reliable whether run locally or from Colab/unzipped folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_image_from_bytes(b):
    return Image.open(io.BytesIO(b)).convert("RGB")

def load_image_from_path_or_url(path_or_url):
    """
    Load an image from:
    - an absolute local path
    - a relative local path (resolved relative to BASE_DIR)
    - an http/https URL
    Returns PIL Image (RGB) or raises Exception.
    """
    try:
        if isinstance(path_or_url, str) and (path_or_url.startswith("http://") or path_or_url.startswith("https://")):
            # requests + PIL
            resp = requests.get(path_or_url, stream=True, timeout=10)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        else:
            # local path: support absolute or relative to BASE_DIR
            if not os.path.isabs(path_or_url):
                candidate = os.path.join(BASE_DIR, path_or_url)
            else:
                candidate = path_or_url
            if not os.path.exists(candidate):
                raise FileNotFoundError(f"Local file not found: {candidate}")
            return Image.open(candidate).convert("RGB")
    except Exception as e:
        raise

def pil_to_cv(img: Image.Image):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    # if single channel, convert to 3-channel first
    if len(img_cv.shape) == 2:
        return Image.fromarray(img_cv)
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def prepare_images(imgA_pil, imgB_pil, width=800):
    # Resize both images to same width while preserving aspect ratio
    def resize_keep_aspect(img, target_w):
        w, h = img.size
        if w == target_w:
            return img
        ratio = target_w / float(w)
        return img.resize((target_w, int(h * ratio)), Image.LANCZOS)

    A = resize_keep_aspect(imgA_pil, width)
    B = resize_keep_aspect(imgB_pil, width)

    # if heights differ, pad smaller one with white background to match height
    if A.size[1] != B.size[1]:
        max_h = max(A.size[1], B.size[1])

        def pad(img, max_h):
            w, h = img.size
            if h == max_h:
                return img
            background = Image.new("RGB", (w, max_h), (255, 255, 255))
            background.paste(img, (0, 0))
            return background

        A = pad(A, max_h)
        B = pad(B, max_h)

    return A, B

def compute_tamper(original_pil, suspect_pil, min_area=100):
    # convert to cv2 BGR
    orig_cv = pil_to_cv(original_pil)
    suspect_cv = pil_to_cv(suspect_pil)

    # ensure same height/width (should be from prepare_images)
    if orig_cv.shape != suspect_cv.shape:
        # as fallback, resize suspect to orig size (keeps processing robust)
        suspect_cv = cv2.resize(suspect_cv, (orig_cv.shape[1], orig_cv.shape[0]))

    # convert to grayscale
    orig_gray = cv2.cvtColor(orig_cv, cv2.COLOR_BGR2GRAY)
    suspect_gray = cv2.cvtColor(suspect_cv, cv2.COLOR_BGR2GRAY)

    # compute SSIM and diff image
    score, diff = ssim(orig_gray, suspect_gray, full=True)
    diff = (diff * 255).astype("uint8")

    # threshold the difference image (Otsu)
    # Use binary inverse so differences become white/foreground, but either is fine.
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # find contours of the thresholded regions
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # draw rectangles around detected differences on copies
    orig_draw = orig_cv.copy()
    suspect_draw = suspect_cv.copy()
    boxes = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:  # ignore tiny noise; adjustable
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h))
        cv2.rectangle(orig_draw, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(suspect_draw, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # make a visual concatenation for display (side-by-side)
    try:
        concat = np.concatenate([orig_draw, suspect_draw], axis=1)
    except Exception:
        # fallback: convert both to same shape explicitly
        h = min(orig_draw.shape[0], suspect_draw.shape[0])
        orig_small = orig_draw[:h, :]
        suspect_small = suspect_draw[:h, :]
        concat = np.concatenate([orig_small, suspect_small], axis=1)

    return {
        "score": float(score),
        "diff": Image.fromarray(diff),
        "thresh": Image.fromarray(thresh),
        "original_marked": cv_to_pil(orig_draw),
        "suspect_marked": cv_to_pil(suspect_draw),
        "side_by_side": cv_to_pil(concat),
        "boxes": boxes
    }

def main():
    st.title("🕵️ PAN Card Tamper Detection")
    st.write(
        """
    This app compares two PAN card images and highlights regions that differ.
    You can either upload both an **original/reference** PAN card and a **suspect** PAN card,
    or upload only the suspect and let the app use a provided template (`pan_template.jpg`) if available.
    """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.header("Reference / Original PAN card")
        uploaded_orig = st.file_uploader("Upload reference (original) image", type=["jpg", "jpeg", "png"], key="orig")
        use_example_orig = st.button("Use example original image", key="example_orig")

    with col2:
        st.header("Suspect PAN card")
        uploaded_suspect = st.file_uploader("Upload suspect image", type=["jpg", "jpeg", "png"], key="suspect")
        use_example_suspect = st.button("Use example tampered image", key="example_suspect")

    # Try to load default template located next to the app
    default_orig_path = os.path.join(BASE_DIR, "pan_template.jpg")

    orig_img = None
    sus_img = None

    if use_example_orig:
        try:
            orig_img = load_image_from_path_or_url("https://www.thestatesman.com/wp-content/uploads/2019/07/pan-card.jpg")
        except Exception as e:
            st.warning(f"Could not load example original: {e}")

    if use_example_suspect:
        try:
            sus_img = load_image_from_path_or_url("https://assets1.cleartax-cdn.com/s/img/20170526124335/Pan4.png")
        except Exception as e:
            st.warning(f"Could not load example suspect: {e}")

    if uploaded_orig is not None:
        try:
            orig_img = load_image_from_bytes(uploaded_orig.read())
        except Exception as e:
            st.error(f"Failed to read uploaded original: {e}")

    if orig_img is None:
        # if not provided yet, try to load local template
        if os.path.exists(default_orig_path):
            try:
                orig_img = load_image_from_path_or_url(default_orig_path)
            except Exception as e:
                st.warning(f"Could not load local template: {e}")

    if uploaded_suspect is not None:
        try:
            sus_img = load_image_from_bytes(uploaded_suspect.read())
        except Exception as e:
            st.error(f"Failed to read uploaded suspect: {e}")

    if orig_img is None or sus_img is None:
        st.info("Please provide both reference and suspect images (either upload them or use the example buttons).")
        st.stop()

    # Ensure same size / alignment
    resize_width = st.sidebar.slider("Resize width for processing (px)", min_value=400, max_value=1400, value=800, step=50)
    orig_img, sus_img = prepare_images(orig_img, sus_img, width=resize_width)

    st.sidebar.markdown("**Detection settings**")
    min_area = st.sidebar.slider("Ignore contours smaller than (area in px)", 10, 5000, 100, 10)
    tamper_threshold = st.sidebar.slider("Tamper SSIM threshold (lower = more sensitive)", 0.50, 0.999, 0.95, 0.01)

    # Run detection
    with st.spinner("Computing similarity and differences..."):
        try:
            result = compute_tamper(orig_img, sus_img, min_area=min_area)
        except Exception as e:
            st.error(f"Error during detection: {e}")
            st.stop()

    score = result["score"]
    st.metric("Structural Similarity Index (SSIM)", f"{score:.4f}")

    # Verdict
    if score < tamper_threshold:
        st.error(f"Likely tampered (SSIM {score:.4f} < threshold {tamper_threshold:.3f})")
    else:
        st.success(f"Images look similar (SSIM {score:.4f} ≥ threshold {tamper_threshold:.3f})")

    st.markdown("---")
    st.subheader("Visual results")

    # Show side-by-side with marked boxes
    st.image(result["side_by_side"], caption="Left: reference (boxes), Right: suspect (boxes)", use_column_width=True)

    cols = st.columns(3)
    cols[0].image(result["diff"], caption="Difference image (grayscale)", use_column_width=True)
    cols[1].image(result["thresh"], caption="Thresholded diff (binary)", use_column_width=True)
    cols[2].write("Detected bounding boxes:")
    if len(result["boxes"]) == 0:
        cols[2].write("No significant differing regions found.")
    else:
        for i, (x, y, w, h) in enumerate(result["boxes"], start=1):
            cols[2].write(f"{i}. x={x}, y={y}, w={w}, h={h}")

    st.markdown(
        """
    ---

    **Notes & tips**
    - This method uses SSIM - it measures structural similarity and highlights regions that changed.
    - If the PAN cards are not aligned (rotated or cropped differently), detection quality drops. Try to ensure cards are similarly aligned or crop/rotate prior to comparison.
    - You can adjust the SSIM threshold and minimum contour area in the sidebar.
    """
    )

if __name__ == "__main__":
    main()
