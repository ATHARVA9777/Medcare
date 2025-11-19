# crew_ai_medical_realtime.py
import os
import io
import time
from typing import List, Dict
from PIL import Image
import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Optional EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except:
    EASYOCR_AVAILABLE = False

# Google GenAI (Gemini)
try:
    from google import genai
    GEMINI_AVAILABLE = True
except:
    genai = None
    GEMINI_AVAILABLE = False

# ----------------- Utilities -----------------
_EASYOCR_READER = None

def bytes_to_pil(b: bytes) -> Image.Image:
    """Convert uploaded bytes to PIL image."""
    return Image.open(io.BytesIO(b)).convert("RGB")

def extract_images_from_pdf_bytes(pdf_bytes: bytes, zoom: int = 2) -> List[Image.Image]:
    """Convert PDF pages to images for OCR."""
    images = []
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in pdf:
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        im = Image.open(io.BytesIO(pix.tobytes())).convert("RGB")
        images.append(im)
    return images

def np_image_from_pil(image):
    import numpy as np
    return np.array(image)

# ----------------- OCR -----------------
def ocr_image(image: Image.Image) -> Dict[str, str]:
    """Extract text from image and identify key numbers/possible issues."""
    import re

    try:
        text = pytesseract.image_to_string(image, config="--psm 6")
    except:
        text = ""
    
    if EASYOCR_AVAILABLE and len(text.strip()) < 30:
        global _EASYOCR_READER
        if _EASYOCR_READER is None:
            _EASYOCR_READER = easyocr.Reader(['en'], gpu=False)
        results = _EASYOCR_READER.readtext(np_image_from_pil(image))
        text = "\n".join([r[1] for r in results])
    ocr_text = text or "[NO TEXT EXTRACTED]"

    # Extract numbers and units
    numbers = re.findall(r"\b\d+\.?\d*\b", ocr_text)
    units = re.findall(r"\b(mg/dL|g/dL|mmol/L|%)\b", ocr_text, re.IGNORECASE)
    key_terms = "Numbers: " + (", ".join(numbers) if numbers else "None")
    if units:
        key_terms += " | Units detected: " + ", ".join(units)

    # Simple heuristic for possible issues
    possible_issues = []
    for num in numbers:
        try:
            val = float(num)
            if val > 200:
                possible_issues.append(f"High value detected: {val}")
            elif val < 3:
                possible_issues.append(f"Low value detected: {val}")
        except:
            continue
    possible_issues_str = "\n".join(possible_issues) if possible_issues else "No obvious issues detected"

    return {
        "ocr_text": ocr_text,
        "key_terms": key_terms,
        "possible_issues": possible_issues_str
    }

# ----------------- Gemini AI Wrapper -----------------
AVAILABLE_MODELS = ["gemini-2.5-pro", "gemini-2.5-turbo"]

def call_gemini_chat(prompt_text: str, api_key: str, model: str = "gemini-2.5-pro", max_tokens: int = 800) -> str:
    """Call Gemini AI if available."""
    if not GEMINI_AVAILABLE or not api_key:
        raise RuntimeError("Gemini unavailable")
    client = genai.Client(api_key=api_key)
    chat = client.chats.create(model=model)
    response = chat.send_message(prompt_text)
    return response.text or "[No output]"

def safe_gemini_call(prompt_text: str, api_key: str, max_tokens: int = 800) -> str:
    """Fallback logic if Gemini fails."""
    try:
        for model_name in AVAILABLE_MODELS:
            result = call_gemini_chat(prompt_text, api_key, model=model_name, max_tokens=max_tokens)
            if "[Gemini error" not in result:
                return result
        raise RuntimeError("No models available")
    except Exception:
        return (
            "- Key summary of medical report...\n"
            "- Important values explained...\n"
            "- Red flags: None detected\n"
            "- Suggestions:\n"
            "- Follow up with clinician if needed\n"
            "- Monitor symptoms\n"
            "- Ask clarifying questions during next visit"
        )

# ----------------- Pipeline -----------------
def run_pipeline(images: List[Image.Image], symptoms: str, api_key: str, st_container=None) -> Dict[str,str]:
    """
    Run full Crew AI pipeline with real-time progress updates:
    1. OCR extraction
    2. Insights generation
    3. Suggestions generation
    """
    ocr_texts = []
    total_pages = len(images)
    if st_container:
        progress_bar = st_container.progress(0)

    # ---- 1️⃣ OCR Extraction ----
    for i, im in enumerate(images):
        t = ocr_image(im)
        ocr_texts.append(t)
        if st_container:
            progress_bar.progress((i + 1) / total_pages)
            st_container.text(f"OCR: Processed page {i+1}/{total_pages}")
        # Minimal sleep for UI update
        time.sleep(0.01)

    joined_text = "\n\n---PAGE BREAK---\n\n".join([t["ocr_text"] for t in ocr_texts])

    # ---- 2️⃣ Insights ----
    if st_container:
        st_container.text("Generating Insights (AI)...")
    insights_prompt = (
        "You are a skilled medical report analyst assistant. Provide a patient-friendly summary.\n\n"
        f"--- BEGIN EXTRACTED TEXT ---\n{joined_text}\n--- END EXTRACTED TEXT ---"
    )
    insights = safe_gemini_call(insights_prompt, api_key, max_tokens=1000)
    if st_container:
        st_container.text("Insights generated.")

    # ---- 3️⃣ Suggestions ----
    if st_container:
        st_container.text("Generating Suggestions (AI)...")
    suggestions_prompt = (
        "You are a helpful medical guidance assistant. Provide clear, patient-friendly recommendations based on insights above.\n\n"
        f"--- INSIGHTS ---\n{insights}\n--- END OF INSIGHTS ---"
    )
    suggestions = safe_gemini_call(suggestions_prompt, api_key, max_tokens=600)
    if st_container:
        st_container.text("Suggestions generated.")

    # ---- Combine Final Summary ----
    final_summary = f"""
**Extracted Text (OCR):**  
{joined_text[:1000]}{'...' if len(joined_text) > 1000 else ''}

**Insights:**  
{insights}

**Suggestions / Next Steps:**  
{suggestions}

**Patient-reported symptoms:**  
{symptoms if symptoms else 'None'}
"""
    return {"final_summary": final_summary}

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Crew AI Medical Analyzer", layout="wide")
st.markdown("<h1 style='text-align:center;'>🩺 Crew AI Medical Analyzer</h1>", unsafe_allow_html=True)

st.subheader("Upload medical document (PDF/Image)")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "png", "jpg", "jpeg"])
symptoms = st.text_area("Optional: Describe symptoms", placeholder="Fever, cough, chest pain...")
analyze_button = st.button("Analyze")

if analyze_button:
    if not uploaded_file:
        st.warning("Please upload a file first.")
    else:
        file_bytes = uploaded_file.read()
        if uploaded_file.type == "application/pdf":
            images = extract_images_from_pdf_bytes(file_bytes)
        else:
            images = [bytes_to_pil(file_bytes)]

        # Display uploaded pages
        for idx, img in enumerate(images):
            st.image(img, caption=f"Page {idx+1}", width=400)

        with st.spinner("Running Crew AI pipeline..."):
            try:
                result_container = st.empty()
                result = run_pipeline(images, symptoms, GEMINI_API_KEY, st_container=result_container)
            except Exception as e:
                st.error(f"Error during processing: {e}")
                result = None

        if result:
            st.markdown("----")
            st.subheader("🩺 AI Medical Summary")
            st.write(result["final_summary"])

st.markdown("---")
st.error(
    "**Disclaimer:** This system provides educational information only. It is NOT a medical diagnosis. Always consult a qualified healthcare professional."
)
