# Crew AI Medical Analyzer

A Streamlit-based application for analyzing medical documents using OCR (Tesseract) and Gemini AI.

## Features
- Upload PDF or image files (PNG, JPG, JPEG)
- Extract text using OCR (Tesseract or EasyOCR fallback)
- Generate insights and suggestions using Google Gemini AI
- Plain-language summaries, key terms explanation, red flags, and next steps

## Prerequisites
- Python 3.8+
- Google Gemini API key (set in `.env` file as `GOOGLE_API_KEY`)

## Installation

1. **Clone or download the repository:**
   - Place `app.py` and `.env` in your working directory.

2. **Install Python dependencies:**
   - Run: `pip install streamlit pillow pymupdf pytesseract python-dotenv google-genai easyocr`

3. **Install Tesseract OCR:**
   - On Windows, use Winget (built-in package manager):
     ```
     winget install --id UB-Mannheim.TesseractOCR -e --accept-package-agreements --accept-source-agreements
     ```
   - Alternatively, download from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH.
   - Verify installation: Open command prompt and run `tesseract --version`

4. **Set up environment variables:**
   - Create a `.env` file in the same directory as `app.py`.
   - Add your Google Gemini API key:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

## Usage

1. Run the application:
   ```
   streamlit run app.py
   ```

2. Upload a medical document (PDF or image).

3. Click "Analyze" to process the document.

4. View OCR text, insights, and suggestions.

## Troubleshooting

- **Tesseract not found:** Ensure Tesseract is installed and in your system's PATH. Restart your terminal/command prompt after installation.
- **Gemini API errors:** Check your API key in `.env` and ensure it's valid.
- **OCR fails:** Try different image formats or use EasyOCR if available.

## Medical Disclaimer
This system provides educational information only and is NOT a medical diagnosis. Always consult a qualified healthcare professional for diagnosis and treatment.

## License
[Add license if applicable]
