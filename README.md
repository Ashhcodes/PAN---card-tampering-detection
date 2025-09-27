# 🕵️ PAN Card Tamper Detection

A **Streamlit app** that detects tampering in PAN card images using **Structural Similarity Index (SSIM)**.  
It compares a **reference PAN card** with a **suspect PAN card** and highlights differences.

---

## ⚙️ How to Run (Windows)

```powershell
# 1. Clone the repo and open the folder
git clone https://github.com/YourUsername/pan-card-tamper-detection.git
cd pan-card-tamper-detection

# 2. Create and activate a virtual environment
py -m venv .venv
.\.venv\Scripts\Activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app
streamlit run main_app.py
