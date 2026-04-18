# Algorithmic Bias Audit — Interactive App
**Muhammad Atif Ahmed · CS-408 Introduction to AI · March 2026**

An interactive Streamlit app for auditing and mitigating algorithmic bias across five ML architectures using the UCI Adult Income Dataset.

---

## 🚀 Deploy in 5 Minutes (Streamlit Community Cloud — FREE)

### Step 1: Push to GitHub
```bash
# Create a new repo on github.com, then:
git init
git add .
git commit -m "Initial bias audit app"
git remote add origin https://github.com/YOUR_USERNAME/bias-audit-app.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to **https://share.streamlit.io**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repo → branch `main` → file `app.py`
5. Click **"Deploy"** → you get a public URL instantly

**Your app will be live at:**  
`https://YOUR_USERNAME-bias-audit-app-app-XXXX.streamlit.app`

---

## 💻 Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```
Opens at **http://localhost:8501** in your browser.

---

## 📁 Project Structure
```
bias_app/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .streamlit/
    └── config.toml         # Dark theme configuration
```

---

## 🔍 App Sections

| Section | Content |
|---------|---------|
| 🏠 Overview | Study summary, key metrics, full results table |
| 📊 Raw Data Bias | Gender/race disparity charts, intersectional audit |
| 🤖 Model Audit | 5-model comparison with 95% CI, paradox scatter |
| 📐 Significance Tests | McNemar's test heatmap & table |
| 🧠 SHAP Explainability | Feature importance, gender-stratified SHAP |
| 🧬 SMOTE Analysis | Mechanism analysis with interactive charts |
| ⚖️ 80% Rule Critique | Limitations, impossibility theorem scatter |
| 🛡️ Bias Mitigation | Interactive slider + Dem. Parity vs Eq. Odds |
| ✅ Conclusions | 5 findings, recommendations, references |

---

## ⚙️ Technical Notes

- **Data**: Loaded automatically from UCI ML Repository on first run
- **Caching**: `@st.cache_data` ensures models train only once per session (~60 s first load)
- **Bootstrap**: B=600 resamples for 95% CIs (reduced from 1,000 for app speed)
- **SHAP**: Computed on N=500 test subset for speed; global ranking is stable at this size

---

## 📦 Dependencies
```
streamlit, pandas, numpy, scikit-learn, xgboost,
imbalanced-learn, fairlearn, shap, plotly, matplotlib, scipy
```
