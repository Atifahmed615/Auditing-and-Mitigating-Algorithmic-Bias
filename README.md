# Algorithmic Bias Audit — Interactive App
**Muhammad Atif Ahmed · CS-408 Introduction to AI · March 2026**

---

## 🚀 Deploy on Streamlit Cloud (FREE)

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Bias audit app"
git remote add origin https://github.com/YOUR_USERNAME/bias-audit-app.git
git push -u origin main
```

### Step 2: Deploy — set Python 3.11 in Advanced Settings

1. Go to **https://share.streamlit.io** → sign in with GitHub
2. Click **"New app"**
3. Select repo → branch `main` → file `app.py`
4. Click **"Advanced settings"** ← before deploying
5. Set **Python version: 3.11** ← CRITICAL
6. Click **"Deploy"**

> ⚠️ Streamlit Cloud defaults to Python 3.14 which cannot build scipy/sklearn.
> Setting Python 3.11 in Advanced Settings fixes all dependency errors.

---

## 💻 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Add Your Notebook & Paper
```
bias-audit-app/
├── app.py
├── requirements.txt
├── .streamlit/config.toml
├── notebook/
│   └── Bias_Audit_Improved.ipynb   ← drag in here
└── paper/
    └── research_paper.pdf          ← drag in here
```
