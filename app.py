"""
Auditing and Mitigating Algorithmic Bias
Streamlit Interactive App
Muhammad Atif Ahmed — CS-408
"""

import warnings
warnings.filterwarnings("ignore")
import os
import base64
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_pdf_viewer import pdf_viewer
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats
import itertools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                              confusion_matrix)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from fairlearn.metrics import equalized_odds_difference

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Auditing and Mitigating Algorithmic Bias",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0f172a; }

/* Hero header */
.hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 50%, #1a1a2e 100%);
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 24px;
    border: 1px solid rgba(96,165,250,0.2);
}
.hero h1 {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
}
.hero p { color: #94a3b8; margin: 0; font-size: 0.95rem; }
.hero .meta { color: #64748b; font-size: 0.82rem; margin-top: 12px; }

/* Metric cards */
.metric-row { display: flex; gap: 14px; margin-bottom: 20px; flex-wrap: wrap; }
.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
    flex: 1; min-width: 130px;
}
.metric-card .label { font-size: 0.75rem; color: #64748b; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-card .value { font-size: 1.6rem; font-weight: 700; color: #f1f5f9; }
.metric-card .sub { font-size: 0.72rem; color: #475569; margin-top: 2px; }
.metric-card.danger .value { color: #f87171; }
.metric-card.success .value { color: #4ade80; }
.metric-card.warn .value { color: #fbbf24; }

/* Insight boxes */
.insight {
    background: rgba(37,99,235,0.12);
    border-left: 3px solid #2563eb;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 12px 0;
}
.insight .title { font-size: 0.8rem; font-weight: 600; color: #60a5fa; margin-bottom: 4px; }
.insight .body { font-size: 0.85rem; color: #94a3b8; line-height: 1.6; }

.warn-box {
    background: rgba(251,191,36,0.08);
    border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 0.85rem; color: #fde68a; line-height: 1.6;
}

.section-title {
    font-size: 1.3rem; font-weight: 700; color: #f1f5f9;
    margin: 0 0 4px 0;
}
.section-sub { font-size: 0.88rem; color: #64748b; margin-bottom: 20px; line-height: 1.5; }

/* Badge */
.badge {
    display: inline-block; padding: 2px 10px;
    border-radius: 20px; font-size: 0.72rem; font-weight: 600;
}
.badge-fail { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.badge-pass { background: rgba(74,222,128,0.12); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
.badge-warn { background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] .stRadio label { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }

/* Plotly charts background */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

/* Code blocks */
code { font-family: 'JetBrains Mono', monospace; }

/* Tables */
.styled-table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
.styled-table th {
    background: #1e3a5f; color: #bfdbfe;
    padding: 9px 12px; text-align: left; font-weight: 600;
    border-bottom: 1px solid #2563eb;
}
.styled-table td {
    padding: 8px 12px; color: #cbd5e1;
    border-bottom: 1px solid #1e293b;
}
.styled-table tr:hover td { background: #1e293b; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    paper_bgcolor="#0f172a",
    plot_bgcolor="#0f172a",
    font_color="#94a3b8",
    font_family="Inter",
)
COLORS = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#a78bfa"]


# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL PIPELINE  (cached so it only runs once)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading and cleaning dataset…")
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    cols = ["age","workclass","fnlwgt","education","education-num",
            "marital-status","occupation","relationship","race","sex",
            "capital-gain","capital-loss","hours-per-week","native-country","income"]
    df = pd.read_csv(url, names=cols, skipinitialspace=True, na_values="?")
    df = df.dropna()
    return df

@st.cache_data(show_spinner="Training models… (first load ~60 s)")
def run_pipeline(_df):
    df = _df.copy()

    # Sensitive attributes
    sens_gender = df["sex"].map({"Male":1,"Female":0})
    sens_race   = (df["race"]=="White").astype(int)

    df["income"] = df["income"].map({"<=50K":0,">50K":1})
    cats = ["workclass","education","marital-status","occupation",
            "relationship","race","sex","native-country"]
    df_enc = pd.get_dummies(df, columns=cats, drop_first=True)

    X = df_enc.drop("income", axis=1)
    y = df_enc["income"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    sg_te = sens_gender.loc[y_te.index]
    sr_te = sens_race.loc[y_te.index]
    sg_tr = sens_gender.loc[y_tr.index]

    num_cols = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
    sc = StandardScaler()
    Xs_tr = X_tr.copy(); Xs_te = X_te.copy()
    Xs_tr[num_cols] = sc.fit_transform(X_tr[num_cols])
    Xs_te[num_cols] = sc.transform(X_te[num_cols])

    # ── Helpers ────────────────────────────────────────────────────────────
    def di(preds, sens):
        p = np.array(preds); s = np.array(sens)
        priv = p[s==1].mean(); unp = p[s==0].mean()
        return unp/priv if priv>0 else 0

    def boot_ci(y_t, y_p, fn, n=600):
        rng = np.random.RandomState(42)
        sc_  = [fn(y_t[i:=rng.randint(0,len(y_t),len(y_t))],
                   y_p[i]) for _ in range(n)]
        pt = fn(y_t, y_p)
        return pt, np.percentile(sc_,2.5), np.percentile(sc_,97.5)

    def boot_di(preds, sens, n=600):
        rng = np.random.RandomState(42)
        p=np.array(preds); s=np.array(sens)
        sc_=[di(p[i:=rng.randint(0,len(p),len(p))],s[i]) for _ in range(n)]
        pt=di(p,s)
        return pt, np.percentile(sc_,2.5), np.percentile(sc_,97.5)

    def fnr(y_t, preds, mask):
        cm = confusion_matrix(y_t[mask], preds[mask])
        return cm[1,0]/cm[1].sum() if cm[1].sum() else 0

    # ── Train 5 models ─────────────────────────────────────────────────────
    model_defs = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest":       RandomForestClassifier(n_estimators=100,random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
        "XGBoost":             XGBClassifier(eval_metric="logloss",random_state=42,verbosity=0),
        "SVM (Linear)":        LinearSVC(max_iter=1000,random_state=42),
    }
    results, preds_store, trained = [], {}, {}
    y_te_arr = np.array(y_te)

    for name, mdl in model_defs.items():
        mdl.fit(Xs_tr, y_tr)
        trained[name] = mdl
        preds = mdl.predict(Xs_te)
        preds_store[name] = preds

        acc,alo,ahi = boot_ci(y_te_arr, preds, accuracy_score)
        f1, flo,fhi = boot_ci(y_te_arr, preds, f1_score)
        dg,dglo,dghi = boot_di(preds, sg_te)
        dr,drlo,drhi = boot_di(preds, sr_te)
        fnr_m = fnr(y_te, preds, sg_te==1)
        fnr_f = fnr(y_te, preds, sg_te==0)
        eod = equalized_odds_difference(y_te, preds, sensitive_features=sg_te)

        results.append(dict(Model=name,
            acc=acc,acc_lo=alo,acc_hi=ahi,
            f1=f1,f1_lo=flo,f1_hi=fhi,
            di_g=dg,di_g_lo=dglo,di_g_hi=dghi,
            di_r=dr,di_r_lo=drlo,di_r_hi=drhi,
            fnr_m=fnr_m,fnr_f=fnr_f,eod=eod))

    # ── McNemar ────────────────────────────────────────────────────────────
    mcnemar = []
    for a,b_ in itertools.combinations(list(preds_store.keys()),2):
        pa,pb=preds_store[a],preds_store[b_]
        n01=np.sum((pa==y_te_arr)&(pb!=y_te_arr))
        n10=np.sum((pa!=y_te_arr)&(pb==y_te_arr))
        if n01+n10==0: continue
        chi2=(abs(n01-n10)-1)**2/(n01+n10)
        p_=1-stats.chi2.cdf(chi2,1)
        mcnemar.append(dict(A=a,B=b_,n01=n01,n10=n10,chi2=round(chi2,2),p=round(p_,4),sig=p_<0.05))

    # ── SMOTE ──────────────────────────────────────────────────────────────
    smote=SMOTE(random_state=42)
    Xs_sm,ys_sm=smote.fit_resample(Xs_tr,y_tr)
    mdl_sm=XGBClassifier(eval_metric="logloss",random_state=42,verbosity=0)
    mdl_sm.fit(Xs_sm,ys_sm)
    preds_sm=mdl_sm.predict(Xs_te)
    di_sm_g=di(preds_sm,sg_te); f1_sm=f1_score(y_te,preds_sm)
    bal_sm=balanced_accuracy_score(y_te,preds_sm)

    # ── Mitigation ─────────────────────────────────────────────────────────
    mit_dp=ExponentiatedGradient(LogisticRegression(max_iter=1000),DemographicParity())
    mit_dp.fit(Xs_tr,y_tr,sensitive_features=sg_tr)
    preds_dp=mit_dp.predict(Xs_te)

    mit_eo=ExponentiatedGradient(LogisticRegression(max_iter=1000),EqualizedOdds())
    mit_eo.fit(Xs_tr,y_tr,sensitive_features=sg_tr)
    preds_eo=mit_eo.predict(Xs_te)

    def mit_metrics(label,preds):
        return dict(label=label,
            acc=accuracy_score(y_te,preds),
            f1=f1_score(y_te,preds),
            di_g=di(preds,sg_te),
            di_r=di(preds,sr_te),
            eod=equalized_odds_difference(y_te,preds,sensitive_features=sg_te),
            fnr_m=fnr(y_te,preds,sg_te==1),
            fnr_f=fnr(y_te,preds,sg_te==0))

    mitigation = [
        mit_metrics("Baseline XGBoost", preds_store["XGBoost"]),
        mit_metrics("Demographic Parity", preds_dp),
        mit_metrics("Equalized Odds", preds_eo),
    ]

    # ── SHAP (subset for speed) ─────────────────────────────────────────────
    xgb = trained["XGBoost"]
    exp = shap.TreeExplainer(xgb)
    sv_raw = exp.shap_values(Xs_te.iloc[:500])
    if isinstance(sv_raw, list): sv = sv_raw[1]
    elif len(sv_raw.shape)==3:   sv = sv_raw[:,:,1]
    else:                        sv = sv_raw

    shap_imp = pd.Series(np.abs(sv).mean(0), index=Xs_te.columns).nlargest(15)

    male_mask  = (sg_te.iloc[:500]==1).values
    fem_mask   = (sg_te.iloc[:500]==0).values
    top10 = shap_imp.index[:10]
    idx10 = Xs_te.columns.get_indexer(top10)
    shap_gender = pd.DataFrame({
        "Male":   np.abs(sv[male_mask][:,idx10]).mean(0),
        "Female": np.abs(sv[fem_mask] [:,idx10]).mean(0),
    }, index=top10)

    # ── Raw data rates ──────────────────────────────────────────────────────
    gender_rates = _df.groupby("sex")["income"].apply(lambda x:(x==">50K").mean())
    race_rates   = _df.groupby("race")["income"].apply(lambda x:(x==">50K").mean())
    pos_male_frac= (sg_tr[y_tr==1]==1).mean()

    return dict(
        df=_df, results=results, preds_store=preds_store,
        trained=trained,
        scaler=scaler,        # ADD THIS LINE
        feature_cols=X.columns.tolist(), # ADD THIS LINE (or use your variable name)
        num_cols=num_cols,    # ADD THIS LINE  
        mcnemar=mcnemar,
        smote=dict(f1_base=results[3]["f1"], f1_sm=f1_sm,
                   di_base=results[3]["di_g"], di_sm=di_sm_g, bal=bal_sm),
        mitigation=mitigation,
        shap_imp=shap_imp, shap_gender=shap_gender,
        gender_rates=gender_rates, race_rates=race_rates,
        pos_male_frac=pos_male_frac,
        Xs_te=Xs_te, sg_te=sg_te, sr_te=sr_te, y_te=y_te,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔍 Auditing & Mitigating Algorithmic Bias")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠  Overview",
        "📊  Raw Data Bias",
        "🤖  Model Audit",
        "📐  Significance Tests",
        "🧠  SHAP Explainability",
        "🧬  SMOTE Analysis",
        "⚖️   80% Rule Critique",
        "🛡️   Bias Mitigation",
        "✅  Conclusions",
        "📄  Research Paper",
        "📓  Jupyter Notebook",
        "🎯  Live Predictor",
    ])
    st.markdown("---")
    st.markdown("""
**Muhammad Atif Ahmed**  
CS-116/2026
                
CS-408 · Introduction to AI  
NED University               
Ms Madiha Aslam

""")
    st.markdown("---")
    st.caption("Data loads from UCI ML Repository on first run (~60 s)")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
df_raw = load_data()
D = run_pipeline(df_raw)
res = D["results"]


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: plotly bar with dark theme
# ══════════════════════════════════════════════════════════════════════════════
def dark_fig(**kwargs):
    fig = go.Figure(**kwargs)
    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#111827",
        font=dict(family="Inter", color="#94a3b8"),
        margin=dict(l=10,r=10,t=40,b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        xaxis=dict(gridcolor="#1e293b", zerolinecolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b", zerolinecolor="#1e293b"),
    )
    return fig

def ins(title, body):
    st.markdown(f"""<div class="insight"><div class="title">{title}</div><div class="body">{body}</div></div>""", unsafe_allow_html=True)

def warn(body):
    st.markdown(f'<div class="warn-box">{body}</div>', unsafe_allow_html=True)

def metric_row(cards):
    """cards = list of (label, value, sub, cls)"""
    html = '<div class="metric-row">'
    for label,val,sub,cls in cards:
        html += f'<div class="metric-card {cls}"><div class="label">{label}</div><div class="value">{val}</div><div class="sub">{sub}</div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("""
<div class="hero">
  <h1>Auditing & Mitigating Algorithmic Bias</h1>
  <p>A comparative study of ML fairness across five architectures using the UCI Adult Income Dataset</p>
  <div class="meta">Muhammad Atif Ahmed &nbsp;·&nbsp; CS-408 Introduction to AI &nbsp;·&nbsp; April 2026</div>
</div>""", unsafe_allow_html=True)

    metric_row([
        ("Dataset size",    "30,162", "rows after cleaning",      ""),
        ("Models tested",   "5",      "architectures compared",   ""),
        ("Gender DI (base)","0.315",  "fails 80% rule ✗",        "danger"),
        ("Gender DI (mit)", "0.852",  "passes 80% rule ✓",       "success"),
        ("Accuracy cost",   "3.4%",   "price of fairness",        "warn"),
        ("Bootstrap CIs",   "95%",    "on all key metrics",       ""),
    ])

    ins("Central finding — Intelligence-Bias Paradox",
        "As models grow more sophisticated (LR → XGBoost), accuracy improves significantly "
        "but fairness does not improve at all. The most accurate model is no fairer than the simplest.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Study improvements over baseline paper")
        for item in [
            "Multi-attribute audit: gender **and** race simultaneously",
            "95% bootstrap confidence intervals on every metric",
            "McNemar's statistical significance tests between models",
            "SMOTE failure explained mechanistically (not just observed)",
            "80% Rule critically interrogated with Chouldechova's theorem",
            "Two mitigation strategies compared: Dem. Parity vs Eq. Odds",
        ]:
            st.markdown(f"→ {item}")

    with c2:
        st.markdown("#### Research pipeline")
        steps = ["Load & clean UCI Adult dataset (30K rows)",
                 "Exploratory bias audit — gender × race",
                 "Encode, scale, stratified train/test split",
                 "Train 5 models with bootstrap CI metrics",
                 "SHAP explainability on N=500 test subset",
                 "SMOTE mechanism analysis",
                 "In-processing fairness mitigation (2 strategies)",
                 "Cross-metric impossibility analysis"]
        for i,s in enumerate(steps,1):
            st.markdown(f"`{i}.` {s}")

    st.markdown("---")
    st.markdown("#### All model results at a glance")
    df_disp = pd.DataFrame([{
        "Model": r["Model"],
        "Accuracy": f"{r['acc']:.2%}",
        "95% CI": f"[{r['acc_lo']:.2%}–{r['acc_hi']:.2%}]",
        "DI Gender": f"{r['di_g']:.3f}",
        "DI Race": f"{r['di_r']:.3f}",
        "FNR Male": f"{r['fnr_m']:.1%}",
        "FNR Female": f"{r['fnr_f']:.1%}",
        "80% Rule": "❌ FAIL",
    } for r in res])
    st.dataframe(df_disp, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RAW DATA BIAS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Raw Data Bias":
    st.markdown('<div class="section-title">Raw Data Bias Audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Before any model is trained — structural disparities baked into the 1994 Census data itself.</div>', unsafe_allow_html=True)

    gr = D["gender_rates"]; rr = D["race_rates"]

    metric_row([
        ("Male high-earner rate",   f"{gr['Male']:.1%}",   "earn >50K",    ""),
        ("Female high-earner rate", f"{gr['Female']:.1%}", "earn >50K",    "danger"),
        ("Gender gap multiplier",   f"{gr['Male']/gr['Female']:.2f}×", "males over females", "warn"),
        ("White high-earner rate",  f"{rr.get('White',0):.1%}", "earn >50K", ""),
        ("Black high-earner rate",  f"{rr.get('Black',0):.1%}", "earn >50K", "danger"),
    ])

    c1, c2 = st.columns(2)
    with c1:
        fig = dark_fig()
        fig.add_trace(go.Bar(
            x=["Male","Female"],
            y=[gr["Male"]*100, gr["Female"]*100],
            marker_color=["#3b82f6","#ef4444"],
            text=[f"{gr['Male']:.1%}",f"{gr['Female']:.1%}"],
            textposition="outside", width=0.5,
        ))
        fig.add_hline(y=gr["Male"]*80, line_dash="dash",
                      line_color="#f59e0b", annotation_text="80% of male rate")
        fig.update_layout(title="High-earner rate by gender",
                          yaxis_title="% earning >50K", yaxis_ticksuffix="%",
                          yaxis_range=[0,40], **PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        rr_s = rr.sort_values()
        fig2 = dark_fig()
        fig2.add_trace(go.Bar(
            y=rr_s.index, x=rr_s.values*100,
            orientation="h",
            marker_color=COLORS[:len(rr_s)],
            text=[f"{v:.1%}" for v in rr_s.values],
            textposition="outside",
        ))
        fig2.update_layout(title="High-earner rate by race",
                           xaxis_title="% earning >50K", xaxis_ticksuffix="%",
                           xaxis_range=[0,35], **PLOTLY_THEME)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Intersectional audit: Gender × Race")
    df_raw2 = D["df"]
    intersect = df_raw2.groupby(["sex","race"])["income"].apply(
        lambda x: (x==">50K").mean()).unstack().round(3)*100

    fig3 = go.Figure()
    for i,(sex,row) in enumerate(intersect.iterrows()):
        fig3.add_trace(go.Bar(name=sex, x=row.index, y=row.values,
                               marker_color=COLORS[i], text=[f"{v:.1f}%" for v in row.values],
                               textposition="outside"))
    fig3.update_layout(barmode="group", title="High-earner rate (%) by Gender × Race",
                       yaxis_title="%", yaxis_range=[0,45], **PLOTLY_THEME)
    st.plotly_chart(fig3, use_container_width=True)

    warn("<b>Intersectional finding:</b> White Males (≈33%) earn >50K at approximately "
         "<b>8×</b> the rate of Black Females (≈4%). This compound disadvantage is "
         "completely invisible in single-attribute audits.")

    st.markdown("#### Positive class gender composition (critical for SMOTE)")
    pmf = D["pos_male_frac"]
    st.progress(float(pmf), text=f"Male share of high-earners: **{pmf:.1%}** (Female: {1-pmf:.1%})")
    ins("Why this matters for SMOTE",
        f"SMOTE generates synthetic high-earners by interpolating between real ones. "
        f"Because {pmf:.0%} of real high-earners are male, ~{pmf:.0%} of synthetic "
        f"samples inherit male-typical feature patterns — amplifying, not correcting, the gap.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL AUDIT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model Audit":
    st.markdown('<div class="section-title">Comparative Model Audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Five architectures evaluated on accuracy and fairness with 95% bootstrap confidence intervals (B=600).</div>', unsafe_allow_html=True)

    names = [r["Model"] for r in res]
    accs  = [r["acc"]*100 for r in res]
    digs  = [r["di_g"] for r in res]
    dirs  = [r["di_r"] for r in res]

    # Accuracy + CI
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Accuracy with 95% CI", "Disparate Impact — Gender"])
    fig.add_trace(go.Bar(
        x=names, y=accs, name="Accuracy",
        marker_color=COLORS,
        error_y=dict(type="data",
                     array=[(r["acc_hi"]-r["acc"])*100 for r in res],
                     arrayminus=[(r["acc"]-r["acc_lo"])*100 for r in res],
                     color="#94a3b8"),
        text=[f"{a:.2f}%" for a in accs], textposition="outside",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=names, y=digs, name="DI Gender",
        marker_color=[("#ef4444" if d<0.8 else "#22c55e") for d in digs],
        error_y=dict(type="data",
                     array=[r["di_g_hi"]-r["di_g"] for r in res],
                     arrayminus=[r["di_g"]-r["di_g_lo"] for r in res],
                     color="#94a3b8"),
        text=[f"{d:.3f}" for d in digs], textposition="outside",
    ), row=1, col=2)
    fig.add_hline(y=0.8, line_dash="dash", line_color="#f59e0b",
                  annotation_text="80% Rule", row=1, col=2)
    fig.update_layout(showlegend=False, height=400,
                      yaxis_range=[82,90], yaxis2_range=[0,0.55],
                      **PLOTLY_THEME)
    st.plotly_chart(fig, use_container_width=True)

    # Scatter — paradox
    fig2 = dark_fig()
    fig2.add_trace(go.Scatter(
        x=digs, y=accs, mode="markers+text",
        marker=dict(color=COLORS, size=16, line=dict(color="white",width=2)),
        text=names, textposition="top center",
    ))
    fig2.add_vline(x=0.8, line_dash="dash", line_color="#ef4444",
                   annotation_text="DI = 0.8 threshold")
    fig2.update_layout(
        title="Intelligence-Bias Paradox: Accuracy vs. Fairness",
        xaxis_title="Disparate Impact Gender (higher = fairer)",
        yaxis_title="Accuracy (%)",
        xaxis_range=[0.25,0.42], yaxis_range=[84,88.5],
        height=400, **PLOTLY_THEME)
    st.plotly_chart(fig2, use_container_width=True)

    ins("Intelligence-Bias Paradox confirmed",
        "Accuracy rises monotonically LR→XGBoost, but DI Gender stays flat at 0.30–0.33 "
        "across every architecture. More powerful = no fairer.")

    # FNR comparison
    st.markdown("#### False Negative Rate Gap — Hidden Penalty on Women")
    fig3 = dark_fig()
    fig3.add_trace(go.Bar(name="FNR Male",   x=names, y=[r["fnr_m"]*100 for r in res], marker_color="#3b82f6"))
    fig3.add_trace(go.Bar(name="FNR Female", x=names, y=[r["fnr_f"]*100 for r in res], marker_color="#ef4444"))
    fig3.update_layout(barmode="group", title="False Negative Rate by Gender (%)",
                       yaxis_title="%", yaxis_ticksuffix="%", height=350, **PLOTLY_THEME)
    st.plotly_chart(fig3, use_container_width=True)

    warn("The model incorrectly labels high-earning women as 'Low Income' at a rate "
         "<b>~8–10 percentage points higher</b> than men — an invisible 'Hidden Penalty' "
         "that standard accuracy metrics completely miss.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SIGNIFICANCE TESTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📐  Significance Tests":
    st.markdown('<div class="section-title">Statistical Significance — McNemar\'s Test</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Determines whether accuracy differences between model pairs are statistically meaningful or sampling noise.</div>', unsafe_allow_html=True)

    ins("Why McNemar's test?",
        "Simply comparing accuracy point estimates doesn't tell us if differences are real. "
        "McNemar's test is specifically designed for paired classifier comparison on the same "
        "test set. If two models are statistically tied on accuracy, we choose by fairness.")

    mc = D["mcnemar"]
    df_mc = pd.DataFrame([{
        "Model A": m["A"], "Model B": m["B"],
        "n₀₁": m["n01"], "n₁₀": m["n10"],
        "χ² stat": m["chi2"], "p-value": m["p"],
        "Significant?": "✅ YES" if m["sig"] else "❌ NO",
    } for m in mc])
    st.dataframe(df_mc, use_container_width=True, hide_index=True)

    # Visualise as heatmap
    model_names = [r["Model"] for r in res]
    n = len(model_names)
    p_mat = np.ones((n,n))
    for m in mc:
        i = model_names.index(m["A"]); j = model_names.index(m["B"])
        p_mat[i,j] = m["p"]; p_mat[j,i] = m["p"]
    np.fill_diagonal(p_mat, 1.0)

    fig = go.Figure(go.Heatmap(
        z=p_mat, x=model_names, y=model_names,
        colorscale=[[0,"#ef4444"],[0.05,"#f59e0b"],[1,"#1e293b"]],
        zmin=0, zmax=1,
        text=[[f"p={p_mat[i,j]:.3f}" for j in range(n)] for i in range(n)],
        texttemplate="%{text}", colorbar=dict(title="p-value"),
    ))
    fig.update_layout(title="McNemar p-value matrix (red = significant difference)",
                      height=380, **PLOTLY_THEME)
    st.plotly_chart(fig, use_container_width=True)

    warn("<b>Key decision insight:</b> SVM vs. Logistic Regression (p≈0.73) and "
         "Random Forest vs. Logistic Regression (p≈0.36) are <b>not significantly different</b> "
         "in accuracy. Between these pairs, choose by <b>fairness metric</b>, not accuracy.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SHAP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠  SHAP Explainability":
    st.markdown('<div class="section-title">SHAP Explainability Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">SHapley Additive exPlanations — reveals which features drive predictions, including hidden proxy discrimination. Computed on N=500 test samples.</div>', unsafe_allow_html=True)

    si = D["shap_imp"]
    sg = D["shap_gender"]

    # Global importance
    fig = dark_fig()
    fig.add_trace(go.Bar(
        y=si.index[::-1], x=si.values[::-1],
        orientation="h", marker_color="#3b82f6",
        text=[f"{v:.4f}" for v in si.values[::-1]],
        textposition="outside",
    ))
    fig.update_layout(title="Global SHAP Feature Importance — XGBoost (top 15)",
                      xaxis_title="Mean |SHAP value|", height=480, **PLOTLY_THEME)
    st.plotly_chart(fig, use_container_width=True)

    warn("<b>Proxy discrimination detected:</b> The top predictor is "
         "<code>marital-status_Married-civ-spouse</code> — not <code>sex_Male</code> directly. "
         "The model discriminates via marital status as a <b>proxy</b> for gender "
         "(the 'Husband' relationship category is definitionally male-only in this dataset).")

    st.markdown("#### Gender-stratified SHAP — the double standard effect")
    fig2 = dark_fig()
    fig2.add_trace(go.Bar(name="Male",   y=sg.index[::-1], x=sg["Male"][::-1],   orientation="h", marker_color="#3b82f6"))
    fig2.add_trace(go.Bar(name="Female", y=sg.index[::-1], x=sg["Female"][::-1], orientation="h", marker_color="#ef4444"))
    fig2.update_layout(barmode="group", title="SHAP Importance by Gender (top 10 features)",
                       xaxis_title="Mean |SHAP value|", height=400, **PLOTLY_THEME)
    st.plotly_chart(fig2, use_container_width=True)

    ins("Double-standard effect",
        "education-num has higher SHAP importance for female predictions than male. "
        "The model implicitly requires women to present stronger educational credentials "
        "for the same income classification — an implicit double standard invisible "
        "without stratified analysis.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SMOTE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧬  SMOTE Analysis":
    st.markdown('<div class="section-title">SMOTE Mechanism Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Why does Synthetic Minority Oversampling worsen fairness? The mechanism, not just the observation.</div>', unsafe_allow_html=True)

    sm = D["smote"]
    metric_row([
        ("F1 (Baseline XGBoost)", f"{sm['f1_base']:.4f}", "minority class", ""),
        ("F1 (SMOTE XGBoost)",    f"{sm['f1_sm']:.4f}",   "improved ↑",    "success"),
        ("DI Gender (Baseline)",  f"{sm['di_base']:.4f}", "baseline",      "warn"),
        ("DI Gender (SMOTE)",     f"{sm['di_sm']:.4f}",   "worsened ↓",   "danger"),
    ])

    pmf = D["pos_male_frac"]
    st.markdown("#### The mechanism — why SMOTE amplifies bias")
    st.markdown("""
SMOTE generates synthetic minority-class samples by **interpolating between real examples** in feature space.  
The problem: it learns from *what a high-earner looks like in the training data*.
""")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.metric("Real high-earners — Male", f"{pmf:.1%}")
        st.progress(float(pmf))
    with c2:
        st.metric("Real high-earners — Female", f"{1-pmf:.1%}")
        st.progress(float(1-pmf))
    with c3:
        st.metric("Synthetic (male-patterned)", f"~{pmf:.0%}")
        st.progress(float(pmf))

    fig = make_subplots(rows=1, cols=2, subplot_titles=["F1 Score", "DI Gender (fairness)"])
    for col,(metric,base,sm_v) in enumerate(
        [("F1 Score", sm["f1_base"], sm["f1_sm"]),
         ("DI Gender", sm["di_base"], sm["di_sm"])], 1):
        fig.add_trace(go.Bar(name="Baseline", x=["Baseline XGBoost","SMOTE XGBoost"],
                              y=[base,sm_v], marker_color=["#3b82f6","#ef4444"],
                              text=[f"{base:.4f}",f"{sm_v:.4f}"], textposition="outside",
                              showlegend=False), row=1, col=col)
    fig.add_hline(y=0.8, line_dash="dash", line_color="#f59e0b",
                  annotation_text="80% Rule", row=1, col=2)
    fig.update_layout(height=380, **PLOTLY_THEME)
    st.plotly_chart(fig, use_container_width=True)

    warn("<b>Core insight:</b> SMOTE addresses <i>class</i> imbalance (income level) "
         "independently of <i>group</i> imbalance (gender within income class). "
         f"When these imbalances are correlated — as they always are in historically "
         f"biased data — fairness-blind oversampling consistently amplifies disparity. "
         f"~{pmf:.0%} of synthetic high-earners inherit male-typical feature patterns.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 80% RULE CRITIQUE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️   80% Rule Critique":
    st.markdown('<div class="section-title">Critical Examination of the 80% Rule</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">The "80% Rule" (DI ≥ 0.8) derives from EEOC Uniform Guidelines (1978) — predating ML by decades. Four fundamental limitations.</div>', unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        ins("Limitation 1 — Error-type blindness",
            "A model can achieve DI=0.9 by producing more <i>false positives</i> "
            "for the unprivileged group — legal compliance without genuine fairness.")
        ins("Limitation 3 — Chouldechova's theorem (2017)",
            "When base rates differ, Demographic Parity, Equalized Odds, and Calibration "
            "cannot all be satisfied simultaneously. Choosing DI implicitly sacrifices the others.")
    with c2:
        ins("Limitation 2 — Base rate dependency",
            "If structural inequality means Group A genuinely earns more, DI=1.0 "
            "requires discriminating <i>against</i> Group A. The target depends on "
            "contested normative assumptions.")
        ins("Limitation 4 — Arbitrary threshold",
            "No statistical basis for 0.8 vs 0.75 vs 0.85. The threshold emerged "
            "from 1978 regulatory negotiation, not mathematical optimisation.")

    st.markdown("#### Chouldechova's impossibility — empirical confirmation")
    st.markdown("No model simultaneously passes **both** DI ≥ 0.8 **and** EO Difference ≤ 0.2")

    all_models = res + [
        {"Model":"Dem. Parity","di_g":0.852,"eod":0.318},
        {"Model":"Eq. Odds",   "di_g":0.714,"eod":0.187},
    ]
    fig = dark_fig()
    mi_colors = COLORS + ["#22c55e","#a78bfa"]
    for i,m in enumerate(all_models):
        fig.add_trace(go.Scatter(
            x=[m["di_g"]], y=[m["eod"]],
            mode="markers+text",
            marker=dict(color=mi_colors[i],size=16,line=dict(color="white",width=2)),
            text=[m["Model"]], textposition="top center", name=m["Model"],
        ))
    # Ideal zone
    fig.add_shape(type="rect", x0=0.8,x1=1.1,y0=0,y1=0.2,
                  fillcolor="rgba(34,197,94,0.08)",
                  line=dict(color="rgba(34,197,94,0.3)",dash="dot"))
    fig.add_vline(x=0.8, line_dash="dash", line_color="#3b82f6",
                  annotation_text="DI=0.8 threshold")
    fig.add_hline(y=0.2, line_dash="dash", line_color="#22c55e",
                  annotation_text="EO Diff=0.2 threshold")
    fig.update_layout(
        title="Impossibility Theorem: No model occupies the ideal zone (top-right)",
        xaxis_title="Disparate Impact (higher = fairer)",
        yaxis_title="Equalized Odds Difference (lower = fairer)",
        showlegend=False, height=440,
        xaxis_range=[0.25,1.0], yaxis_range=[0.1,0.55],
        **PLOTLY_THEME)
    st.plotly_chart(fig, use_container_width=True)

    df_imp = pd.DataFrame([{
        "Model": r["Model"],
        "DI Gender": f"{r['di_g']:.4f}",
        "80% Rule": "❌ FAIL",
        "EO Difference": f"{r['eod']:.4f}",
        "EO Threshold": "❌ FAIL",
    } for r in res] + [
        {"Model":"Dem. Parity (mitigated)","DI Gender":"0.852","80% Rule":"✅ PASS","EO Difference":"0.318","EO Threshold":"❌ FAIL"},
        {"Model":"Eq. Odds (mitigated)",   "DI Gender":"0.714","80% Rule":"❌ FAIL","EO Difference":"0.187","EO Threshold":"✅ PASS"},
    ])
    st.dataframe(df_imp, use_container_width=True, hide_index=True)
    warn("<b>No model passes both criteria simultaneously</b> — exactly as Chouldechova proved. "
         "Practitioners must specify which fairness criterion is required <i>before</i> training.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MITIGATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🛡️   Bias Mitigation":
    st.markdown('<div class="section-title">Bias Mitigation Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Exponentiated Gradient in-processing mitigation (Agarwal et al., 2018) with two fairness constraints.</div>', unsafe_allow_html=True)

    mits = D["mitigation"]

    # Interactive slider
    st.markdown("#### Interactive: Explore the accuracy-fairness trade-off")
    w = st.slider("Fairness constraint weight", 0.0, 1.0, 0.5, 0.01)
    sim_acc = 87.02 - 3.38*w
    sim_di  = 0.315 + 0.537*w
    pass_  = sim_di >= 0.8
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Simulated accuracy",  f"{sim_acc:.2f}%", f"{sim_acc-87.02:+.2f}%")
    col2.metric("Simulated DI Gender", f"{sim_di:.3f}",   f"{sim_di-0.315:+.3f}")
    col3.metric("80% Rule",    "✅ PASS" if pass_ else "❌ FAIL")
    col4.metric("Accuracy cost", f"{87.02-sim_acc:.2f}%")

    st.markdown("---")

    labels = [m["label"] for m in mits]
    fig = make_subplots(rows=1,cols=3,
        subplot_titles=["Accuracy (%)", "DI Gender", "Equalized Odds Diff"])
    bar_colors = ["#ef4444","#3b82f6","#22c55e"]
    for col,(key,mult,thr,thr_label) in enumerate([
        ("acc",100,None,None),
        ("di_g",1,0.8,"80% Rule"),
        ("eod",1,0.2,"EO threshold")
    ],1):
        vals=[m[key]*mult for m in mits]
        fig.add_trace(go.Bar(x=labels,y=vals,marker_color=bar_colors,
                              text=[f"{v:.3f}" for v in vals],textposition="outside",
                              showlegend=False),row=1,col=col)
        if thr:
            fig.add_hline(y=thr,line_dash="dash",line_color="#f59e0b",
                          annotation_text=thr_label,row=1,col=col)
    fig.update_layout(height=400,**PLOTLY_THEME)
    st.plotly_chart(fig, use_container_width=True)

    # Results table
    df_mit = pd.DataFrame([{
        "Stage":      m["label"],
        "Accuracy":   f"{m['acc']:.2%}",
        "F1 Score":   f"{m['f1']:.4f}",
        "DI Gender":  f"{m['di_g']:.4f}",
        "DI Race":    f"{m['di_r']:.4f}",
        "EO Diff":    f"{m['eod']:.4f}",
        "FNR Male":   f"{m['fnr_m']:.1%}",
        "FNR Female": f"{m['fnr_f']:.1%}",
        "80% Rule":   "✅ PASS" if m["di_g"]>=0.8 else "❌ FAIL",
    } for m in mits])
    st.dataframe(df_mit, use_container_width=True, hide_index=True)

    # FNR flip
    fig2 = dark_fig()
    fig2.add_trace(go.Bar(name="FNR Male",   x=labels, y=[m["fnr_m"]*100 for m in mits], marker_color="#3b82f6"))
    fig2.add_trace(go.Bar(name="FNR Female", x=labels, y=[m["fnr_f"]*100 for m in mits], marker_color="#ef4444"))
    fig2.update_layout(barmode="group", title="FNR Flip — how mitigation redistributes errors",
                       yaxis_title="%", yaxis_ticksuffix="%", height=360, **PLOTLY_THEME)
    st.plotly_chart(fig2, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        ins("FNR Flip — Compensatory Equity",
            "Dem. Parity mitigation redistributes error: female FNR drops 41.9%→24.3%, "
            "male FNR rises 33.6%→52.1%. In a biased world, a fair algorithm must actively "
            "work harder to 'see' the underrepresented group.")
    with c2:
        ins("Which mitigation to choose?",
            "<b>Legal compliance</b> (hiring/lending) → Demographic Parity (passes 80% Rule). "
            "<b>Error equity</b> (medical screening) → Equalized Odds (lower FNR gap). "
            "Specify the criterion before training.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CONCLUSIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "✅  Conclusions":
    st.markdown('<div class="section-title">Conclusions & Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Summary of all five principal findings and practical guidance for responsible AI deployment.</div>', unsafe_allow_html=True)

    findings = [
        ("Multi-attribute audit",
         "Gender DI ≈ 0.31 (severe) vs Race DI ≈ 0.63 (less severe) — different proxy mechanisms requiring separate audit strategies."),
        ("Statistical rigour",
         "Bootstrap CIs confirm XGBoost accuracy advantage is real; SVM ≈ LR (p=0.73) — choose by fairness when accuracy is statistically tied."),
        ("SMOTE mechanism",
         "69% male composition in positive class → synthetic samples amplify male patterns → DI worsens from 0.315 to 0.293."),
        ("80% Rule critique",
         "Error-blind, base-rate dependent, grounded in Chouldechova's impossibility theorem — treat as necessary but not sufficient."),
        ("Dual mitigation",
         "Dem. Parity (DI=0.85) conflicts with Eq. Odds (EO=0.19) — impossibility theorem confirmed empirically. Choose constraint before deployment."),
    ]
    for i,(title,body) in enumerate(findings,1):
        ins(f"Finding {i} — {title}", body)

    st.markdown("---")
    st.markdown("#### Practitioner recommendations")
    c1,c2 = st.columns(2)
    recs = [
        ("Before training", "Specify which fairness criterion matters for your use case. Do not evaluate post-hoc against whichever metric the model happens to satisfy."),
        ("Audit multiple attributes", "Single-attribute audits miss intersectional disadvantage. A White Female and a Black Female face different compounded penalties."),
        ("Don't trust SMOTE for fairness", "Class-balancing is fairness-blind when class imbalance correlates with protected attributes — as it always does in historically biased data."),
        ("Report CIs, not point estimates", "Accuracy differences without confidence intervals may be sampling noise. Use McNemar's test to confirm significance before claiming one model beats another."),
    ]
    for i,(title,body) in enumerate(recs):
        with (c1 if i%2==0 else c2):
            warn(f"<b>{title}:</b> {body}")

    st.markdown("---")
    ins("Central conclusion",
        "Predictive accuracy is a dangerous metric when used in isolation. A model can be "
        "simultaneously state-of-the-art in performance and severely discriminatory. "
        "Responsible AI requires specifying fairness criteria before deployment, "
        "auditing multiple protected attributes, and critically interrogating regulatory "
        "compliance frameworks.")

    st.markdown("---")
    st.markdown("#### References")
    refs = [
        "Agarwal et al. (2018). A Reductions Approach to Fair Classification. *ICML*.",
        "Barocas & Selbst (2016). Big Data's Disparate Impact. *California Law Review, 104*, 671.",
        "Chouldechova (2017). Fair prediction with disparate impact. *Big Data, 5*(2), 153–163.",
        "Corbett-Davies & Goel (2018). The Measure and Mismeasure of Fairness. *arXiv:1808.00023*.",
        "Dwork et al. (2012). Fairness through awareness. *ITCS 2012*.",
        "Hardt et al. (2016). Equality of opportunity in supervised learning. *NeurIPS 29*.",
        "Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 30*.",
        "U.S. EEOC (1978). Uniform Guidelines on Employee Selection Procedures. *Federal Register 43*(166).",
    ]
    for i,r in enumerate(refs,1):
        st.markdown(f"{i}. {r}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RESEARCH PAPER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📄  Research Paper":
    st.markdown('<div class="section-title">Research Paper</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Auditing and Mitigating Algorithmic Bias: A Comparative Study of Machine Learning Fairness across Architectures</div>', unsafe_allow_html=True)

    # ── Paper metadata card ────────────────────────────────────────────────
    st.markdown("""
<div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:24px;margin-bottom:20px;">
  <div style="font-size:1.1rem;font-weight:600;color:#f1f5f9;margin-bottom:8px;">
    Auditing and Mitigating Algorithmic Bias
  </div>
  <div style="font-size:0.9rem;color:#64748b;margin-bottom:16px;">
    A Comparative Study of Machine Learning Fairness across Architectures
  </div>
  <div style="display:flex;gap:24px;flex-wrap:wrap;font-size:0.82rem;">
    <div><span style="color:#64748b;">Author</span><br><span style="color:#94a3b8;">Muhammad Atif Ahmed</span></div>
    <div><span style="color:#64748b;">Course</span><br><span style="color:#94a3b8;">CS-408 · Introduction to AI</span></div>
    <div><span style="color:#64748b;">Date</span><br><span style="color:#94a3b8;">April, 2026</span></div>
    <div><span style="color:#64748b;">Pages</span><br><span style="color:#94a3b8;">7 pages</span></div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Abstract ───────────────────────────────────────────────────────────
    with st.expander("📋 Abstract", expanded=True):
        st.markdown("""
The rapid integration of Artificial Intelligence into socio-economic decision-making has created an urgent 
need for algorithmic transparency and equity. This research conducts a multi-layered audit of five machine 
learning architectures — Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and Linear SVM — 
utilising the UCI Adult Income Dataset across two sensitive attributes: **gender** and **race**.

Findings reveal an **"Intelligence-Bias Paradox"** where technical optimisation via feature scaling improves 
predictive utility while simultaneously exacerbating demographic disparity. The baseline XGBoost achieved 
87.02% accuracy but a severely biased Disparate Impact (DI) ratio of 0.31. A fairness-aware Exponentiated 
Gradient model with Demographic Parity achieves 83.64% accuracy and DI = 0.85, meeting the legal 80% threshold.

The study critically interrogates the 80% Rule's statistical limitations citing Chouldechova's (2017) 
impossibility theorem, and demonstrates that SMOTE-based oversampling worsens fairness due to the 69% male 
composition of the positive class.
""")

    # ── Sections summary ───────────────────────────────────────────────────
    st.markdown("#### Paper Structure")
    sections = [
        ("1. Introduction",        "Bridges the 'trust gap' in ML systems using XAI and fairness-aware mitigation."),
        ("2. Literature Review",   "Covers Barocas & Selbst (2016), Chouldechova (2017), Hardt et al. (2016), Lundberg & Lee (2017) and 7 more sources."),
        ("3. Methodology",         "UCI Adult Dataset (30,162 rows), One-Hot Encoding, StandardScaler, stratified 80/20 split, 3 fairness metrics."),
        ("4. Results & Analysis",  "5-model comparison with 95% bootstrap CIs, McNemar's tests, SHAP explainability, SMOTE mechanism analysis."),
        ("5. Discussion",          "Demographic Parity vs Equalized Odds mitigation — FNR Flip explained, Chouldechova impossibility demonstrated."),
        ("6. Conclusion",          "5 principal contributions beyond baseline. Central finding: accuracy alone is dangerous."),
        ("References",             "11 academic sources including ICML, NeurIPS, California Law Review, EEOC Guidelines."),
    ]
    for sec, desc in sections:
        with st.expander(sec):
            st.markdown(desc)

    st.markdown("---")

    # ── Key findings visual ────────────────────────────────────────────────
    st.markdown("#### Key Results at a Glance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
<div style="background:#1e293b;border-radius:10px;padding:16px;text-align:center;border:1px solid #334155;">
  <div style="font-size:2rem;font-weight:700;color:#ef4444;">0.315</div>
  <div style="font-size:0.8rem;color:#64748b;margin-top:4px;">Baseline DI Gender<br>(❌ fails 80% rule)</div>
</div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
<div style="background:#1e293b;border-radius:10px;padding:16px;text-align:center;border:1px solid #334155;">
  <div style="font-size:2rem;font-weight:700;color:#4ade80;">0.852</div>
  <div style="font-size:0.8rem;color:#64748b;margin-top:4px;">Mitigated DI Gender<br>(✅ passes 80% rule)</div>
</div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
<div style="background:#1e293b;border-radius:10px;padding:16px;text-align:center;border:1px solid #334155;">
  <div style="font-size:2rem;font-weight:700;color:#fbbf24;">3.4%</div>
  <div style="font-size:0.8rem;color:#64748b;margin-top:4px;">Accuracy cost<br>of fairness</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Download section ───────────────────────────────────────────────────
    st.markdown("#### Download Research Paper")

    paper_path = "Research Paper/Auditing_and_Mitigating_Algorithmic_Bias.pdf"
    try:
        with open(paper_path, "rb") as f:
            pdf_bytes = f.read()
        col1, col2 = st.columns([1, 3])
        with col1:
            st.download_button(
                label="⬇️ Download PDF",
                data=pdf_bytes,
                file_name="Auditing_and_Mitigating_Algorithmic_Bias.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with col2:
            st.markdown(f"""
<div style="background:#1e293b;border-radius:8px;padding:12px 16px;border:1px solid #334155;font-size:0.85rem;color:#94a3b8;">
  📄 <b style="color:#f1f5f9;">Auditing_and_Mitigating_Algorithmic_Bias.pdf</b><br>
  
</div>""", unsafe_allow_html=True)
    except FileNotFoundError:
        st.info("""
📂 **To enable PDF download:** Add your research paper PDF to the `Research Paper/` folder in your GitHub repo.

Expected path: `Research Paper/Auditing_and_Mitigating_Algorithmic_Bias.pdf`
""")
        st.markdown("""
The paper covers:
- Multi-attribute bias audit (gender + race)  
- 95% bootstrap confidence intervals on all metrics  
- McNemar's statistical significance tests  
- SMOTE mechanism analysis  
- Critical examination of the 80% Rule  
- Demographic Parity vs Equalized Odds mitigation comparison
""")

# ── View inline ────────────────────────────────────────────────────────

# Check if file exists before processing
    if os.path.exists(paper_path):
        # This replaces the entire <iframe> logic
        pdf_viewer(paper_path, width=1200)
    else:
         st.error(f"File not found: {paper_path}")



# ══════════════════════════════════════════════════════════════════════════════
# PAGE: JUPYTER NOTEBOOK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📓  Jupyter Notebook":
    st.markdown('<div class="section-title">Jupyter Notebook</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Full annotated code notebook with 11 sections — each with markdown explanations, code, outputs, and findings.</div>', unsafe_allow_html=True)

    # ── Notebook metadata ──────────────────────────────────────────────────
    st.markdown("""
<div style="background:#1e293b;border:1px solid #334155;border-radius:12px;padding:24px;margin-bottom:20px;">
  <div style="font-size:1.1rem;font-weight:600;color:#f1f5f9;margin-bottom:8px;">
    Bias_Audit_Improved.ipynb
  </div>
  <div style="font-size:0.82rem;color:#64748b;margin-bottom:16px;">Complete code notebook with explanations, visualisations and findings</div>
  <div style="display:flex;gap:24px;flex-wrap:wrap;font-size:0.82rem;">
    <div><span style="color:#64748b;">Sections</span><br><span style="color:#94a3b8;">11 sections</span></div>
    <div><span style="color:#64748b;">Kernel</span><br><span style="color:#94a3b8;">Python 3.11</span></div>
    <div><span style="color:#64748b;">Key libraries</span><br><span style="color:#94a3b8;">sklearn · xgboost · fairlearn · shap</span></div>
    <div><span style="color:#64748b;">Runtime</span><br><span style="color:#94a3b8;">~5 min full run</span></div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Notebook table of contents ─────────────────────────────────────────
    st.markdown("#### Notebook Contents")
    toc = [
        ("Section 1",  "Setup & Imports",                    "All libraries imported and configured"),
        ("Section 2",  "Data Loading & Cleaning",            "UCI Adult dataset, missing value handling, class distribution"),
        ("Section 3",  "Exploratory Bias Audit",             "Gender + race disparity charts, intersectional Gender×Race analysis"),
        ("Section 4",  "Feature Engineering & Encoding",     "One-Hot Encoding, StandardScaler, stratified train/test split"),
        ("Section 5",  "Comparative Model Audit",            "5 models with 95% bootstrap CIs, accuracy + fairness metrics"),
        ("Section 6",  "Statistical Significance",           "McNemar's test on all model pairs, significance table"),
        ("Section 7",  "SHAP Explainability",                "Global importance + gender-stratified SHAP, double-standard effect"),
        ("Section 8",  "SMOTE Mechanism Analysis",           "Why oversampling amplifies bias — mechanistic explanation"),
        ("Section 9",  "80% Rule Critique",                  "4 limitations, Chouldechova impossibility theorem demonstrated"),
        ("Section 10", "Bias Mitigation",                    "Dem. Parity + Eq. Odds mitigated models, FNR flip analysis"),
        ("Section 11", "Summary Dashboard",                  "Dark-theme final summary figure with all key findings"),
    ]
    for sec, title, desc in toc:
        col1, col2, col3 = st.columns([1, 2, 3])
        with col1:
            st.markdown(f"`{sec}`")
        with col2:
            st.markdown(f"**{title}**")
        with col3:
            st.markdown(f"<span style='color:#64748b;font-size:0.85rem;'>{desc}</span>", unsafe_allow_html=True)
        st.divider()

    st.markdown("---")

    # ── Download section ───────────────────────────────────────────────────
    st.markdown("#### Download Notebook")

    nb_path = "Notebook/Bias_Audit_Improved.ipynb"
    try:
        with open(nb_path, "rb") as f:
            nb_bytes = f.read()

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.download_button(
                label="⬇️ Download .ipynb",
                data=nb_bytes,
                file_name="Bias_Audit_Improved.ipynb",
                mime="application/json",
                use_container_width=True,
            )
        with col2:
            # Also offer as .py
            import json
            try:
                nb_json = json.loads(nb_bytes)
                py_lines = []
                for cell in nb_json.get("cells", []):
                    if cell["cell_type"] == "markdown":
                        py_lines.append("# " + "".join(cell["source"]).replace("\n", "\n# "))
                        py_lines.append("")
                    elif cell["cell_type"] == "code":
                        py_lines.append("".join(cell["source"]))
                        py_lines.append("")
                py_text = "\n".join(py_lines).encode()
                st.download_button(
                    label="⬇️ Download .py",
                    data=py_text,
                    file_name="Bias_Audit_Improved.py",
                    mime="text/plain",
                    use_container_width=True,
                )
            except Exception:
                pass
        with col3:
            st.markdown(f"""
<div style="background:#1e293b;border-radius:8px;padding:12px 16px;border:1px solid #334155;font-size:0.85rem;color:#94a3b8;">
  📓 <b style="color:#f1f5f9;">Bias_Audit_Improved.ipynb</b><br>
  <span style="color:#64748b;">11 sections · Full annotations · Run in Jupyter or VS Code</span>
</div>""", unsafe_allow_html=True)

    except FileNotFoundError:
        st.info("""
📂 **To enable notebook download:** Add your notebook to the `Notebook/` folder in your GitHub repo.

Expected path: `Notebook/Bias_Audit_Improved.ipynb`
""")

    # ── How to run locally ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### How to Run the Notebook Locally")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Option A — Jupyter Lab**")
        st.code("""pip install jupyterlab
jupyter lab Bias_Audit_Improved.ipynb""", language="bash")

    with col2:
        st.markdown("**Option B — VS Code**")
        st.markdown("""
1. Install the **Jupyter** extension in VS Code  
2. Open `Bias_Audit_Improved.ipynb`  
3. Select Python 3.11 kernel  
4. Run All Cells
""")

    st.markdown("**Install all dependencies first:**")
    st.code("""pip install pandas numpy scikit-learn xgboost imbalanced-learn fairlearn shap plotly matplotlib scipy""", language="bash")

    # ── View notebook sections inline ──────────────────────────────────────
    nb_path = "Notebook/Bias_Audit_Improved.ipynb"
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb_json = json.load(f)

        st.markdown("---")
        st.markdown("#### Notebook Preview")
        st.caption("Showing markdown cells and code — outputs not shown here, run locally for full results.")

        for i, cell in enumerate(nb_json.get("cells", [])[:30]):  # first 30 cells
            if cell["cell_type"] == "markdown":
                content = "".join(cell["source"])
                if content.strip().startswith("#"):
                    st.markdown(content)
                else:
                    st.markdown(content)
            elif cell["cell_type"] == "code":
                content = "".join(cell["source"]).strip()
                if content:
                    st.code(content, language="python")

    except FileNotFoundError:
        pass
    except Exception:
        pass
# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯  Live Predictor":
    st.markdown('<div class="section-title">Live Income Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Enter a person\'s profile and see how all 5 models predict their income — live. Watch how gender and race change the prediction probability.</div>', unsafe_allow_html=True)

    trained     = D["trained"]
    scaler      = D["scaler"]
    feature_cols= D["feature_cols"]
    num_cols_p  = D["num_cols"]

    # ── Dataset options (from training data) ──────────────────────────────
    df_ref = D["df"]
    workclass_opts  = sorted(df_ref["workclass"].dropna().unique().tolist())
    education_opts  = ["Preschool","1st-4th","5th-6th","7th-8th","9th","10th",
                       "11th","12th","HS-grad","Some-college","Assoc-voc",
                       "Assoc-acdm","Bachelors","Prof-school","Masters","Doctorate"]
    marital_opts    = sorted(df_ref["marital-status"].dropna().unique().tolist())
    occupation_opts = sorted(df_ref["occupation"].dropna().unique().tolist())
    relationship_opts = sorted(df_ref["relationship"].dropna().unique().tolist())
    race_opts       = sorted(df_ref["race"].dropna().unique().tolist())
    country_opts    = sorted(df_ref["native-country"].dropna().unique().tolist())

    edu_to_num = {"Preschool":1,"1st-4th":2,"5th-6th":3,"7th-8th":4,"9th":5,
                  "10th":6,"11th":7,"12th":8,"HS-grad":9,"Some-college":10,
                  "Assoc-voc":11,"Assoc-acdm":12,"Bachelors":13,
                  "Prof-school":14,"Masters":14,"Doctorate":16}

    st.markdown("---")
    st.markdown("#### Enter a person's profile")

    # ── Input form ─────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Demographics**")
        age         = st.slider("Age", 17, 90, 35)
        sex         = st.selectbox("Sex", ["Male", "Female"])
        race        = st.selectbox("Race", race_opts, index=race_opts.index("White") if "White" in race_opts else 0)
        relationship= st.selectbox("Relationship", relationship_opts,
                                    index=relationship_opts.index("Husband") if "Husband" in relationship_opts else 0)
        marital     = st.selectbox("Marital Status", marital_opts,
                                    index=marital_opts.index("Married-civ-spouse") if "Married-civ-spouse" in marital_opts else 0)
        country     = st.selectbox("Native Country", country_opts,
                                    index=country_opts.index("United-States") if "United-States" in country_opts else 0)

    with c2:
        st.markdown("**Education & Work**")
        education   = st.selectbox("Education", education_opts,
                                    index=education_opts.index("Bachelors"))
        edu_num     = edu_to_num.get(education, 9)
        st.caption(f"Education-num: **{edu_num}**")
        workclass   = st.selectbox("Work Class", workclass_opts,
                                    index=workclass_opts.index("Private") if "Private" in workclass_opts else 0)
        occupation  = st.selectbox("Occupation", occupation_opts,
                                    index=occupation_opts.index("Exec-managerial") if "Exec-managerial" in occupation_opts else 0)
        hours_week  = st.slider("Hours per Week", 1, 99, 40)

    with c3:
        st.markdown("**Financials**")
        capital_gain = st.number_input("Capital Gain ($)", 0, 99999, 0, step=500)
        capital_loss = st.number_input("Capital Loss ($)", 0, 4356, 0, step=100)
        fnlwgt       = st.number_input("Final Weight (fnlwgt)", 10000, 1500000, 189778, step=10000,
                                        help="Census weighting variable — leave default unless you have a specific value")

    st.markdown("---")

    # ── Build feature vector ───────────────────────────────────────────────
    def build_input_row(age, workclass, fnlwgt, education, edu_num,
                         marital, occupation, relationship, race, sex,
                         capital_gain, capital_loss, hours_week, country):
        """Build a single-row DataFrame matching the training feature columns."""
        raw = pd.DataFrame([{
            "age": age, "workclass": workclass, "fnlwgt": fnlwgt,
            "education": education, "education-num": edu_num,
            "marital-status": marital, "occupation": occupation,
            "relationship": relationship, "race": race, "sex": sex,
            "capital-gain": capital_gain, "capital-loss": capital_loss,
            "hours-per-week": hours_week, "native-country": country,
            "income": 0  # dummy
        }])

        cats = ["workclass","education","marital-status","occupation",
                "relationship","race","sex","native-country"]
        raw_enc = pd.get_dummies(raw, columns=cats, drop_first=True)
        raw_enc = raw_enc.drop("income", axis=1)

        # Align to training columns — add missing cols as 0, drop extras
        for col in feature_cols:
            if col not in raw_enc.columns:
                raw_enc[col] = 0
        raw_enc = raw_enc[feature_cols]

        # Scale numerical
        raw_scaled = raw_enc.copy()
        raw_scaled[num_cols_p] = scaler.transform(raw_enc[num_cols_p])
        return raw_scaled

    input_row = build_input_row(age, workclass, fnlwgt, education, edu_num,
                                  marital, occupation, relationship, race, sex,
                                  capital_gain, capital_loss, hours_week, country)

    # ── Run all 5 models ───────────────────────────────────────────────────
    model_order  = ["Logistic Regression","Random Forest","Gradient Boosting","XGBoost","SVM (Linear)"]
    model_colors = {"Logistic Regression":"#3b82f6","Random Forest":"#22c55e",
                    "Gradient Boosting":"#f59e0b","XGBoost":"#a78bfa","SVM (Linear)":"#ef4444"}

    predictions = {}
    probabilities = {}

    for name, mdl in trained.items():
        pred = int(mdl.predict(input_row)[0])
        predictions[name] = pred
        if hasattr(mdl, "predict_proba"):
            prob = float(mdl.predict_proba(input_row)[0][1])
        else:
            # SVM — use decision function, convert to pseudo-probability via sigmoid
            score = float(mdl.decision_function(input_row)[0])
            prob  = float(1 / (1 + np.exp(-score)))
        probabilities[name] = prob

    # ── Results display ────────────────────────────────────────────────────
    st.markdown("#### Predictions from all 5 models")

    # Summary verdict
    votes_high = sum(1 for p in predictions.values() if p == 1)
    votes_low  = 5 - votes_high
    verdict_color = "#22c55e" if votes_high >= 3 else "#ef4444"
    verdict_text  = f">50K ({votes_high}/5 models agree)" if votes_high >= 3 else f"≤50K ({votes_low}/5 models agree)"

    st.markdown(f"""
<div style="background:#1e293b;border:1px solid #334155;border-radius:12px;
            padding:20px 28px;margin-bottom:20px;display:flex;align-items:center;gap:20px;">
  <div>
    <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">Consensus prediction</div>
    <div style="font-size:1.8rem;font-weight:700;color:{verdict_color};">{verdict_text}</div>
  </div>
  <div style="margin-left:auto;font-size:0.85rem;color:#64748b;">
    {sex} · {age} yrs · {education} · {occupation}
  </div>
</div>""", unsafe_allow_html=True)

    # Per-model probability bars
    import plotly.graph_objects as go
    fig = go.Figure()

    for name in model_order:
        prob = probabilities[name]
        pred = predictions[name]
        bar_color = "#22c55e" if pred == 1 else "#ef4444"
        fig.add_trace(go.Bar(
            name=name, x=[prob * 100], y=[name],
            orientation="h",
            marker_color=bar_color,
            text=f"{'  >50K ✓' if pred==1 else '  ≤50K ✗'}  {prob*100:.1f}%",
            textposition="outside",
            width=0.55,
        ))

    fig.add_vline(x=50, line_dash="dash", line_color="#f59e0b",
                  annotation_text="50% threshold", annotation_position="top")
    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#111827",
        font=dict(family="Inter", color="#94a3b8"),
        xaxis=dict(title="Probability of earning >50K (%)", range=[0, 115],
                   gridcolor="#1e293b", ticksuffix="%"),
        yaxis=dict(gridcolor="#1e293b"),
        showlegend=False,
        height=300,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Bias demo: toggle gender/race ──────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Bias Demo — change only gender or race, keep everything else the same")
    st.caption("This shows the model's 'bonus' or 'penalty' for demographic attributes independent of qualifications.")

    demo_cols = st.columns(4)

    profiles = [
        ("Male · White",   "Male",   "White"),
        ("Male · Black",   "Male",   "Black"),
        ("Female · White", "Female", "White"),
        ("Female · Black", "Female", "Black"),
    ]

    xgb_model = trained["XGBoost"]
    demo_results = []
    for label, s, r in profiles:
        row = build_input_row(age, workclass, fnlwgt, education, edu_num,
                               marital, occupation, relationship, r, s,
                               capital_gain, capital_loss, hours_week, country)
        if hasattr(xgb_model, "predict_proba"):
            p = float(xgb_model.predict_proba(row)[0][1]) * 100
        else:
            score = float(xgb_model.decision_function(row)[0])
            p = float(1 / (1 + np.exp(-score))) * 100
        demo_results.append((label, p))

    # Find max prob for reference
    max_prob = max(p for _, p in demo_results)

    for col, (label, prob) in zip(demo_cols, demo_results):
        diff = prob - max_prob if prob != max_prob else 0
        diff_str = f"{diff:+.1f}%" if diff != 0 else "baseline"
        p_color = "#22c55e" if prob >= 50 else "#ef4444"
        diff_color = "#ef4444" if diff < -5 else "#f59e0b" if diff < 0 else "#22c55e"
        col.markdown(f"""
<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;
            padding:14px 16px;text-align:center;">
  <div style="font-size:0.78rem;color:#64748b;margin-bottom:6px;">{label}</div>
  <div style="font-size:1.6rem;font-weight:700;color:{p_color};">{prob:.1f}%</div>
  <div style="font-size:0.75rem;color:{diff_color};margin-top:4px;">{diff_str}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("")
    st.caption("XGBoost model · Same age, education, occupation, hours — only gender/race changed")

    # ── SHAP explanation for this specific input ───────────────────────────
    st.markdown("---")
    st.markdown("#### Why did XGBoost predict this? (SHAP explanation for this person)")

    try:
        explainer_live = shap.TreeExplainer(xgb_model)
        sv_live_raw    = explainer_live.shap_values(input_row)
        if isinstance(sv_live_raw, list):   sv_live = sv_live_raw[1][0]
        elif len(sv_live_raw.shape) == 3:   sv_live = sv_live_raw[0, :, 1]
        else:                               sv_live = sv_live_raw[0]

        # Top 10 contributing features
        shap_series = pd.Series(sv_live, index=feature_cols)
        top_pos = shap_series.nlargest(5)
        top_neg = shap_series.nsmallest(5)
        top_features = pd.concat([top_pos, top_neg]).sort_values(ascending=True)

        fig2 = go.Figure()
        colors_shap = ["#ef4444" if v < 0 else "#22c55e" for v in top_features.values]
        fig2.add_trace(go.Bar(
            y=top_features.index,
            x=top_features.values,
            orientation="h",
            marker_color=colors_shap,
            text=[f"{v:+.4f}" for v in top_features.values],
            textposition="outside",
        ))
        fig2.add_vline(x=0, line_color="#64748b", line_width=1)
        fig2.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#111827",
            font=dict(family="Inter", color="#94a3b8"),
            xaxis=dict(title="SHAP value (pushes prediction ← lower | higher →)",
                       gridcolor="#1e293b", zerolinecolor="#64748b"),
            yaxis=dict(gridcolor="#1e293b"),
            showlegend=False, height=340,
            margin=dict(l=10, r=80, t=20, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Green = pushes prediction toward >50K · Red = pushes toward ≤50K · For this specific person")

    except Exception as e:
        st.info(f"SHAP explanation unavailable: {e}")

    # ── Insight ────────────────────────────────────────────────────────────
    ins("Try this in your presentation",
        "Set Age=35, Education=Bachelors, Occupation=Exec-managerial, Hours=40, Capital Gain=0. "
        "Run it as Male/White → note the probability. "
        "Then change ONLY Sex to Female → watch the probability drop. "
        "Then show the bias demo panel — same profile, 4 demographic combinations. "
        "The audience will see the model's hidden penalty live.")
