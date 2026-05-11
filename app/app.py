# ============================================
# SWASTHAI - Full App with Backend
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, sys, time
from datetime import datetime

sys.path.append(r"D:\Python\PythonProject\SwasthAI")
from src.config import MODEL_DIR
from src.database import (
    init_db, save_prediction, get_all_predictions,
    get_prediction_by_id, get_analytics, delete_prediction
)

st.set_page_config(
    page_title="SwasthAI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Initialize DB ─────────────────────────────────────────────
init_db()

# ─── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
    --bg:#0a0f1e; --surface:#111827; --card:#1a2235;
    --border:#2a3a55; --accent:#3b82f6; --accent2:#06b6d4;
    --danger:#ef4444; --warn:#f59e0b; --success:#10b981;
    --text:#f1f5f9; --muted:#94a3b8;
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:var(--bg)!important;color:var(--text)!important;}
.stApp{background:var(--bg)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:2rem 2rem 4rem!important;max-width:1100px!important;}
section[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
section[data-testid="stSidebar"] *{color:var(--text)!important;}
.hero{text-align:center;padding:2.5rem 1rem 1.5rem;background:radial-gradient(ellipse at 50% 0%,#1e3a5f55 0%,transparent 70%);border-bottom:1px solid var(--border);margin-bottom:2rem;}
.hero-badge{display:inline-block;font-size:11px;letter-spacing:.15em;text-transform:uppercase;color:var(--accent2);border:1px solid var(--accent2);padding:4px 14px;border-radius:20px;margin-bottom:1rem;}
.hero h1{font-family:'DM Serif Display',serif;font-size:clamp(2rem,5vw,3rem);font-weight:400;margin:0 0 .6rem;background:linear-gradient(135deg,#f1f5f9 30%,#93c5fd 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hero p{color:var(--muted);font-size:.95rem;max-width:480px;margin:0 auto;line-height:1.7;}
.section-label{font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:.8rem;}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:1.5rem;}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1rem 1.2rem;}
.kpi .lbl{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;}
.kpi .val{font-size:1.8rem;font-weight:600;margin-top:2px;}
.kpi .sub{font-size:11px;color:var(--muted);margin-top:2px;}
.result-hero{border-radius:20px;padding:2rem;text-align:center;margin:1.2rem 0;}
.result-hero h2{font-family:'DM Serif Display',serif;font-size:1.8rem;font-weight:400;margin:.4rem 0;}
.result-high{background:linear-gradient(135deg,#450a0a,#7f1d1d);border:1px solid #ef444455;}
.result-moderate{background:linear-gradient(135deg,#431407,#78350f);border:1px solid #f59e0b55;}
.result-low{background:linear-gradient(135deg,#052e16,#14532d);border:1px solid #10b98155;}
.stButton>button{background:linear-gradient(135deg,var(--accent),var(--accent2))!important;color:white!important;border:none!important;border-radius:12px!important;padding:.75rem 1.5rem!important;font-family:'DM Sans',sans-serif!important;font-weight:600!important;width:100%;}
.pill{display:inline-block;padding:5px 12px;border-radius:20px;font-size:12px;font-weight:500;margin:3px;}
.pill-d{background:#450a0a;color:#fca5a5;border:1px solid #ef444440;}
.pill-w{background:#431407;color:#fcd34d;border:1px solid #f59e0b40;}
.pill-o{background:#052e16;color:#6ee7b7;border:1px solid #10b98140;}
.rec-card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.1rem;}
.history-row{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:.9rem 1.1rem;margin-bottom:.6rem;display:flex;align-items:center;gap:16px;}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;}
.badge-r{background:#450a0a;color:#fca5a5;}
.badge-g{background:#052e16;color:#6ee7b7;}
.badge-w{background:#431407;color:#fcd34d;}
.bar-track{background:var(--border);border-radius:4px;height:7px;margin:4px 0 10px;}
.bar-fill{height:100%;border-radius:4px;}
label,.stSlider label{color:var(--muted)!important;font-size:13px!important;font-weight:500!important;}
.stNumberInput input{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:10px!important;color:var(--text)!important;}
.stTextInput input{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:10px!important;color:var(--text)!important;}
hr{border-color:var(--border)!important;}
</style>
""", unsafe_allow_html=True)

# ─── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model_and_scaler():
    with open(os.path.join(MODEL_DIR,"best_model.pkl"),'rb') as f: model=pickle.load(f)
    with open(os.path.join(MODEL_DIR,"scaler.pkl"),'rb') as f: scaler=pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

# ─── Sidebar Navigation ────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.5rem 0 1rem;'>
      <div style='font-family:DM Serif Display,serif;font-size:1.6rem;'>🫀 SwasthAI</div>
      <div style='font-size:11px;color:#94a3b8;margin-top:4px;'>Diabetes Risk System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔍  Predict", "📋  Patient History", "📊  Analytics"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    analytics = get_analytics()
    st.markdown(f"""
    <div style='font-size:12px;color:#94a3b8;'>
      <div style='margin-bottom:6px;'>Total predictions: <b style='color:#f1f5f9;'>{analytics['total']}</b></div>
      <div style='margin-bottom:6px;'>High risk flagged: <b style='color:#ef4444;'>{analytics['high_risk']}</b></div>
      <div style='margin-bottom:6px;'>Low risk cleared: <b style='color:#10b981;'>{analytics['low_risk']}</b></div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE 1 — PREDICT
# ══════════════════════════════════════════════
if "Predict" in page:

    st.markdown("""
    <div class="hero">
      <div class="hero-badge">AI-Powered Health Screening</div>
      <h1>SwasthAI</h1>
      <p>Enter patient vitals. Our AI will assess diabetes risk instantly.</p>
    </div>
    """, unsafe_allow_html=True)

    # Patient name
    patient_name = st.text_input("Patient Name (optional)", placeholder="e.g. Rahul Sharma")

    st.markdown('<div class="section-label">Patient vitals</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Glucose** *(mg/dL)*")
        glucose = st.slider("G",44,200,110,label_visibility="collapsed")
        st.markdown(f"<div style='text-align:center;font-size:2rem;font-weight:700;color:{'#ef4444' if glucose>140 else '#f59e0b' if glucose>110 else '#10b981'}'>{glucose}</div>",unsafe_allow_html=True)
        st.markdown("<br>**BMI**")
        bmi = st.slider("B",18.0,67.0,25.0,0.1,label_visibility="collapsed")
        st.markdown(f"<div style='text-align:center;font-size:2rem;font-weight:700;color:{'#ef4444' if bmi>30 else '#f59e0b' if bmi>25 else '#10b981'}'>{bmi:.1f}</div>",unsafe_allow_html=True)
        st.markdown("<br>**Age** *(years)*")
        age = st.slider("A",21,81,30,label_visibility="collapsed")
        st.markdown(f"<div style='text-align:center;font-size:2rem;font-weight:700;color:{'#f59e0b' if age>45 else '#10b981'}'>{age}</div>",unsafe_allow_html=True)

    with c2:
        st.markdown("**Blood Pressure** *(mm Hg)*")
        bp = st.slider("BP",24,122,72,label_visibility="collapsed")
        st.markdown(f"<div style='text-align:center;font-size:2rem;font-weight:700;color:{'#f59e0b' if bp>90 else '#10b981'}'>{bp}</div>",unsafe_allow_html=True)
        st.markdown("<br>**Insulin** *(IU/mL)*")
        insulin = st.slider("I",14,846,80,label_visibility="collapsed")
        st.markdown(f"<div style='text-align:center;font-size:2rem;font-weight:700;color:var(--text)'>{insulin}</div>",unsafe_allow_html=True)
        st.markdown("<br>**Skin Thickness** *(mm)*")
        skin = st.slider("S",7,99,20,label_visibility="collapsed")
        st.markdown(f"<div style='text-align:center;font-size:2rem;font-weight:700;color:var(--text)'>{skin}</div>",unsafe_allow_html=True)

    with c3:
        st.markdown("**Pregnancies**")
        preg = st.number_input("P",0,20,1,label_visibility="collapsed")
        st.markdown(f"<div style='text-align:center;font-size:2rem;font-weight:700;color:var(--text)'>{preg}</div>",unsafe_allow_html=True)
        st.markdown("<br>**Diabetes Pedigree Function**")
        dpf = st.slider("D",0.07,2.50,0.47,0.01,label_visibility="collapsed")
        st.markdown(f"<div style='text-align:center;font-size:2rem;font-weight:700;color:{'#f59e0b' if dpf>0.8 else '#10b981'}'>{dpf:.2f}</div>",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    predict_btn = st.button("🔍  Analyze & Predict Diabetes Risk")

    if predict_btn:
        with st.spinner("Running AI analysis..."):
            time.sleep(0.7)

        cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                'Insulin','BMI','DiabetesPedigreeFunction','Age']
        pdf = pd.DataFrame([[preg,glucose,bp,skin,insulin,bmi,dpf,age]],columns=cols)
        psc = pd.DataFrame(scaler.transform(pdf),columns=cols)

        pred   = model.predict(psc)[0]
        prob   = model.predict_proba(psc)[0]
        no_pct = round(prob[0]*100,1)
        ye_pct = round(prob[1]*100,1)

        if pred==1 and ye_pct>=70:   risk="Very High"; css="result-high";   color="#ef4444"; icon="🔴"; msg="Strong indicators detected. Immediate consultation recommended."
        elif pred==1 and ye_pct>=50: risk="High";      css="result-high";   color="#ef4444"; icon="🔴"; msg="Several indicators elevated. See a doctor soon."
        elif pred==1:                risk="Moderate";  css="result-moderate";color="#f59e0b"; icon="🟡"; msg="Some indicators present. Schedule a checkup."
        elif no_pct>=80:             risk="Very Low";  css="result-low";    color="#10b981"; icon="🟢"; msg="Indicators look healthy. Maintain your lifestyle."
        else:                        risk="Low";       css="result-low";    color="#10b981"; icon="🟢"; msg="Most indicators normal. Stay active and get checkups."

        # Save to database
        pred_id = save_prediction(
            patient_name or "Anonymous", age, preg, glucose,
            bp, skin, insulin, bmi, dpf,
            int(pred), prob[0], prob[1], risk
        )

        st.markdown("---")
        st.markdown(f"""
        <div class="result-hero {css}">
          <div style='font-size:2rem;'>{icon}</div>
          <h2>{risk} Risk of Diabetes</h2>
          <p style='opacity:.85;font-size:.9rem;'>{msg}</p>
          <div style='font-size:11px;margin-top:.8rem;opacity:.5;'>Record #{pred_id} saved to history</div>
        </div>
        """, unsafe_allow_html=True)

        ra, rb, rc = st.columns(3)
        with ra:
            st.markdown(f"<div style='text-align:center;'><div style='font-size:3rem;font-weight:700;color:#10b981;'>{no_pct}%</div><div style='font-size:12px;color:#94a3b8;'>No Diabetes</div></div>",unsafe_allow_html=True)
        with rb:
            st.markdown(f"<div style='text-align:center;'><div style='font-size:3rem;font-weight:700;color:{color};'>{ye_pct}%</div><div style='font-size:12px;color:#94a3b8;'>Has Diabetes</div></div>",unsafe_allow_html=True)
        with rc:
            st.markdown(f"<div style='text-align:center;'><div style='font-size:2rem;font-weight:700;color:{color};'>{risk}</div><div style='font-size:12px;color:#94a3b8;'>Risk Level</div></div>",unsafe_allow_html=True)

        st.markdown(f"""
        <div style='margin:12px 0 4px;font-size:12px;color:#94a3b8;'>No Diabetes — {no_pct}%</div>
        <div class='bar-track'><div class='bar-fill' style='width:{no_pct}%;background:#10b981;'></div></div>
        <div style='margin:4px 0;font-size:12px;color:#94a3b8;'>Has Diabetes — {ye_pct}%</div>
        <div class='bar-track'><div class='bar-fill' style='width:{ye_pct}%;background:{color};'></div></div>
        """,unsafe_allow_html=True)

        st.markdown("---")
        fa, fb = st.columns(2)
        with fa:
            st.markdown('<div class="section-label">Risk factors</div>',unsafe_allow_html=True)
            factors = []
            factors.append(("pill-d" if glucose>140 else "pill-w" if glucose>110 else "pill-o", f"Glucose {glucose}"))
            factors.append(("pill-d" if bmi>30 else "pill-w" if bmi>25 else "pill-o", f"BMI {bmi:.1f}"))
            factors.append(("pill-w" if age>45 else "pill-o", f"Age {age}"))
            factors.append(("pill-d" if dpf>0.8 else "pill-w" if dpf>0.5 else "pill-o", f"DPF {dpf:.2f}"))
            factors.append(("pill-w" if bp>90 else "pill-o", f"BP {bp}"))
            st.markdown("".join(f'<span class="pill {c}">{l}</span>' for c,l in factors),unsafe_allow_html=True)

        with fb:
            st.markdown('<div class="section-label">AI focus (feature importance)</div>',unsafe_allow_html=True)
            feats=[("Glucose",28.1),("BMI",17.0),("Age",12.6),("DPF",11.1),("Insulin",10.3)]
            for name,val in feats:
                st.markdown(f"""
                <div style='display:flex;align-items:center;gap:10px;margin-bottom:7px;'>
                  <div style='font-size:12px;color:#94a3b8;width:80px;flex-shrink:0;'>{name}</div>
                  <div class='bar-track' style='flex:1;margin:0;'>
                    <div class='bar-fill' style='width:{val/28.1*100:.0f}%;background:linear-gradient(90deg,#3b82f6,#06b6d4);'></div>
                  </div>
                  <div style='font-size:12px;color:#f1f5f9;width:36px;text-align:right;'>{val}%</div>
                </div>""",unsafe_allow_html=True)

        # Recommendations
        st.markdown("---")
        st.markdown('<div class="section-label">Recommendations</div>',unsafe_allow_html=True)
        rx1,rx2,rx3 = st.columns(3)
        if pred==1:
            recs=[("🏥","See a doctor","Book a consultation within 2 weeks for full clinical evaluation."),
                  ("🥗","Adjust diet","Cut sugar and refined carbs. Add vegetables and whole grains."),
                  ("🏃","Exercise daily","30 min of moderate activity daily reduces diabetes risk significantly.")]
        else:
            recs=[("✅","Stay active","150 min of moderate exercise per week keeps your risk low."),
                  ("🩺","Regular checkups","Blood sugar test every 6 months as a preventive measure."),
                  ("💧","Healthy habits","Balanced diet, good sleep, and low stress protect you long term.")]
        for col,(ic,title,desc) in zip([rx1,rx2,rx3],recs):
            with col:
                st.markdown(f"""
                <div class="rec-card">
                  <div style='font-size:1.4rem;margin-bottom:.4rem;'>{ic}</div>
                  <div style='font-weight:600;margin-bottom:.3rem;'>{title}</div>
                  <div style='font-size:12px;color:#94a3b8;'>{desc}</div>
                </div>""",unsafe_allow_html=True)

        st.markdown("<br><div style='text-align:center;font-size:11px;color:#475569;'>⚠️ SwasthAI is for educational purposes only. Not a medical diagnosis. Always consult a doctor.</div>",unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 2 — HISTORY
# ══════════════════════════════════════════════
elif "History" in page:
    st.markdown('<div style="padding-top:1rem;"></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-label">Patient prediction history</div>',unsafe_allow_html=True)
    st.markdown("### 📋 All Predictions")

    rows = get_all_predictions()

    if not rows:
        st.info("No predictions yet. Go to the Predict page to get started!")
    else:
        # Search
        search = st.text_input("🔍 Search by patient name", placeholder="Type a name...")

        col_h = ['ID','Timestamp','Patient','Age','Glucose','BMI','Prediction','Diabetes %','Risk']
        df = pd.DataFrame(rows, columns=col_h)
        df['Prediction'] = df['Prediction'].map({1:"Has Diabetes",0:"No Diabetes"})
        df['Diabetes %'] = df['Diabetes %'].apply(lambda x: f"{x*100:.1f}%")

        if search:
            df = df[df['Patient'].str.contains(search, case=False, na=False)]

        st.markdown(f"<div style='font-size:13px;color:#94a3b8;margin-bottom:1rem;'>Showing {len(df)} records</div>",unsafe_allow_html=True)

        for _, row in df.iterrows():
            risk_badge = "badge-r" if "High" in str(row['Risk']) else "badge-w" if "Moderate" in str(row['Risk']) else "badge-g"
            pred_badge = "badge-r" if row['Prediction']=="Has Diabetes" else "badge-g"
            st.markdown(f"""
            <div class="history-row">
              <div style='font-size:12px;color:#64748b;width:30px;'>#{row['ID']}</div>
              <div style='flex:1;'>
                <div style='font-weight:500;'>{row['Patient']}</div>
                <div style='font-size:12px;color:#94a3b8;'>Age {row['Age']} · Glucose {row['Glucose']} · BMI {row['BMI']}</div>
              </div>
              <div><span class='badge {pred_badge}'>{row['Prediction']}</span></div>
              <div><span class='badge {risk_badge}'>{row['Risk']}</span></div>
              <div style='font-size:12px;color:#64748b;'>{str(row['Timestamp'])[:16]}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Export data**")
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇️  Download as CSV",
            data=csv,
            file_name=f"swasthai_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )


# ══════════════════════════════════════════════
# PAGE 3 — ANALYTICS
# ══════════════════════════════════════════════
elif "Analytics" in page:
    st.markdown('<div style="padding-top:1rem;"></div>',unsafe_allow_html=True)
    st.markdown('<div class="section-label">Analytics dashboard</div>',unsafe_allow_html=True)
    st.markdown("### 📊 Prediction Analytics")

    a = get_analytics()

    if a['total'] == 0:
        st.info("No data yet. Make some predictions first!")
    else:
        # KPI strip
        pct = round(a['high_risk']/a['total']*100,1) if a['total'] else 0
        st.markdown(f"""
        <div class="kpi-grid">
          <div class="kpi"><div class="lbl">Total screened</div><div class="val" style="color:#93c5fd;">{a['total']}</div><div class="sub">patients</div></div>
          <div class="kpi"><div class="lbl">High risk</div><div class="val" style="color:#ef4444;">{a['high_risk']}</div><div class="sub">{pct}% of total</div></div>
          <div class="kpi"><div class="lbl">Low risk</div><div class="val" style="color:#10b981;">{a['low_risk']}</div><div class="sub">{100-pct}% of total</div></div>
          <div class="kpi"><div class="lbl">Avg glucose</div><div class="val" style="color:#f1f5f9;">{a['avg_glucose']}</div><div class="sub">mg/dL</div></div>
          <div class="kpi"><div class="lbl">Avg BMI</div><div class="val" style="color:#f1f5f9;">{a['avg_bmi']}</div><div class="sub">kg/m²</div></div>
          <div class="kpi"><div class="lbl">Avg age</div><div class="val" style="color:#f1f5f9;">{a['avg_age']}</div><div class="sub">years</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Risk distribution
        if a['risk_dist']:
            st.markdown("---")
            ca, cb = st.columns(2)
            with ca:
                st.markdown('<div class="section-label">Risk distribution</div>',unsafe_allow_html=True)
                for risk_level, count in a['risk_dist']:
                    pct_r = round(count/a['total']*100,1)
                    color = "#ef4444" if "High" in str(risk_level) else "#f59e0b" if "Moderate" in str(risk_level) else "#10b981"
                    st.markdown(f"""
                    <div style='margin-bottom:10px;'>
                      <div style='display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px;'>
                        <span style='color:#94a3b8;'>{risk_level}</span>
                        <span style='color:#f1f5f9;font-weight:500;'>{count} ({pct_r}%)</span>
                      </div>
                      <div class='bar-track'>
                        <div class='bar-fill' style='width:{pct_r}%;background:{color};'></div>
                      </div>
                    </div>""",unsafe_allow_html=True)

            with cb:
                st.markdown('<div class="section-label">Diabetic vs healthy — avg values</div>',unsafe_allow_html=True)
                d_avg = a['diabetic_avg']
                h_avg = a['healthy_avg']
                if d_avg[0] and h_avg[0]:
                    comparisons = [
                        ("Avg glucose", d_avg[0], h_avg[0], 200),
                        ("Avg BMI",     d_avg[1], h_avg[1], 50),
                        ("Avg age",     d_avg[2], h_avg[2], 80),
                    ]
                    for label, dv, hv, mx in comparisons:
                        dv = round(dv,1); hv = round(hv,1)
                        st.markdown(f"""
                        <div style='margin-bottom:12px;'>
                          <div style='font-size:12px;color:#94a3b8;margin-bottom:4px;'>{label}</div>
                          <div style='display:flex;align-items:center;gap:8px;font-size:12px;'>
                            <span style='color:#ef4444;width:90px;'>Diabetic: {dv}</span>
                            <div class='bar-track' style='flex:1;margin:0;'>
                              <div class='bar-fill' style='width:{dv/mx*100:.0f}%;background:#ef4444;'></div>
                            </div>
                          </div>
                          <div style='display:flex;align-items:center;gap:8px;font-size:12px;margin-top:3px;'>
                            <span style='color:#10b981;width:90px;'>Healthy: {hv}</span>
                            <div class='bar-track' style='flex:1;margin:0;'>
                              <div class='bar-fill' style='width:{hv/mx*100:.0f}%;background:#10b981;'></div>
                            </div>
                          </div>
                        </div>""",unsafe_allow_html=True)

        # Daily activity
        if a['daily']:
            st.markdown("---")
            st.markdown('<div class="section-label">Recent daily activity</div>',unsafe_allow_html=True)
            max_day = max(r[1] for r in a['daily'])
            for day, count in reversed(a['daily']):
                pct_d = round(count/max_day*100)
                st.markdown(f"""
                <div style='display:flex;align-items:center;gap:12px;margin-bottom:8px;'>
                  <div style='font-size:12px;color:#94a3b8;width:100px;flex-shrink:0;'>{day}</div>
                  <div class='bar-track' style='flex:1;margin:0;'>
                    <div class='bar-fill' style='width:{pct_d}%;background:linear-gradient(90deg,#3b82f6,#06b6d4);'></div>
                  </div>
                  <div style='font-size:12px;color:#f1f5f9;width:50px;text-align:right;'>{count} predictions</div>
                </div>""",unsafe_allow_html=True)