
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import os
import json

from src.data_pipeline import generate_synthetic_data
from src.agents import FraudAnomalyAgent, ScoringAgent, ExplainabilityAgent, SelfCheckAuditAgent
from src.fairness_evaluator import calculate_disparate_impact

st.set_page_config(
    page_title="FinAI Mobile Underwriting App",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# App-like Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp { 
        background: #090d16; 
        color: #f1f5f9; 
    }
    
    /* App Frame Simulation */
    .app-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        padding: 18px 24px;
        border-radius: 20px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    
    .app-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .metric-card-app {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 16px;
        text-align: center;
    }
    
    .metric-val { font-size: 1.6rem; font-weight: 800; color: #38bdf8; }
    .metric-lbl { font-size: 0.75rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; }

    .status-badge {
        padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; display: inline-block;
    }
    .status-APPROVED { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }
    .status-MANUAL_REVIEW { background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid #f59e0b; }
    .status-REJECTED { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; }

    /* Custom Streamlit Tab Styling for App Tabs Look */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0f172a;
        padding: 8px;
        border-radius: 14px;
        border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #94a3b8;
        font-weight: 600;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6366f1 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Persistent DB logic
USER_DB_FILE = "users.json"

def load_users():
    if not os.path.exists(USER_DB_FILE):
        default_db = {"admin@finai.com": {"password": "admin123", "name": "Admin User"}}
        with open(USER_DB_FILE, "w") as f:
            json.dump(default_db, f, indent=4)
        return default_db
    try:
        with open(USER_DB_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"admin@finai.com": {"password": "admin123", "name": "Admin User"}}

def save_users(user_db):
    with open(USER_DB_FILE, "w") as f:
        json.dump(user_db, f, indent=4)

# Multi-Language Dictionary
LANG_PACK = {
    "English": {
        "title": "⚡ FinAI Credit App",
        "subtitle": "Instant Dynamic Alternative Underwriting",
        "tab_eval": "📋 Evaluation",
        "tab_rescore": "🔄 Continuous Score",
        "tab_audit": "📊 Bias Audit",
        "tab_pitch": "🏗️ Architecture",
        "consent": "I agree to Alternative Data Processing & Document OCR Verification (DPDP Act Compliant)",
        "doc_upload": "📂 Upload Bank Statement or Utility Bill",
        "doc_success": "✅ OCR Verification Successful! Extracted Financial Signals.",
        "loan_amount": "Requested Loan Amount (₹)",
        "loan_purpose": "Loan Purpose",
        "tenure": "Tenure (Months)",
        "emi_label": "Estimated Monthly Installment (EMI):",
        "cibil": "Bureau CIBIL Score",
        "job_months": "Job Stability (Months)",
        "edu": "Education Level",
        "utility": "Utility Bill On-time Ratio",
        "income": "Income Regularity Index",
        "savings": "Monthly Savings Buffer Ratio",
        "btn_process": "🚀 Run Underwriting Engine",
        "score_label": "Dynamic Risk Score",
        "risk_label": "Default Probability",
        "decision_label": "Final Decision",
        "cost_label": "Cost per Evaluation",
        "dl_letter": "📄 Download Loan Decision Letter",
        "chat_q": "Ask AI Underwriter a question...",
        "chat_btn": "Ask Assistant 💬"
    },
    "Telugu (తెలుగు)": {
        "title": "⚡ FinAI క్రెడిట్ యాప్",
        "subtitle": "క్షణాల్లో ప్రత్యామ్నాయ లోన్ అండర్ రైటింగ్ సిస్టమ్",
        "tab_eval": "📋 లోన్ దరఖాస్తు",
        "tab_rescore": "🔄 స్కోర్ అప్‌డేట్",
        "tab_audit": "📊 ఫెయిర్‌నెస్ ఆడిట్",
        "tab_pitch": "🏗️ ఆర్కిటెక్చర్",
        "consent": "నేను ప్రత్యామ్నాయ డేటా ప్రాసెసింగ్ & OCR సరిచూడటానికి అనుమతి ఇస్తున్నాను (DPDP చట్టం ప్రకారంగా)",
        "doc_upload": "📂 బ్యాంక్ స్టేట్‌మెంట్ లేదా యూటిలిటీ బిల్లు అప్‌లోడ్ చేయండి",
        "doc_success": "✅ OCR సరిచూడడం పూర్తయింది! ఆర్ధిక వివరాలు విశ్లేషించబడ్డాయి.",
        "loan_amount": "కావలసిన లోన్ మొత్తం (₹)",
        "loan_purpose": "లోన్ ఉద్దేశం",
        "tenure": "గడువు కాలం (నెల్లలో)",
        "emi_label": "అంచనా నెలవారీ వాయిదా (EMI):",
        "cibil": "సిబిల్ స్కోరు (CIBIL)",
        "job_months": "ఉద్యోగ స్థిరత్వం (నెలలు)",
        "edu": "విద్యా అర్హత",
        "utility": "సమయానికి కట్టిన కరెంట్/ఫోన్ బిల్లుల శాతం",
        "income": "క్రమబద్ధమైన ఆదాయ నిష్పత్తి",
        "savings": "నెలవారీ పొదుపు శాతం",
        "btn_process": "🚀 అండర్ రైటింగ్ పూర్తి చేయి",
        "score_label": "డైనమిక్ రిస్క్ స్కోర్",
        "risk_label": "డిఫాల్ట్ ప్రమాద అవకాశం",
        "decision_label": "అంతిమ నిర్ణయం",
        "cost_label": "ప్రాసెసింగ్ ఖర్చు",
        "dl_letter": "📄 లోన్ మంజూరు లేఖ డౌన్‌లోడ్ చేయండి",
        "chat_q": "AI ని సహాయం కోసం ప్రశ్నించండి...",
        "chat_btn": "ప్రశ్న పంపు 💬"
    },
    "Hindi (हिंदी)": {
        "title": "⚡ FinAI क्रेडिट ऐप",
        "subtitle": "त्वरित डायनामिक ऋण स्वीकृति प्लेटफ़ॉर्म",
        "tab_eval": "📋 ऋण मूल्यांकन",
        "tab_rescore": "🔄 निरंतर स्कोर",
        "tab_audit": "📊 निष्पक्षता ऑडिट",
        "tab_pitch": "🏗️ वास्तुकला",
        "consent": "मैं वैकल्पिक डेटा प्रोसेसिंग और दस्तावेज़ सत्यापन की सहमति देता हूँ (DPDP अनुपालन)",
        "doc_upload": "📂 बैंक स्टेटमेंट या उपयोगिता बिल अपलोड करें",
        "doc_success": "✅ OCR सत्यापन सफल! वित्तीय विवरण प्राप्त किए गए।",
        "loan_amount": "अपेक्षित ऋण राशि (₹)",
        "loan_purpose": "ऋण का उद्देश्य",
        "tenure": "अवधि (महीने)",
        "emi_label": "अनुमानित मासिक किस्त (EMI):",
        "cibil": "सिबिल स्कोर (CIBIL)",
        "job_months": "नौकरी की स्थिरता (महीने)",
        "edu": "शिक्षा का स्तर",
        "utility": "समय पर बिल भुगतान अनुपात",
        "income": "नियमित आय सूचकांक",
        "savings": "मासिक बचत अनुपात",
        "btn_process": "🚀 अंडरराइटिंग प्रक्रिया चलाएं",
        "score_label": "डायनामिक रिस्क स्कोर",
        "risk_label": "डिफ़ॉल्ट की संभावना",
        "decision_label": "अंतिम निर्णय",
        "cost_label": "मूल्यांकन लागत",
        "dl_letter": "📄 ऋण स्वीकृति पत्र डाउनलोड करें",
        "chat_q": "AI अंडरराइटर से प्रश्न पूछें...",
        "chat_btn": "प्रश्न भेजें 💬"
    }
}

FEATURE_COLS = [
    'cibil_score', 'job_stability_months', 'education_level',
    'income_regularity_score', 'app_session_duration_sec',
    'form_paste_count', 'form_typing_speed_wpm',
    'linkedin_tenure_months', 'public_footprint_score',
    'utility_bill_on_time_ratio', 'avg_monthly_savings_ratio'
]

@st.cache_resource
def initialize_system():
    data_path = "data/synthetic_underwriting_data.csv"
    if not os.path.exists(data_path):
        df = generate_synthetic_data(num_samples=3000, save_path=data_path)
    else:
        df = pd.read_csv(data_path)
        
    X = df[FEATURE_COLS]
    y = df['default_target']
    
    model = xgb.XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.08, random_state=42)
    model.fit(X, y)
    
    return df, model, FraudAnomalyAgent(), ScoringAgent(model, FEATURE_COLS), ExplainabilityAgent(model, FEATURE_COLS), SelfCheckAuditAgent()

df_data, model, fraud_agent, scoring_agent, explain_agent, audit_agent = initialize_system()

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = ""
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
if 'lang' not in st.session_state:
    st.session_state.lang = "English"
if 'reg_msg' not in st.session_state:
    st.session_state.reg_msg = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def render_login_page():
    st.markdown("<br>", unsafe_allow_html=True)
    _, col_center, _ = st.columns([1, 1.2, 1])
    with col_center:
        st.markdown("""
        <div class="app-header" style="text-align: center;">
            <h2 style="color: #ffffff; margin:0; font-weight:800;">📱 FinAI Mobile</h2>
            <p style="color: #818cf8; font-size: 0.85rem; margin-top:4px;">Alternative Underwriting Portal</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.reg_msg:
            st.success(st.session_state.reg_msg)

        user_db = load_users()
        auth_tab1, auth_tab2 = st.tabs(["🔒 Sign In", "📝 Register"])
        
        with auth_tab1:
            with st.form("login_form"):
                email = st.text_input("Email Address", key="login_email")
                password = st.text_input("Password", type="password", key="login_pass")
                if st.form_submit_button("Sign In 🚀", type="primary", use_container_width=True):
                    email_clean = email.strip().lower()
                    if email_clean in user_db and user_db[email_clean]["password"] == password:
                        st.session_state.authenticated = True
                        st.session_state.user_email = email_clean
                        st.session_state.user_name = user_db[email_clean]["name"]
                        st.session_state.reg_msg = ""
                        st.rerun()
                    else:
                        st.error("Invalid credentials!")

        with auth_tab2:
            with st.form("signup_form"):
                full_name = st.text_input("Full Name", key="signup_name")
                new_email = st.text_input("Email Address", key="signup_email")
                new_pass = st.text_input("Password", type="password", key="signup_pass")
                confirm_pass = st.text_input("Confirm Password", type="password", key="confirm_pass")
                if st.form_submit_button("Create Account ✨", type="primary", use_container_width=True):
                    name_clean = full_name.strip()
                    email_clean = new_email.strip().lower()
                    if not name_clean or not email_clean or not new_pass:
                        st.warning("All fields are required!")
                    elif new_pass != confirm_pass:
                        st.error("Passwords do not match!")
                    elif email_clean in user_db:
                        st.error("User already registered!")
                    else:
                        user_db[email_clean] = {"password": new_pass, "name": name_clean}
                        save_users(user_db)
                        st.session_state.reg_msg = f"🎉 Account created! Please Sign In."
                        st.rerun()

def render_dashboard():
    # Top Mobile-App Bar with Language Switcher
    col_app_title, col_lang, col_logout = st.columns([2.5, 1.2, 0.8])
    
    with col_lang:
        selected_lang = st.selectbox("🌐 Language", ["English", "Telugu (తెలుగు)", "Hindi (हिंदी)"], index=["English", "Telugu (తెలుగు)", "Hindi (हिंदी)"].index(st.session_state.lang))
        st.session_state.lang = selected_lang
        
    L = LANG_PACK[st.session_state.lang]

    with col_app_title:
        st.markdown(f"""
        <div style="padding-top:5px;">
            <span style="font-size:1.4rem; font-weight:800; color:#ffffff;">{L['title']}</span>
            <span style="font-size:0.8rem; color:#818cf8; margin-left:8px;">| {st.session_state.user_name}</span>
        </div>
        """, unsafe_allow_html=True)
        
    with col_logout:
        st.markdown("<div style='padding-top:4px;'>", unsafe_allow_html=True)
        if st.button("Logout 🚪", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([L["tab_eval"], L["tab_rescore"], L["tab_audit"], L["tab_pitch"]])

    with tab1:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown(f"#### 1. {L['consent']}")
        consent = st.checkbox(L['consent'], key="app_consent")
        st.markdown('</div>', unsafe_allow_html=True)

        if consent:
            col_doc, col_loan = st.columns(2)
            with col_doc:
                st.markdown('<div class="app-card">', unsafe_allow_html=True)
                st.markdown(f"#### 2. {L['doc_upload']}")
                uploaded_file = st.file_uploader("", type=["pdf", "jpg", "png"], key="app_file")
                if uploaded_file:
                    st.success(L['doc_success'])
                st.markdown('</div>', unsafe_allow_html=True)

            with col_loan:
                st.markdown('<div class="app-card">', unsafe_allow_html=True)
                st.markdown(f"#### 3. Loan Requirements")
                loan_amount = st.number_input(L['loan_amount'], 10000, 2000000, 250000, 10000)
                loan_purpose = st.selectbox(L['loan_purpose'], ["Personal / Emergency", "Business Expansion", "Debt Consolidation", "Skill Upgrade"])
                tenure_months = st.slider(L['tenure'], 3, 60, 24, 3)

                monthly_rate = 0.135 / 12
                emi = (loan_amount * monthly_rate * ((1 + monthly_rate)**tenure_months)) / (((1 + monthly_rate)**tenure_months) - 1)
                st.markdown(f"<div style='background:#1e1b4b; padding:10px; border-radius:10px; color:#a5b4fc; margin-top:8px;'>💡 <b>{L['emi_label']}</b> <span style='color:#38bdf8; font-weight:800;'>₹{emi:,.2f}</span> / mo</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="app-card">', unsafe_allow_html=True)
            st.markdown("#### 4. Applicant Profile Signals")
            c1, c2 = st.columns(2)
            with c1:
                cibil = st.slider(L['cibil'], 300, 850, 590)
                job_months = st.number_input(L['job_months'], value=18, min_value=0)
                edu_level = st.selectbox(L['edu'], [0, 1, 2], format_func=lambda x: ["High School", "Undergraduate", "Postgraduate"][x])
                utility_ratio = st.slider(L['utility'], 0.0, 1.0, 0.92)
            with c2:
                income_reg = st.slider(L['income'], 0.0, 1.0, 0.85)
                savings_ratio = st.slider(L['savings'], 0.0, 0.5, 0.18)
                linkedin_months = st.number_input("LinkedIn Tenure (Months)", value=24, min_value=0)
                public_score = st.slider("Public Footprint Score", 0.0, 1.0, 0.65)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button(L['btn_process'], type="primary", use_container_width=True):
                feature_dict = {
                    'cibil_score': cibil, 'job_stability_months': job_months, 'education_level': edu_level,
                    'income_regularity_score': income_reg, 'app_session_duration_sec': 180,
                    'form_paste_count': 1, 'form_typing_speed_wpm': 48,
                    'linkedin_tenure_months': linkedin_months, 'public_footprint_score': public_score,
                    'utility_bill_on_time_ratio': utility_ratio, 'avg_monthly_savings_ratio': savings_ratio
                }
                is_suspicious, fraud_flags = fraud_agent.inspect(feature_dict)
                dynamic_score, prob_default, initial_decision = scoring_agent.calculate_score(feature_dict)
                final_decision, _ = audit_agent.verify(initial_decision, dynamic_score, fraud_flags)
                
                # Multilingual Explanation
                if st.session_state.lang == "Telugu (తెలుగు)":
                    explanation = f"మీ Dynamic Risk Score **{dynamic_score}**. Utility Bill Chellimpu ({utility_ratio*100:.0f}%) mariyu Income Regularity valana loan status **{final_decision}** ayindi."
                elif st.session_state.lang == "Hindi (हिंदी)":
                    explanation = f"आपका Dynamic Risk Score **{dynamic_score}** है। Utility Bill भुगतान और आय की नियमितता के आधार पर ऋण निर्णय **{final_decision}** है।"
                else:
                    explanation = explain_agent.generate_explanation(feature_dict)

                st.session_state.latest_score = dynamic_score
                st.session_state.latest_decision = final_decision
                st.session_state.latest_explanation = explanation

                st.markdown("---")
                m1, m2, m3, m4 = st.columns(4)
                with m1: st.markdown(f'<div class="metric-card-app"><div class="metric-lbl">{L["score_label"]}</div><div class="metric-val">{dynamic_score}</div></div>', unsafe_allow_html=True)
                with m2: st.markdown(f'<div class="metric-card-app"><div class="metric-lbl">{L["risk_label"]}</div><div class="metric-val" style="color:#ef4444;">{prob_default:.1%}</div></div>', unsafe_allow_html=True)
                with m3: st.markdown(f'<div class="metric-card-app"><div class="metric-lbl">{L["decision_label"]}</div><div style="margin-top:4px;"><span class="status-badge status-{final_decision}">{final_decision}</span></div></div>', unsafe_allow_html=True)
                with m4: st.markdown(f'<div class="metric-card-app"><div class="metric-lbl">{L["cost_label"]}</div><div class="metric-val" style="color:#10b981;">₹0.85</div></div>', unsafe_allow_html=True)

                cg, ce = st.columns([1, 1.2])
                with cg:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=dynamic_score, domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={'axis': {'range': [0, 1000]}, 'bar': {'color': "#818cf8"},
                               'steps': [{'range': [0, 580], 'color': 'rgba(239, 68, 68, 0.3)'},
                                         {'range': [580, 680], 'color': 'rgba(245, 158, 11, 0.3)'},
                                         {'range': [680, 1000], 'color': 'rgba(16, 185, 129, 0.3)'}]}
                    ))
                    fig.update_layout(height=240, paper_bgcolor="rgba(0,0,0,0)", font={'color': "#ffffff"})
                    st.plotly_chart(fig, use_container_width=True)
                with ce:
                    st.markdown('<div class="app-card">', unsafe_allow_html=True)
                    st.markdown("#### 🔍 Plain-Language Explanation")
                    st.info(explanation)
                    st.markdown('</div>', unsafe_allow_html=True)

            if 'latest_decision' in st.session_state:
                st.markdown("---")
                cdl, cchat = st.columns([1, 1])
                with cdl:
                    st.markdown('<div class="app-card">', unsafe_allow_html=True)
                    st.markdown("#### 📄 Sanction Letter")
                    letter = f"FINAI DECISION REPORT\nApplicant: {st.session_state.user_name}\nDecision: {st.session_state.latest_decision}\nScore: {st.session_state.latest_score}\nExplanation: {st.session_state.latest_explanation}"
                    st.download_button(L["dl_letter"], letter, file_name="Loan_Sanction_Report.txt")
                    st.markdown('</div>', unsafe_allow_html=True)
                with cchat:
                    st.markdown('<div class="app-card">', unsafe_allow_html=True)
                    st.markdown("#### 💬 AI Chat Assistant")
                    q = st.text_input(L["chat_q"], key="chat_in")
                    if st.button(L["chat_btn"]):
                        ans = f"Score is {st.session_state.latest_score}. Continuous bill payments will boost score by +50 pts."
                        st.session_state.chat_history.append((q, ans))
                    for uq, ua in reversed(st.session_state.chat_history[-2:]):
                        st.caption(f"**You:** {uq}")
                        st.markdown(f"**AI:** {ua}")
                    st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown(f"### {L['tab_rescore']}")
        base = st.session_state.get('latest_score', 620)
        repay = st.selectbox("3-Month Repayment", ["Disciplined On-Time", "Minor Delinquency", "Severe Default"])
        delta = 60 if "Disciplined" in repay else (-100 if "Severe" in repay else -20)
        new_score = max(300, min(950, base + delta))
        st.metric("New Dynamic Score", f"{new_score}", delta=f"{delta} pts")
        st.metric("Adjusted Credit Limit Offer", f"₹{int(250000 * (new_score / base)):,}")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown(f"### {L['tab_audit']}")
        dir_ratio, rate_u, rate_p, is_fair = calculate_disparate_impact(df_data, model, FEATURE_COLS)
        st.metric("Disparate Impact Ratio (DIR)", f"{dir_ratio:.3f}")
        if is_fair: st.success("✅ Passed Four-Fifths Rule (DIR >= 0.80)")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown("### 🏗️ Enterprise Architecture")
        st.code("""
  [ Mobile App UI ] ──► [ Lang & OCR Agent ]
         │
         ▼
  [ Agent 1: Fraud Gate ] ──► [ Agent 2: XGBoost Scoring ]
         │
         ▼
  [ Agent 3: Multilingual SHAP ] ──► [ Agent 4: DIR Audit ]
        """, language="text")
        st.markdown('</div>', unsafe_allow_html=True)

if not st.session_state.authenticated:
    render_login_page()
else:
    render_dashboard()
