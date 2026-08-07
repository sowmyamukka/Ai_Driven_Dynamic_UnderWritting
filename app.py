
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
    page_title="FinAI Mobile Credit App",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Glassmorphic Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp { 
        background: radial-gradient(circle at top right, #1e1b4b 0%, #090d16 50%, #030712 100%);
        color: #f8fafc;
    }
    
    .app-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.3);
    }

    .doc-checklist {
        background: rgba(15, 23, 42, 0.6);
        border: 1px dashed rgba(99, 102, 241, 0.4);
        border-radius: 12px;
        padding: 12px 16px;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .doc-checklist ul {
        margin: 0;
        padding-left: 20px;
        color: #cbd5e1;
        font-size: 0.85rem;
    }

    .profile-pill {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid #6366f1;
        color: #a5b4fc;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 12px;
    }

    .metric-card-app {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 18px 14px;
        text-align: center;
    }
    
    .metric-val { 
        font-size: 1.8rem; 
        font-weight: 800; 
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-lbl { 
        font-size: 0.75rem; 
        color: #94a3b8; 
        font-weight: 700; 
        text-transform: uppercase;
    }

    .status-badge {
        padding: 8px 18px; 
        border-radius: 30px; 
        font-weight: 800; 
        font-size: 0.85rem; 
        display: inline-block;
    }
    .status-APPROVED { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid #10b981; }
    .status-MANUAL_REVIEW { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid #f59e0b; }
    .status-REJECTED { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid #ef4444; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(15, 23, 42, 0.8);
        padding: 8px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        color: #94a3b8;
        font-weight: 700;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
        color: #ffffff !important;
    }

    .info-box-app {
        background: rgba(56, 189, 248, 0.1);
        border-left: 4px solid #38bdf8;
        padding: 14px 18px;
        border-radius: 12px;
        color: #e0f2fe;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

USER_DB_FILE = "users.json"

def load_users():
    if not os.path.exists(USER_DB_FILE):
        default_db = {"admin@finai.com": {"password": "admin123", "name": "Admin Underwriter"}}
        with open(USER_DB_FILE, "w") as f:
            json.dump(default_db, f, indent=4)
        return default_db
    try:
        with open(USER_DB_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"admin@finai.com": {"password": "admin123", "name": "Admin Underwriter"}}

def save_users(user_db):
    with open(USER_DB_FILE, "w") as f:
        json.dump(user_db, f, indent=4)

LANG_PACK = {
    "English": {
        "title": "⚡ FinAI Credit App",
        "tab_eval": "📋 Loan Application",
        "tab_rescore": "🔄 Dynamic Re-Score",
        "tab_audit": "📊 Bias Governance",
        "tab_pitch": "🏗️ Architecture",
        "consent": "I give explicit consent for Alternative Data Processing & OCR Verification (DPDP Act Compliant)",
        "doc_upload": "📂 Alternative Document Intake",
        "doc_success": "✅ OCR Agent Success! Alternative Financial Signals Verified.",
        "loan_amount": "Requested Loan Amount (₹)",
        "loan_purpose": "Loan Purpose",
        "tenure": "Tenure (Months)",
        "emi_label": "Estimated Monthly Installment (EMI):",
        "cibil": "Bureau CIBIL Score (Baseline)",
        "job_months": "Job Stability (Months)",
        "edu": "Education Level",
        "utility": "Utility Bill Payment On-time Ratio",
        "income": "Income Regularity Index",
        "savings": "Monthly Savings Buffer Ratio",
        "btn_process": "🚀 Process Underwriting Engine",
        "score_label": "Dynamic Risk Score",
        "risk_label": "Default Probability",
        "decision_label": "Final Decision",
        "cost_label": "Cost per Decision",
        "dl_letter": "📄 Download Sanction Letter"
    },
    "Telugu (తెలుగు)": {
        "title": "⚡ FinAI క్రెడిట్ యాప్",
        "tab_eval": "📋 లోన్ దరఖాస్తు",
        "tab_rescore": "🔄 స్కోర్ అప్‌డేట్",
        "tab_audit": "📊 ఫెయిర్‌నెస్ ఆడిట్",
        "tab_pitch": "🏗️ ఆర్కిటెక్చర్",
        "consent": "నేను ప్రత్యామ్నాయ డేటా ప్రాసెసింగ్ & OCR సరిచూడడానికి అనుమతి ఇస్తున్నాను (DPDP చట్టం ప్రకారంగా)",
        "doc_upload": "📂 ప్రత్యామ్నాయ పత్రాల సరిచూడడం",
        "doc_success": "✅ OCR సరిచూడడం పూర్తయింది! వివరాలు విశ్లేషించబడ్డాయి.",
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
        "dl_letter": "📄 లోన్ మంజూరు లేఖ డౌన్‌లోడ్ చేయండి"
    },
    "Hindi (हिंदी)": {
        "title": "⚡ FinAI क्रेडिट ऐप",
        "tab_eval": "📋 ऋण मूल्यांकन",
        "tab_rescore": "🔄 निरंतर स्कोर",
        "tab_audit": "📊 निष्पक्षता ऑडिट",
        "tab_pitch": "🏗️ वास्तुकला",
        "consent": "मैं वैकल्पिक डेटा प्रोसेसिंग और दस्तावेज़ सत्यापन की सहमति देता हूँ (DPDP अनुपालन)",
        "doc_upload": "📂 वैकल्पिक दस्तावेज़ सत्यापन",
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
        "dl_letter": "📄 ऋण स्वीकृति पत्र डाउनलोड करें"
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

def render_login_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col_center, _ = st.columns([1, 1.2, 1])
    with col_center:
        st.markdown("""
        <div class="app-card" style="text-align: center;">
            <h2 style="margin:0; font-weight:800; background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">📱 FinAI Credit Portal</h2>
            <p style="color: #94a3b8; font-size: 0.85rem; margin-top:6px;">Next-Gen Mobile Underwriting Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        user_db = load_users()
        auth_tab1, auth_tab2 = st.tabs(["🔒 Sign In", "📝 Register Account"])
        
        with auth_tab1:
            with st.form("login_form"):
                email = st.text_input("Work Email Address", key="login_email")
                password = st.text_input("Password", type="password", key="login_pass")
                if st.form_submit_button("Sign In 🚀", type="primary", use_container_width=True):
                    email_clean = email.strip().lower()
                    if email_clean in user_db and user_db[email_clean]["password"] == password:
                        st.session_state.authenticated = True
                        st.session_state.user_email = email_clean
                        st.session_state.user_name = user_db[email_clean]["name"]
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")

        with auth_tab2:
            with st.form("signup_form"):
                full_name = st.text_input("Full Name", key="signup_name")
                new_email = st.text_input("Work Email Address", key="signup_email")
                new_pass = st.text_input("Create Password", type="password", key="signup_pass")
                if st.form_submit_button("Create Account ✨", type="primary", use_container_width=True):
                    user_db[new_email.strip().lower()] = {"password": new_pass, "name": full_name.strip()}
                    save_users(user_db)
                    st.success("🎉 Account created! Please sign in.")

def render_dashboard():
    c_title, c_lang, c_logout = st.columns([2.5, 1.2, 0.8])
    with c_lang:
        selected_lang = st.selectbox("🌐 Language", ["English", "Telugu (తెలుగు)", "Hindi (हिंदी)"], index=["English", "Telugu (తెలుగు)", "Hindi (हिंदी)"].index(st.session_state.lang))
        st.session_state.lang = selected_lang
        
    L = LANG_PACK[st.session_state.lang]

    with c_title:
        st.markdown(f"<span style='font-size:1.5rem; font-weight:800;'>{L['title']}</span> | User: <b>{st.session_state.user_name}</b>", unsafe_allow_html=True)
        
    with c_logout:
        if st.button("Logout 🚪", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs([L["tab_eval"], L["tab_rescore"], L["tab_audit"], L["tab_pitch"]])

    with tab1:
        # Section 1: Consent
        st.markdown(f"#### 1. Consent & Verification")
        consent = st.checkbox(L['consent'], key="app_consent")
        
        if consent:
            st.markdown("---")
            
            # Section 2 & 3: Columns layout without extra empty cards
            c_doc, c_loan = st.columns(2)
            
            with c_doc:
                st.markdown(f"#### 2. {L['doc_upload']}")
                
                applicant_type = st.selectbox("Select Applicant Category:", [
                    "🎓 Student Profile",
                    "🚲 Gig Economy Worker (Swiggy/Zomato/Uber)",
                    "🏪 Small Business / Vendor",
                    "💼 Salaried / Standard Applicant"
                ])
                
                if "Student" in applicant_type:
                    st.markdown("""
                    <div class="doc-checklist">
                        <b style="color:#a5b4fc;">📄 Required Documents for Students:</b>
                        <ul>
                            <li>College Student ID Card / Bonafide Certificate</li>
                            <li>Semester Marks Memo or Admission Fee Receipt</li>
                            <li>Parent / Co-applicant Aadhaar / Income Proof</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                elif "Gig Economy" in applicant_type:
                    st.markdown("""
                    <div class="doc-checklist">
                        <b style="color:#a5b4fc;">📄 Required Documents for Gig Workers:</b>
                        <ul>
                            <li>Gig App Dashboard Screenshot (Swiggy/Zomato/Uber)</li>
                            <li>3-Month Bank/UPI Earnings Statement</li>
                            <li>Recent Electricity / Mobile Bill</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                elif "Small Business" in applicant_type:
                    st.markdown("""
                    <div class="doc-checklist">
                        <b style="color:#a5b4fc;">📄 Required Documents for Small Business:</b>
                        <ul>
                            <li>Shop Establishment / Trade / GST License</li>
                            <li>UPI Merchant Daily QR Transaction Statement</li>
                            <li>Utility Bill (Shop / Residence)</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="doc-checklist">
                        <b style="color:#a5b4fc;">📄 Required Documents for Salaried:</b>
                        <ul>
                            <li>Latest 3-Month Bank Salary Statement</li>
                            <li>Company ID or Employment Offer Letter</li>
                            <li>Recent Utility Bill</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                uploaded_file = st.file_uploader("Upload Verification Document", type=["pdf", "jpg", "png"], key="app_file")
                if uploaded_file:
                    st.success(L['doc_success'])

            with c_loan:
                st.markdown(f"#### 3. Loan Purpose & Amount")
                loan_amount = st.number_input(L['loan_amount'], 5000, 2000000, 150000, 5000)
                
                loan_purpose = st.selectbox(L['loan_purpose'], [
                    "Personal / Emergency", 
                    "Student / Education Loan", 
                    "Business Tool / Asset Purchase", 
                    "Skill Upgrade / Certification", 
                    "Other / Custom Reason"
                ])
                
                custom_reason = ""
                if loan_purpose == "Other / Custom Reason":
                    custom_reason = st.text_area("📝 Please specify your loan purpose reason in detail:", placeholder="e.g., Laptop purchase for freelancing, Shop inventory, Agriculture tools...")
                
                tenure_months = st.slider(L['tenure'], 3, 60, 12, 3)
                monthly_rate = 0.12 / 12
                emi = (loan_amount * monthly_rate * ((1 + monthly_rate)**tenure_months)) / (((1 + monthly_rate)**tenure_months) - 1)
                st.markdown(f"<div class='info-box-app'>💡 <b>{L['emi_label']}</b> <span style='color:#38bdf8; font-weight:800;'>₹{emi:,.2f}</span> / mo</div>", unsafe_allow_html=True)

            st.markdown("---")

            # Section 4: Applicant Dynamic Profile Signals
            if "Student" in applicant_type or loan_purpose == "Student / Education Loan":
                st.markdown('<span class="profile-pill">🎓 Student Alternative Profile Activated</span>', unsafe_allow_html=True)
                st.markdown("#### 4. Student & Academic Signals (Thin-File Flow)")
                
                cs1, cs2 = st.columns(2)
                with cs1:
                    college_tier = st.selectbox("College / Institution Tier", ["Tier 1 (IIT/NIT/BITS/IIM)", "Tier 2 (State Govt/Recognized)", "Tier 3 / Local College"])
                    cgpa = st.slider("Current Academic GPA / Percentage", 5.0, 10.0, 8.2, 0.1)
                    utility_ratio = st.slider(L['utility'], 0.0, 1.0, 0.95)
                with cs2:
                    has_guarantor = st.radio("Co-applicant / Parent Income Support?", ["Yes", "No"])
                    skill_certs = st.selectbox("Online Certifications (Coursera/Udemy/NPTEL)", ["2+ Active Certifications", "1 Certification", "None"])
                    savings_ratio = st.slider("Pocket Savings Buffer Ratio", 0.0, 0.5, 0.15)
                
                cibil = 610
                job_months = 0
                edu_level = 1
                income_reg = 0.70 if has_guarantor == "Yes" else 0.40
                public_score = 0.75 if cgpa > 8.0 else 0.50
                linkedin_months = 12

            elif "Gig" in applicant_type:
                st.markdown('<span class="profile-pill">🚲 Gig Economy Cash-Flow Profile</span>', unsafe_allow_html=True)
                st.markdown("#### 4. Gig Worker Cash-Flow Signals")
                cg1, cg2 = st.columns(2)
                with cg1:
                    cibil = st.slider(L['cibil'], 300, 850, 580)
                    job_months = st.number_input("Gig Platform Tenure (Months)", value=14, min_value=0)
                    utility_ratio = st.slider(L['utility'], 0.0, 1.0, 0.92)
                with cg2:
                    income_reg = st.slider("Weekly Earnings Regularity", 0.0, 1.0, 0.85)
                    savings_ratio = st.slider("UPI Daily Savings Buffer", 0.0, 0.5, 0.20)
                    public_score = 0.65
                    edu_level = 1
                    linkedin_months = 6

            else:
                st.markdown('<span class="profile-pill">💼 Standard Alternative Profile</span>', unsafe_allow_html=True)
                st.markdown("#### 4. Applicant Profile Signals")
                c1, c2 = st.columns(2)
                with c1:
                    cibil = st.slider(L['cibil'], 300, 850, 610)
                    job_months = st.number_input(L['job_months'], value=12, min_value=0)
                    edu_level = st.selectbox(L['edu'], [0, 1, 2], format_func=lambda x: ["High School", "Undergraduate", "Postgraduate"][x])
                    utility_ratio = st.slider(L['utility'], 0.0, 1.0, 0.88)
                with c2:
                    income_reg = st.slider(L['income'], 0.0, 1.0, 0.80)
                    savings_ratio = st.slider(L['savings'], 0.0, 0.5, 0.15)
                    linkedin_months = st.number_input("LinkedIn Tenure (Months)", value=18, min_value=0)
                    public_score = st.slider("Public Footprint Score", 0.0, 1.0, 0.60)

            st.markdown("<br>", unsafe_allow_html=True)

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
                
                if "Student" in applicant_type or loan_purpose == "Student / Education Loan":
                    dynamic_score = min(900, dynamic_score + 65)
                    if initial_decision == "REJECTED":
                        initial_decision = "MANUAL_REVIEW"
                elif "Gig" in applicant_type:
                    dynamic_score = min(900, dynamic_score + 40)

                final_decision, _ = audit_agent.verify(initial_decision, dynamic_score, fraud_flags)
                
                if "Student" in applicant_type or loan_purpose == "Student / Education Loan":
                    explanation = f"Student Academic Index & Utility Payment Consistency ({utility_ratio*100:.0f}%) applied. Routed to Education Micro-Credit Track with Score **{dynamic_score}**."
                elif "Gig" in applicant_type:
                    explanation = f"Gig Weekly Earnings Regularity ({income_reg*100:.0f}%) & UPI Cash-flow verified. Approved for Micro-Credit Line with Score **{dynamic_score}**."
                else:
                    explanation = explain_agent.generate_explanation(feature_dict)

                st.session_state.latest_score = dynamic_score
                st.session_state.latest_decision = final_decision
                st.session_state.latest_explanation = explanation

                st.markdown("---")
                
                m1, m2, m3, m4 = st.columns(4)
                with m1: 
                    st.markdown(f'<div class="metric-card-app"><div class="metric-lbl">{L["score_label"]}</div><div class="metric-val">{dynamic_score}</div></div>', unsafe_allow_html=True)
                with m2: 
                    st.markdown(f'<div class="metric-card-app"><div class="metric-lbl">{L["risk_label"]}</div><div class="metric-val" style="color:#ef4444;">{prob_default:.1%}</div></div>', unsafe_allow_html=True)
                with m3: 
                    st.markdown(f'<div class="metric-card-app"><div class="metric-lbl">{L["decision_label"]}</div><div style="margin-top:6px;"><span class="status-badge status-{final_decision}">{final_decision}</span></div></div>', unsafe_allow_html=True)
                with m4: 
                    st.markdown(f'<div class="metric-card-app"><div class="metric-lbl">{L["cost_label"]}</div><div class="metric-val" style="color:#10b981;">₹0.85</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                cg, ce = st.columns([1, 1.2])
                with cg:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=dynamic_score, domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={'axis': {'range': [0, 1000]}, 'bar': {'color': "#818cf8"},
                               'steps': [{'range': [0, 580], 'color': 'rgba(239, 68, 68, 0.3)'},
                                         {'range': [580, 680], 'color': 'rgba(245, 158, 11, 0.3)'},
                                         {'range': [680, 1000], 'color': 'rgba(16, 185, 129, 0.3)'}]}
                    ))
                    fig.update_layout(height=240, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#ffffff"})
                    st.plotly_chart(fig, use_container_width=True)

                with ce:
                    st.markdown("#### 🔍 AI Underwriting Rationale")
                    st.info(explanation)
                    if custom_reason:
                        st.caption(f"📌 **Custom Reason Logged:** {custom_reason}")

    with tab2:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown(f"### {L['tab_rescore']}")
        base = st.session_state.get('latest_score', 630)
        repay = st.selectbox("3-Month Repayment Record", ["Disciplined On-Time", "Minor Delinquency", "Severe Default"])
        delta = 60 if "Disciplined" in repay else (-100 if "Severe" in repay else -20)
        new_score = max(300, min(950, base + delta))
        st.metric("New Dynamic Score", f"{new_score}", delta=f"{delta} pts")
        st.metric("Adjusted Credit Limit Offer", f"₹{int(150000 * (new_score / base)):,}")
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
  [ Mobile UI Intake ] ──► [ Applicant Dynamic Guidance Router ]
         │
         ▼
  [ Agent 1: Fraud Gate ] ──► [ Agent 2: XGBoost Potential Engine ]
         │
         ▼
  [ Agent 3: Multilingual SHAP ] ──► [ Agent 4: DIR Audit & Step-Up Ladder ]
        """, language="text")
        st.markdown('</div>', unsafe_allow_html=True)

if not st.session_state.authenticated:
    render_login_page()
else:
    render_dashboard()
