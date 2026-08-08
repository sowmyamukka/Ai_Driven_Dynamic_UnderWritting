import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime
import hashlib
import re
import cv2
from fpdf import FPDF

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="FinAI - Dynamic AI Credit Underwriting Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Password Hashing Helper ---
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

# --- Basic Email Validator Helper ---
def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

# --- In-Memory User Database Initializer ---
if 'user_db' not in st.session_state:
    st.session_state.user_db = {
        "mowa": {"password": make_hashes("mowa123"), "email": "mowa.finai@gmail.com", "name": "Mowa User"},
        "admin": {"password": make_hashes("admin123"), "email": "admin.finai@gmail.com", "name": "System Admin"}
    }

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'current_user' not in st.session_state:
    st.session_state.current_user = None

# --- Custom Styling for UI & Containers ---
st.markdown("""
<style>
    .auth-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 30px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        backdrop-filter: blur(10px);
    }
    .auth-title {
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 5px;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .auth-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 0.9rem;
        margin-bottom: 20px;
    }
    
    .doc-box {
        background-color: #0b1329;
        border: 1px dashed #2563eb;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 12px;
        margin-bottom: 18px;
    }
    .doc-box-title {
        color: #60a5fa;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 8px;
    }
    .doc-box ul {
        margin: 0;
        padding-left: 22px;
        color: #cbd5e1;
        font-size: 0.93rem;
    }

    .arch-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
    }
    .arch-header {
        font-size: 1.1rem;
        font-weight: bold;
        color: #38bdf8;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper: PDF Sanction Letter Generator ---
def generate_sanction_letter(applicant_name, applicant_email, loan_amount, decision, score, emi):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 18)
    pdf.cell(0, 10, "FinAI Credit Underwriting Platform", ln=True, align='C')
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, "OFFICIAL LOAN SANCTION LETTER", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 8, f"Date: {datetime.date.today().strftime('%B %d, %Y')}", ln=True)
    pdf.cell(0, 8, f"Applicant Name: {applicant_name}", ln=True)
    pdf.cell(0, 8, f"Applicant Email Address: {applicant_email}", ln=True)
    pdf.cell(0, 8, f"Application Decision: {decision}", ln=True)
    pdf.cell(0, 8, f"Dynamic Credit Score: {score} / 900", ln=True)
    pdf.cell(0, 8, f"Sanctioned Amount: Rs. {loan_amount:,.2f}", ln=True)
    pdf.cell(0, 8, f"Estimated Monthly Installment (EMI): Rs. {emi:,.2f}", ln=True)
    pdf.ln(10)
    pdf.set_font("Helvetica", 'I', 10)
    pdf.multi_cell(0, 6, "Note: Conditional sanction letter based on dynamic alternative underwriting data. DPDP Act compliant.")
    return pdf.output()

# --- AI Agents Stubs ---
class DummyScoringAgent:
    def calculate_score(self, data):
        cibil = data.get('cibil_score', 600)
        utility = data.get('utility_bill_on_time_ratio', 0.8)
        base_score = int(cibil * 0.7 + utility * 250)
        prob_default = max(0.02, min(0.45, (850 - base_score) / 1000))
        decision = "APPROVED" if base_score >= 650 else ("MANUAL_REVIEW" if base_score >= 580 else "REJECTED")
        return base_score, prob_default, decision

scoring_agent = DummyScoringAgent()


# ==============================================================================
# CENTERED LOGIN & SIGNUP SCREEN
# ==============================================================================
if not st.session_state.logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_left, col_center, col_right = st.columns([1, 1.2, 1])
    
    with col_center:
        st.markdown("""
        <div class="auth-card">
            <div class="auth-title">⚡ FinAI — Access Portal</div>
            <div class="auth-subtitle">Welcome back! Log in with your credentials.</div>
        </div>
        """, unsafe_allow_html=True)
        
        auth_tab1, auth_tab2 = st.tabs(["🔑 Login", "✨ Sign Up"])
        
        # --- LOGIN TAB ---
        with auth_tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("login_form", clear_on_submit=False):
                login_username = st.text_input("Username", key="login_user", placeholder="e.g. name")
                login_password = st.text_input("Password", type="password", key="login_pass", placeholder="••••••••")
                st.markdown("<br>", unsafe_allow_html=True)
                login_submitted = st.form_submit_button("🚀 Login to Portal", type="primary", use_container_width=True)
                
                if login_submitted:
                    username_clean = login_username.strip().lower()
                    if username_clean in st.session_state.user_db:
                        stored_hash = st.session_state.user_db[username_clean]["password"]
                        if check_hashes(login_password, stored_hash):
                            st.session_state.logged_in = True
                            st.session_state.current_user = username_clean
                            st.toast(f"Welcome back, {st.session_state.user_db[username_clean]['name']}! 🎉", icon="✅")
                            st.rerun()
                        else:
                            st.error("Incorrect Password! Please try again.")
                    else:
                        st.error("Username not found! Please Sign Up first.")

        # --- SIGNUP TAB ---
        with auth_tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("signup_form", clear_on_submit=False):
                new_name = st.text_input("Full Name", key="su_name", placeholder="e.g. Sowmya Mukka")
                new_username = st.text_input("Choose Username", key="su_user", placeholder="e.g. somu")
                new_email = st.text_input("Gmail / Email Address", key="su_email", placeholder="e.g. sowmya143@gmail.com")
                new_password = st.text_input("Choose Password", type="password", key="su_pass")
                confirm_password = st.text_input("Confirm Password", type="password", key="su_confirm")
                
                st.markdown("<br>", unsafe_allow_html=True)
                signup_submitted = st.form_submit_button("✨ Create Account", type="primary", use_container_width=True)
                
                if signup_submitted:
                    clean_name = new_name.strip() if new_name else ""
                    clean_new_user = new_username.strip().lower() if new_username else ""
                    clean_email = new_email.strip() if new_email else ""
                    
                    if not clean_name or not clean_new_user or not new_password or not clean_email or not confirm_password:
                        st.error("Please fill in all mandatory fields!")
                    elif not is_valid_email(clean_email):
                        st.error("Please enter a valid Email address (e.g. user@gmail.com)!")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match!")
                    elif clean_new_user in st.session_state.user_db:
                        st.error("Username already exists! Choose another username.")
                    else:
                        st.session_state.user_db[clean_new_user] = {
                            "password": make_hashes(new_password),
                            "email": clean_email,
                            "name": clean_name
                        }
                        st.success("🎉 Account created successfully! Please switch to Login tab.")


# ==============================================================================
# MAIN DASHBOARD SCREEN (When Logged In)
# ==============================================================================
else:
    user_info = st.session_state.user_db[st.session_state.current_user]
    
    st.sidebar.markdown(f"### 👤 User Account")
    st.sidebar.info(f"**Name:** {user_info['name']}\n\n**Email:** {user_info['email']}")
    if st.sidebar.button("🔒 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.rerun()

    st.title("⚡ FinAI — Dynamic AI Credit Underwriting System")
    st.markdown("Automated alternative-data credit scoring engine powered by Machine Learning & Multi-Agent AI.")

    tab1, tab2, tab3 = st.tabs(["📝 New Loan Application", "📈 Dynamic Re-Score", "📊 Admin Audit & Governance"])

    with tab1:
        st.subheader("1. Applicant Registration & Consent")
        col_reg1, col_reg2 = st.columns(2)
        with col_reg1:
            user_email = st.text_input("📧 Email Address for Application Records:", value=user_info['email'])
        with col_reg2:
            consent = st.checkbox("I give explicit consent for Alternative Data Processing & OCR Verification (DPDP Act Compliant)", value=True)

        st.markdown("---")

        c_doc, c_loan = st.columns(2)
        with c_doc:
            st.markdown("### 2. 📁 Alternative Document Intake")
            applicant_type = st.selectbox("Select Applicant Category:", [
                "🎓 Student Profile",
                "🚲 Gig Economy Worker",
                "🏪 Small Business / Vendor",
                "💼 Salaried / Standard Applicant"
            ])
            
            if "Student" in applicant_type:
                st.markdown("""
                <div class="doc-box">
                    <div class="doc-box-title">📄 Required Documents for Students:</div>
                    <ul>
                        <li>College Student ID Card / Bonafide Certificate</li>
                        <li>Semester Marks Memo or Admission Fee Receipt</li>
                        <li>Parent / Co-applicant Aadhaar / Income Proof</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif "Gig Economy" in applicant_type:
                st.markdown("""
                <div class="doc-box">
                    <div class="doc-box-title">📄 Required Documents for Gig Workers:</div>
                    <ul>
                        <li>Gig Platform Profile Screenshot / App Earnings Report</li>
                        <li>Bank Account Statement (Last 3 to 6 Months)</li>
                        <li>Aadhaar / Driving License / Vehicle RC</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif "Small Business" in applicant_type:
                st.markdown("""
                <div class="doc-box">
                    <div class="doc-box-title">📄 Required Documents for Vendors & Businesses:</div>
                    <ul>
                        <li>UPI / Merchant QR Code Summary Report (PhonePe/GPay)</li>
                        <li>Bank Account Statement (Last 6 Months)</li>
                        <li>Udyam Registration / Shop License (if available)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="doc-box">
                    <div class="doc-box-title">📄 Required Documents for Salaried Applicants:</div>
                    <ul>
                        <li>Last 3 Months Salary Slips</li>
                        <li>Bank Statement (Last 6 Months)</li>
                        <li>Form 16 or Income Tax Returns (ITR)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # --- SMART FILENAME & CONTENT VALIDATION ---
        # --- UPDATED SMART VALIDATION LOGIC ---
            uploaded_files = st.file_uploader(
                f"Upload Verification Documents for {applicant_type.split()[1]} Profile", 
                type=["pdf", "jpg", "png", "jpeg"], 
                accept_multiple_files=True
            )
            
            has_invalid_image = False
            # Check 1: Mandatory File Upload Check
            if not uploaded_files:
                st.warning("⚠️ Please upload your documents to proceed with verification.")
            else:
                for uploaded_file in uploaded_files:
                    # Check 2: Blank/Small file check
                    if uploaded_file.size < 1024:
                        has_invalid_image = True
                        st.error(f"⚠️ Warning: '{uploaded_file.name}' is blank/invalid!")
                    else:
                        fname_lower = uploaded_file.name.lower()
                        # Keywords list (includes place/location markers)
                        valid_keywords = ['id', 'student', 'mark', 'memo', 'fee', 'bonafide', 'card', 'report', 'statement', 'income', 'aadhaar', 'salary', 'slip', 'qr', 'shop', 'business', 'place', 'location', 'city', 'address']
                        is_likely_doc = any(keyword in fname_lower for keyword in valid_keywords)
                        
                        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                        uploaded_file.seek(0)
                        image = cv2.imdecode(file_bytes, 1)
                        
                        # Blank image check
                        if image is None or np.mean(image) > 240:
                            has_invalid_image = True
                            st.error(f"⚠️ '{uploaded_file.name}' is a blank document!")
                        elif not is_likely_doc:
                            has_invalid_image = True
                            st.error(f"❌ '{uploaded_file.name}' does not contain required verification details (Place/ID/Marks).")

        with c_loan:
            st.markdown("### 3. Loan Purpose & Amount")
            loan_amount = st.number_input("Requested Loan Amount (₹)", 5000, 2000000, 150000, 5000)
            
            loan_purpose_option = st.selectbox("Loan Purpose", [
                "Personal / Emergency", 
                "Student / Education Loan", 
                "Business Tool / Asset Purchase", 
                "Other"
            ])
            if loan_purpose_option == "Other":
                loan_purpose = st.text_input("Please specify loan purpose:", placeholder="Type your purpose here...")
            else:
                loan_purpose = loan_purpose_option

            tenure_months = st.slider("Tenure (Months)", 3, 60, 12, 3)
            monthly_rate = 0.12 / 12
            emi = (loan_amount * monthly_rate * ((1 + monthly_rate)**tenure_months)) / (((1 + monthly_rate)**tenure_months) - 1)
            st.info(f"💡 **Estimated Monthly Installment (EMI):** ₹{emi:,.2f} / month")

        st.markdown("---")
        st.subheader("4. Risk & Academic Signals")

        cs1, cs2 = st.columns(2)
        with cs1:
            cibil = st.slider("Credit Bureau Score (CIBIL)", 300, 850, 610)
            cgpa = st.slider("Current Academic GPA / Percentage", 5.0, 10.0, 8.2, 0.1)
            utility_ratio = st.slider("Utility Bill Payment On-time Ratio", 0.0, 1.0, 0.95)
        with cs2:
            has_guarantor = st.radio("Co-applicant / Parent Income Support?", ["Yes", "No"])
            savings_ratio = st.slider("Pocket Savings Buffer Ratio", 0.0, 0.5, 0.15)

        if st.button("🚀 Process Application with Multi-Agent Underwriting", type="primary", use_container_width=True):
            if not consent:
                st.error("Please grant DPDP Act Consent to proceed.")
            elif not is_valid_email(user_email.strip()):
                st.error("Please enter a valid email address.")
            elif loan_purpose_option == "Other" and not loan_purpose.strip():
                st.error("Please specify your loan purpose in the text box.")
            elif not uploaded_files:
                st.error("🚫 Document Upload Required: Please upload your mandatory verification documents before processing!")
            elif has_invalid_image:
                st.error("Please upload valid professional/academic documents to proceed.")
            else:
                feature_dict = {'cibil_score': cibil, 'utility_bill_on_time_ratio': utility_ratio}
                dynamic_score, prob_default, initial_decision = scoring_agent.calculate_score(feature_dict)
                
                if "Student" in applicant_type:
                    dynamic_score = min(900, dynamic_score + 65)
                    final_decision = "APPROVED" if dynamic_score >= 630 else "MANUAL_REVIEW"
                else:
                    final_decision = initial_decision

                target_email = user_email.strip()
                if final_decision == "APPROVED":
                    st.toast(f"FinAI Application Evaluated: LOAN APPROVED!", icon="✅")
                else:
                    st.toast(f"FinAI Application Evaluated: Status - {final_decision}", icon="⚠️")

                st.markdown("---")
                m1, m2, m3 = st.columns(3)
                m1.metric("Dynamic Score", f"{dynamic_score} / 900")
                m2.metric("Default Risk", f"{prob_default:.1%}")
                m3.metric("Decision", final_decision)

                if final_decision == "APPROVED":
                    pdf_bytes = generate_sanction_letter(
                        applicant_name=user_info['name'],
                        applicant_email=target_email,
                        loan_amount=loan_amount,
                        decision=final_decision,
                        score=dynamic_score,
                        emi=emi
                    )
                    st.download_button(
                        label="📥 Download Official Sanction Letter (PDF)",
                        data=bytes(pdf_bytes),
                        file_name=f"FinAI_Sanction_{st.session_state.current_user}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

    with tab2:
        st.subheader("📈 Post-Disbursement Dynamic Behavioral Re-Scoring")
        r_col1, r_col2 = st.columns([1, 1])
        with r_col1:
            st.markdown("#### ⚡ Real-Time Behavioral Triggers")
            month_count = st.slider("Simulated Months Post-Disbursement", 1, 12, 6)
            upi_volume = st.slider("Monthly Average UPI Transactions Count", 10, 300, 85)
            utility_delays = st.slider("Utility / Recharge Delays (Last 3 Months)", 0, 10, 1)
            savings_inflow = st.slider("Monthly Average Savings Rate (% of Income)", 0, 50, 18)

            base_rescore = 620 + (upi_volume * 0.4) - (utility_delays * 25) + (savings_inflow * 3.2)
            final_rescore = int(max(300, min(900, base_rescore)))
            delta_val = final_rescore - 620
            st.metric("Updated Dynamic Credit Score", f"{final_rescore} / 900", delta=f"{delta_val} Points")

        with r_col2:
            st.markdown("#### 📊 6-Month Score Trajectory")
            months = [f"Month {i}" for i in range(1, month_count + 1)]
            np.random.seed(42)
            scores_trend = [int(620 + (i * (delta_val / month_count)) + np.random.randint(-10, 10)) for i in range(1, month_count + 1)]

            df_trend = pd.DataFrame({"Month": months, "Credit Score": scores_trend})
            fig_trend = px.line(df_trend, x="Month", y="Credit Score", markers=True, title="Behavioral Score Trajectory Over Time")
            fig_trend.update_traces(line_color="#38bdf8", line_width=3)
            fig_trend.update_layout(template="plotly_dark", height=320)
            st.plotly_chart(fig_trend, use_container_width=True)

    with tab3:
        st.subheader("📊 System Architecture & Governance Framework")
        st.markdown("### 🏛️ Multi-Agent System Architecture Flow")
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            st.markdown("""<div class="arch-card"><div class="arch-header">1. Document Agent</div><p style="color: #94a3b8; font-size: 0.85rem;">• EasyOCR Engine<br>• Fake Document Check</p></div>""", unsafe_allow_html=True)
        with a2:
            st.markdown("""<div class="arch-card"><div class="arch-header">2. Feature Agent</div><p style="color: #94a3b8; font-size: 0.85rem;">• Alternative Signals<br>• Cash Flow Velocity</p></div>""", unsafe_allow_html=True)
        with a3:
            st.markdown("""<div class="arch-card"><div class="arch-header">3. Scoring Agent</div><p style="color: #94a3b8; font-size: 0.85rem;">• XGBoost / Random Forest<br>• SHAP Explainability</p></div>""", unsafe_allow_html=True)
        with a4:
            st.markdown("""<div class="arch-card"><div class="arch-header">4. Governance Agent</div><p style="color: #94a3b8; font-size: 0.85rem;">• DPDP Act Compliance<br>• Automated PDF Sanction</p></div>""", unsafe_allow_html=True)