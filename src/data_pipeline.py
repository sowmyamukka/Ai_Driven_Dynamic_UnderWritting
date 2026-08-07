import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_samples=2500, save_path="data/synthetic_underwriting_data.csv"):
    np.random.seed(42)
    
    # Protected Attribute (for Fairness & Disparate Impact testing)
    gender = np.random.choice([0, 1], size=num_samples, p=[0.45, 0.55]) # 0: Female/Unprivileged, 1: Male
    
    # 1. Baseline Bureau Score
    cibil_score = np.random.randint(300, 850, num_samples)
    
    # 2. Employment & Education Signals
    job_stability_months = np.random.randint(1, 120, num_samples)
    education_level = np.random.choice([0, 1, 2], size=num_samples, p=[0.3, 0.5, 0.2]) # HighSchool, Undergrad, Postgrad
    income_regularity_score = np.random.uniform(0.2, 1.0, num_samples)
    
    # 3. Digital Engagement Patterns
    app_session_duration_sec = np.random.randint(30, 900, num_samples)
    form_paste_count = np.random.poisson(lam=1.2, size=num_samples)
    form_typing_speed_wpm = np.random.normal(loc=45, scale=15, size=num_samples)
    
    # 4. Social & Professional Presence
    linkedin_tenure_months = np.random.randint(0, 100, num_samples)
    public_footprint_score = np.random.uniform(0.1, 1.0, num_samples)
    
    # 5. Financial Discipline (Alternative Cash-Flow)
    utility_bill_on_time_ratio = np.random.uniform(0.4, 1.0, num_samples)
    avg_monthly_savings_ratio = np.random.uniform(0.0, 0.5, num_samples)

    # Risk Equation (Probability of Default)
    # Higher score = higher probability of default
    risk_score_logits = (
        (850 - cibil_score) * 0.003
        - (income_regularity_score * 1.8)
        - (utility_bill_on_time_ratio * 2.2)
        - (avg_monthly_savings_ratio * 1.5)
        - (job_stability_months * 0.015)
        + (form_paste_count * 0.4)
        + np.where(form_typing_speed_wpm > 100, 1.5, 0)
        + np.random.normal(0, 0.5, num_samples)
    )
    
    # Convert logits to probability
    prob_default = 1 / (1 + np.exp(-risk_score_logits))
    default_target = (prob_default > 0.45).astype(int)

    df = pd.DataFrame({
        'gender': gender,
        'cibil_score': cibil_score,
        'job_stability_months': job_stability_months,
        'education_level': education_level,
        'income_regularity_score': income_regularity_score,
        'app_session_duration_sec': app_session_duration_sec,
        'form_paste_count': form_paste_count,
        'form_typing_speed_wpm': form_typing_speed_wpm,
        'linkedin_tenure_months': linkedin_tenure_months,
        'public_footprint_score': public_footprint_score,
        'utility_bill_on_time_ratio': utility_bill_on_time_ratio,
        'avg_monthly_savings_ratio': avg_monthly_savings_ratio,
        'default_target': default_target
    })
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    return df

if __name__ == "__main__":
    generate_synthetic_data()
    print("Dataset generated and saved successfully.")