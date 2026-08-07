import xgboost as xgb
import pandas as pd
import numpy as np
import shap

class FraudAnomalyAgent:
    """Agent 1: Fast Heuristic and Velocity Fraud Check"""
    def inspect(self, applicant_data):
        flags = []
        if applicant_data.get('form_paste_count', 0) > 4:
            flags.append("High application field copy-paste frequency detected.")
        if applicant_data.get('form_typing_speed_wpm', 40) > 110:
            flags.append("Bot-like typing velocity (>110 WPM).")
        if applicant_data.get('public_footprint_score', 0.5) < 0.15 and applicant_data.get('linkedin_tenure_months', 0) == 0:
            flags.append("Zero online professional trace identified.")
            
        is_suspicious = len(flags) > 0
        return is_suspicious, flags

class ScoringAgent:
    """Agent 2: Core ML Risk Classifier"""
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def calculate_score(self, feature_dict):
        df_input = pd.DataFrame([feature_dict])[self.feature_names]
        prob_default = float(self.model.predict_proba(df_input)[0][1])
        
        # Scale score from 0 to 1000
        dynamic_score = int((1.0 - prob_default) * 1000)
        
        if dynamic_score >= 680:
            decision = "APPROVED"
        elif dynamic_score >= 580:
            decision = "MANUAL_REVIEW"
        else:
            decision = "REJECTED"
            
        return dynamic_score, prob_default, decision

class ExplainabilityAgent:
    """Agent 3: SHAP-Driven Plain Language Reasoning Engine"""
    def __init__(self, model, feature_names):
        self.explainer = shap.TreeExplainer(model)
        self.feature_names = feature_names

    def generate_explanation(self, feature_dict):
        df_input = pd.DataFrame([feature_dict])[self.feature_names]
        shap_vals = self.explainer.shap_values(df_input)[0]
        
        pairs = sorted(zip(self.feature_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)
        
        positive_factors = []
        negative_factors = []
        
        feature_labels = {
            'cibil_score': 'Traditional CIBIL Bureau Record',
            'utility_bill_on_time_ratio': 'On-Time Utility Payment Habits',
            'income_regularity_score': 'Income Consistency',
            'job_stability_months': 'Employment Tenure',
            'avg_monthly_savings_ratio': 'Monthly Cash Savings Buffer',
            'linkedin_tenure_months': 'Professional Network Footprint',
            'form_paste_count': 'Application Interaction Pattern'
        }
        
        for feat, val in pairs:
            label = feature_labels.get(feat, feat)
            if val < -0.05: # Negative SHAP value reduces default risk (Positive impact on decision)
                positive_factors.append(f"{label}")
            elif val > 0.05: # Positive SHAP value increases default risk (Negative impact)
                negative_factors.append(f"{label}")
                
        reasoning = []
        if positive_factors:
            reasoning.append(f"Strengths: {', '.join(positive_factors[:3])}.")
        if negative_factors:
            reasoning.append(f"Risk Factors: {', '.join(negative_factors[:3])}.")
            
        return " ".join(reasoning) if reasoning else "Score is balanced across baseline parameters."

class SelfCheckAuditAgent:
    """Agent 4: Quality & Compliance Self-Check Gate"""
    def verify(self, decision, score, fraud_flags):
        if len(fraud_flags) > 0 and decision == "APPROVED":
            return "MANUAL_REVIEW", "Self-Check Rule Triggered: Decision downgraded due to unresolved fraud flags."
        return decision, "Passed automated compliance validation."