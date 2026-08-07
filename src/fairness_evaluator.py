import numpy as np
import pandas as pd

def calculate_disparate_impact(df, model, feature_names, protected_col='gender'):
    """
    Computes Disparate Impact Ratio = P(Approved | Unprivileged) / P(Approved | Privileged)
    Under Indian and US regulatory standards, DIR should be >= 0.80 (Four-Fifths Rule).
    """
    X = df[feature_names]
    probs = model.predict_proba(X)[:, 1]
    scores = (1.0 - probs) * 1000
    df['predicted_approved'] = (scores >= 650).astype(int)
    
    unprivileged_group = df[df[protected_col] == 0]
    privileged_group = df[df[protected_col] == 1]
    
    approval_rate_unprivileged = unprivileged_group['predicted_approved'].mean()
    approval_rate_privileged = privileged_group['predicted_approved'].mean()
    
    if approval_rate_privileged == 0:
        dir_ratio = 1.0
    else:
        dir_ratio = approval_rate_unprivileged / approval_rate_privileged
        
    is_fair = dir_ratio >= 0.80
    return dir_ratio, approval_rate_unprivileged, approval_rate_privileged, is_fair