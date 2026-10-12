import pandas as pd

def create_features(df_clean):
    df_feat = df_clean.copy()
    
    # 1. Balance tiers (wealth proxy)
    df_feat['balance_tier'] = pd.cut(
        df_feat['balance'],
        bins=[-np.inf, 0, 500, 2000, 10000, np.inf],
        labels=[0, 1, 2, 3, 4]  # negative → very high
    ).astype(int)
    
    # 2. Contact intensity (campaign engagement signal)
    df_feat['contact_intensity'] = np.log1p(df_feat['campaign'])
    
    # 3. Previous engagement score
    # pdays = -1 means never contacted previously
    df_feat['prev_contacted'] = (df_feat['pdays'] != -1).astype(int)
    df_feat['days_since_contact'] = df_feat['pdays'].replace(-1, 999)  # 999 = never
    
    # 4. Age group
    df_feat['age_group'] = pd.cut(
        df_feat['age'],
        bins=[0, 25, 35, 50, 65, 100],
        labels=[0, 1, 2, 3, 4]  # Gen Z → Retirees
    ).astype(int)
    
    # 5. Financial stress indicator
    df_feat['financial_stress'] = (
        df_feat['default'] + df_feat['housing'] + df_feat['loan']
    )  # 0–3 score
    
    # 6. Engagement quality (longer calls = more interested)
    df_feat['high_engagement'] = (df_feat['duration'] > df_feat['duration'].median()).astype(int)
    
    # 7. Encode remaining categoricals (label encode)
    le = LabelEncoder()
    for col in ['job', 'marital', 'education', 'contact', 'poutcome', 'month']:
        df_feat[col + '_enc'] = le.fit_transform(df_feat[col])
        
    return df_feat