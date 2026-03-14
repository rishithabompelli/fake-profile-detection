import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess(df):
    num_features = df[['followers', 'following', 'follower_following_ratio',
                        'posts', 'has_profile_pic', 'username_randomness',
                        'suspicious_links_in_bio', 'verified',
                        'bio_length']].fillna(0).values

    scaler = StandardScaler()
    num_features = scaler.fit_transform(num_features)
    joblib.dump(scaler, 'models/scaler.pkl')

    X = num_features
    y = df['is_fake'].values

    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X, y)

    return X_resampled, y_resampled