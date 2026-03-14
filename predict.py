import numpy as np
import joblib
from tensorflow.keras.models import load_model

def preprocess_input(followers, following, follower_following_ratio,
                     posts, has_profile_pic, username_randomness,
                     suspicious_links_in_bio, verified, bio_length):

    scaler = joblib.load('models/scaler.pkl')
    num = np.array([[followers, following, follower_following_ratio,
                     posts, has_profile_pic, username_randomness,
                     suspicious_links_in_bio, verified, bio_length]])
    num_scaled = scaler.transform(num)
    return num_scaled

def predict(followers, following, follower_following_ratio,
            posts, has_profile_pic, username_randomness,
            suspicious_links_in_bio, verified, bio_length):

    features = preprocess_input(
        followers, following, follower_following_ratio,
        posts, has_profile_pic, username_randomness,
        suspicious_links_in_bio, verified, bio_length)

    model = load_model('models/lstm_model.h5')
    features_lstm = features.reshape((features.shape[0], 1, features.shape[1]))
    prob = model.predict(features_lstm)[0][0]
    label = 'FAKE' if prob > 0.5 else 'REAL'

    return {
        'label': label,
        'confidence': round(float(prob) * 100, 2)
    }