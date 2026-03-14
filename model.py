import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os
from preprocess import load_data, preprocess
 
def train_random_forest(X_train, y_train):
    print('Training Random Forest...')
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, 'models/rf_model.pkl')
    print('Random Forest model saved.')
    return rf
 
def train_svm(X_train, y_train):
    print('Training SVM...')
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    joblib.dump(svm, 'models/svm_model.pkl')
    print('SVM model saved.')
    return svm
 
def train_lstm(X_train, y_train, input_dim):
    print('Training LSTM...')
    X_train_lstm = X_train.reshape(
        (X_train.shape[0], 1, X_train.shape[1]))
 
    model = Sequential([
        LSTM(128, input_shape=(1, input_dim), return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
 
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
 
    early_stop = EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True)
 
    model.fit(X_train_lstm, y_train,
              epochs=20, batch_size=64,
              validation_split=0.2,
              callbacks=[early_stop])
 
    model.save('models/lstm_model.h5')
    print('LSTM model saved.')
    return model
 
def evaluate_model(model, X_test, y_test, name, is_lstm=False):
    if is_lstm:
        X_test = X_test.reshape(
            (X_test.shape[0], 1, X_test.shape[1]))
        y_pred = (model.predict(X_test) > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
 
    print(f'\n--- {name} Results ---')
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(classification_report(y_test, y_pred))
 
if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    df = load_data('data/profiles.csv')
    X, y = preprocess(df)
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
 
    rf = train_random_forest(X_train, y_train)
    evaluate_model(rf, X_test, y_test, 'Random Forest')
 
    svm = train_svm(X_train, y_train)
    evaluate_model(svm, X_test, y_test, 'SVM')
 
    lstm = train_lstm(X_train, y_train, X_train.shape[1])
    evaluate_model(lstm, X_test, y_test, 'LSTM', is_lstm=True)