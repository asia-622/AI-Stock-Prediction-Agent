import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import utils

MODEL_PATH = "model/model.h5"

def create_lstm_model(input_shape):
    """Create LSTM model architecture"""
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def load_or_create_model():
    """Load existing model or create new one"""
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    else:
        # Create simple model for immediate use
        model = create_lstm_model((60, 1))
        model.save(MODEL_PATH)
        return model

@tf.function
def predict_next_price(df):
    """Main prediction function"""
    model = load_or_create_model()
    
    # Prepare data
    lookback = 60
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['close']].values)
    
    if len(scaled_data) < lookback + 1:
        # Not enough data, use simple trend prediction
        latest_price = df['close'].iloc[-1]
        trend = df['close'].pct_change().tail(10).mean()
        predicted_price = latest_price * (1 + trend)
        predictions = np.array([latest_price, predicted_price])
    else:
        # LSTM prediction
        X = []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
        
        X = np.array(X)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Predict next price
        predicted_scaled = model.predict(X[-1:].reshape(1, lookback, 1), verbose=0)
        predicted_price = scaler.inverse_transform([[predicted_scaled[0,0]]])[0,0]
        
        latest_price = df['close'].iloc[-1]
        predictions = np.array([latest_price, predicted_price])
    
    return predictions, float(latest_price), float(predicted_price)
