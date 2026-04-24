import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def clean_stock_data(df):
    """Memory-efficient data cleaning for large datasets"""
    
    # Convert to lowercase and standardize column names
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Required columns
    required_cols = ['date', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        # Try alternative names
        col_mapping = {
            'date': ['date', 'datetime', 'time', 'timestamp'],
            'close': ['close', 'closing_price', 'close_price']
        }
        
        for target, alternatives in col_mapping.items():
            for alt in alternatives:
                if alt in df.columns:
                    df[target] = df[alt]
                    break
    
    # Ensure we have required columns
    if 'date' not in df.columns or 'close' not in df.columns:
        return pd.DataFrame()
    
    # Convert date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Convert close to numeric
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['close'])
    
    # Optional columns (fill NaN with forward fill then 0)
    optional_cols = ['open', 'high', 'low', 'volume']
    for col in optional_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(method='ffill').fillna(0)
        else:
            df[col] = 0
    
    # Remove outliers (prices < $0.01 or extreme jumps)
    df = df[(df['close'] > 0.01) & 
            (df['close'].pct_change().abs() < 0.5)].dropna()
    
    return df

def prepare_lstm_data(df, lookback=60):
    """Prepare sequential data for LSTM"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['close']].values)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler
