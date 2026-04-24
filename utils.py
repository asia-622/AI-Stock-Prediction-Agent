import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def clean_stock_data(df):
    """Memory-efficient data cleaning for large datasets"""
    
    # Convert to lowercase and standardize column names
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Required columns mapping
    col_mapping = {
        'date': ['date', 'datetime', 'time', 'timestamp', 'Date', 'Datetime'],
        'close': ['close', 'closing_price', 'close_price', 'Close', 'Closing_Price']
    }
    
    # Find required columns
    date_col = None
    close_col = None
    
    for target, alternatives in col_mapping.items():
        for alt in alternatives:
            if alt in df.columns:
                if target == 'date':
                    date_col = alt
                elif target == 'close':
                    close_col = alt
                break
    
    if date_col is None or close_col is None:
        print(f"Missing required columns. Found date: {date_col}, close: {close_col}")
        return pd.DataFrame()
    
    # Rename to standard names
    df['date'] = pd.to_datetime(df[date_col], errors='coerce')
    df['close'] = pd.to_numeric(df[close_col], errors='coerce')
    df = df[['date', 'close']].dropna()
    df = df.sort_values('date').reset_index(drop=True)
    
    # Optional columns
    optional = ['open', 'high', 'low', 'volume', 'Open', 'High', 'Low', 'Volume']
    for col in optional:
        if col in df.columns:
            df[col.lower()] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill optional columns
    for col in ['open', 'high', 'low', 'volume']:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(method='ffill').fillna(0)
    
    # Remove outliers
    df = df[(df['close'] > 0.01)]
    df['pct_change'] = df['close'].pct_change()
    df = df[df['pct_change'].abs() < 0.5].drop('pct_change', axis=1)
    
    return df

def prepare_lstm_data(df, lookback=60):
    """Prepare sequential data (for local TensorFlow use)"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['close']].values)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler
