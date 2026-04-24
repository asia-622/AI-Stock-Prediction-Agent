import pandas as pd
import numpy as np

def clean_stock_data(df):
    """Clean stock data - Compatible with all pandas versions"""
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Find date column
    date_cols = ['date', 'datetime', 'time']
    date_col = next((col for col in date_cols if col in df.columns), None)
    
    if date_col is None:
        # Try to find any date-like column
        for col in df.columns:
            if 'date' in col or df[col].dtype == 'object':
                date_col = col
                break
    
    # Find close column
    close_cols = ['close', 'closing_price', 'close_price']
    close_col = next((col for col in close_cols if col in df.columns), None)
    
    if close_col is None:
        return pd.DataFrame()
    
    # Convert date
    df['date'] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Convert close to numeric
    df['close'] = pd.to_numeric(df[close_col], errors='coerce')
    
    # Drop invalid rows
    df = df[['date', 'close']].dropna().sort_values('date').reset_index(drop=True)
    
    if len(df) < 2:
        return pd.DataFrame()
    
    # Add volume column if exists
    vol_col = next((col for col in df.columns if 'vol' in col.lower()), None)
    if vol_col:
        df['volume'] = pd.to_numeric(df[vol_col], errors='coerce').fillna(0)
    else:
        df['volume'] = 1000000  # Default volume
    
    # Clean outliers (prices > 0.01 and reasonable changes)
    df = df[df['close'] > 0.01].copy()
    df['pct_change'] = df['close'].pct_change()
    df = df[df['pct_change'].abs() < 0.5].drop('pct_change', axis=1)
    
    return df
