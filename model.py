import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self):
        self.prophet_model = None
        self.lr_model = LinearRegression()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = MinMaxScaler()
    
    def fit(self, df):
        """Train all models"""
        # Prepare Prophet data
        prophet_df = df[['date', 'close']].rename(columns={'date': 'ds', 'close': 'y'})
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        
        # Train Prophet
        self.prophet_model = Prophet(daily_seasonality=True, weekly_seasonality=True)
        self.prophet_model.fit(prophet_df)
        
        # Prepare ML data
        df_ml = df.copy()
        df_ml['day'] = df_ml['date'].dt.dayofyear
        df_ml['month'] = df_ml['date'].dt.month
        df_ml['day_of_week'] = df_ml['date'].dt.dayofweek
        df_ml['price_lag1'] = df_ml['close'].shift(1)
        df_ml['price_lag5'] = df_ml['close'].shift(5)
        df_ml['price_ma5'] = df_ml['close'].rolling(5).mean()
        df_ml['price_ma20'] = df_ml['close'].rolling(20).mean()
        
        df_ml = df_ml.dropna()
        X = df_ml[['day', 'month', 'day_of_week', 'price_lag1', 'price_lag5', 'price_ma5', 'price_ma20']]
        y = df_ml['close']
        
        # Train ML models
        self.lr_model.fit(X, y)
        self.rf_model.fit(X, y)
        
        # Scale data
        self.scaler.fit(df[['close']])
    
    def predict_next_price(self, df):
        """Predict next closing price using ensemble"""
        if len(df) < 20:
            # Simple trend for small datasets
            latest_price = df['close'].iloc[-1]
            trend = df['close'].pct_change().tail(10).mean()
            predicted_price = latest_price * (1 + trend * 1.1)
            predictions = np.array([latest_price, predicted_price])
            return predictions, float(latest_price), float(predicted_price)
        
        # Fit models if not fitted
        if self.prophet_model is None:
            self.fit(df)
        
        latest_price = df['close'].iloc[-1]
        
        # Prophet prediction
        future = self.prophet_model.make_future_dataframe(periods=1)
        prophet_forecast = self.prophet_model.predict(future)
        prophet_pred = prophet_forecast['yhat'].iloc[-1]
        
        # ML prediction
        df_ml = df.copy()
        df_ml['date'] = pd.to_datetime(df_ml['date'])
        df_ml['day'] = df_ml['date'].dt.dayofyear
        df_ml['month'] = df_ml['date'].dt.month
        df_ml['day_of_week'] = df_ml['date'].dt.dayofweek
        df_ml['price_lag1'] = df_ml['close'].shift(1)
        df_ml['price_lag5'] = df_ml['close'].shift(5)
        df_ml['price_ma5'] = df_ml['close'].rolling(5).mean()
        df_ml['price_ma20'] = df_ml['close'].rolling(20).mean()
        
        latest_features = df_ml[['day', 'month', 'day_of_week', 'price_lag1', 'price_lag5', 'price_ma5', 'price_ma20']].iloc[-1:].fillna(method='ffill')
        lr_pred = self.lr_model.predict(latest_features)[0]
        rf_pred = self.rf_model.predict(latest_features)[0]
        
        # Ensemble prediction (weighted average)
        predicted_price = (prophet_pred * 0.5 + lr_pred * 0.25 + rf_pred * 0.25)
        
        # Generate prediction series
        predictions = np.linspace(latest_price, predicted_price, 10)
        
        return predictions, float(latest_price), float(predicted_price)

# Global predictor instance
predictor = StockPredictor()

def predict_next_price(df):
    """Main prediction function - compatible with Streamlit Cloud"""
    return predictor.predict_next_price(df)
