import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="AI Stock Agent", layout="wide")

# ===========================================
# BIG CLEAR TITLE
# ===========================================
st.markdown("""
<div style='text-align: center; padding: 2rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
            color: white; border-radius: 20px; margin-bottom: 2rem;'>
    <h1 style='font-size: 3rem; margin: 0;'>🤖 AI STOCK PREDICTION AGENT</h1>
    <p style='font-size: 1.2rem; opacity: 0.9;'>Upload CSV → Get Instant Trading Signals</p>
</div>
""", unsafe_allow_html=True)

# File Upload
st.markdown("### 📁 **Upload Your Stock Data**")
uploaded_file = st.file_uploader("", type="csv")

if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        # Auto-detect columns
        if 'close' not in df.columns.str.lower():
            df['close'] = df.iloc[:, -1]  # Use last column
        
        if 'date' not in df.columns.str.lower():
            df['date'] = pd.date_range(start='2023-01-01', periods=len(df))
        else:
            df['date'] = pd.to_datetime(df['date'])
        
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close']).reset_index(drop=True)
        
        # Success
        st.success(f"✅ **Data Loaded: {len(df)} rows**")
        
        # Show data
        st.subheader("📊 **Your Data**")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Price chart
        st.subheader("📈 **Stock Price Chart**")
        fig1 = px.line(df, x='date', y='close', title="Current Price Trend")
        st.plotly_chart(fig1, use_container_width=True)
        
        # AI PREDICTION
        if st.button("🎯 **AI PREDICT NEXT PRICE**", type="primary", use_container_width=True):
            # Simple AI prediction
            current_price = df['close'].iloc[-1]
            trend = df['close'].pct_change().tail(5).mean()
            next_price = current_price * (1 + trend * 1.05)
            
            # 10-day forecast
            days = np.arange(10)
            predictions = current_price + trend * current_price * days * 1.02
            
            # Trading signal
            change_pct = (next_price - current_price) / current_price * 100
            if change_pct > 2:
                signal = "🟢 **BUY** - Strong Uptrend!"
            elif change_pct > -2:
                signal = "🟡 **HOLD** - Sideways"
            else:
                signal = "🔴 **SELL** - Downtrend"
            
            # RESULTS SECTION
            st.markdown("## 🎯 **AI PREDICTION RESULTS**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            with col2:
                st.metric("Next Day", f"${next_price:.2f}", f"{change_pct:+.1f}%")
            with col3:
                st.metric("AI Signal", signal)
            
            # Forecast chart
            st.markdown("### 📊 **10-Day AI Forecast**")
            fig2 = px.line(x=days, y=predictions, 
                          title="AI Predicted Prices",
                          labels={'x': 'Days', 'y': 'Price $'})
            st.plotly_chart(fig2, use_container_width=True)
            
            # Analysis
            st.markdown("### 🤖 **AI Analysis**")
            st.success(f"**{signal}** | Trend: {trend*100:+.1f}% | Ready to trade!")
            
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.info("Upload CSV with 'Date' and 'Close' columns")

else:
    # Demo
    st.info("👆 **Upload CSV file to analyze**")
    st.markdown("""
    **Works with:**
    ```
    Date,Close
    2024-01-01,150.25
    2024-01-02,152.80
    ```
    """)
