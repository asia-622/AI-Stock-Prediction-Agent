import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="AI Stock Agent", layout="wide")

# ===================================
# MAIN TITLE - ALWAYS VISIBLE
# ===================================
st.markdown("""
<style>
h1 { 
    background: linear-gradient(90deg, #667eea, #764ba2); 
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent; 
    font-size: 3rem; 
    text-align: center;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# 🤖 **AI STOCK PREDICTION AGENT**")
st.markdown("**Upload ANY CSV → Get Trading Signals Instantly**")

# File upload
uploaded_file = st.file_uploader("📁 **UPLOAD CSV**", type="csv")

if uploaded_file:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        # Use LAST 2 columns as price data
        if len(df.columns) >= 2:
            df['price'] = pd.to_numeric(df.iloc[:, -1], errors='coerce')  # Last column
            df['time'] = range(len(df))  # Simple index
        else:
            df['price'] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
            df['time'] = range(len(df))
        
        # Clean data
        df = df.dropna(subset=['price']).reset_index(drop=True)
        
        if len(df) == 0:
            st.error("❌ No numeric data found!")
            st.stop()
        
        st.success(f"✅ **LOADED {len(df)} ROWS**")
        
        # Data preview
        st.subheader("📊 **YOUR DATA**")
        st.dataframe(df[['time', 'price']].head(10))
        
        # Price chart
        st.subheader("📈 **PRICE CHART**")
        fig = px.line(df, x='time', y='price', 
                     title="Your Stock Data",
                     labels={'time':'Time', 'price':'Price'})
        st.plotly_chart(fig, use_container_width=True)
        
        # AI BUTTON
        if st.button("🚀 **AI PREDICT**", type="primary"):
            # AI Prediction
            current = df['price'].iloc[-1]
            avg_change = df['price'].pct_change().mean()
            next_price = current * (1 + avg_change * 1.1)
            
            # Forecast
            forecast = [current]
            for i in range(9):
                forecast.append(forecast[-1] * (1 + avg_change * 0.02))
            
            change_pct = (next_price - current) / current * 100
            
            # Signal
            if change_pct > 1:
                signal = "🟢 **BUY**"
                color = "inverse"
            elif change_pct > -1:
                signal = "🟡 **HOLD**"
                color = "normal"
            else:
                signal = "🔴 **SELL**"
                color = "normal"
            
            # DISPLAY RESULTS
            st.markdown("## 🎯 **AI RESULTS**")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("**NOW**", f"${current:.2f}")
            with c2:
                st.metric("**NEXT**", f"${next_price:.2f}", f"{change_pct:+.1f}%")
            with c3:
                st.metric("**SIGNAL**", signal, delta=change_pct)
            
            # Forecast chart
            st.markdown("### 📊
**10-DAY FORECAST**")
            fig2 = px.line(x=range(10), y=forecast, 
                          title="AI Prediction",
                          labels={'x':'Days', 'y':'Price $'})
            fig2.update_layout(template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)
            
            st.balloons()
            
    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("""
    👆 **UPLOAD CSV FILE**
    
    **Works with ANY data like:**
    ```
    150.25
    152.80  
    149.10
    ```
    """)
