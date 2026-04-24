import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")

st.title("🤖 AI Stock Prediction Agent")

uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Use last column as price data
    df['Price'] = pd.to_numeric(df.iloc[:, -1], errors='coerce')
    df = df.dropna(subset=['Price']).reset_index(drop=True)
    
    st.success("Data loaded successfully: {} rows".format(len(df)))
    
    # Show data preview
    st.subheader("Data Preview")
    st.dataframe(df.head(10))
    
    # Price chart
    st.subheader("Price Chart")
    fig = px.line(df, x=df.index, y='Price', title="Stock Price")
    st.plotly_chart(fig, use_container_width=True)
    
    # AI Prediction
    if st.button("Generate AI Prediction"):
        current_price = df['Price'].iloc[-1]
        trend = df['Price'].pct_change().tail(10).mean()
        predicted_price = current_price * (1 + trend)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        with col2:
            st.metric("Predicted Price", f"${predicted_price:.2f}")
        
        change_pct = (predicted_price - current_price) / current_price * 100
        if change_pct > 0:
            st.success("🟢 BUY Signal")
        else:
            st.warning("🔴 SELL Signal")
        
        st.balloons()

else:
    st.info("Please upload a CSV file with price data")
