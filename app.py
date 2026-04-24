import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import local modules with error handling
try:
    import model
    import utils
    import agent
except ImportError as e:
    st.error(f"Missing module: {e}")
    st.stop()

st.set_page_config(page_title="AI Stock Prediction", page_icon="📈", layout="wide")

# SAFE CSS - Won't break layout
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"]  {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}
h1 { color: #ffffff !important; font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem; }
h2 { color: #e0e7ff !important; font-size: 1.8rem; }
h3 { color: #c7d2fe !important; font-size: 1.4rem; }
.metric-container { background: rgba(255,255,255,0.15); border-radius: 15px; padding: 1.5rem; margin: 1rem 0; border: 1px solid rgba(255,255,255,0.2); }
.stMetric label { color: #e2e8f0 !important; font-weight: 600; font-size: 1rem; }
.stMetric div div { color: #ffffff !important; font-size: 2rem !important; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

def main():
    # Title
    st.title("🚀 AI Stock Prediction Agent")
    st.markdown("**Upload CSV → Get AI Trading Signals Instantly**")
    
    # File upload
    uploaded_file = st.file_uploader("📁 **Upload Stock CSV**", type="csv")
    
    if uploaded_file is not None:
        try:
            with st.spinner('🔄 Processing your data...'):
                # Read and clean data
                df = pd.read_csv(uploaded_file)
                df = utils.clean_stock_data(df)
                
                if df.empty:
                    st.error("❌ **No valid data found!**")
                    st.info("**Required columns:** Date (2024-01-01), Close (numbers)")
                    st.stop()
            
            # Success message
            st.success(f"✅ **{len(df):,} rows loaded successfully!**")
            
            # Show preview
            st.markdown("### 📊 **Data Preview**")
            st.dataframe(df.tail(10), use_container_width=True)
            
            # Charts
            st.markdown("### 📈 **Stock Chart**")
            fig = model.create_charts(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction button
            if st.button("🎯 **RUN AI ANALYSIS**", type="primary", use_container_width=True):
                with st.spinner('🤖 AI predicting future price...'):
                    predictions, latest_price, predicted_price = model.predict_next_price(df)
                    insights = agent.get_trading_insights(latest_price, predicted_price, df)
                
                # Results
                show_results(latest_price, predicted_price, predictions, insights)
                
        except Exception as e:
            st.error(f"❌ **Processing Error**: {str(e)}")
            st.info("**Try:** Check Date/Close columns format")
    else:
        # Welcome screen
        st.info("👆 **Upload your CSV file to start!**")
        st.markdown("### 📋 **Supported Formats**")
        st.markdown("""
        
