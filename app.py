import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

import model
import utils
import agent

st.set_page_config(
    page_title="AI Stock Prediction Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .main { background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #2a1b4d 100%); padding: 2rem; }
    .stApp { background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #2a1b4d 100%); }
    h1, h2, h3 { font-family: 'Inter', sans-serif; color: #ffffff; font-weight: 700; }
    .metric-card { background: rgba(255,255,255,0.1); backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.2); border-radius: 20px; padding: 2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
    .stMetric > label { color: #a0a0ff !important; font-weight: 500; }
    .stMetric > div > div { color: #ffffff !important; font-size: 2rem !important; font-weight: 700; }
    .upload-area { background: rgba(255,255,255,0.05); border: 2px dashed rgba(160,160,255,0.5); border-radius: 15px; padding: 3rem; text-align: center; }
    .ai-insight { background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2)); border: 1px solid rgba(102,126,234,0.3); border-radius: 15px; padding: 1.5rem; }
    </style>
""", unsafe_allow_html=True)

def main():
    st.title("🚀 AI Stock Prediction Agent")
    st.markdown("### Premium AI-powered stock analysis with unlimited file upload")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📁 Upload Stock Data")
        uploaded_file = st.file_uploader(
            "CSV files (Unlimited size - 200MB+)",
            type="csv",
            help="Upload CSV with Date and Close columns"
        )
    
    with col2:
        st.markdown("### 📊 Requirements")
        st.markdown("• **Date** (YYYY-MM-DD)\n• **Close** price\n• **Optional**: Open, High, Low, Volume")
    
    if uploaded_file is not None:
        try:
            with st.spinner("🔄 Processing dataset..."):
                df = pd.read_csv(uploaded_file, low_memory=False)
                df = utils.clean_stock_data(df)
                
                if df.empty:
                    st.error("❌ No valid data. Needs 'Date' and 'Close' columns.")
                    st.stop()
            
            st.success(f"✅ Loaded {len(df):,} rows!")
            
            st.markdown("### 👀 Data Preview")
            st.dataframe(df.tail(10), use_container_width=True)
            
            fig = create_stock_charts(df)
            st.plotly_chart(fig, use_container_width=True)
            
            if st.button("🎯 Run AI Analysis", key="analyze"):
                with st.spinner("🤖 AI analyzing..."):
                    predictions, latest_price, predicted_price = model.predict_next_price(df)
                    ai_insights = agent.get_trading_insights(latest_price, predicted_price, df)
                    display_results(latest_price, predicted_price, predictions, ai_insights, df)
                    
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Check CSV format: Date (YYYY-MM-DD), Close (numbers)")
    else:
        st.info("👆 Upload CSV to start!")
        st.markdown("### 📋 Sample Format")
        sample = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02'],
            'Close': [100.0, 102.5],
            'Volume': [1000000, 1200000]
        })
        st.dataframe(sample)

def create_stock_charts(df):
    fig = make_subplots(rows=2, cols=1, subplot_titles=('📈 Price', '📊 Volume'), vertical_spacing=0.1, row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Scatter(x=df['date'], y=df['close'], name='Close', line=dict(color='#667eea', width=2)), row=1, col=1)
    
    if len(df) >= 20:
        df['ma20'] = df['close'].rolling(20).mean()
        fig.add_trace(go.Scatter(x=df['date'], y=df['ma20'], name='MA20', line=dict(color='#764ba2')), row=1, col=1)
    
    fig.add_trace(go.Bar(x=df['date'], y=df['volume'], name='Volume', marker_color='rgba(102,126,234,0.6)'), row=2, col=1)
    
    fig.update_layout(height=600, showlegend=True, title_text="Stock Analysis", xaxis_rangeslider_visible=False, template='plotly_dark')
    return fig

def display_results(latest_price, predicted_price, predictions, ai_insights, df):
    st.markdown("## 🎯 AI RESULTS")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        delta = ((predicted_price - latest_price) / latest_price) * 100
        st.metric("Latest Price", f"${latest_price:.2f}", f"{delta:+.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Predicted", f"${predicted_price:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card ai-insight">', unsafe_allow_html=True)
        st.metric("Recommendation", ai_insights['recommendation'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Risk", ai_insights['risk_level'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 📊 Prediction Chart")
    fig_pred = px.line(x=list(range(len(predictions))), y=predictions, title="AI Price Forecast")
    fig_pred.update_layout(template='plotly_dark')
    st.plotly_chart(fig_pred, use_container_width=True)
    
    st.markdown("### 🤖 AI Insights")
    st.markdown(f"""
    **Trend**: {ai_insights['trend']}  
    **Confidence**: {ai_insights['confidence']:.0f}%  
    **Analysis**: {ai_insights['explanation']}
    """)
    
    st.markdown("---")
    st.markdown("*Educational use only. Not financial advice.*")

if __name__ == "__main__":
    main()
