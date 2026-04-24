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
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="AI Stock Prediction Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium dark theme with glassmorphism
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #2a1b4d 100%);
        padding: 2rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #2a1b4d 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #ffffff;
        font-weight: 700;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .stMetric > label {
        color: #a0a0ff !important;
        font-weight: 500;
    }
    
    .stMetric > div > div {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700;
    }
    
    .upload-area {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(160, 160, 255, 0.5);
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
    }
    
    .btn-primary {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-weight: 600;
        color: white;
        font-size: 1.1rem;
    }
    
    .ai-insight {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.title("🚀 AI Stock Prediction Agent")
    st.markdown("### Premium LSTM-powered stock analysis with unlimited file upload (200MB+)")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    theme = st.sidebar.selectbox("Theme", ["Dark (Recommended)", "Light"])
    
    # File Upload Section
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📁 Upload Your Stock Data")
        uploaded_file = st.file_uploader(
            "Choose CSV file (Unlimited size - up to 200MB+)",
            type="csv",
            help="Upload your stock CSV with Date, Open, High, Low, Close, Volume columns"
        )
    
    with col2:
        st.markdown("### 📊 Supported Formats")
        st.markdown("""
        - **CSV files only**
        - **No row limits** (1K to 1M+ rows)
        - **Required columns**: Date, Close (minimum)
        - **Optional**: Open, High, Low, Volume
        """)
    
    if uploaded_file is not None:
        try:
            # Memory-efficient large file reading
            with st.spinner("🔄 Processing large dataset..."):
                df = pd.read_csv(uploaded_file, low_memory=False)
                df = utils.clean_stock_data(df)
                
                if df.empty:
                    st.error("❌ No valid data found. Please check your CSV format.")
                    st.stop()
            
            st.success(f"✅ Loaded {len(df):,} rows successfully!")
            
            # Data Preview
            st.markdown("### 👀 Data Preview")
            st.dataframe(df.tail(10), use_container_width=True)
            
            # Charts
            fig = create_stock_charts(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Run AI Analysis
            if st.button("🎯 Run AI Analysis & Predict", key="analyze", help="LSTM Prediction + AI Insights"):
                with st.spinner("🤖 AI analyzing market trends..."):
                    # LSTM Prediction
                    predictions, latest_price, predicted_price = model.predict_next_price(df)
                    
                    # AI Agent Insights
                    ai_insights = agent.get_trading_insights(latest_price, predicted_price, df)
                    
                    # Results Dashboard
                    display_results(latest_price, predicted_price, predictions, ai_insights, df)
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("💡 Tip: Ensure your CSV has 'Date' and 'Close' columns with proper formats.")
    else:
        st.info("👆 Please upload a CSV file to get started!")
        st.markdown("### 🎯 Sample Data Structure")
        sample_data = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Open': [100.0, 101.5, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [99.0, 100.5, 101.0],
            'Close': [102.5, 103.0, 104.5],
            'Volume': [1000000, 1200000, 1100000]
        })
        st.dataframe(sample_data)

def create_stock_charts(df):
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('📈 Stock Price', '📊 Volume'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Price line
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Close'], 
                  name='Close Price', line=dict(color='#667eea', width=2)),
        row=1, col=1
    )
    
    # Moving averages
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    
    if len(df) >= 50:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['MA_20'], 
                      name='MA 20', line=dict(color='#764ba2', width=1.5)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['MA_50'], 
                      name='MA 50', line=dict(color='#f093fb', width=1.5)),
            row=1, col=1
        )
    
    # Volume
    fig.add_trace(
        go.Bar(x=df['Date'], y=df['Volume'], name='Volume',
               marker_color='rgba(102, 126, 234, 0.6)'),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="Stock Price Analysis",
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )
    
    return fig

def display_results(latest_price, predicted_price, predictions, ai_insights, df):
    st.markdown("---")
    st.markdown("## 🎯 AI PREDICTION RESULTS")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        delta = ((predicted_price - latest_price) / latest_price) * 100
        st.metric("Latest Price", f"${latest_price:.2f}", f"{delta:+.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Predicted Price", f"${predicted_price:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card ai-insight">', unsafe_allow_html=True)
        st.metric("AI Recommendation", ai_insights['recommendation'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Risk Level", ai_insights['risk_level'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction Chart
    st.markdown("### 📊 Predicted vs Actual")
    fig_pred = px.line(x=list(range(len(predictions))), 
                      y=predictions.flatten(), 
                      title="LSTM Price Predictions",
                      labels={'x': 'Time Steps', 'y': 'Price ($)'})
    fig_pred.update_layout(template='plotly_dark')
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # AI Insights
    st.markdown("### 🤖 AI Trading Assistant")
    st.markdown(f"""
    **Market Trend**: {ai_insights['trend']}  
    **Confidence**: {ai_insights['confidence']:.1f}%  
    **Analysis**: {ai_insights['explanation']}
    """)
    
    st.markdown("---")
    st.markdown("*Disclaimer: This is for educational purposes only. Not financial advice.*")

if __name__ == "__main__":
    main()
