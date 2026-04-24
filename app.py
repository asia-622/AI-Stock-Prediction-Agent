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

st.set_page_config(page_title="AI Stock Prediction", page_icon="📈", layout="wide")

# PERFECT COLORS - HIGH VISIBILITY
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
.main { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 35%, #0f3460 100%); padding: 2rem 1rem; }
h1 { color: #ffffff !important; font-size: 3rem; font-weight: 700; text-shadow: 0 2px 10px rgba(0,0,0,0.5); }
h2 { color: #e0e7ff !important; font-size: 2rem; font-weight: 600; }
h3 { color: #c7d2fe !important; font-size: 1.5rem; }
.metric-card { background: rgba(255,255,255,0.15) !important; backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.25); border-radius: 20px; padding: 1.5rem; box-shadow: 0 8px 32px rgba(0,0,0,0.4); }
.stMetric > label { color: #e2e8f0 !important; font-size: 1.1rem !important; font-weight: 600 !important; }
.stMetric > div > div { color: #ffffff !important; font-size: 2.5rem !important; font-weight: 800 !important; text-shadow: 0 1px 3px rgba(0,0,0,0.5); }
.upload-area { background: rgba(255,255,255,0.08); border: 2px dashed #60a5fa; border-radius: 15px; padding: 2rem; color: #e2e8f0; }
.ai-insight { background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(139,92,246,0.25)); border: 1px solid rgba(99,102,241,0.5); border-radius: 15px; padding: 1.5rem; color: #e2e8f0; }
.stButton > button { background: linear-gradient(45deg, #3b82f6, #8b5cf6); color: white !important; border-radius: 12px; font-weight: 600; border: none; padding: 0.75rem 2rem; font-size: 1.1rem; }
.element-container .dataframe { background: rgba(255,255,255,0.1); border-radius: 12px; color: #e2e8f0 !important; }
.dataframe th { background: rgba(99,102,241,0.3) !important; color: #ffffff !important; }
.dataframe td { color: #e2e8f0 !important; }
.stSuccess { background: rgba(34,197,94,0.2); border: 1px solid rgba(34,197,94,0.5); color: #dcfce7; }
.stError { background: rgba(239,68,68,0.2); border: 1px solid rgba(239,68,68,0.5); color: #fecaca; }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🚀 AI Stock Prediction Agent")
    st.markdown("**Upload CSV → Get AI Predictions Instantly**")
    
    col1, col2 = st.columns([3,1])
    
    with col1:
        uploaded_file = st.file_uploader("📁 Upload Stock CSV", type="csv", 
                                       help="Date + Close columns required")
    
    with col2:
        st.markdown("""
        **✅ Works with:**
        - Apple, Tesla, Bitcoin
        - Any stock CSV
        - 200MB+ files OK!
        """)
    
    if uploaded_file:
        try:
            with st.spinner("🔄 Processing..."):
                df = pd.read_csv(uploaded_file)
                df = utils.clean_stock_data(df)
                
                if df.empty:
                    st.error("❌ No valid data. Needs Date + Close columns.")
                    st.stop()
            
            st.success(f"✅ Loaded **{len(df):,} rows**")
            
            st.markdown("### 📊 Data Preview")
            st.dataframe(df.tail(10), use_container_width=True)
            
            # Charts
            fig = model.create_charts(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # AI Analysis Button
            if st.button("🎯 **RUN AI PREDICTION**", type="primary"):
                with st.spinner("🤖 AI Analyzing Trends..."):
                    predictions, latest_price, predicted_price = model.predict_next_price(df)
                    ai_insights = agent.get_trading_insights(latest_price, predicted_price, df)
                    display_results(latest_price, predicted_price, predictions, ai_insights)
                    
        except Exception as e:
            st.error(f"❌ **Error**: {str(e)}")
            st.info("💡 **Tip**: CSV needs 'Date' (2024-01-01) and 'Close' (numbers)")
    else:
        st.info("👆 **Upload CSV to start analysis**")
        st.markdown("### 📋 Sample Format")
        sample_df = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Close': [100.0, 102.5, 105.2],
            'Volume': [1000000, 1200000, 1100000]
        })
        st.dataframe(sample_df)

def display_results(latest, predicted, predictions, insights):
    st.markdown("## 🎯 **AI PREDICTION RESULTS**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        delta = ((predicted - latest) / latest) * 100
        st.metric("**Current Price**", f"${latest:.2f}", f"{delta:+.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("**Next Price**", f"${predicted:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card ai-insight">', unsafe_allow_html=True)
        st.metric("**AI Signal**", insights['recommendation'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("**Risk Level**", insights['risk_level'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction Chart
    st.markdown("### 📈 **Price Forecast**")
    fig = px.line(x=list(range(10)), y=predictions, 
                  title="AI Predicted Price Movement",
                  labels={'x':'Days', 'y':'Price ($)'})
    fig.update_layout(template='plotly_dark', height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # AI Insights
    st.markdown("### 🤖 **AI Trading Analysis**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Trend**: {insights['trend']}")
        st.markdown(f"**Confidence**: {insights['confidence']:.0f}%")
    with col2:
        st.markdown(f"**Volatility**: {insights['volatility']:.1f}%")
        st.markdown(f"**Change**: {insights['change_pct']:+.1f}%")
    
    st.markdown("**Recommendation**: " + insights['explanation'])

if __name__ == "__main__":
    main()
