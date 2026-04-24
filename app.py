import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

try:
    import model
    import utils
    import agent
except:
    st.error("Install modules: pip install -r requirements.txt")
    st.stop()

st.set_page_config(page_title="AI Stock Prediction", page_icon="📈", layout="wide")

# FIXED CSS - No syntax errors
st.markdown("""
<style>
h1 { color: #ffffff !important; font-size: 2.5rem; font-weight: 700; }
h2 { color: #e0e7ff !important; font-size: 1.8rem; }
.metric-container { 
    background: rgba(255,255,255,0.15); 
    border-radius: 15px; 
    padding: 1.5rem; 
    margin: 1rem 0; 
    border: 1px solid rgba(255,255,255,0.2);
}
.stMetric label { color: #e2e8f0 !important; font-weight: 600; }
.stMetric div div { color: #ffffff !important; font-size: 2rem !important; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🚀 AI Stock Prediction Agent")
    
    uploaded_file = st.file_uploader("📁 Upload CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df = utils.clean_stock_data(df)
            
            if df.empty:
                st.error("No valid data! Need Date + Close columns.")
                return
            
            st.success(f"✅ Loaded {len(df)} rows")
            st.dataframe(df.tail())
            
            fig = model.create_charts(df)
            st.plotly_chart(fig, use_container_width=True)
            
            if st.button("🎯 RUN AI PREDICTION"):
                predictions, latest, predicted = model.predict_next_price(df)
                insights = agent.get_trading_insights(latest, predicted, df)
                show_results(latest, predicted, predictions, insights)
                
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("👆 Upload CSV file")

def show_results(latest, predicted, predictions, insights):
    st.markdown("## 🎯 RESULTS")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        delta = ((predicted-latest)/latest)*100
        st.metric("Current", f"${latest:.2f}", f"{delta:+.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Predicted", f"${predicted:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Signal", insights['recommendation'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    fig = px.line(x=range(10), y=predictions, title="Price Prediction")
    st.plotly_chart(fig)
    
    st.markdown(f"**{insights['explanation']}**")

if __name__ == "__main__":
    main()
