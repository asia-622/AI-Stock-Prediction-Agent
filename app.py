def display_results(latest, predicted, predictions, insights):
    st.markdown("## 🎯 **AI PREDICTION RESULTS**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        delta = ((predicted - latest) / latest) * 100
        st.metric("**Current**", f"${latest:.2f}", f"{delta:+.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("**Predicted**", f"${predicted:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card ai-insight">', unsafe_allow_html=True)
        st.metric("**Signal**", insights['recommendation'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("**Risk**", insights['risk_level'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chart
    st.markdown("### 📈 **AI Price Forecast**")
    fig = px.line(x=list(range(10)), y=predictions, 
                  title="Next 10 Days Prediction",
                  labels={'x':'Days', 'y':'Price ($)'})
    fig.update_layout(template='plotly_dark', height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # AI Analysis
    st.markdown("### 🤖 **Smart Analysis**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**📊 Trend**: {insights['trend']}")
        st.markdown(f"**🎯 Confidence**: **{insights['confidence']}%**")
    with col2:
        st.markdown(f"**⚡ Volatility**: **{insights['volatility']:.0f}%**")
        st.markdown(f"**💹 Change**: **{insights['change_pct']:+.1f}%**")
    
    st.markdown("### 💡 **AI Recommendation**")
    st.markdown(f"**{insights['recommendation']}**  -  {insights['explanation']}")
    
    st.markdown("---")
    st.caption("*For education only. Not financial advice.*")
