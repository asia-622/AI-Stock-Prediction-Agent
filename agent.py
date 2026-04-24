def get_trading_insights(latest_price, predicted_price, df):
    """AI Agent for trading decisions"""
    
    price_change = (predicted_price - latest_price) / latest_price * 100
    volatility = df['close'].pct_change().std() * 100
    recent_trend = df['close'].tail(20).pct_change().mean() * 100
    
    # Determine recommendation
    if price_change > 3:
        recommendation = "🟢 BUY"
        confidence = min(95, 70 + abs(price_change))
    elif price_change > 0:
        recommendation = "🟡 HOLD"
        confidence = min(85, 60 + price_change * 10)
    elif price_change > -3:
        recommendation = "🟡 HOLD"
        confidence = min(75, 50 - abs(price_change) * 5)
    else:
        recommendation = "🔴 SELL"
        confidence = min(90, 65 + abs(price_change))
    
    # Risk assessment
    if volatility < 2:
        risk_level = "🟢 LOW"
    elif volatility < 5:
        risk_level = "🟡 MEDIUM"
    else:
        risk_level = "🔴 HIGH"
    
    # Trend direction
    trend = "🟢 UPWARD" if recent_trend > 0 else "🔴 DOWNWARD"
    
    # Generate explanation
    explanations = {
        "🟢 BUY": f"Strong upward momentum detected ({price_change:+.2f}%). Recent trend: {recent_trend:+.2f}%. Low risk entry point.",
        "🔴 SELL": f"Significant downward pressure ({price_change:+.2f}%). High volatility suggests exit strategy.",
        "🟡 HOLD": f"Sideways market with {price_change:+.2f}% predicted change. Monitor for breakout."
    }
    
    return {
        'recommendation': recommendation,
        'risk_level': risk_level,
        'trend': trend,
        'confidence': confidence,
        'explanation': explanations.get(recommendation, "Analyzing market conditions..."),
        'price_change_pct': price_change
    }
