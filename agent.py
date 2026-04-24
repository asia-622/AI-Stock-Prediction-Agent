import numpy as np

def get_trading_insights(latest_price, predicted_price, df):
    """AI Trading Agent - Complete with all metrics"""
    
    # Calculations
    change_pct = ((predicted_price - latest_price) / latest_price) * 100
    df['returns'] = df['close'].pct_change()
    volatility = df['returns'].std() * 100 * np.sqrt(252)  # Annualized volatility
    trend = df['close'].tail(20).pct_change().mean() * 100
    
    # Trading Signal
    if change_pct > 3:
        recommendation = "🟢 **STRONG BUY**"
        confidence = 90
    elif change_pct > 1:
        recommendation = "🟢 **BUY**"
        confidence = 75
    elif change_pct > -1:
        recommendation = "🟡 **HOLD**"
        confidence = 60
    elif change_pct > -3:
        recommendation = "🟠 **WEAK SELL**"
        confidence = 70
    else:
        recommendation = "🔴 **STRONG SELL**"
        confidence = 85
    
    # Risk Level
    if volatility < 20:
        risk_level = "🟢 **LOW**"
    elif volatility < 40:
        risk_level = "🟡 **MEDIUM**"
    else:
        risk_level = "🔴 **HIGH**"
    
    # Trend
    if trend > 1:
        trend_text = "🟢 **BULLISH**"
    elif trend > -1:
        trend_text = "🟡 **SIDEWAYS**"
    else:
        trend_text = "🔴 **BEARISH**"
    
    # Explanation
    explanations = {
        "🟢 **STRONG BUY**": f"**Powerful upside** ({change_pct:+.1f}%) with low volatility ({volatility:.0f}%)",
        "🟢 **BUY**": f"**Good entry** ({change_pct:+.1f}%) - bullish trend",
        "🟡 **HOLD**": f"**Wait for breakout** ({change_pct:+.1f}%) - sideways market",
        "🟠 **WEAK SELL**": f"**Minor pullback** ({change_pct:+.1f}%) - reduce position",
        "🔴 **STRONG SELL**": f"**Exit now** ({change_pct:+.1f}%) - high risk"
    }
    
    return {
        'recommendation': recommendation,
        'risk_level': risk_level,
        'trend': trend_text,
        'confidence': confidence,
        'volatility': volatility,
        'change_pct': change_pct,
        'explanation': explanations.get(recommendation, "Analyzing...")
    }
