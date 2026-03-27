import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# =========================
# 页面配置与专业UI
# =========================
st.set_page_config(layout="wide", page_title="老练交易员：量价趋势洞察系统 (Pro)")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; color: #1e293b; }
    .stMetric { background-color: #ffffff; border: 1px solid #e2e8f0; padding: 15px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .stAlert { border-radius: 8px; }
    .trader-insight { background-color: #f8fafc; border-left: 5px solid #3b82f6; padding: 20px; margin: 15px 0; border-radius: 0 10px 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .section-title { color: #1e40af; font-weight: bold; margin-top: 20px; border-bottom: 2px solid #e2e8f0; padding-bottom: 5px; }
    .price-tag { font-size: 2.5rem; font-weight: 800; color: #1e293b; }
    .stock-name { font-size: 1.2rem; color: #64748b; margin-bottom: 10px; }
    .decision-box { background-color: #fef3c7; border-left: 5px solid #f59e0b; padding: 15px; margin: 10px 0; border-radius: 0 8px 8px 0; }
    .data-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #e2e8f0; }
    .data-label { font-weight: 600; color: #475569; }
    .data-value { color: #1e293b; font-weight: 500; }
    .positive { color: #dc2626; }
    .negative { color: #16a34a; }
    </style>
    """, unsafe_allow_html=True)

st.title("🦅 老练交易员：量价趋势洞察系统 (Pro)")
st.markdown("""
本系统模拟职业交易员的深度决策逻辑，动态分析趋势、量价、资金流向，并结合风控给出实战建议。
> **核心原则**：趋势定方向，量价定强弱，结构定买卖，风控定仓位。
""")

# =========================
# 侧边栏控制台
# =========================
st.sidebar.header("🛠️ 交易控制台")
ticker_input = st.sidebar.text_input("输入标的代码 (如 000001, AAPL)", "000001")
period_select = st.sidebar.selectbox("回测周期 (显示长度)", ["1y", "2y", "5y"], index=0)

# 多周期权重调节
st.sidebar.subheader("📊 多周期权重")
w_d = st.sidebar.slider("日线权重", 0.0, 1.0, 0.5, 0.05)
w_w = st.sidebar.slider("周线权重", 0.0, 1.0, 0.3, 0.05)
w_h = st.sidebar.slider("小时线权重", 0.0, 1.0, 0.2, 0.05)
# 归一化
total = w_d + w_w + w_h
if total > 0:
    w_d /= total
    w_w /= total
    w_h /= total

# =========================
# 工具函数
# =========================
def normalize_ticker(ticker):
    ticker = ticker.strip().upper()
    if re.match(r"^\d{6}$", ticker):
        if ticker.startswith(("60", "68")):
            return ticker + ".SS"
        elif ticker.startswith(("00", "30")):
            return ticker + ".SZ"
    return ticker

@st.cache_data(ttl=86400)
def get_stock_info(ticker):
    try:
        t = yf.Ticker(normalize_ticker(ticker))
        info = t.info
        name = info.get('longName') or info.get('shortName') or ticker
        return name
    except:
        return ticker

@st.cache_data(ttl=3600)
def get_data(ticker, period="5y", interval="1d"):
    try:
        ticker_norm = normalize_ticker(ticker)
        # 确保获取足够长的数据用于指标计算
        fetch_period = "max" if period in ["5y", "max"] else "5y"
        df = yf.download(ticker_norm, period=fetch_period, interval=interval, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except Exception as e:
        st.error(f"数据获取失败 ({ticker}): {e}")
        return None

# =========================
# 高级指标计算（一次性完成）
# =========================
def add_advanced_indicators(df):
    if df is None or len(df) < 60:
        return df
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # 均线
    df["MA5"] = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    df["MA120"] = close.rolling(120).mean()

    # 量能
    df["VOL_MA20"] = volume.rolling(20).mean()
    df["Vol_Ratio"] = volume / df["VOL_MA20"].replace(0, np.nan)

    # 动态量比阈值（基于最近60日）
    vol_ratio_series = df["Vol_Ratio"].dropna()
    if len(vol_ratio_series) >= 60:
        vol_mean = vol_ratio_series.tail(60).mean()
        vol_std = vol_ratio_series.tail(60).std()
        df["Vol_Ratio_Threshold_High"] = vol_mean + vol_std
        df["Vol_Ratio_Threshold_Low"] = vol_mean - vol_std
    else:
        df["Vol_Ratio_Threshold_High"] = 1.5
        df["Vol_Ratio_Threshold_Low"] = 0.8

    # OBV
    df['OBV'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df['OBV_MA20'] = df['OBV'].rolling(20).mean()

    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ATR
    high_low = high - low
    high_prev_close = np.abs(high - close.shift())
    low_prev_close = np.abs(low - close.shift())
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # ADX (14)
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)
    atr_14 = df["ATR"]
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df["ADX"] = dx.rolling(14).mean()
    df["+DI"] = plus_di
    df["-DI"] = minus_di

    # CMF (Chaikin Money Flow, 20期)
    mf_mult = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mf_volume = mf_mult * volume
    df["CMF"] = mf_volume.rolling(20).sum() / volume.rolling(20).sum()

    return df

# =========================
# 关键支撑/阻力位识别
# =========================
def find_support_resistance(df, lookback=50):
    """基于近期高低点识别关键支撑阻力"""
    recent = df.tail(lookback)
    # 局部高点：前后各5根K线内最高
    highs = recent["High"]
    resistance_candidates = []
    for i in range(5, len(highs)-5):
        if highs.iloc[i] == highs.iloc[i-5:i+6].max():
            resistance_candidates.append(highs.iloc[i])
    # 局部低点
    lows = recent["Low"]
    support_candidates = []
    for i in range(5, len(lows)-5):
        if lows.iloc[i] == lows.iloc[i-5:i+6].min():
            support_candidates.append(lows.iloc[i])

    # 取最近且显著的（取均值附近）
    if resistance_candidates:
        resistance = np.mean(resistance_candidates[-3:])  # 最近三个高点平均
    else:
        resistance = df["High"].tail(20).max()
    if support_candidates:
        support = np.mean(support_candidates[-3:])
    else:
        support = df["Low"].tail(20).min()
    return support, resistance

# =========================
# OBV背离检测
# =========================
def detect_obv_divergence(df, lookback=20):
    """检测OBV与价格的顶背离/底背离"""
    close = df["Close"].tail(lookback)
    obv = df["OBV"].tail(lookback)
    # 找价格峰值和OBV峰值
    price_peaks = (close == close.rolling(5, center=True).max())
    obv_peaks = (obv == obv.rolling(5, center=True).max())
    # 找价格谷底
    price_troughs = (close == close.rolling(5, center=True).min())
    obv_troughs = (obv == obv.rolling(5, center=True).min())

    # 顶背离：价格创新高，OBV未创新高
    price_high_idx = close[price_peaks].index
    if len(price_high_idx) >= 2:
        last_high = price_high_idx[-1]
        prev_high = price_high_idx[-2]
        if close[last_high] > close[prev_high] and obv[last_high] <= obv[prev_high]:
            return "顶背离"
    # 底背离：价格创新低，OBV未创新低
    price_low_idx = close[price_troughs].index
    if len(price_low_idx) >= 2:
        last_low = price_low_idx[-1]
        prev_low = price_low_idx[-2]
        if close[last_low] < close[prev_low] and obv[last_low] >= obv[prev_low]:
            return "底背离"
    return "无背离"

# =========================
# 动态评分与决策
# =========================
def detailed_analysis(df):
    if df is None or len(df) < 60:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price = latest["Close"]

    # 获取关键指标
    ma5 = latest.get("MA5", price)
    ma10 = latest.get("MA10", price)
    ma20 = latest.get("MA20", price)
    ma60 = latest.get("MA60", price)
    ma120 = latest.get("MA120", price)
    vol_ratio = latest["Vol_Ratio"]
    rsi = latest["RSI"]
    adx = latest.get("ADX", 20)
    plus_di = latest.get("+DI", 25)
    minus_di = latest.get("-DI", 25)
    cmf = latest.get("CMF", 0)
    obv = latest["OBV"]
    obv_ma = latest["OBV_MA20"]

    # 动态量比阈值
    vol_high_th = latest.get("Vol_Ratio_Threshold_High", 1.5)
    vol_low_th = latest.get("Vol_Ratio_Threshold_Low", 0.8)

    # 趋势判断（结合ADX）
    is_bullish_alignment = ma5 > ma10 > ma20
    is_above_ma60 = price > ma60
    if adx > 25:
        trend_strength = "强趋势"
        if plus_di > minus_di:
            trend_desc = "强势多头"
            trend_score = 30
        else:
            trend_desc = "强势空头"
            trend_score = 0
    else:
        trend_strength = "震荡/弱趋势"
        if is_bullish_alignment and is_above_ma60:
            trend_desc = "多头初期"
            trend_score = 20
        elif price > ma20 and ma20 < ma60:
            trend_desc = "震荡筑底"
            trend_score = 10
        else:
            trend_desc = "空头趋势"
            trend_score = 0

    # 量价结构（动态阈值）
    price_change = price - prev["Close"]
    if price > prev["Close"] and vol_ratio > vol_high_th:
        vpa_signal = "放量进攻"
        vpa_score = 20
        vpa_detail = f"价格上涨+放量：当前量比{vol_ratio:.2f}，超过动态上轨{vol_high_th:.2f}，买盘积极"
    elif price < prev["Close"] and vol_ratio < vol_low_th and price > ma20:
        vpa_signal = "良性回踩"
        vpa_score = 15
        vpa_detail = f"价格下跌但缩量：量比{vol_ratio:.2f}低于动态下轨{vol_low_th:.2f}，抛压减轻"
    elif price > prev["Close"] and vol_ratio < vol_low_th:
        vpa_signal = "诱多背离"
        vpa_score = 5
        vpa_detail = f"价格上涨但缩量：量比低于下轨，上涨乏力"
    elif price < prev["Close"] and vol_ratio > vol_high_th:
        vpa_signal = "恐慌放量"
        vpa_score = 0
        vpa_detail = f"价格下跌+放量：恐慌盘涌出"
    else:
        vpa_signal = "中性"
        vpa_score = 10
        vpa_detail = f"量比{vol_ratio:.2f}处于正常区间"

    # RSI + 趋势结合调整
    if trend_desc in ["强势多头", "多头初期"]:
        overbought = 80
        oversold = 30
    else:
        overbought = 70
        oversold = 40

    if oversold < rsi < overbought:
        rsi_status = "适度强势"
        rsi_score = 15
        rsi_detail = f"RSI={rsi:.1f}，处于{oversold}-{overbought}健康区间"
    elif rsi >= overbought:
        rsi_status = "超买预警"
        rsi_score = 5
        rsi_detail = f"RSI={rsi:.1f}，超过{overbought}，短期过热"
    elif rsi <= oversold:
        rsi_status = "超卖反弹"
        rsi_score = 5
        rsi_detail = f"RSI={rsi:.1f}，低于{oversold}，超卖可能反弹"
    else:
        rsi_status = "中性"
        rsi_score = 10

    # 资金流向（CMF + OBV背离）
    if cmf > 0.1:
        money_status = "资金持续流入"
        money_score = 15
        money_detail = f"CMF={cmf:.2f}>0.1，显示资金积极介入"
    elif cmf < -0.1:
        money_status = "资金持续流出"
        money_score = 0
        money_detail = f"CMF={cmf:.2f}<-0.1，资金撤离"
    else:
        money_status = "资金平衡"
        money_score = 8
        money_detail = f"CMF={cmf:.2f}，资金博弈均衡"

    # OBV背离
    obv_div = detect_obv_divergence(df)
    if obv_div == "顶背离":
        money_score -= 5
        money_detail += "；⚠️ OBV顶背离，警惕反转"
    elif obv_div == "底背离":
        money_score += 5
        money_detail += "；✅ OBV底背离，潜在买点"

    # 综合评分
    total_score = trend_score + vpa_score + rsi_score + money_score

    # 关键支撑/阻力位
    support, resistance = find_support_resistance(df)

    # 止损：取支撑位下方一点，或基于ATR的止损，取两者中较紧的
    atr = latest.get("ATR", price * 0.02)
    stop_loss_from_support = support * 0.98
    stop_loss_from_atr = price - 2 * atr
    stop_loss = max(stop_loss_from_support, stop_loss_from_atr)  # 取较高的止损（更紧）
    take_profit = resistance * 0.98 if resistance > price else price + 3 * atr  # 阻力位或ATR目标

    # 盈亏比
    risk = price - stop_loss
    reward = take_profit - price
    risk_reward = reward / risk if risk > 0 else 0

    # 仓位建议（假设账户风险2%）
    position_size_pct = 0.02 / (risk / price) if risk > 0 else 0
    position_size_pct = min(position_size_pct, 0.5)  # 单票不超过50%仓位

    # 操作建议
    if total_score >= 60:
        status = "🔥 强力做多"
        advice = f"各项指标共振，建议买入/加仓。仓位建议：{position_size_pct:.0%}。"
        action_color = "success"
    elif total_score >= 40:
        status = "📈 谨慎看多"
        advice = f"趋势向好但需观察量能，可轻仓参与。仓位建议：{position_size_pct:.0%}。"
        action_color = "info"
    elif total_score >= 20:
        status = "🔄 震荡观望"
        advice = "方向不明，建议等待明确信号。"
        action_color = "warning"
    else:
        status = "❄️ 空头回避"
        advice = "空头排列，资金流出，建议空仓。"
        action_color = "error"

    return {
        "price": price,
        "change": (price - prev["Close"]) / prev["Close"] * 100,
        "trend": trend_desc,
        "trend_strength": trend_strength,
        "vpa": vpa_signal,
        "status": status,
        "advice": advice,
        "action_color": action_color,
        "rsi": rsi,
        "vol_ratio": vol_ratio,
        "adx": adx,
        "cmf": cmf,
        "obv_div": obv_div,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "support": support,
        "resistance": resistance,
        "risk_reward": risk_reward,
        "position_size": position_size_pct,
        # 详细分解
        "trend_score": trend_score,
        "vpa_score": vpa_score,
        "rsi_score": rsi_score,
        "money_score": money_score,
        "total_score": total_score,
        "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60, "ma120": ma120,
        "vpa_detail": vpa_detail,
        "rsi_detail": rsi_detail,
        "money_detail": money_detail,
        "atr": atr,
    }

def get_score(df):
    """快速评分（供多周期加权）"""
    if df is None or len(df) < 20:
        return 0
    latest = df.iloc[-1]
    score = 0
    # 趋势
    if latest["Close"] > latest.get("MA20", 0): score += 15
    if latest.get("MA20", 0) > latest.get("MA60", 0): score += 15
    if latest.get("MA5", 0) > latest.get("MA10", 0): score += 10
    # 量能
    vol_ratio = latest.get("Vol_Ratio", 1)
    vol_high = latest.get("Vol_Ratio_Threshold_High", 1.5)
    if vol_ratio > vol_high: score += 15
    # 资金
    if latest.get("OBV", 0) > latest.get("OBV_MA20", 0): score += 15
    # RSI
    rsi = latest.get("RSI", 50)
    if 40 < rsi < 75: score += 15
    # 涨跌
    if latest["Close"] > df.iloc[-2]["Close"]: score += 15
    return score

# =========================
# 主界面
# =========================
if st.sidebar.button("执行深度洞察"):
    with st.spinner("正在解析市场微观结构..."):
        stock_name = get_stock_info(ticker_input)
        df_d_raw = get_data(ticker_input, period_select, "1d")
        df_w_raw = get_data(ticker_input, "5y", "1wk")
        df_h_raw = get_data(ticker_input, "1mo", "60m")

    if df_d_raw is not None and len(df_d_raw) > 60:
        df_d = add_advanced_indicators(df_d_raw)
        analysis = detailed_analysis(df_d)

        if analysis:
            # 头部信息
            st.markdown(f'<div class="stock-name">{stock_name} ({ticker_input})</div>', unsafe_allow_html=True)
            col_p1, col_p2 = st.columns([2, 3])
            with col_p1:
                color = "#16a34a" if analysis['change'] < 0 else "#dc2626"
                st.markdown(
                    f'<div class="price-tag">¥{analysis["price"]:.2f} <span style="font-size:1.5rem; color:{color};">{analysis["change"]:+.2f}%</span></div>',
                    unsafe_allow_html=True)

            # 多周期评分
            score_d = get_score(df_d)
            score_w = get_score(df_w_raw) if df_w_raw is not None else 0
            score_h = get_score(df_h_raw) if df_h_raw is not None else 0
            avg_score = w_d * score_d + w_w * score_w + w_h * score_h

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("综合操盘评分", f"{avg_score:.1f}", delta=f"{score_d - 50:.1f}")
            m2.metric("趋势定性", analysis['trend'])
            m3.metric("量价结构", analysis['vpa'])
            m4.metric("RSI强弱", f"{analysis['rsi']:.1f}")

            if analysis['action_color'] == "success":
                st.success(f"**【操盘建议】** {analysis['status']} —— {analysis['advice']}")
            elif analysis['action_color'] == "warning":
                st.warning(f"**【操盘建议】** {analysis['status']} —— {analysis['advice']}")
            elif analysis['action_color'] == "error":
                st.error(f"**【操盘建议】** {analysis['status']} —— {analysis['advice']}")
            else:
                st.info(f"**【操盘建议】** {analysis['status']} —— {analysis['advice']}")

            # 决策推导面板
            st.markdown('<div class="section-title">🔍 决策推导过程</div>', unsafe_allow_html=True)

            # 趋势判断
            with st.expander("📈 **第一步：趋势判断（均线+ADX）**", expanded=True):
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.write("**均线排列**")
                    st.markdown(f"""
- MA5: {analysis['ma5']:.2f}
- MA10: {analysis['ma10']:.2f}
- MA20: {analysis['ma20']:.2f}
- MA60: {analysis['ma60']:.2f}
- 当前价: {analysis['price']:.2f}
                    """)
                with col_t2:
                    st.write("**趋势强度**")
                    st.markdown(f"""
- ADX: {analysis['adx']:.1f} (>25为强趋势)
- +DI: {df_d['+DI'].iloc[-1]:.1f}
- -DI: {df_d['-DI'].iloc[-1]:.1f}
- 结论: {analysis['trend_strength']} {analysis['trend']}
- 得分: {analysis['trend_score']}/30
                    """)

            # 量价结构
            with st.expander("💹 **第二步：量价结构（动态阈值）**", expanded=True):
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    st.write(f"**量比: {analysis['vol_ratio']:.2f}**")
                    st.write(f"动态上轨: {df_d['Vol_Ratio_Threshold_High'].iloc[-1]:.2f}")
                    st.write(f"动态下轨: {df_d['Vol_Ratio_Threshold_Low'].iloc[-1]:.2f}")
                with col_v2:
                    st.write(f"**信号**: {analysis['vpa']}")
                    st.write(f"**详情**: {analysis['vpa_detail']}")
                st.write(f"得分: {analysis['vpa_score']}/20")

            # RSI
            with st.expander("📊 **第三步：RSI强弱（结合趋势）**", expanded=True):
                st.write(f"RSI: {analysis['rsi']:.1f}")
                st.write(f"详情: {analysis['rsi_detail']}")
                st.write(f"得分: {analysis['rsi_score']}/15")

            # 资金流向
            with st.expander("💰 **第四步：资金流向（CMF+OBV背离）**", expanded=True):
                st.write(f"CMF: {analysis['cmf']:.3f} ( >0.1流入，<-0.1流出 )")
                st.write(f"OBV背离: {analysis['obv_div']}")
                st.write(f"详情: {analysis['money_detail']}")
                st.write(f"得分: {analysis['money_score']}/15")

            # 综合评分
            st.markdown('<div class="section-title">🎯 综合评分</div>', unsafe_allow_html=True)
            col_score = st.columns(5)
            col_score[0].metric("趋势得分", f"{analysis['trend_score']}/30")
            col_score[1].metric("量价得分", f"{analysis['vpa_score']}/20")
            col_score[2].metric("RSI得分", f"{analysis['rsi_score']}/15")
            col_score[3].metric("资金得分", f"{analysis['money_score']}/15")
            col_score[4].metric("总得分", f"{analysis['total_score']}/80")
            st.info(f"**评分说明**：总分{analysis['total_score']}/80，≥60强力做多，40-60谨慎看多，20-40震荡观望，<20空头回避。")

            # 风控与仓位
            st.markdown('<div class="section-title">🛡️ 交易计划与风控</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("止损位", f"¥{analysis['stop_loss']:.2f}")
            c2.metric("目标位", f"¥{analysis['take_profit']:.2f}")
            c3.metric("盈亏比", f"{analysis['risk_reward']:.2f}")
            c4.metric("建议仓位", f"{analysis['position_size']:.0%}")
            with st.expander("📋 **风控计算详解**"):
                st.markdown(f"""
**支撑/阻力识别**：
- 关键支撑: ¥{analysis['support']:.2f}
- 关键阻力: ¥{analysis['resistance']:.2f}

**止损设定**：
- 基于支撑: {analysis['support']*0.98:.2f}
- 基于ATR: {analysis['price'] - 2*analysis['atr']:.2f}
- 最终取两者较高（更紧）: **¥{analysis['stop_loss']:.2f}**

**仓位计算**：
- 单笔风险 = 账户资金 × 2%
- 建议仓位 = 2% / (止损幅度) = {analysis['position_size']:.0%}
            """)

            # 老手洞察
            st.markdown('<div class="section-title">🕵️ 老手洞察 (Trader\'s Insight)</div>', unsafe_allow_html=True)
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                # 筹码稳定性模拟
                vol_trend = "温和放量" if df_d["Vol_Ratio"].tail(5).mean() > 1.2 else "缩量洗盘" if df_d["Vol_Ratio"].tail(5).mean() < 0.8 else "存量博弈"
                st.write(f"**量能趋势**: {vol_trend}")
                st.write("地量往往意味着变盘前兆，当前量能水平需关注后续变化。")
            with col_in2:
                st.write("**主力意图**")
                if analysis['cmf'] > 0.1 and analysis['trend'] in ["强势多头","多头初期"]:
                    st.write("✅ CMF持续流入+趋势向上，主力可能正在吸筹/拉升。")
                elif analysis['cmf'] < -0.1 and analysis['trend'] == "空头趋势":
                    st.write("⚠️ 资金流出+空头趋势，主力派发迹象明显，规避。")
                elif analysis['obv_div'] == "底背离" and analysis['trend'] == "震荡筑底":
                    st.write("🔍 OBV底背离，暗示下跌末端，可关注底部反转。")
                else:
                    st.write("资金行为不明确，等待更清晰信号。")

            # 图表
            st.markdown('<div class="section-title">📈 量价图谱</div>', unsafe_allow_html=True)
            days_map = {"1y": 252, "2y": 504, "5y": 1260}
            display_df = df_d.tail(days_map.get(period_select, 252))

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=display_df.index, open=display_df["Open"], high=display_df["High"],
                                         low=display_df["Low"], close=display_df["Close"], name="K线"), row=1, col=1)
            for ma, color in zip(["MA5", "MA20", "MA60"], ["#6366f1", "#f59e0b", "#10b981"]):
                fig.add_trace(go.Scatter(x=display_df.index, y=display_df[ma], name=ma, line=dict(width=1.5, color=color)), row=1, col=1)

            v_colors = ['#dc2626' if display_df.iloc[i]['Close'] >= display_df.iloc[i]['Open'] else '#16a34a' for i in range(len(display_df))]
            fig.add_trace(go.Bar(x=display_df.index, y=display_df["Volume"], name="成交量", marker_color=v_colors), row=2, col=1)
            fig.add_trace(go.Scatter(x=display_df.index, y=display_df["VOL_MA20"], name="量均线", line=dict(color="#f59e0b")), row=2, col=1)

            fig.update_layout(height=800, template="plotly_white", showlegend=True, xaxis_rangeslider_visible=False,
                              margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("数据不足（至少需要60个交易日），请尝试其他标的或稍后再试。")
else:
    st.info("👈 在左侧控制台输入代码开始分析。")
