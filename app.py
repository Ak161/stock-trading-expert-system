import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# =========================
# 页面配置与专业UI ()
# =========================
st.set_page_config(layout="wide", page_title="老练交易员：量价趋势洞察系统 ")

# 自定义CSS：优化配色，减轻黑色背景，增加专业感
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

st.title("🦅 老练交易员：量价趋势洞察系统 ")
st.markdown("""
本系统致力于还原**职业交易员**的决策逻辑，并完整展示每一步推导过程。
> **核心逻辑：** 趋势定仓位，量价定强弱，结构定买卖，ATR定风控。
""")


# =========================
# 工具函数：代码标准化与名称获取
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


# =========================
# 数据引擎：获取与清洗
# =========================
@st.cache_data(ttl=3600)
def get_data(ticker, period="2y", interval="1d"):
    try:
        ticker_norm = normalize_ticker(ticker)
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
# 核心指标库
# =========================
def add_advanced_indicators(df):
    if df is None or len(df) < 2: return df
    df = df.copy()
    close = df["Close"]

    # 均线系统
    df["MA5"] = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()
    df["MA120"] = close.rolling(120).mean()

    # 量能
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()
    df["Vol_Ratio"] = df["Volume"] / df["VOL_MA20"].replace(0, np.nan)

    # OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_MA20'] = df['OBV'].rolling(20).mean()

    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ATR
    high_low = df["High"] - df["Low"]
    high_prev_close = np.abs(df["High"] - df["Close"].shift())
    low_prev_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    return df


# =========================
# 详细决策推导引擎
# =========================
def detailed_analysis(df):
    """返回包含详细推导过程的分析结果"""
    if df is None or len(df) < 20:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price = latest["Close"]
    vol_ratio = latest["Vol_Ratio"]
    rsi = latest["RSI"]

    # 获取均线数据
    ma5 = latest.get("MA5", price)
    ma10 = latest.get("MA10", price)
    ma20 = latest.get("MA20", price)
    ma60 = latest.get("MA60", price)
    ma120 = latest.get("MA120", price)

    # ========== 第一步：趋势判断 ==========
    trend_analysis = {
        "ma5_vs_ma10": ma5 > ma10,
        "ma10_vs_ma20": ma10 > ma20,
        "ma20_vs_ma60": ma20 > ma60,
        "price_vs_ma20": price > ma20,
        "price_vs_ma60": price > ma60,
    }

    # 趋势定性
    is_bullish_alignment = ma5 > ma10 > ma20
    is_above_ma60 = price > ma60

    if is_bullish_alignment and is_above_ma60:
        trend_desc = "强力多头"
        trend_score = 25
    elif price > ma20 > ma60:
        trend_desc = "多头初期"
        trend_score = 20
    elif price > ma20 and ma20 < ma60:
        trend_desc = "震荡筑底"
        trend_score = 10
    else:
        trend_desc = "空头趋势"
        trend_score = 0

    # ========== 第二步：量价结构判断 ==========
    price_change = price - prev["Close"]
    vol_ratio_prev5 = df["Vol_Ratio"].tail(5).mean()

    if price > prev["Close"] and vol_ratio > 1.5:
        vpa_signal = "放量进攻"
        vpa_score = 20
        vpa_detail = f"价格上涨+放量：当前量比{vol_ratio:.2f}，远超1.5阈值，表现强势"
    elif price < prev["Close"] and vol_ratio < 0.8 and price > ma20:
        vpa_signal = "良性回踩"
        vpa_score = 15
        vpa_detail = f"价格下跌但缩量：当前量比{vol_ratio:.2f}，低于0.8阈值，说明抛压不重"
    elif price > prev["Close"] and vol_ratio < 0.6:
        vpa_signal = "诱多背离"
        vpa_score = 5
        vpa_detail = f"价格上涨但缩量：当前量比{vol_ratio:.2f}，低于0.6阈值，上涨乏力"
    elif price < prev["Close"] and vol_ratio > 2.0:
        vpa_signal = "恐慌放量"
        vpa_score = 0
        vpa_detail = f"价格下跌+放量：当前量比{vol_ratio:.2f}，远超2.0阈值，抛压严重"
    else:
        vpa_signal = "中性"
        vpa_score = 10
        vpa_detail = f"量价结构平衡：当前量比{vol_ratio:.2f}，处于正常范围"

    # ========== 第三步：RSI强弱判断 ==========
    if 40 < rsi < 75:
        rsi_status = "适度强势"
        rsi_score = 15
        rsi_detail = f"RSI为{rsi:.1f}，处于40-75的健康区间，表示适度强势"
    elif rsi >= 75:
        rsi_status = "严重超买"
        rsi_score = 5
        rsi_detail = f"RSI为{rsi:.1f}，超过75，处于超买状态，有回调风险"
    elif rsi <= 40:
        rsi_status = "严重超卖"
        rsi_score = 5
        rsi_detail = f"RSI为{rsi:.1f}，低于40，处于超卖状态，可能反弹"
    else:
        rsi_status = "中性"
        rsi_score = 10
        rsi_detail = f"RSI为{rsi:.1f}，处于中性区间"

    # ========== 第四步：资金流向判断 ==========
    obv_val = latest.get("OBV", 0)
    obv_ma = latest.get("OBV_MA20", 0)

    if obv_val > obv_ma:
        obv_status = "资金持续流入"
        obv_score = 15
        obv_detail = f"OBV({obv_val:.0f}) > OBV_MA20({obv_ma:.0f})，资金呈净流入"
    else:
        obv_status = "资金处于流出"
        obv_score = 5
        obv_detail = f"OBV({obv_val:.0f}) < OBV_MA20({obv_ma:.0f})，资金呈净流出"

    # ========== 综合评分 ==========
    total_score = trend_score + vpa_score + rsi_score + obv_score

    # ========== 操作建议与风控 ==========
    atr = latest.get("ATR", price * 0.02)

    if trend_desc in ["强力多头", "多头初期"]:
        stop_loss = max(ma20 * 0.98, price - 2 * atr)
        take_profit = price + 3 * atr
        if vpa_signal == "放量进攻":
            status = "🔥 趋势共振爆发"
            advice = "多头排列+放量突破。老练交易员在此刻会果断持有，关注量能持续性。"
            action_color = "success"
        elif vpa_signal == "良性回踩":
            status = "🧼 缩量洗盘支撑"
            advice = "趋势向上中的缩量调整，只要不跌破20日线，是理想的逢低布局机会。"
            action_color = "success"
        else:
            status = "📈 趋势持仓期"
            advice = "趋势稳健，建议以20日线作为移动止损线继续持股。"
            action_color = "success"
    elif trend_desc == "空头趋势":
        resistance = min(ma20 * 1.02, price + 2 * atr)
        stop_loss = price - 1.5 * atr
        take_profit = resistance
        status = "❄️ 弱势观望"
        advice = "处于下降通道，上方压力重重。严格执行交易纪律，保持空仓。"
        action_color = "error"
    else:
        stop_loss = price - 2 * atr
        take_profit = price + 2 * atr
        status = "🔄 震荡选择期"
        advice = "多空博弈激烈，方向不明。建议轻仓短线，或等待放量突破。"
        action_color = "info"

    change = (price - prev["Close"]) / prev["Close"] * 100

    return {
        # 基础数据
        "price": price,
        "change": change,
        "trend": trend_desc,
        "vpa": vpa_signal,
        "status": status,
        "advice": advice,
        "action_color": action_color,
        "rsi": rsi,
        "vol_ratio": vol_ratio,
        "stop_loss": stop_loss,
        "take_profit": take_profit,

        # 详细推导数据
        "trend_analysis": trend_analysis,
        "trend_score": trend_score,
        "vpa_score": vpa_score,
        "rsi_score": rsi_score,
        "obv_score": obv_score,
        "total_score": total_score,

        # 均线数据
        "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60, "ma120": ma120,

        # 详细说明
        "vpa_detail": vpa_detail,
        "rsi_status": rsi_status,
        "rsi_detail": rsi_detail,
        "obv_status": obv_status,
        "obv_detail": obv_detail,

        # 历史对比
        "vol_ratio_prev5": vol_ratio_prev5,
        "atr": atr,
    }


# =========================
# 评分引擎
# =========================
def get_score(df):
    if df is None or len(df) < 20: return 0
    df = add_advanced_indicators(df)
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0

    ma20 = latest.get("MA20", 0)
    ma60 = latest.get("MA60", 0)
    ma5 = latest.get("MA5", 0)
    ma10 = latest.get("MA10", 0)

    if latest["Close"] > ma20: score += 15
    if ma20 > ma60: score += 15
    if ma5 > ma10: score += 10
    if latest.get("Vol_Ratio", 0) > 1.2: score += 15
    if latest.get("OBV", 0) > latest.get("OBV_MA20", 0): score += 15
    rsi = latest.get("RSI", 50)
    if 40 < rsi < 75: score += 15
    if latest["Close"] > prev["Close"]: score += 15

    return score


# =========================
# 侧边栏
# =========================
st.sidebar.header("🛠️ 交易控制台")
ticker_input = st.sidebar.text_input("输入标的代码 (如 000001, AAPL)", "000001")
period_select = st.sidebar.selectbox("回测周期 (显示长度)", ["1y", "2y", "5y"], index=0)

if st.sidebar.button("执行深度洞察"):
    with st.spinner("正在解析市场微观结构..."):
        stock_name = get_stock_info(ticker_input)
        df_d_raw = get_data(ticker_input, period_select, "1d")
        df_w_raw = get_data(ticker_input, "5y", "1wk")
        df_h_raw = get_data(ticker_input, "1mo", "60m")

    if df_d_raw is not None and len(df_d_raw) > 5:
        df_d = add_advanced_indicators(df_d_raw)
        analysis = detailed_analysis(df_d)

        if analysis:
            # --- 第一部分：核心价格面板 ---
            st.markdown(f'<div class="stock-name">{stock_name} ({ticker_input})</div>', unsafe_allow_html=True)
            col_p1, col_p2 = st.columns([2, 3])
            with col_p1:
                # 修复：中国A股市场红涨绿跌
                color = "#16a34a" if analysis['change'] < 0 else "#dc2626"
                st.markdown(
                    f'<div class="price-tag">¥{analysis["price"]:.2f} <span style="font-size:1.5rem; color:{color};">{analysis["change"]:+.2f}%</span></div>',
                    unsafe_allow_html=True)

            # --- 第二部分：核心指标 ---
            st.markdown('<div class="section-title">📊 核心分析报告</div>', unsafe_allow_html=True)
            score_d = get_score(df_d)
            score_w = get_score(df_w_raw)
            score_h = get_score(df_h_raw)
            avg_score = (score_d * 0.5 + score_w * 0.3 + score_h * 0.2)

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

            # --- 第三部分：决策推导面板 ---
            st.markdown('<div class="section-title">🔍 决策推导过程</div>', unsafe_allow_html=True)

            # 趋势判断推导
            with st.expander("📈 **第一步：趋势判断** - 从均线排列看方向", expanded=True):
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.write("**均线排列对标：**")
                    st.markdown(f"""
- MA5 = ¥{analysis['ma5']:.2f}
- MA10 = ¥{analysis['ma10']:.2f}
- MA20 = ¥{analysis['ma20']:.2f}
- MA60 = ¥{analysis['ma60']:.2f}
- 当前价 = ¥{analysis['price']:.2f}
                    """)
                with col_t2:
                    st.write("**判断逻辑：**")
                    checks = []
                    if analysis['trend_analysis']['ma5_vs_ma10']:
                        checks.append("✓ MA5 > MA10（短期向上）")
                    else:
                        checks.append("✗ MA5 < MA10（短期向下）")
                    if analysis['trend_analysis']['ma10_vs_ma20']:
                        checks.append("✓ MA10 > MA20（中期向上）")
                    else:
                        checks.append("✗ MA10 < MA20（中期向下）")
                    if analysis['trend_analysis']['ma20_vs_ma60']:
                        checks.append("✓ MA20 > MA60（长期向上）")
                    else:
                        checks.append("✗ MA20 < MA60（长期向下）")
                    if analysis['trend_analysis']['price_vs_ma20']:
                        checks.append("✓ 价格 > MA20（站上支撑）")
                    else:
                        checks.append("✗ 价格 < MA20（跌破支撑）")
                    for check in checks:
                        st.write(check)

                st.markdown(f"**结论：{analysis['trend']}** (得分: {analysis['trend_score']}/25分)")

            # 量价结构推导
            with st.expander("💹 **第二步：量价结构** - 从成交量看强弱", expanded=True):
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    st.write("**量能数据对标：**")
                    st.markdown(f"""
- 当前量比 = {analysis['vol_ratio']:.2f}
- 近5日平均量比 = {analysis['vol_ratio_prev5']:.2f}
- 判断标准：
  - > 1.5 = 放量
  - 0.8-1.5 = 正常
  - < 0.8 = 缩量
                    """)
                with col_v2:
                    st.write("**价量配合：**")
                    st.markdown(f"""
- 价格变化：{analysis['change']:+.2f}%
- 量价信号：{analysis['vpa']}
- 详细说明：{analysis['vpa_detail']}
                    """)

                st.markdown(f"**结论：{analysis['vpa']}** (得分: {analysis['vpa_score']}/20分)")

            # RSI强弱推导
            with st.expander("📊 **第三步：RSI强弱** - 从相对强弱指数看动能", expanded=True):
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.write("**RSI数据：**")
                    st.markdown(f"""
- 当前RSI = {analysis['rsi']:.1f}
- 判断标准：
  - > 75 = 超买
  - 40-75 = 健康
  - < 40 = 超卖
                    """)
                with col_r2:
                    st.write("**强弱判断：**")
                    st.markdown(f"""
- 状态：{analysis['rsi_status']}
- 说明：{analysis['rsi_detail']}
                    """)

                st.markdown(f"**结论：{analysis['rsi_status']}** (得分: {analysis['rsi_score']}/15分)")

            # 资金流向推导
            with st.expander("💰 **第四步：资金流向** - 从OBV看主力意图", expanded=True):
                st.write("**资金流向分析：**")
                st.markdown(f"""
- {analysis['obv_detail']}
- **含义**：如果股价下跌但资金流入，可能存在"主力吸筹"现象
- **结论**：{analysis['obv_status']} (得分: {analysis['obv_score']}/15分)
                """)

            # 综合评分
            st.markdown('<div class="section-title">🎯 综合评分</div>', unsafe_allow_html=True)
            col_score1, col_score2, col_score3, col_score4, col_score5 = st.columns(5)
            with col_score1:
                st.metric("趋势得分", f"{analysis['trend_score']}/25")
            with col_score2:
                st.metric("量价得分", f"{analysis['vpa_score']}/20")
            with col_score3:
                st.metric("RSI得分", f"{analysis['rsi_score']}/15")
            with col_score4:
                st.metric("资金得分", f"{analysis['obv_score']}/15")
            with col_score5:
                st.metric("总得分", f"{analysis['total_score']}/75")

            st.info(
                f"**综合评分说明**：总分{analysis['total_score']}分。得分越高，多头信号越强。得分<30分建议观望，30-50分可轻仓参与，>50分可积极参与。")

            # --- 第四部分：老手洞察 ---
            st.markdown('<div class="section-title">🕵️ 老手洞察 (Trader\'s Insight)</div>', unsafe_allow_html=True)
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                st.write("**筹码稳定性：**")
                v_avg = df_d["Vol_Ratio"].tail(5).mean()
                vol_trend = "温和放量" if v_avg > 1.2 else "缩量洗盘" if v_avg < 0.8 else "存量博弈"
                st.write(f"近5日表现：**{vol_trend}**。老手会观察地量是否出现，那往往是变盘前奏。")
            with col_in2:
                st.write("**资金动向：**")
                obv_val = df_d["OBV"].iloc[-1]
                obv_ma = df_d["OBV_MA20"].iloc[-1]
                obv_status = "资金持续流入" if obv_val > obv_ma else "资金处于流出"
                st.write(f"OBV指标显示：**{obv_status}**。如果股价下跌但资金流入，可能存在'主力吸筹'。")

            # --- 第五部分：风控预案 ---
            st.markdown('<div class="section-title">🛡️ 交易计划预案</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("防守位 (止损)", f"¥{analysis['stop_loss']:.2f}")
                st.caption("基于20日均线与波动率(ATR)动态计算")
            with c2:
                st.metric("攻击位 (目标)", f"¥{analysis['take_profit']:.2f}")
                st.caption("基于预期波动率的目标位")
            with c3:
                current_p = analysis['price']
                risk_reward = (analysis['take_profit'] - current_p) / (
                    max(0.01, abs(current_p - analysis['stop_loss'])))
                st.metric("盈亏比预期", f"{risk_reward:.2f}")
                st.caption("建议盈亏比 > 1.5 时参与")

            # 风控计算过程
            with st.expander("📋 **风控计算详解**"):
                st.markdown(f"""
**止损位计算：**
- 基础值1：MA20 × 0.98 = {analysis['ma20']:.2f} × 0.98 = {analysis['ma20'] * 0.98:.2f}
- 基础值2：当前价 - 2×ATR = {analysis['price']:.2f} - 2×{analysis['atr']:.2f} = {analysis['price'] - 2 * analysis['atr']:.2f}
- 最终止损 = MAX(两个值) = **¥{analysis['stop_loss']:.2f}**

**止盈位计算：**
- 当前价 + 3×ATR = {analysis['price']:.2f} + 3×{analysis['atr']:.2f} = **¥{analysis['take_profit']:.2f}**

**盈亏比计算：**
- 盈利空间 = {analysis['take_profit']:.2f} - {analysis['price']:.2f} = {analysis['take_profit'] - analysis['price']:.2f}
- 亏损空间 = {analysis['price']:.2f} - {analysis['stop_loss']:.2f} = {analysis['price'] - analysis['stop_loss']:.2f}
- 盈亏比 = 盈利空间 / 亏损空间 = **{risk_reward:.2f}**
                """)

            # --- 第六部分：图表 ---
            st.markdown('<div class="section-title">📈 量价图谱</div>', unsafe_allow_html=True)
            days_map = {"1y": 252, "2y": 504, "5y": 1260}
            display_df = df_d.tail(days_map.get(period_select, 252))

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=display_df.index, open=display_df["Open"], high=display_df["High"],
                                         low=display_df["Low"], close=display_df["Close"], name="K线"), row=1, col=1)
            for ma, color in zip(["MA5", "MA20", "MA60"], ["#6366f1", "#f59e0b", "#10b981"]):
                fig.add_trace(
                    go.Scatter(x=display_df.index, y=display_df[ma], name=ma, line=dict(width=1.5, color=color)), row=1,
                    col=1)

            v_colors = ['#dc2626' if display_df.iloc[i]['Close'] >= display_df.iloc[i]['Open'] else '#16a34a' for i in
                        range(len(display_df))]
            fig.add_trace(go.Bar(x=display_df.index, y=display_df["Volume"], name="成交量", marker_color=v_colors),
                          row=2, col=1)
            fig.add_trace(
                go.Scatter(x=display_df.index, y=display_df["VOL_MA20"], name="量均线", line=dict(color="#f59e0b")),
                row=2, col=1)

            fig.update_layout(height=800, template="plotly_white", showlegend=True, xaxis_rangeslider_visible=False,
                              margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("未能加载有效数据，请检查代码。")
else:
    st.info("👈 在左侧控制台输入代码开始分析。")
