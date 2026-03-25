import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# 页面配置与标题
# =========================
st.set_page_config(layout="wide", page_title="顺势交易专家系统 Pro")
st.title("📈 顺势交易专家系统 Pro（量价深度分析版）")
st.markdown("""
本系统基于“顺势而为”核心逻辑：**20日线定趋势，量价配合定买点，动态止损定生死。**
""")


# =========================
# 股票代码标准化
# =========================
def normalize_ticker(ticker):
    ticker = ticker.strip().upper()
    if re.match(r"^\d{6}$", ticker):
        if ticker.startswith(("60", "68")):
            return ticker + ".SS"
        elif ticker.startswith(("00", "30")):
            return ticker + ".SZ"
    return ticker


# =========================
# 数据获取与处理
# =========================
@st.cache_data(ttl=3600)
def get_stock_data(ticker, period="1y", interval="1d"):
    try:
        ticker_norm = normalize_ticker(ticker)
        df = yf.download(ticker_norm, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        # 处理 MultiIndex 列名（yfinance 新版特性）
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return None


# =========================
# 技术指标计算
# =========================
def calculate_indicators(df):
    df = df.copy()
    # 价格均线
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA60"] = df["Close"].rolling(window=60).mean()
    # 均线斜率（判断方向）
    # 避免除以零，并确保有足够数据计算斜率
    df["MA20_Slope"] = df["MA20"].diff(3) / df["MA20"].shift(3).replace(0, np.nan) * 100

    # 成交量均线
    df["VOL_MA20"] = df["Volume"].rolling(window=20).mean()
    # 量比
    df["Vol_Ratio"] = df["Volume"] / df["VOL_MA20"]

    # ATR (用于动态止损)
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df["ATR"] = true_range.rolling(14).mean()

    return df.dropna()


# =========================
# 核心逻辑：量价与结构判断
# =========================
def analyze_market_structure(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    price = latest["Close"]
    ma20 = latest["MA20"]
    ma20_slope = latest["MA20_Slope"]
    vol_ratio = latest["Vol_Ratio"]

    # 1. 趋势定性 (MA20)
    is_uptrend = price > ma20 and ma20_slope > 0

    # 2. 量价特征
    # 放量突破定义更严格：今日收盘价高于昨日最高价，且量比显著放大
    is_volume_breakout = (vol_ratio > 1.5) and (latest["Close"] > prev["High"])
    # 缩量上涨动能减弱的定义：价格上涨但成交量明显萎缩
    is_divergence = (latest["Close"] > prev["Close"]) and (latest["Volume"] < prev["Volume"] * 0.8)

    # 3. 市场位置 (过去30日)
    recent_high = df["High"].tail(30).max()
    recent_low = df["Low"].tail(30).min()
    # 避免除以零
    if (recent_high - recent_low) == 0:
        pos = 0.5  # 无法判断，取中性
    else:
        pos = (price - recent_low) / (recent_high - recent_low)

    # 4. 结构分类与操作建议
    status = ""
    advice = ""

    if is_uptrend:
        if is_volume_breakout:
            status = "🔥 主升浪突破"
            advice = "市场情绪高涨，资金积极涌入，趋势加速确立。可考虑顺势介入或加仓，但需注意风险控制。"
        elif is_divergence and pos > 0.8:  # 高位缩量上涨
            status = "⚠️ 高位量价背离"
            advice = "股价在高位缩量上涨，表明追涨动能不足，主力可能正在悄然派发。建议逐步减仓，并密切关注趋势反转信号。"
        elif price < ma20 * 1.02 and vol_ratio < 1.0:  # 缩量回踩20日线
            status = "🧼 缩量回踩均线"
            advice = "股价在上升趋势中缩量回踩20日均线，若能获得支撑并企稳，是潜在的低吸机会。观察是否放量反弹。"
        else:
            status = "📈 趋势健康持仓"
            advice = "股价沿20日均线稳健上涨，趋势良好。继续持股，享受趋势带来的收益。"
    else:  # 非上升趋势
        if price < ma20 and vol_ratio > 1.5:  # 放量下跌
            status = "💀 恐慌放量下跌"
            advice = "市场恐慌情绪蔓延，大量资金出逃，短期内难以止跌。应坚决规避，切勿盲目抄底。"
        elif pos < 0.2 and vol_ratio < 0.5:  # 低位横盘缩量
            status = "💤 低位横盘缩量"
            advice = "股价在底部区域极度缩量横盘，抛压枯竭，主力可能正在吸筹。耐心等待放量突破，确认趋势反转。"
        else:
            status = "🔄 震荡盘整/弱势"
            advice = "市场缺乏明确方向，多空力量均衡或空头占优。建议观望，等待趋势明朗。"

    # 5. 动态止损位 (基于 ATR 和 MA20)
    # 止损位应始终低于当前价格，且考虑波动性
    # 初始止损可以设定为买入价下方3%-5%或ATR倍数
    # 动态止损：在上升趋势中，MA20是重要的支撑线，可以作为止损参考。
    # 如果价格跌破MA20，则MA20不再是止损位，应考虑ATR或其他百分比止损。
    # 这里我们设定一个相对保守的止损，确保它在当前价格之下。
    # 止损位 = 当前价格 - 2 * ATR，同时确保不低于MA20（如果MA20在下方）
    # 避免止损位高于当前价格的问题
    atr_stop = price - 2 * latest["ATR"]

    if is_uptrend:  # 上升趋势中，MA20是重要支撑
        # 止损位取MA20和ATR止损的较高值，但不能高于当前价格
        stop_loss = max(ma20, atr_stop)
        stop_loss = min(stop_loss, price * 0.98)  # 确保止损位在当前价格的2%以下
    else:  # 非上升趋势，主要以ATR止损为主
        stop_loss = atr_stop
        stop_loss = min(stop_loss, price * 0.95)  # 确保止损位在当前价格的5%以下

    # 确保止损位不为负数
    stop_loss = max(0.01, stop_loss)

    # 6. 动态止盈位 (基于ATR或历史高点)
    # 止盈策略可以多样化，这里提供一个基于ATR的简单追踪止盈
    # 当价格上涨超过一定ATR倍数时，可以考虑部分止盈，或者使用移动止损作为止盈
    # 简单止盈：当前价格 + 2 * ATR，但这不是一个严格的止盈点，更多是参考
    take_profit = price + 2 * latest["ATR"]
    # 也可以考虑使用最近一段时间的高点作为参考
    # take_profit = df["High"].tail(5).max() * 0.98 # 略低于近期高点

    return {
        "status": status,
        "advice": advice,
        "stop_loss": stop_loss,
        "take_profit": take_profit,  # 新增止盈位
        "is_uptrend": is_uptrend,
        "vol_ratio": vol_ratio,
        "ma20_slope": ma20_slope
    }


# =========================
# 多周期共振评分
# =========================
def get_timeframe_score(df):
    if df is None or df.empty: return 0
    df = calculate_indicators(df)
    latest = df.iloc[-1]
    score = 0
    # 价格在20日线上 (25分)
    if latest["Close"] > latest["MA20"]: score += 25
    # 20日线方向向上 (25分)
    if latest["MA20_Slope"] > 0: score += 25
    # 量价配合突破 (25分) - 调整为量比大于1.2且收盘价高于前一日收盘价
    if latest["Vol_Ratio"] > 1.2 and latest["Close"] > df.iloc[-2]["Close"]: score += 25
    # 20日线在60日线上 (25分)
    if latest["MA20"] > latest["MA60"]: score += 25
    return score


# =========================
# 侧边栏：参数输入
# =========================
st.sidebar.header("🔍 策略参数")
ticker_input = st.sidebar.text_input("输入股票代码 (如 000001 或 AAPL)", "000001")
analysis_btn = st.sidebar.button("开始深度分析")

if ticker_input:
    # 获取多周期数据
    with st.spinner("正在获取多周期数据并进行深度计算..."):
        df_d = get_stock_data(ticker_input, "1y", "1d")
        df_w = get_stock_data(ticker_input, "5y", "1wk")
        df_h = get_stock_data(ticker_input, "1mo", "60m")

    if df_d is not None and not df_d.empty:
        df_d = calculate_indicators(df_d)
        analysis = analyze_market_structure(df_d)

        # 多周期评分
        score_d = get_timeframe_score(df_d)
        score_w = get_timeframe_score(df_w)
        score_h = get_timeframe_score(df_h)

        # 确保所有周期都有数据才计算平均分
        valid_scores = [s for s in [score_d, score_w, score_h] if s is not None]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

        # --- 第一部分：核心指标概览 ---
        st.subheader("🎯 核心诊断结论")
        c1, c2, c3, c4, c5 = st.columns(5)  # 增加一列显示止盈位
        c1.metric("综合评分", f"{avg_score:.1f}",
                  delta=f"{avg_score - 50:.1f}" if avg_score > 50 else f"{avg_score - 50:.1f}")
        c2.metric("当前结构", analysis['status'])
        c3.metric("建议止损位", f"{analysis['stop_loss']:.2f}")
        c4.metric("建议止盈位", f"{analysis['take_profit']:.2f}")  # 显示止盈位
        c5.metric("日线量比", f"{analysis['vol_ratio']:.2f}")

        st.info(f"💡 **操作建议：** {analysis['advice']}")

        # --- 第二部分：多周期共振 ---
        st.subheader("🌊 多周期共振分析")
        col_d, col_w, col_h = st.columns(3)
        col_d.metric("日线级别", f"{score_d}分", "短期趋势")
        col_w.metric("周线级别", f"{score_w}分", "中期大势")
        col_h.metric("小时级别", f"{score_h}分", "精确入场")

        # --- 第三部分：可视化图表 ---
        st.subheader("📊 量价趋势图谱")

        # 创建子图: 上部K线+均线, 下部成交量
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03, subplot_titles=("价格与均线系统", "成交量与量能均线"),
                            row_width=[0.3, 0.7])

        # K线图
        fig.add_trace(go.Candlestick(x=df_d.index, open=df_d["Open"], high=df_d["High"],
                                     low=df_d["Low"], close=df_d["Close"], name="K线"), row=1, col=1)
        # 均线
        fig.add_trace(
            go.Scatter(x=df_d.index, y=df_d["MA20"], name="MA20 (趋势线)", line=dict(color="orange", width=2)), row=1,
            col=1)
        fig.add_trace(
            go.Scatter(x=df_d.index, y=df_d["MA60"], name="MA60 (生命线)", line=dict(color="blue", width=1.5)), row=1,
            col=1)

        # 成交量
        # 确保颜色判断基于当日收盘价和开盘价
        colors = ["red" if df_d.iloc[i]["Close"] < df_d.iloc[i]["Open"] else "green" for i in range(len(df_d))]
        fig.add_trace(go.Bar(x=df_d.index, y=df_d["Volume"], name="成交量", marker_color=colors, opacity=0.7), row=2,
                      col=1)
        fig.add_trace(
            go.Scatter(x=df_d.index, y=df_d["VOL_MA20"], name="成交量均线", line=dict(color="purple", width=1)), row=2,
            col=1)

        # 止损线和止盈线
        fig.add_hline(y=analysis['stop_loss'], line_dash="dash", line_color="red", annotation_text="止损线", row=1,
                      col=1)
        fig.add_hline(y=analysis['take_profit'], line_dash="dot", line_color="green", annotation_text="止盈线", row=1,
                      col=1)

        fig.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False,
                          margin=dict(l=50, r=50, b=50, t=50))
        st.plotly_chart(fig, use_container_width=True)

        # --- 第四部分：风险提示 ---
        with st.expander("📝 策略执行手册"):
            st.write("""
            1. **顺势原则**：综合评分 > 70分且日线、周线共振时，胜率最高。
            2. **量价配合**：突破时成交量需超过均量1.5倍，否则视为“弱突破”，易回踩。
            3. **止损执行**：若股价跌破“建议止损位”，应无条件减仓或离场，防止趋势反转带来的大幅亏损。
            4. **假突破识别**：放量突破后，若3日内缩量跌回突破点下方，大概率为假突破。
            5. **止盈策略**：止盈并非一蹴而就，可采用分批止盈或移动止损（如跌破20日线）的方式锁定利润。
            """)
    else:
        st.warning("未能获取到有效的股票数据，请检查代码输入是否正确（如 A股 000001，美股 AAPL）。")
else:
    st.info("👈 请在左侧输入股票代码开始分析。")