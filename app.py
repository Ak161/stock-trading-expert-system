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
st.set_page_config(layout="wide", page_title="量价趋势深度分析系统")
st.title("📊 量价趋势深度分析系统")
st.markdown("""
本系统专注于**量价关系**与**多周期共振**分析。通过深度挖掘成交量与股价变动的内在逻辑，识别主力资金动向与市场情绪。
核心逻辑：**均线定方向，量价定强弱，波动定风控。**
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
    df["MA20_Slope"] = df["MA20"].diff(3) / df["MA20"].shift(3).replace(0, np.nan) * 100

    # 成交量均线
    df["VOL_MA20"] = df["Volume"].rolling(window=20).mean()
    # 量比 (当前成交量 / 20日平均成交量)
    df["Vol_Ratio"] = df["Volume"] / df["VOL_MA20"]

    # ATR (平均真实波幅，用于动态止损)
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

    # 1. 趋势定性
    is_positive_trend = price > ma20 and ma20_slope > 0

    # 2. 量价特征
    # 放量突破：量比显著放大且收盘价高于昨日最高
    is_volume_breakout = (vol_ratio > 1.5) and (latest["Close"] > prev["High"])
    # 量价背离：价格上涨但成交量萎缩
    is_divergence = (latest["Close"] > prev["Close"]) and (latest["Volume"] < prev["Volume"] * 0.8)

    # 3. 市场位置 (过去30日相对位置)
    recent_high = df["High"].tail(30).max()
    recent_low = df["Low"].tail(30).min()
    if (recent_high - recent_low) == 0:
        pos = 0.5
    else:
        pos = (price - recent_low) / (recent_high - recent_low)

    # 4. 结构分类与操作建议
    status = ""
    advice = ""

    if is_positive_trend:
        if is_volume_breakout:
            status = "🔥 放量加速突破"
            advice = "当前量价齐升，资金抢筹明显。操作上可考虑加仓或持有，关注量能是否持续，若缩量滞涨则需警惕。"
        elif is_divergence and pos > 0.8:
            status = "⚠️ 高位量能枯竭"
            advice = "价格虽创新高但量能萎缩，属于典型的量价背离。建议逢高减仓，严守止损位，防止高位跳水。"
        elif price < ma20 * 1.02 and vol_ratio < 1.0:
            status = "🧼 缩量回踩支撑"
            advice = "股价缩量回落至均线附近，洗盘特征明显。若能在此位置企稳并再次放量，是理想的二次介入点。"
        else:
            status = "📈 稳健放量上行"
            advice = "趋势运行健康，成交量配合良好。建议继续持有，关注20日均线的支撑力度。"
    else:
        if price < ma20 and vol_ratio > 1.5:
            status = "💀 恐慌放量杀跌"
            advice = "放量跌破关键位，主力出货迹象明显。应果断离场规避，不宜盲目抄底，等待市场情绪平复。"
        elif pos < 0.2 and vol_ratio < 0.5:
            status = "💤 低位地量横盘"
            advice = "地量见地价，市场交投极度冷清。建议保持耐心观察，等待一根放量长阳线确认反转信号。"
        else:
            status = "🔄 弱势震荡筑底"
            advice = "方向不明，量能杂乱。操作上建议观望为主，等待量价结构清晰后再行决策。"

    # 5. 动态风控位
    atr_stop = price - 2 * latest["ATR"]
    if is_positive_trend:
        stop_loss = max(ma20, atr_stop)
        stop_loss = min(stop_loss, price * 0.98)
    else:
        stop_loss = atr_stop
        stop_loss = min(stop_loss, price * 0.95)

    stop_loss = max(0.01, stop_loss)
    take_profit = price + 2.5 * latest["ATR"]

    return {
        "status": status,
        "advice": advice,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "is_positive_trend": is_positive_trend,
        "vol_ratio": vol_ratio,
        "ma20_slope": ma20_slope
    }


# =========================
# 评分逻辑说明
# =========================
def get_timeframe_score(df):
    if df is None or df.empty: return 0
    df = calculate_indicators(df)
    latest = df.iloc[-1]
    score = 0
    # 1. 价格位置 (25分): 价格在20均线上方说明短期强势
    if latest["Close"] > latest["MA20"]: score += 25
    # 2. 趋势斜率 (25分): 20均线向上说明趋势处于扩张期
    if latest["MA20_Slope"] > 0: score += 25
    # 3. 量价配合 (25分): 收盘价上涨且量比大于1.2，说明上涨有量能支撑
    if latest["Vol_Ratio"] > 1.2 and latest["Close"] > df.iloc[-2]["Close"]: score += 25
    # 4. 长短周期配合 (25分): 20均线在60均线上方，说明多头排列
    if latest["MA20"] > latest["MA60"]: score += 25
    return score


# =========================
# 侧边栏：参数输入
# =========================
st.sidebar.header("🔍 股票深度诊断")
ticker_input = st.sidebar.text_input("输入代码 (如 000001, AAPL)", "000001")
analysis_btn = st.sidebar.button("开始量价分析")

if ticker_input:
    with st.spinner("正在抓取全球金融数据并进行量价建模..."):
        df_d = get_stock_data(ticker_input, "1y", "1d")
        df_w = get_stock_data(ticker_input, "5y", "1wk")
        df_h = get_stock_data(ticker_input, "1mo", "60m")

    if df_d is not None and not df_d.empty:
        df_d = calculate_indicators(df_d)
        analysis = analyze_market_structure(df_d)

        score_d = get_timeframe_score(df_d)
        score_w = get_timeframe_score(df_w)
        score_h = get_timeframe_score(df_h)

        valid_scores = [s for s in [score_d, score_w, score_h] if s is not None]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

        # --- 第一部分：核心诊断结论 ---
        st.subheader("🎯 量价诊断核心结论")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("多周期综合评分", f"{avg_score:.1f}")
        c2.metric("当前量价结构", analysis['status'])
        c3.metric("防守止损位", f"{analysis['stop_loss']:.2f}")
        c4.metric("目标止盈位", f"{analysis['take_profit']:.2f}")
        c5.metric("当日量比", f"{analysis['vol_ratio']:.2f}")

        st.success(f"💡 **专业操作建议：** {analysis['advice']}")

        # --- 第二部分：多周期共振深度说明 ---
        st.subheader("🌊 多周期共振分析")
        col_d, col_w, col_h = st.columns(3)

        with col_d:
            st.metric("日线级别 (Day)", f"{score_d}分")
            st.caption("**日线定义**：代表短期1-2周的波段趋势。分数高表示短期量价配合理想。")
        with col_w:
            st.metric("周线级别 (Week)", f"{score_w}分")
            st.caption("**周线定义**：代表中期3-6个月的大趋势。周线高分是安全边际的保障。")
        with col_h:
            st.metric("小时级别 (60m)", f"{score_h}分")
            st.caption("**小时定义**：代表当日或次日的精确入场时机。用于捕捉日内波动点。")

        with st.expander("📝 评分逻辑与数值说明"):
            st.markdown("""
            ### 1. 分数是怎么算的？
            每个周期总分100分，由四个维度构成（各25分）：
            - **价格站稳**：收盘价 > 20日均线。
            - **趋势向上**：20日均线斜率为正。
            - **量价齐升**：当日上涨且量比 > 1.2（说明不是虚涨）。
            - **多头排列**：20日线在60日线上方（中期趋势支撑）。

            ### 2. 综合评分含义：
            - **80-100分**：**强力共振**。多周期趋势向上且量能充沛，极具操作价值。
            - **60-80分**：**趋势良好**。存在局部量价背离或周期不统一，建议轻仓参与。
            - **40-60分**：**震荡徘徊**。多空博弈激烈，量能无序，建议观望。
            - **40分以下**：**弱势区域**。趋势向下且可能伴随放量杀跌，严禁抄底。

            ### 3. 量价关键指标：
            - **量比 (Vol Ratio)**：衡量相对成交强度。>1.5 为放量，<0.5 为缩量。
            - **ATR (波动率)**：用于计算止损。止损位设在 2 倍 ATR 波动之外，防止被市场噪音震荡出局。
            """)

        # --- 第三部分：可视化图表 ---
        st.subheader("📊 量价趋势图谱")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03, subplot_titles=("价格走势与风控系统", "成交量与量能均线"),
                            row_width=[0.3, 0.7])

        fig.add_trace(go.Candlestick(x=df_d.index, open=df_d["Open"], high=df_d["High"],
                                     low=df_d["Low"], close=df_d["Close"], name="K线"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_d.index, y=df_d["MA20"], name="20日参考线", line=dict(color="orange", width=2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=df_d.index, y=df_d["MA60"], name="60日生命线", line=dict(color="blue", width=1.5)),
                      row=1, col=1)

        colors = ["#EF5350" if df_d.iloc[i]["Close"] < df_d.iloc[i]["Open"] else "#26A69A" for i in range(len(df_d))]
        fig.add_trace(go.Bar(x=df_d.index, y=df_d["Volume"], name="成交量", marker_color=colors, opacity=0.7), row=2,
                      col=1)
        fig.add_trace(
            go.Scatter(x=df_d.index, y=df_d["VOL_MA20"], name="成交量均线", line=dict(color="purple", width=1)), row=2,
            col=1)

        fig.add_hline(y=analysis['stop_loss'], line_dash="dash", line_color="red", annotation_text="止损防线", row=1,
                      col=1)
        fig.add_hline(y=analysis['take_profit'], line_dash="dot", line_color="green", annotation_text="目标位", row=1,
                      col=1)

        fig.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False,
                          margin=dict(l=50, r=50, b=50, t=50))
        st.plotly_chart(fig, use_container_width=True)

        # --- 第四部分：量价交易纪律 ---
        with st.expander("🛡️ 量价交易风控纪律"):
            st.write("""
            1. **不量不入**：没有成交量配合的突破多为假突破，需谨慎对待。
            2. **周期服从**：当小时级别与日线级别冲突时，以日线为准；当日线与周线冲突时，以周线为准。
            3. **量价背离必减**：价格创新高但成交量萎缩，是主力派发的早期信号，必须执行减仓。
            4. **严格止损**：止损位是生存底线，一旦跌破必须无条件执行，切勿产生幻觉。
            5. **动态止盈**：随着价格上涨，应不断上调止损位（移动止损），锁住既得利润。
            """)
    else:
        st.warning("未能获取到有效的股票数据，请检查代码输入是否正确（如 A股 000001，美股 AAPL）。")
else:
    st.info("👈 请在左侧输入股票代码开始量价深度分析。")
