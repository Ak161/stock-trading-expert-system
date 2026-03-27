import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import openai

# =========================
# 页面配置与专业UI
# =========================
st.set_page_config(layout="wide", page_title="交易员：量价趋势洞察系统")

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
    .tooltip { border-bottom: 1px dotted #999; cursor: help; }
    </style>
    """, unsafe_allow_html=True)

st.title("🦅 老练交易员：量价趋势洞察系统 (Pro)")
st.markdown("""
本系统模拟职业交易员的深度决策逻辑，动态分析趋势、量价、资金流向、技术形态及背离信号，并结合AI给出实战建议。
> **核心原则**：趋势定方向，量价定强弱，结构定买卖，风控定仓位。
""")

# =========================
# 侧边栏控制台
# =========================
st.sidebar.header("🛠️ 交易控制台")
ticker_input = st.sidebar.text_input("输入标的代码 (如 000001, AAPL)", "000001")
period_select = st.sidebar.selectbox("回测周期 (显示长度)", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

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

# AI 分析开关
st.sidebar.subheader("🤖 AI 智能分析")
enable_ai = st.sidebar.checkbox("启用AI深度分析 (deepseek-v3.2)", value=False)
if enable_ai:
    ai_api_key = st.sidebar.text_input("AIHubMix API Key", type="password", value="<AIHUBMIX_API_KEY>")
    ai_base_url = st.sidebar.text_input("API Base URL", value="https://aihubmix.com/v1")
    ai_model = st.sidebar.selectbox("选择模型", ["deepseek-v3.2", "gpt-4", "claude-3-opus", "gemini-pro"], index=0)

# 帮助文档折叠
with st.sidebar.expander("📖 使用帮助"):
    st.markdown("""
    **指标说明**  
    - **均线**: 趋势方向，多头排列(MA5>MA10>MA20)看涨  
    - **MACD**: 快慢线金叉/死叉判断多空  
    - **RSI**: >70超买，<30超卖  
    - **乖离率**: 股价偏离均线程度，过大易回归  
    - **CMF**: 资金流向，>0.1流入，<-0.1流出  
    - **MFI**: 资金流量指数，>70超买，<30超卖  
    - **OBV背离**: 价格与量能不一致时预示反转  
    - **K线形态**: 锤头线、吞没形态等反转信号  
    - **价格形态**: 箱体、三角形、旗形整理  
    - **止损**: 基于关键支撑和ATR动态计算  
    """)

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
        pe = info.get('trailingPE')
        pb = info.get('priceToBook')
        return name, pe, pb
    except:
        return ticker, None, None

@st.cache_data(ttl=3600)
def get_data(ticker, period="1y", interval="1d"):
    try:
        ticker_norm = normalize_ticker(ticker)
        df = yf.download(ticker_norm, period=period, interval=interval, progress=False)
        if df.empty:
            st.warning(f"未获取到数据，请检查代码 {ticker} 是否正确，或尝试其他周期。")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except Exception as e:
        st.error(f"数据获取失败 ({ticker}): {e}")
        return None

# =========================
# 高级指标计算
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

    # 成交量均线
    df["VOL_MA5"] = volume.rolling(5).mean()
    df["VOL_MA10"] = volume.rolling(10).mean()
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

    # 乖离率 BIAS
    df["BIAS"] = (close - df["MA20"]) / df["MA20"] * 100

    # ATR
    high_low = high - low
    high_prev_close = np.abs(high - close.shift())
    low_prev_close = np.abs(low - close.shift())
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # ADX
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

    # CMF
    mf_mult = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mf_volume = mf_mult * volume
    df["CMF"] = mf_volume.rolling(20).sum() / volume.rolling(20).sum()

    # MFI
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    mfi_ratio = positive_flow / negative_flow.replace(0, np.nan)
    df["MFI"] = 100 - (100 / (1 + mfi_ratio))

    # MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    return df

# =========================
# 关键支撑/阻力位识别
# =========================
def find_support_resistance(df, lookback=50):
    recent = df.tail(lookback)
    highs = recent["High"]
    resistance_candidates = []
    for i in range(5, len(highs)-5):
        if highs.iloc[i] == highs.iloc[i-5:i+6].max():
            resistance_candidates.append(highs.iloc[i])
    lows = recent["Low"]
    support_candidates = []
    for i in range(5, len(lows)-5):
        if lows.iloc[i] == lows.iloc[i-5:i+6].min():
            support_candidates.append(lows.iloc[i])

    if resistance_candidates:
        resistance = np.mean(resistance_candidates[-3:])
    else:
        resistance = df["High"].tail(20).max()
    if support_candidates:
        support = np.mean(support_candidates[-3:])
    else:
        support = df["Low"].tail(20).min()
    return support, resistance

# =========================
# 量价组合信号
# =========================
def volume_price_signals(df):
    signals = []
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    vol_ratio = latest["Vol_Ratio"]
    price_change = (latest["Close"] - prev["Close"]) / prev["Close"] * 100

    if price_change > 0 and vol_ratio > 1.2:
        signals.append(f"放量上涨 (涨幅{price_change:.2f}%, 量比{vol_ratio:.2f})")
    elif price_change < 0 and vol_ratio > 1.2:
        signals.append(f"放量下跌 (跌幅{price_change:.2f}%, 量比{vol_ratio:.2f})")

    recent_vol_ratios = df["Vol_Ratio"].tail(3).values
    if all(v > 1.2 for v in recent_vol_ratios):
        signals.append("连续3日放量，动能强劲")

    vol_ma5 = latest["VOL_MA5"]
    vol_ma10 = latest["VOL_MA10"]
    vol_ma20 = latest["VOL_MA20"]
    signals.append(f"成交量均线: MA5={vol_ma5:.0f}, MA10={vol_ma10:.0f}, MA20={vol_ma20:.0f}")

    if price_change > 0 and vol_ratio < 0.8:
        signals.append("量价背离：价涨量缩，上涨乏力")
    elif price_change < 0 and vol_ratio > 1.2:
        signals.append("量价背离：价跌量增，恐慌抛售")

    return signals

# =========================
# K线形态识别
# =========================
def detect_candle_patterns(df):
    patterns = []
    if len(df) < 2:
        return patterns
    today = df.iloc[-1]
    yesterday = df.iloc[-2]

    body = abs(today["Close"] - today["Open"])
    lower_shadow = min(today["Open"], today["Close"]) - today["Low"]
    upper_shadow = today["High"] - max(today["Open"], today["Close"])
    if lower_shadow > 2 * body and upper_shadow < body:
        patterns.append("锤头线（可能反转）")

    if today["Close"] > today["Open"] and yesterday["Close"] < yesterday["Open"]:
        if today["Close"] > yesterday["Open"] and today["Open"] < yesterday["Close"]:
            patterns.append("看涨吞没（底部反转信号）")
    if today["Close"] < today["Open"] and yesterday["Close"] > yesterday["Open"]:
        if today["Open"] > yesterday["Close"] and today["Close"] < yesterday["Open"]:
            patterns.append("看跌吞没（顶部反转信号）")

    return patterns

# =========================
# 形态识别（箱体、三角形、旗形）
# =========================
def detect_price_patterns(df, lookback=30):
    patterns = []
    recent = df.tail(lookback)
    highs = recent["High"]
    lows = recent["Low"]
    close = recent["Close"]

    price_range = highs.max() - lows.min()
    avg_range = recent["ATR"].mean() if "ATR" in recent.columns else price_range
    if price_range < avg_range * 0.5:
        patterns.append("箱体整理 (横盘震荡)")

    recent_highs = highs.tail(10).values
    recent_lows = lows.tail(10).values
    high_decreasing = all(recent_highs[i] > recent_highs[i+1] for i in range(len(recent_highs)-1))
    low_increasing = all(recent_lows[i] < recent_lows[i+1] for i in range(len(recent_lows)-1))
    if high_decreasing and low_increasing:
        patterns.append("对称三角形 (即将突破)")

    if len(recent) >= 20:
        price_change_20 = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] * 100
        if abs(price_change_20) > 10:
            recent_5_change = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100
            if abs(recent_5_change) < 5:
                patterns.append("旗形整理 (趋势中继)")

    return patterns

# =========================
# 背离检测（MACD、RSI、OBV）
# =========================
def detect_macd_divergence(df, lookback=20):
    if len(df) < lookback:
        return "无背离"
    close = df["Close"].tail(lookback)
    macd = df["MACD"].tail(lookback)
    price_peaks = (close == close.rolling(5, center=True).max())
    macd_peaks = (macd == macd.rolling(5, center=True).max())
    price_troughs = (close == close.rolling(5, center=True).min())
    macd_troughs = (macd == macd.rolling(5, center=True).min())

    price_high_idx = close[price_peaks].index
    if len(price_high_idx) >= 2:
        last_high = price_high_idx[-1]
        prev_high = price_high_idx[-2]
        if close[last_high] > close[prev_high] and macd[last_high] <= macd[prev_high]:
            return "MACD顶背离"
    price_low_idx = close[price_troughs].index
    if len(price_low_idx) >= 2:
        last_low = price_low_idx[-1]
        prev_low = price_low_idx[-2]
        if close[last_low] < close[prev_low] and macd[last_low] >= macd[prev_low]:
            return "MACD底背离"
    return "无背离"

def detect_rsi_divergence(df, lookback=20):
    if len(df) < lookback:
        return "无背离"
    close = df["Close"].tail(lookback)
    rsi = df["RSI"].tail(lookback)
    price_peaks = (close == close.rolling(5, center=True).max())
    rsi_peaks = (rsi == rsi.rolling(5, center=True).max())
    price_troughs = (close == close.rolling(5, center=True).min())
    rsi_troughs = (rsi == rsi.rolling(5, center=True).min())

    price_high_idx = close[price_peaks].index
    if len(price_high_idx) >= 2:
        last_high = price_high_idx[-1]
        prev_high = price_high_idx[-2]
        if close[last_high] > close[prev_high] and rsi[last_high] <= rsi[prev_high]:
            return "RSI顶背离"
    price_low_idx = close[price_troughs].index
    if len(price_low_idx) >= 2:
        last_low = price_low_idx[-1]
        prev_low = price_low_idx[-2]
        if close[last_low] < close[prev_low] and rsi[last_low] >= rsi[prev_low]:
            return "RSI底背离"
    return "无背离"

def detect_obv_divergence(df, lookback=20):
    close = df["Close"].tail(lookback)
    obv = df["OBV"].tail(lookback)
    price_peaks = (close == close.rolling(5, center=True).max())
    obv_peaks = (obv == obv.rolling(5, center=True).max())
    price_troughs = (close == close.rolling(5, center=True).min())
    obv_troughs = (obv == obv.rolling(5, center=True).min())

    price_high_idx = close[price_peaks].index
    if len(price_high_idx) >= 2:
        last_high = price_high_idx[-1]
        prev_high = price_high_idx[-2]
        if close[last_high] > close[prev_high] and obv[last_high] <= obv[prev_high]:
            return "OBV顶背离"
    price_low_idx = close[price_troughs].index
    if len(price_low_idx) >= 2:
        last_low = price_low_idx[-1]
        prev_low = price_low_idx[-2]
        if close[last_low] < close[prev_low] and obv[last_low] >= obv[prev_low]:
            return "OBV底背离"
    return "无背离"

# =========================
# 筹码集中度定性描述
# =========================
def describe_chip_concentration(df):
    vol_series = df["Volume"].tail(20)
    vol_cv = vol_series.std() / vol_series.mean() if vol_series.mean() != 0 else 1
    close = df["Close"].iloc[-1]
    high = df["High"].tail(20).max()
    low = df["Low"].tail(20).min()
    price_position = (close - low) / (high - low) if high != low else 0.5

    if "Turnover" in df.columns:
        turnover = df["Turnover"].iloc[-1]
        if turnover > 5:
            concentration = "筹码分散，活跃度高"
        elif turnover > 2:
            concentration = "筹码相对集中，交易活跃"
        else:
            concentration = "筹码高度集中，交投清淡"
    else:
        if vol_cv < 0.5 and price_position > 0.3 and price_position < 0.7:
            concentration = "筹码集中，支撑较强"
        elif vol_cv > 1:
            concentration = "筹码分散，容易大起大落"
        else:
            concentration = "筹码一般，波动中等"
    return concentration

# =========================
# 动态评分与决策
# =========================
def detailed_analysis(df, pe=None, pb=None):
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
    bias = latest["BIAS"]
    adx = latest.get("ADX", 20)
    plus_di = latest.get("+DI", 25)
    minus_di = latest.get("-DI", 25)
    cmf = latest.get("CMF", 0)
    mfi = latest.get("MFI", 50)
    obv = latest["OBV"]
    obv_ma = latest["OBV_MA20"]
    macd = latest.get("MACD", 0)
    macd_signal = latest.get("MACD_signal", 0)
    macd_hist = latest.get("MACD_hist", 0)

    vol_high_th = latest.get("Vol_Ratio_Threshold_High", 1.5)
    vol_low_th = latest.get("Vol_Ratio_Threshold_Low", 0.8)

    # 趋势判断
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

    # 量价结构
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

    # RSI
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

    # 乖离率
    bias_abs = abs(bias)
    if bias_abs > 8:
        bias_status = "严重偏离均线，回归概率大"
        bias_score = 5
    elif bias_abs > 5:
        bias_status = "中度偏离，注意回归"
        bias_score = 10
    else:
        bias_status = "正常偏离"
        bias_score = 15

    # 资金流向
    money_score = 0
    money_detail = ""
    if cmf > 0.1:
        money_status = "资金持续流入"
        money_score += 8
        money_detail += f"CMF={cmf:.2f}>0.1，资金积极介入；"
    elif cmf < -0.1:
        money_status = "资金持续流出"
        money_score += 0
        money_detail += f"CMF={cmf:.2f}<-0.1，资金撤离；"
    else:
        money_status = "资金平衡"
        money_score += 5
        money_detail += f"CMF={cmf:.2f}，资金博弈均衡；"

    if mfi > 70:
        money_status += "，MFI超买"
        money_detail += f"MFI={mfi:.1f}>70，资金过热；"
        money_score -= 2
    elif mfi < 30:
        money_status += "，MFI超卖"
        money_detail += f"MFI={mfi:.1f}<30，资金过度悲观；"
        money_score += 2
    else:
        money_detail += f"MFI={mfi:.1f}正常；"

    obv_div = detect_obv_divergence(df)
    if obv_div == "顶背离":
        money_score -= 5
        money_detail += "⚠️ OBV顶背离，警惕反转；"
    elif obv_div == "底背离":
        money_score += 5
        money_detail += "✅ OBV底背离，潜在买点；"
    else:
        money_detail += "OBV无背离；"

    total_score = trend_score + vpa_score + rsi_score + money_score

    support, resistance = find_support_resistance(df)

    atr = latest.get("ATR", price * 0.02)
    stop_loss_from_support = support * 0.98
    stop_loss_from_atr = price - 2 * atr
    stop_loss = max(stop_loss_from_support, stop_loss_from_atr)
    take_profit = resistance * 0.98 if resistance > price else price + 3 * atr

    risk = price - stop_loss
    reward = take_profit - price
    risk_reward = reward / risk if risk > 0 else 0

    position_size_pct = 0.02 / (risk / price) if risk > 0 else 0
    position_size_pct = min(position_size_pct, 0.5)

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

    chip_concentration = describe_chip_concentration(df)

    volume_price_signals_list = volume_price_signals(df)
    candle_patterns = detect_candle_patterns(df)
    price_patterns = detect_price_patterns(df)
    macd_div = detect_macd_divergence(df)
    rsi_div = detect_rsi_divergence(df)

    return {
        "price": price,
        "change": (price - prev["Close"]) / prev["Close"] * 100,
        "volume": latest["Volume"],
        "vol_ma20": latest["VOL_MA20"],
        "vol_ratio": vol_ratio,
        "pe": pe,
        "pb": pb,
        "chip_concentration": chip_concentration,
        "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60, "ma120": ma120,
        "trend": trend_desc,
        "trend_strength": trend_strength,
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "rsi": rsi,
        "bias": bias,
        "rsi_status": rsi_status,
        "bias_status": bias_status,
        "vpa_signal": vpa_signal,
        "vpa_detail": vpa_detail,
        "cmf": cmf,
        "mfi": mfi,
        "obv_div": obv_div,
        "money_status": money_status,
        "money_detail": money_detail,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "support": support,
        "resistance": resistance,
        "risk_reward": risk_reward,
        "position_size": position_size_pct,
        "status": status,
        "advice": advice,
        "action_color": action_color,
        "trend_score": trend_score,
        "vpa_score": vpa_score,
        "rsi_score": rsi_score,
        "money_score": money_score,
        "bias_score": bias_score,
        "total_score": total_score,
        "atr": atr,
        "volume_price_signals": volume_price_signals_list,
        "candle_patterns": candle_patterns,
        "price_patterns": price_patterns,
        "macd_divergence": macd_div,
        "rsi_divergence": rsi_div,
    }

def get_score(df):
    if df is None or len(df) < 20:
        return 0
    latest = df.iloc[-1]
    score = 0
    if latest["Close"] > latest.get("MA20", 0): score += 15
    if latest.get("MA20", 0) > latest.get("MA60", 0): score += 15
    if latest.get("MA5", 0) > latest.get("MA10", 0): score += 10
    vol_ratio = latest.get("Vol_Ratio", 1)
    vol_high = latest.get("Vol_Ratio_Threshold_High", 1.5)
    if vol_ratio > vol_high: score += 15
    if latest.get("OBV", 0) > latest.get("OBV_MA20", 0): score += 15
    rsi = latest.get("RSI", 50)
    if 40 < rsi < 75: score += 15
    if latest["Close"] > df.iloc[-2]["Close"]: score += 15
    return score

# =========================
# AI 分析函数
# =========================
def ai_analysis(analysis, ticker, stock_name, api_key, base_url, model):
    prompt = f"""
    请基于以下数据对股票 {stock_name} ({ticker}) 进行专业的技术面分析，给出交易建议。

    【基本信息】
    价格: {analysis['price']:.2f} ({analysis['change']:+.2f}%)
    成交量: {analysis['volume']:.0f}，20日均量: {analysis['vol_ma20']:.0f}，量比: {analysis['vol_ratio']:.2f}
    市盈率PE: {analysis['pe'] if analysis['pe'] else 'N/A'}，市净率PB: {analysis['pb'] if analysis['pb'] else 'N/A'}
    筹码集中度: {analysis['chip_concentration']}

    【趋势指标】
    均线排列: MA5={analysis['ma5']:.2f}, MA10={analysis['ma10']:.2f}, MA20={analysis['ma20']:.2f}, MA60={analysis['ma60']:.2f}, MA120={analysis['ma120']:.2f}
    趋势定性: {analysis['trend']} ({analysis['trend_strength']})
    ADX: {analysis['adx']:.1f}，+DI={analysis['plus_di']:.1f}，-DI={analysis['minus_di']:.1f}
    MACD: DIF={analysis['macd']:.3f}, DEA={analysis['macd_signal']:.3f}, 柱状线={analysis['macd_hist']:.3f}

    【超买超卖】
    RSI: {analysis['rsi']:.1f} ({analysis['rsi_status']})
    乖离率BIAS: {analysis['bias']:.2f}% ({analysis['bias_status']})

    【量价资金】
    量价信号: {analysis['vpa_signal']} — {analysis['vpa_detail']}
    CMF: {analysis['cmf']:.3f}, MFI: {analysis['mfi']:.1f}
    OBV背离: {analysis['obv_div']}
    量价组合信号: {', '.join(analysis['volume_price_signals']) if analysis['volume_price_signals'] else '无'}

    【技术形态】
    K线形态: {', '.join(analysis['candle_patterns']) if analysis['candle_patterns'] else '无'}
    价格形态: {', '.join(analysis['price_patterns']) if analysis['price_patterns'] else '无'}

    【背离信号】
    MACD背离: {analysis['macd_divergence']}
    RSI背离: {analysis['rsi_divergence']}
    OBV背离: {analysis['obv_div']}

    【风险与支撑】
    支撑位: {analysis['support']:.2f}, 压力位: {analysis['resistance']:.2f}
    建议止损: {analysis['stop_loss']:.2f}, 目标位: {analysis['take_profit']:.2f}
    盈亏比: {analysis['risk_reward']:.2f}

    请结合以上指标给出综合判断，并说明理由。最后给出明确的交易建议（买入/卖出/观望）及仓位建议。
    """
    try:
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一位经验丰富的交易员，擅长技术分析和基本面分析。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI分析调用失败：{str(e)}"

# =========================
# 主界面
# =========================
if st.sidebar.button("执行深度洞察"):
    with st.spinner("正在解析市场微观结构..."):
        stock_name, pe, pb = get_stock_info(ticker_input)
        df_d_raw = get_data(ticker_input, period_select, "1d")
        df_w_raw = get_data(ticker_input, period_select, "1wk")   # 周线使用相同周期
        df_h_raw = get_data(ticker_input, "1mo", "60m")           # 小时线最多取1个月

    if df_d_raw is not None and len(df_d_raw) > 20:
        df_d = add_advanced_indicators(df_d_raw)
        analysis = detailed_analysis(df_d, pe, pb)

        if analysis:
            # 头部信息
            st.markdown(f'<div class="stock-name">{stock_name} ({ticker_input})</div>', unsafe_allow_html=True)
            col_p1, col_p2 = st.columns([2, 3])
            with col_p1:
                color = "#16a34a" if analysis['change'] < 0 else "#dc2626"
                st.markdown(
                    f'<div class="price-tag">¥{analysis["price"]:.2f} <span style="font-size:1.5rem; color:{color};">{analysis["change"]:+.2f}%</span></div>',
                    unsafe_allow_html=True)
                vol_ratio = analysis['vol_ratio']
                vol_text = "放量" if vol_ratio > 1.2 else "缩量" if vol_ratio < 0.8 else "平量"
                st.write(f"成交量：{analysis['volume']:,.0f} | 20日均量：{analysis['vol_ma20']:,.0f} | {vol_text}（量比{vol_ratio:.2f}）")
                pe_str = f"{analysis['pe']:.2f}" if analysis['pe'] else "N/A"
                pb_str = f"{analysis['pb']:.2f}" if analysis['pb'] else "N/A"
                st.write(f"市盈率(PE)：{pe_str}  市净率(PB)：{pb_str}")
                st.write(f"平均成本参考：MA20={analysis['ma20']:.2f}，MA60={analysis['ma60']:.2f}")
                st.write(f"筹码集中度：{analysis['chip_concentration']}")

            # 多周期评分
            score_d = get_score(df_d)
            score_w = get_score(df_w_raw) if df_w_raw is not None else 0
            score_h = get_score(df_h_raw) if df_h_raw is not None else 0
            avg_score = w_d * score_d + w_w * score_w + w_h * score_h

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("综合操盘评分", f"{avg_score:.1f}", delta=f"{score_d - 50:.1f}")
            m2.metric("趋势定性", analysis['trend'])
            m3.metric("量价结构", analysis['vpa_signal'])
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
            with st.expander("📈 **第一步：趋势判断（均线+MACD+ADX）**", expanded=True):
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.write("**均线排列**")
                    st.markdown(f"""
- MA5: {analysis['ma5']:.2f}
- MA10: {analysis['ma10']:.2f}
- MA20: {analysis['ma20']:.2f}
- MA60: {analysis['ma60']:.2f}
- MA120: {analysis['ma120']:.2f}
- 当前价: {analysis['price']:.2f}
                    """)
                    st.write(f"**排列状态**：{'多头排列' if analysis['ma5']>analysis['ma10']>analysis['ma20'] else '非多头排列'}")
                with col_t2:
                    st.write("**趋势强度**")
                    st.markdown(f"""
- ADX: {analysis['adx']:.1f} (>25为强趋势)
- +DI: {analysis['plus_di']:.1f}
- -DI: {analysis['minus_di']:.1f}
- 结论: {analysis['trend_strength']} {analysis['trend']}
- 得分: {analysis['trend_score']}/30
                    """)
                    st.write("**MACD**")
                    st.markdown(f"""
- DIF: {analysis['macd']:.3f}
- DEA: {analysis['macd_signal']:.3f}
- 柱状线: {analysis['macd_hist']:.3f}
- 状态: {'金叉' if analysis['macd']>analysis['macd_signal'] else '死叉'}
                    """)

            with st.expander("💹 **第二步：量价组合信号**", expanded=False):
                if analysis['volume_price_signals']:
                    for sig in analysis['volume_price_signals']:
                        st.write(f"- {sig}")
                else:
                    st.write("无特殊量价信号")

            with st.expander("📊 **第三步：RSI+乖离率（超买超卖）**", expanded=False):
                st.write(f"RSI: {analysis['rsi']:.1f} — {analysis['rsi_status']}")
                st.write(f"乖离率(BIAS): {analysis['bias']:.2f}% — {analysis['bias_status']}")

            with st.expander("💰 **第四步：资金流向（CMF+MFI+OBV）**", expanded=False):
                st.write(f"CMF: {analysis['cmf']:.3f} ( >0.1流入，<-0.1流出 )")
                st.write(f"MFI: {analysis['mfi']:.1f} ( >70超买，<30超卖 )")
                st.write(f"OBV背离: {analysis['obv_div']}")
                st.write(f"详情: {analysis['money_detail']}")

            with st.expander("📐 **第五步：技术形态识别**", expanded=False):
                st.write("**K线组合**")
                if analysis['candle_patterns']:
                    for pat in analysis['candle_patterns']:
                        st.write(f"- {pat}")
                else:
                    st.write("无明显K线形态")
                st.write("**价格形态**")
                if analysis['price_patterns']:
                    for pat in analysis['price_patterns']:
                        st.write(f"- {pat}")
                else:
                    st.write("无明显价格形态")

            with st.expander("⚠️ **第六步：背离信号**", expanded=False):
                st.write(f"MACD背离: {analysis['macd_divergence']}")
                st.write(f"RSI背离: {analysis['rsi_divergence']}")
                st.write(f"OBV背离: {analysis['obv_div']}")

            with st.expander("🛡️ **第七步：风险与支撑点**", expanded=False):
                st.write(f"关键支撑位: ¥{analysis['support']:.2f}")
                st.write(f"关键压力位: ¥{analysis['resistance']:.2f}")
                st.write(f"止损位: ¥{analysis['stop_loss']:.2f} (基于支撑和ATR)")
                st.write(f"目标位: ¥{analysis['take_profit']:.2f}")
                st.write(f"盈亏比: {analysis['risk_reward']:.2f}")
                st.write(f"建议仓位: {analysis['position_size']:.0%}")
                st.write(f"筹码集中度: {analysis['chip_concentration']}")

            # 综合判断逻辑（人话版）
            st.markdown('<div class="section-title">🧠 综合判断逻辑（人话版）</div>', unsafe_allow_html=True)
            st.markdown(f"""
**先看趋势**：{analysis['trend']}（{analysis['trend_strength']}，MACD{'金叉' if analysis['macd']>analysis['macd_signal'] else '死叉'}）

**看短期热冷**：RSI {analysis['rsi']:.1f}（{analysis['rsi_status']}），乖离率{analysis['bias']:.2f}%（{analysis['bias_status']}）

**看动力**：成交量{analysis['vpa_signal']}，量比{analysis['vol_ratio']:.2f}，OBV{analysis['obv_div']}，CMF={analysis['cmf']:.2f}，MFI={analysis['mfi']:.1f}

**看稳定性**：筹码{analysis['chip_concentration']}，关键支撑¥{analysis['support']:.2f}

**技术形态**：K线{', '.join(analysis['candle_patterns']) if analysis['candle_patterns'] else '无明显信号'}，价格形态{', '.join(analysis['price_patterns']) if analysis['price_patterns'] else '无'}

**背离信号**：MACD{analysis['macd_divergence']}，RSI{analysis['rsi_divergence']}，OBV{analysis['obv_div']}

**设置止损**：建议止损¥{analysis['stop_loss']:.2f}，目标¥{analysis['take_profit']:.2f}，盈亏比{analysis['risk_reward']:.2f}

**最终建议**：{analysis['advice']}
""")

            # AI 分析区块
            if enable_ai:
                st.markdown('<div class="section-title">🤖 AI 智能分析报告</div>', unsafe_allow_html=True)
                if ai_api_key and ai_api_key != "<AIHUBMIX_API_KEY>":
                    with st.spinner("AI 正在深度思考..."):
                        ai_result = ai_analysis(analysis, ticker_input, stock_name, ai_api_key, ai_base_url, ai_model)
                    st.markdown(ai_result)
                else:
                    st.warning("请填写有效的 API Key 以启用 AI 分析。")

            # 图表
            st.markdown('<div class="section-title">📈 量价图谱</div>', unsafe_allow_html=True)
            days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 252, "2y": 504, "5y": 1260}
            display_df = df_d.tail(days_map.get(period_select, 252))

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=display_df.index, open=display_df["Open"], high=display_df["High"],
                                         low=display_df["Low"], close=display_df["Close"], name="K线"), row=1, col=1)
            for ma, color in zip(["MA5", "MA20", "MA60"], ["#6366f1", "#f59e0b", "#10b981"]):
                if ma in display_df.columns:
                    fig.add_trace(go.Scatter(x=display_df.index, y=display_df[ma], name=ma, line=dict(width=1.5, color=color)), row=1, col=1)

            v_colors = ['#dc2626' if display_df.iloc[i]['Close'] >= display_df.iloc[i]['Open'] else '#16a34a' for i in range(len(display_df))]
            fig.add_trace(go.Bar(x=display_df.index, y=display_df["Volume"], name="成交量", marker_color=v_colors), row=2, col=1)
            if "VOL_MA20" in display_df.columns:
                fig.add_trace(go.Scatter(x=display_df.index, y=display_df["VOL_MA20"], name="量均线", line=dict(color="#f59e0b")), row=2, col=1)

            fig.update_layout(height=800, template="plotly_white", showlegend=True, xaxis_rangeslider_visible=False,
                              margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.caption("💡 数据缓存1小时，如需最新数据请等待或重启应用。")
    else:
        st.warning("数据不足（至少需要20个交易日），请尝试其他标的或稍后再试。")
else:
    st.info("👈 在左侧控制台输入代码开始分析。")
