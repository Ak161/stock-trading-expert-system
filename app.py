# -*- coding: utf-8 -*-
import os
import re
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from openai import OpenAI

# =========================
# 常量
# =========================
CHART_DAYS = {
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 252,
    "2y": 504,
    "5y": 1260,
}
NEW_TOKEN_MODEL_PREFIX = ("gpt-5", "gpt-4.1", "o3", "o4")

# 交易模式配置
TRADE_MODES = {
    "趋势": {"atr_mult": 2.5, "rsi_ob": 80, "rsi_os": 20, "stop_atr": 2.0, "take_profit_atr": 3.0},
    "短线": {"atr_mult": 1.5, "rsi_ob": 75, "rsi_os": 25, "stop_atr": 1.5, "take_profit_atr": 2.2},
    "超短": {"atr_mult": 1.2, "rsi_ob": 70, "rsi_os": 30, "stop_atr": 1.0, "take_profit_atr": 1.5},
    "波段": {"atr_mult": 2.0, "rsi_ob": 85, "rsi_os": 15, "stop_atr": 2.0, "take_profit_atr": 2.8},
}

# =========================
# 页面配置
# =========================
st.set_page_config(layout="wide", page_title="交易员：量价趋势洞察系统")

# 高阶样式（保持不变）
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400;14..32,500;14..32,600;14..32,700&display=swap');
    * { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
    .stApp { background: radial-gradient(circle at 10% 20%, rgba(245,247,250,1) 0%, rgba(235,240,250,1) 100%); }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }
    .glass-card {
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(12px);
        border-radius: 32px;
        padding: 1.2rem 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.02);
        border: 1px solid rgba(255,255,255,0.5);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover { transform: translateY(-2px); box-shadow: 0 16px 40px rgba(0,0,0,0.12); background: rgba(255,255,255,0.85); }
    .risk-box {
        background-color: #fee2e2;
        border-left: 5px solid #ef4444;
        border-radius: 16px;
        padding: 0.8rem 1.2rem;
        margin: 1rem 0;
        color: #991b1b;
        font-weight: 500;
    }
    .hot-tag {
        display: inline-block;
        background: linear-gradient(135deg, #f97316, #ef4444);
        color: white;
        border-radius: 40px;
        padding: 0.2rem 0.8rem;
        font-size: 0.7rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .info-tag { background: #3b82f6; }
    .stock-header { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
    .stock-name { font-size: 1.3rem; font-weight: 600; color: #1e293b; letter-spacing: -0.01em; }
    .stock-meta { font-size: 0.85rem; color: #475569; background: rgba(0,0,0,0.04); padding: 4px 10px; border-radius: 40px; }
    .price-tag { font-size: 2.8rem; font-weight: 700; line-height: 1.1; color: #0f172a; letter-spacing: -0.02em; }
    .metrics-grid { display: flex; flex-wrap: wrap; gap: 16px; margin: 20px 0 12px; }
    .metric-card {
        background: white;
        border-radius: 24px;
        padding: 12px 20px;
        flex: 1;
        min-width: 120px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
        border: 1px solid #eef2ff;
        transition: all 0.2s;
    }
    .metric-card:hover { border-color: #cbd5e1; background: #fafcff; }
    .metric-label { font-size: 0.8rem; font-weight: 500; color: #64748b; text-transform: uppercase; letter-spacing: 0.02em; }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #0f172a; margin-top: 4px; }
    .stButton > button {
        border-radius: 60px;
        font-weight: 500;
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        color: white;
        border: none;
        padding: 0.5rem 1.2rem;
        transition: all 0.2s;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 8px 20px rgba(37,99,235,0.3); }
    section[data-testid="stSidebar"] { background: rgba(255,255,255,0.92); backdrop-filter: blur(8px); border-right: 1px solid rgba(203,213,225,0.4); }
    .js-plotly-plot .plotly .main-svg { border-radius: 24px; background: rgba(255,255,255,0.7); box-shadow: 0 4px 12px rgba(0,0,0,0.03); }
    hr { margin: 1rem 0; border-color: #e2e8f0; }
    .chat-wrap { background: rgba(255,255,255,0.75); backdrop-filter: blur(12px); border-radius: 28px; padding: 1rem; border: 1px solid rgba(255,255,255,0.5); }
    h1 { font-weight: 800; background: linear-gradient(120deg, #1e3a8a, #3b82f6); background-clip: text; -webkit-background-clip: text; color: transparent; margin-bottom: 0; }
</style>
""", unsafe_allow_html=True)

st.title("🦅 交易员：量价趋势洞察系统")
st.caption("核心原则：趋势定方向，量价定强弱，结构定买卖，风控定仓位。")

# =========================
# 工具函数
# =========================
def is_a_share_code(ticker: str) -> bool:
    return bool(re.fullmatch(r"\d{6}", ticker.strip().upper()))

def normalize_ticker_for_yf(ticker: str) -> str:
    t = ticker.strip().upper()
    if is_a_share_code(t):
        if t.startswith(("60", "68")):
            return f"{t}.SS"
        if t.startswith(("00", "30")):
            return f"{t}.SZ"
    return t

def market_colors(ticker: str):
    if is_a_share_code(ticker):
        return "#dc2626", "#16a34a"
    return "#16a34a", "#dc2626"

def use_max_completion_tokens(model: str) -> bool:
    if "deepseek" in model.lower():
        return False
    return isinstance(model, str) and model.startswith(NEW_TOKEN_MODEL_PREFIX)

def standardize_ohlcv(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    rename_map = {
        "日期": "Date", "时间": "Date", "开盘": "Open", "最高": "High",
        "最低": "Low", "收盘": "Close", "成交量": "Volume", "成交量(股)": "Volume",
        "成交量(手)": "Volume",
    }
    out.rename(columns=rename_map, inplace=True)
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.set_index("Date")
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index, errors="coerce")
        except Exception:
            return None
    need_cols = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in out.columns for c in need_cols):
        return None
    out = out[need_cols].copy()
    for c in need_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    return out

# 简单重试装饰器
def retry(max_attempts=2, delay=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_info(ticker: str, source: str = "yfinance"):
    name = ticker.strip().upper()
    pe = None
    pb = None
    if source == "akshare" and is_a_share_code(name):
        try:
            import akshare as ak
            info_df = ak.stock_individual_info_em(symbol=name)
            if not info_df.empty and {"item", "value"}.issubset(info_df.columns):
                info_map = dict(zip(info_df["item"], info_df["value"]))
                name = info_map.get("股票简称", name)
        except Exception:
            pass
    try:
        yf_ticker = yf.Ticker(normalize_ticker_for_yf(ticker))
        info = yf_ticker.info or {}
        name = info.get("longName") or info.get("shortName") or name
        pe = info.get("trailingPE")
        pb = info.get("priceToBook")
    except Exception:
        pass
    return name, pe, pb

@st.cache_data(ttl=900, show_spinner=False)
def get_data_yfinance(ticker: str, period: str = "1y", interval: str = "1d"):
    try:
        t = normalize_ticker_for_yf(ticker)
        raw = yf.download(
            t, period=period, interval=interval, progress=False,
            auto_adjust=False, prepost=False, threads=False,
        )
        return standardize_ohlcv(raw)
    except Exception:
        return None

@st.cache_data(ttl=900, show_spinner=False)
def get_data_akshare(ticker: str, period: str = "1y", interval: str = "1d"):
    import akshare as ak
    code = ticker.strip().upper()
    if not is_a_share_code(code):
        return None
    days = CHART_DAYS.get(period, 252)
    now = datetime.now()
    if interval in ("1d", "1wk"):
        ak_period = "daily" if interval == "1d" else "weekly"
        # 多拉一些数据避免边界不足
        start_date = (now - timedelta(days=int(days * 1.5 + 30))).strftime("%Y%m%d")
        end_date = now.strftime("%Y%m%d")
        raw = ak.stock_zh_a_hist(
            symbol=code, period=ak_period, start_date=start_date,
            end_date=end_date, adjust="qfq",
        )
        if raw is None or raw.empty:
            return None
        raw = raw.rename(columns={
            "日期": "Date", "开盘": "Open", "最高": "High", "最低": "Low",
            "收盘": "Close", "成交量": "Volume",
        })
        df = standardize_ohlcv(raw)
        if df is None:
            return None
        if interval == "1d":
            return df.tail(days + 20)
        weekly_bars = max(int(days / 5) + 10, 52)
        return df.tail(weekly_bars)
    elif interval == "60m":
        start_dt = (now - timedelta(days=40)).strftime("%Y-%m-%d 09:30:00")
        end_dt = now.strftime("%Y-%m-%d 15:00:00")
        raw = ak.stock_zh_a_hist_min_em(
            symbol=code, period="60", start_date=start_dt,
            end_date=end_dt, adjust="",
        )
        if raw is None or raw.empty:
            return None
        raw = raw.rename(columns={
            "时间": "Date", "日期": "Date", "开盘": "Open", "最高": "High",
            "最低": "Low", "收盘": "Close", "成交量": "Volume",
        })
        df = standardize_ohlcv(raw)
        if df is None:
            return None
        return df.tail(320)
    return None

def get_data(ticker: str, period: str = "1y", interval: str = "1d", source: str = "yfinance"):
    notes = []
    df = None
    if source == "akshare":
        try:
            df = get_data_akshare(ticker, period, interval)
            if df is None or df.empty:
                notes.append(f"akshare {interval} 无有效数据，已回退 yfinance。")
                # 对于小时线，yfinance不支持A股，直接跳过回退
                if interval == "60m" and is_a_share_code(ticker):
                    notes.append("⚠️ yfinance 不支持 A 股小时线，无法回退。请使用日线或更换数据源。")
                    return None, notes
                df = get_data_yfinance(ticker, period, interval)
        except ImportError:
            notes.append("akshare 未安装（pip install akshare），已回退 yfinance。")
            if interval == "60m" and is_a_share_code(ticker):
                notes.append("⚠️ yfinance 不支持 A 股小时线，无法回退。")
                return None, notes
            df = get_data_yfinance(ticker, period, interval)
        except Exception as e:
            notes.append(f"akshare {interval} 获取失败：{e}，已回退 yfinance。")
            if interval == "60m" and is_a_share_code(ticker):
                notes.append("⚠️ yfinance 不支持 A 股小时线，无法回退。")
                return None, notes
            df = get_data_yfinance(ticker, period, interval)
    else:
        if interval == "60m" and is_a_share_code(ticker):
            notes.append("yfinance 不支持 A 股小时线，无法获取数据。")
            return None, notes
        df = get_data_yfinance(ticker, period, interval)
    if df is None or df.empty:
        notes.append(f"{source} {interval} 无有效数据。")
    if df is not None and len(df) > 0:
        return df, notes
    return None, notes

def resample_to_weekly(df_daily: pd.DataFrame):
    if df_daily is None or df_daily.empty:
        return None
    w = (df_daily.resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna())
    return w

# ----- 技术指标函数（完整保留）-----
def add_advanced_indicators(df: pd.DataFrame):
    if df is None or len(df) < 20:
        return df
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    volume = out["Volume"]
    for w in [5, 10, 20, 60, 120]:
        out[f"MA{w}"] = close.rolling(w, min_periods=max(2, w // 2)).mean()
    for w in [5, 10, 20]:
        out[f"VOL_MA{w}"] = volume.rolling(w, min_periods=max(2, w // 2)).mean()
    out["Vol_Ratio"] = volume / out["VOL_MA20"].replace(0, np.nan)
    vr_mean = out["Vol_Ratio"].rolling(60, min_periods=20).mean()
    vr_std = out["Vol_Ratio"].rolling(60, min_periods=20).std()
    out["Vol_Ratio_Threshold_High"] = (vr_mean + vr_std).fillna(1.5)
    out["Vol_Ratio_Threshold_Low"] = (vr_mean - vr_std).fillna(0.8).clip(lower=0.3)
    direction = np.sign(close.diff()).fillna(0)
    out["OBV"] = (direction * volume).cumsum()
    out["OBV_MA20"] = out["OBV"].rolling(20, min_periods=5).mean()
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_down = down.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    out["RSI"] = (100 - 100 / (1 + rs)).clip(0, 100)
    out["BIAS"] = (close - out["MA20"]) / out["MA20"] * 100
    tr = pd.concat([(high - low).abs(), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    out["ATR"] = tr.rolling(14, min_periods=5).mean()
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr14 = tr.rolling(14, min_periods=5).sum().replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=out.index).rolling(14, min_periods=5).sum() / tr14
    minus_di = 100 * pd.Series(minus_dm, index=out.index).rolling(14, min_periods=5).sum() / tr14
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    out["+DI"] = plus_di
    out["-DI"] = minus_di
    out["ADX"] = dx.rolling(14, min_periods=5).mean()
    mf_mult = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mf_vol = mf_mult * volume
    out["CMF"] = (mf_vol.rolling(20, min_periods=5).sum() / volume.rolling(20, min_periods=5).sum().replace(0, np.nan))
    tp = (high + low + close) / 3
    money_flow = tp * volume
    pos_flow = money_flow.where(tp > tp.shift(1), 0.0)
    neg_flow = money_flow.where(tp < tp.shift(1), 0.0).abs()
    pos_sum = pos_flow.rolling(14, min_periods=5).sum()
    neg_sum = neg_flow.rolling(14, min_periods=5).sum()
    mfi_ratio = pos_sum / neg_sum.replace(0, np.nan)
    out["MFI"] = (100 - 100 / (1 + mfi_ratio)).clip(0, 100)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out

def find_support_resistance(df: pd.DataFrame, lookback: int = 80):
    recent = df.tail(lookback)
    highs = recent["High"]
    lows = recent["Low"]
    resistance_candidates = []
    support_candidates = []
    if len(recent) >= 11:
        for i in range(5, len(recent) - 5):
            h_slice = highs.iloc[i - 5: i + 6]
            l_slice = lows.iloc[i - 5: i + 6]
            if highs.iloc[i] == h_slice.max():
                resistance_candidates.append(highs.iloc[i])
            if lows.iloc[i] == l_slice.min():
                support_candidates.append(lows.iloc[i])
    resistance = np.mean(resistance_candidates[-3:]) if resistance_candidates else highs.tail(20).max()
    support = np.mean(support_candidates[-3:]) if support_candidates else lows.tail(20).min()
    return float(support), float(resistance)

def volume_price_signals(df: pd.DataFrame):
    signals = []
    if df is None or len(df) < 3:
        return signals
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    vol_ratio = latest.get("Vol_Ratio", np.nan)
    price_change = (latest["Close"] - prev["Close"]) / prev["Close"] * 100
    if pd.notna(vol_ratio):
        if price_change > 0 and vol_ratio > 1.2:
            signals.append(f"放量上涨（涨幅{price_change:.2f}% / 量比{vol_ratio:.2f}）")
        elif price_change < 0 and vol_ratio > 1.2:
            signals.append(f"放量下跌（跌幅{price_change:.2f}% / 量比{vol_ratio:.2f}）")
        recent_vr = df["Vol_Ratio"].tail(3).dropna().values
        if len(recent_vr) == 3 and all(v > 1.2 for v in recent_vr):
            signals.append("连续3期放量，动能较强")
        if price_change > 0 and vol_ratio < 0.8:
            signals.append("价涨量缩，上攻动能偏弱")
        elif price_change < 0 and vol_ratio > 1.2:
            signals.append("价跌量增，短线抛压较重")
    signals.append(
        f"量均线：MA5={latest.get('VOL_MA5', np.nan):.0f}, MA10={latest.get('VOL_MA10', np.nan):.0f}, MA20={latest.get('VOL_MA20', np.nan):.0f}"
    )
    return signals

def detect_candle_patterns(df: pd.DataFrame):
    patterns = []
    if df is None or len(df) < 2:
        return patterns
    t = df.iloc[-1]
    y = df.iloc[-2]
    body = abs(t["Close"] - t["Open"])
    lower_shadow = min(t["Open"], t["Close"]) - t["Low"]
    upper_shadow = t["High"] - max(t["Open"], t["Close"])
    if body > 0 and lower_shadow > 2 * body and upper_shadow < body:
        patterns.append("锤头线（潜在反转）")
    if t["Close"] > t["Open"] and y["Close"] < y["Open"]:
        if t["Close"] > y["Open"] and t["Open"] < y["Close"]:
            patterns.append("看涨吞没")
    if t["Close"] < t["Open"] and y["Close"] > y["Open"]:
        if t["Open"] > y["Close"] and t["Close"] < y["Open"]:
            patterns.append("看跌吞没")
    return patterns

def detect_price_patterns(df: pd.DataFrame, lookback: int = 30):
    patterns = []
    if df is None or len(df) < lookback:
        return patterns
    recent = df.tail(lookback)
    highs = recent["High"]
    lows = recent["Low"]
    close = recent["Close"]
    price_range = highs.max() - lows.min()
    avg_price = close.mean()
    if avg_price > 0 and price_range / avg_price < 0.08:
        patterns.append("箱体整理")
    if len(recent) >= 15:
        x = np.arange(15)
        h_slope = np.polyfit(x, highs.tail(15).values, 1)[0]
        l_slope = np.polyfit(x, lows.tail(15).values, 1)[0]
        if h_slope < 0 and l_slope > 0:
            patterns.append("收敛三角形（待突破）")
    if len(recent) >= 20:
        chg20 = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] * 100
        chg5 = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100
        if abs(chg20) > 10 and abs(chg5) < 4:
            patterns.append("旗形整理（趋势中继）")
    return patterns

def _last_two_pivots(series: pd.Series, mode: str = "high", window: int = 5):
    roll = series.rolling(window, center=True, min_periods=window)
    if mode == "high":
        pivots = series[series == roll.max()].dropna()
    else:
        pivots = series[series == roll.min()].dropna()
    if len(pivots) >= 2:
        return pivots.index[-2], pivots.index[-1]
    return None, None

def detect_divergence(price: pd.Series, indicator: pd.Series, lookback: int = 30):
    p = price.tail(lookback).dropna()
    i = indicator.reindex(p.index).dropna()
    p = p.reindex(i.index)
    if len(p) < 10 or len(i) < 10:
        return "无背离"
    h1, h2 = _last_two_pivots(p, mode="high")
    if h1 is not None and h2 is not None and pd.notna(i.loc[h1]) and pd.notna(i.loc[h2]):
        if p.loc[h2] > p.loc[h1] and i.loc[h2] <= i.loc[h1]:
            return "顶背离"
    l1, l2 = _last_two_pivots(p, mode="low")
    if l1 is not None and l2 is not None and pd.notna(i.loc[l1]) and pd.notna(i.loc[l2]):
        if p.loc[l2] < p.loc[l1] and i.loc[l2] >= i.loc[l1]:
            return "底背离"
    return "无背离"

def detect_macd_divergence(df: pd.DataFrame, lookback: int = 30):
    return detect_divergence(df["Close"], df["MACD"], lookback=lookback)

def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 30):
    return detect_divergence(df["Close"], df["RSI"], lookback=lookback)

def detect_obv_divergence(df: pd.DataFrame, lookback: int = 30):
    return detect_divergence(df["Close"], df["OBV"], lookback=lookback)

def describe_chip_concentration(df: pd.DataFrame):
    vol = df["Volume"].tail(20)
    if vol.mean() == 0:
        return "筹码一般"
    vol_cv = vol.std() / vol.mean()
    close = df["Close"].iloc[-1]
    high20 = df["High"].tail(20).max()
    low20 = df["Low"].tail(20).min()
    pos = (close - low20) / (high20 - low20) if high20 != low20 else 0.5
    if vol_cv < 0.5 and 0.3 < pos < 0.7:
        return "筹码集中，支撑相对较强"
    if vol_cv > 1.0:
        return "筹码分散，波动可能较大"
    return "筹码中性，波动中等"

def get_structured_signals(df: pd.DataFrame):
    signals = []
    if df is None or len(df) < 3:
        return signals
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    vol_ratio = latest.get("Vol_Ratio", 1.0)
    price_change = (latest["Close"] - prev["Close"]) / prev["Close"] * 100
    if price_change > 0 and vol_ratio > 1.2:
        signals.append({"type": "bullish", "name": "放量上涨", "strength": min(1.0, vol_ratio / 2.0)})
    elif price_change < 0 and vol_ratio > 1.2:
        signals.append({"type": "bearish", "name": "放量下跌", "strength": min(1.0, vol_ratio / 2.0)})
    elif price_change > 0 and vol_ratio < 0.8:
        signals.append({"type": "neutral", "name": "价涨量缩", "strength": 0.4})
    elif price_change < 0 and vol_ratio < 0.8:
        signals.append({"type": "bullish", "name": "缩量下跌", "strength": 0.6})
    if latest["Close"] > latest["MA20"]:
        signals.append({"type": "bullish", "name": "站上MA20", "strength": 0.7})
    else:
        signals.append({"type": "bearish", "name": "跌破MA20", "strength": 0.7})
    if latest["MACD"] > latest["MACD_signal"]:
        signals.append({"type": "bullish", "name": "MACD金叉", "strength": 0.8})
    elif latest["MACD"] < latest["MACD_signal"]:
        signals.append({"type": "bearish", "name": "MACD死叉", "strength": 0.8})
    return signals

def generate_trade_signal(df: pd.DataFrame):
    if df is None or len(df) < 20:
        return []
    triggers = []
    latest = df.iloc[-1]
    high_20 = df["High"].rolling(20).max()
    if latest["Close"] > high_20.iloc[-2]:
        triggers.append("突破20日新高")
    if latest["Close"] > latest["MA20"] and latest["Vol_Ratio"] < 0.8:
        triggers.append("缩量回踩MA20")
    if latest["Vol_Ratio"] > 1.5 and latest["Close"] > df["Close"].shift(1).iloc[-1]:
        triggers.append("放量启动")
    if (df["MA5"].iloc[-1] > df["MA10"].iloc[-1] > df["MA20"].iloc[-1] and
            not (df["MA5"].iloc[-2] > df["MA10"].iloc[-2] > df["MA20"].iloc[-2])):
        triggers.append("均线多头排列形成")
    return triggers

def analyze_main_behavior(df: pd.DataFrame, analysis: dict):
    if df is None or analysis is None:
        return {"behavior": "无法判断", "confidence": 0, "description": "数据不足"}
    latest = df.iloc[-1]
    vol_ratio = analysis.get("vol_ratio", 1.0)
    price_change = analysis.get("change", 0)
    cmf = analysis.get("cmf", 0)
    obv_div = analysis.get("obv_div", "")
    rsi = analysis.get("rsi", 50)
    macd_hist = analysis.get("macd_hist", 0)
    chip = analysis.get("chip_concentration", "")
    if abs(price_change) < 2 and vol_ratio > 0.8 and vol_ratio < 1.5 and cmf > 0.05 and "集中" in chip:
        return {"behavior": "吸筹阶段", "confidence": 0.7, "description": "主力低位收集筹码，价格波动小，资金温和流入。"}
    if price_change < 0 and vol_ratio < 0.8 and cmf > -0.1 and "集中" in chip:
        return {"behavior": "洗盘阶段", "confidence": 0.75, "description": "主力清洗浮筹，下跌缩量，可关注止跌信号。"}
    if price_change > 0 and vol_ratio > 1.2 and cmf > 0.1 and rsi > 50 and macd_hist > 0:
        return {"behavior": "拉升阶段", "confidence": 0.85, "description": "主力主动拉升，量价配合良好，趋势强劲。"}
    if vol_ratio > 1.5 and (abs(price_change) < 1 or price_change < 0) and cmf < -0.05 and "顶背离" in obv_div:
        return {"behavior": "出货阶段", "confidence": 0.8, "description": "主力高位派发，量能异常，风险较大。"}
    return {"behavior": "主力观望/分歧", "confidence": 0.5, "description": "主力无明显动作，跟随市场波动。"}

def evaluate_dip_entry(df: pd.DataFrame, analysis: dict, mode: str):
    if df is None or analysis is None or len(df) < 20:
        return {"score": 0, "action": "不试仓", "reason": "数据不足", "suggested_position": 0, "stop_loss": None, "target": None}
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price_change = (latest["Close"] - prev["Close"]) / prev["Close"] * 100
    vol_ratio = analysis.get("vol_ratio", 1.0)
    rsi = analysis.get("rsi", 50)
    support = analysis.get("support", latest["Close"] * 0.95)
    price = analysis.get("price", latest["Close"])
    atr = analysis.get("atr", price * 0.02)
    mode_config = TRADE_MODES.get(mode, TRADE_MODES["短线"])
    atr_mult = mode_config["atr_mult"]
    if price_change > -2:
        return {"score": 0, "action": "不试仓", "reason": "跌幅不足，非急跌", "suggested_position": 0, "stop_loss": None, "target": None}
    score = 0
    reasons = []
    if vol_ratio < 0.8:
        score += 30
        reasons.append("缩量下跌，洗盘可能")
    else:
        score -= 20
        reasons.append("放量下跌，抛压重")
    distance_to_support = (price - support) / price * 100
    if 0 < distance_to_support < 3:
        score += 30
        reasons.append(f"靠近支撑位 {support:.2f}（距离{distance_to_support:.1f}%）")
    elif distance_to_support < 0:
        score -= 40
        reasons.append("已跌破支撑，风险大")
    else:
        score += 10
        reasons.append("离支撑尚远，可能继续下跌")
    if rsi < 30:
        score += 20
        reasons.append(f"RSI={rsi:.1f}超卖区")
    elif rsi < 40:
        score += 10
        reasons.append(f"RSI={rsi:.1f}接近超卖")
    cmf = analysis.get("cmf", 0)
    if cmf > 0:
        score += 15
        reasons.append("CMF为正，资金未大幅流出")
    else:
        score -= 10
        reasons.append("CMF为负，资金流出")
    score = max(0, min(100, score))
    if score >= 70:
        action = "积极试仓"
        suggested_position = 0.05
        stop_loss = price - atr_mult * atr
        target = price + atr_mult * atr * 1.5
    elif score >= 50:
        action = "谨慎试仓"
        suggested_position = 0.02
        stop_loss = price - atr_mult * atr * 0.8
        target = price + atr_mult * atr * 1.2
    else:
        action = "不试仓"
        suggested_position = 0
        stop_loss = None
        target = None
    return {
        "score": score,
        "action": action,
        "reason": "；".join(reasons),
        "suggested_position": suggested_position,
        "stop_loss": stop_loss,
        "target": target,
    }

def detailed_analysis(df: pd.DataFrame, pe=None, pb=None, mode="短线"):
    if df is None or len(df) < 20:   # 降低门槛到20根
        return None
    mode_config = TRADE_MODES.get(mode, TRADE_MODES["短线"])
    atr_multiplier = mode_config["atr_mult"]
    rsi_ob = mode_config["rsi_ob"]
    rsi_os = mode_config["rsi_os"]
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(latest["Close"])
    prev_close = float(prev["Close"])
    change_pct = (price - prev_close) / prev_close * 100
    def _v(col, default):
        val = latest.get(col, np.nan)
        if pd.isna(val):
            if col in df.columns and not df[col].dropna().empty:
                return float(df[col].dropna().iloc[-1])
            return float(default)
        return float(val)
    ma5 = _v("MA5", price)
    ma10 = _v("MA10", price)
    ma20 = _v("MA20", price)
    ma60 = _v("MA60", ma20)
    ma120 = _v("MA120", ma60)
    vol_ratio = _v("Vol_Ratio", 1.0)
    vol_high_th = _v("Vol_Ratio_Threshold_High", 1.5)
    vol_low_th = _v("Vol_Ratio_Threshold_Low", 0.8)
    rsi = _v("RSI", 50)
    bias = _v("BIAS", 0)
    adx = _v("ADX", 20)
    plus_di = _v("+DI", 25)
    minus_di = _v("-DI", 25)
    cmf = _v("CMF", 0)
    mfi = _v("MFI", 50)
    macd = _v("MACD", 0)
    macd_signal = _v("MACD_signal", 0)
    macd_hist = _v("MACD_hist", 0)
    is_bull = ma5 > ma10 > ma20
    is_bear = ma5 < ma10 < ma20
    if adx >= 25:
        trend_strength = "强趋势"
    else:
        trend_strength = "弱趋势/震荡"
    if is_bull and price > ma60 and plus_di >= minus_di:
        trend_desc = "强势多头" if adx >= 25 else "多头结构"
    elif is_bear and price < ma60 and minus_di > plus_di:
        trend_desc = "强势空头" if adx >= 25 else "空头结构"
    elif price >= ma20:
        trend_desc = "震荡偏多"
    else:
        trend_desc = "震荡偏空"
    trend_score = 0
    if price > ma20:
        trend_score += 8
    if price > ma60:
        trend_score += 8
    if is_bull:
        trend_score += 8
    if adx >= 25 and plus_di > minus_di:
        trend_score += 6
    elif adx >= 25 and plus_di < minus_di:
        trend_score -= 3
    trend_score = int(np.clip(trend_score, 0, 30))
    if change_pct > 0 and vol_ratio > vol_high_th:
        vpa_signal = "放量进攻"
        vpa_score = 20
        vpa_detail = f"上涨 + 放量，量比{vol_ratio:.2f} > 动态上轨{vol_high_th:.2f}"
    elif change_pct < 0 and vol_ratio < vol_low_th and price > ma20:
        vpa_signal = "缩量回踩"
        vpa_score = 15
        vpa_detail = f"回踩缩量，量比{vol_ratio:.2f} < 动态下轨{vol_low_th:.2f}"
    elif change_pct > 0 and vol_ratio < vol_low_th:
        vpa_signal = "价涨量缩"
        vpa_score = 6
        vpa_detail = "上攻动能一般，警惕冲高回落"
    elif change_pct < 0 and vol_ratio > vol_high_th:
        vpa_signal = "放量下杀"
        vpa_score = 2
        vpa_detail = "短线恐慌盘释放，注意防守"
    else:
        vpa_signal = "中性"
        vpa_score = 10
        vpa_detail = f"量比{vol_ratio:.2f}，处于常态区间"
    if rsi_os <= rsi <= rsi_ob:
        rsi_status = "健康强势"
        rsi_score = 15
    elif rsi > rsi_ob:
        rsi_status = "超买预警"
        rsi_score = 5
    elif rsi < rsi_os:
        rsi_status = "超卖反弹区"
        rsi_score = 8
    else:
        rsi_status = "中性"
        rsi_score = 10
    bias_abs = abs(bias)
    if bias_abs <= 5:
        bias_status = "偏离正常"
        bias_score = 10
    elif bias_abs <= 8:
        bias_status = "中度偏离"
        bias_score = 6
    else:
        bias_status = "严重偏离，警惕回归"
        bias_score = 2
    obv_div_short = detect_obv_divergence(df)
    obv_div_text = f"OBV{obv_div_short}" if obv_div_short != "无背离" else "无背离"
    money_score = 0
    money_detail = []
    if cmf > 0.1:
        money_score += 8
        money_detail.append(f"CMF={cmf:.2f}，资金偏流入")
        money_status = "资金流入"
    elif cmf < -0.1:
        money_score += 2
        money_detail.append(f"CMF={cmf:.2f}，资金偏流出")
        money_status = "资金流出"
    else:
        money_score += 5
        money_detail.append(f"CMF={cmf:.2f}，资金中性")
        money_status = "资金平衡"
    if 40 <= mfi <= 70:
        money_score += 5
        money_detail.append(f"MFI={mfi:.1f}，资金温和")
    elif mfi > 80:
        money_score -= 1
        money_detail.append(f"MFI={mfi:.1f}，过热")
    elif mfi < 25:
        money_score += 2
        money_detail.append(f"MFI={mfi:.1f}，超卖区")
    if obv_div_short == "底背离":
        money_score += 4
        money_detail.append("OBV底背离，潜在转强")
    elif obv_div_short == "顶背离":
        money_score -= 4
        money_detail.append("OBV顶背离，警惕转弱")
    money_score = int(np.clip(money_score, 0, 20))
    money_detail_text = "；".join(money_detail)
    support, resistance = find_support_resistance(df)
    atr = _v("ATR", price * 0.02)
    if atr <= 0:
        atr = price * 0.02
    stop_loss = min(support * 0.98, price - atr_multiplier * atr)
    if stop_loss >= price:
        stop_loss = price - atr_multiplier * atr
    take_profit = max(resistance * 0.99, price + 2.2 * atr)
    if take_profit <= price:
        take_profit = price + 2.2 * atr
    risk = max(price - stop_loss, price * 0.004)
    reward = max(take_profit - price, price * 0.006)
    risk_reward = reward / risk if risk > 0 else 0
    position_size_pct = min(0.02 / (risk / price), 0.5) if risk > 0 else 0
    total_score = int(np.clip(trend_score + vpa_score + rsi_score + bias_score + money_score, 0, 100))
    if total_score >= 75:
        status = "🔥 强力做多"
        advice = f"可考虑分批买入/加仓，建议仓位上限约 {position_size_pct:.0%}。"
        action_color = "success"
    elif total_score >= 60:
        status = "📈 谨慎看多"
        advice = f"结构偏多，可轻中仓跟随，建议仓位约 {position_size_pct:.0%}。"
        action_color = "info"
    elif total_score >= 45:
        status = "🔄 震荡观望"
        advice = "方向不够清晰，建议等待放量突破或关键位确认。"
        action_color = "warning"
    else:
        status = "❄️ 空头回避"
        advice = "趋势弱且资金不佳，优先防守，减少参与。"
        action_color = "error"
    return {
        "price": price,
        "change": change_pct,
        "volume": float(latest["Volume"]),
        "vol_ma20": _v("VOL_MA20", latest["Volume"]),
        "vol_ratio": vol_ratio,
        "pe": pe,
        "pb": pb,
        "chip_concentration": describe_chip_concentration(df),
        "ma5": ma5,
        "ma10": ma10,
        "ma20": ma20,
        "ma60": ma60,
        "ma120": ma120,
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
        "obv_div": obv_div_text,
        "money_status": money_status,
        "money_detail": money_detail_text,
        "stop_loss": float(stop_loss),
        "take_profit": float(take_profit),
        "support": float(support),
        "resistance": float(resistance),
        "risk_reward": float(risk_reward),
        "position_size": float(position_size_pct),
        "status": status,
        "advice": advice,
        "action_color": action_color,
        "trend_score": trend_score,
        "vpa_score": vpa_score,
        "rsi_score": rsi_score,
        "bias_score": bias_score,
        "money_score": money_score,
        "total_score": total_score,
        "atr": atr,
        "volume_price_signals": volume_price_signals(df),
        "candle_patterns": detect_candle_patterns(df),
        "price_patterns": detect_price_patterns(df),
        "macd_divergence": detect_macd_divergence(df),
        "rsi_divergence": detect_rsi_divergence(df),
    }

def get_score(df: pd.DataFrame):
    if df is None or len(df) < 10:
        return 0
    dfi = df.copy()
    if "MA20" not in dfi.columns:
        dfi = add_advanced_indicators(dfi)
    if dfi is None or len(dfi) < 10:
        return 0
    latest = dfi.iloc[-1]
    prev = dfi.iloc[-2] if len(dfi) >= 2 else latest
    score = 0
    if latest["Close"] > latest.get("MA20", np.nan):
        score += 20
    if latest.get("MA20", np.nan) > latest.get("MA60", np.nan):
        score += 15
    if latest.get("MA5", np.nan) > latest.get("MA10", np.nan) > latest.get("MA20", np.nan):
        score += 15
    vr = latest.get("Vol_Ratio", 1)
    if vr > 1.2:
        score += 10
    elif vr < 0.8:
        score += 4
    if latest.get("MACD", 0) > latest.get("MACD_signal", 0):
        score += 15
    rsi = latest.get("RSI", 50)
    if 45 <= rsi <= 70:
        score += 15
    elif rsi < 35:
        score += 8
    if latest.get("CMF", 0) > 0:
        score += 10
    if latest["Close"] > prev["Close"]:
        score += 10
    adx = latest.get("ADX", 20)
    if adx >= 25:
        score += 5
    return int(np.clip(score, 0, 100))

def build_chart(df: pd.DataFrame, period: str, ticker: str, support=None, resistance=None):
    if df is None or df.empty:
        return None
    bars = CHART_DAYS.get(period, 252)
    show_df = df.tail(bars).copy()
    if show_df.empty:
        return None
    up_color, down_color = market_colors(ticker)
    vol_colors = np.where(show_df["Close"] >= show_df["Open"], up_color, down_color)
    macd_bar_colors = np.where(show_df["MACD_hist"] >= 0, up_color, down_color)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.56, 0.18, 0.26],
        specs=[[{}], [{}], [{"secondary_y": True}]],
    )
    fig.add_trace(go.Candlestick(
        x=show_df.index, open=show_df["Open"], high=show_df["High"],
        low=show_df["Low"], close=show_df["Close"], name="K线",
        increasing_line_color=up_color, decreasing_line_color=down_color,
        increasing_fillcolor=up_color, decreasing_fillcolor=down_color,
    ), row=1, col=1)
    for ma, color in [("MA5", "#6366f1"), ("MA20", "#f59e0b"), ("MA60", "#10b981")]:
        if ma in show_df.columns:
            fig.add_trace(go.Scatter(x=show_df.index, y=show_df[ma], name=ma, line=dict(color=color, width=1.4)), row=1, col=1)
    if support is not None:
        fig.add_hline(y=support, row=1, col=1, line_dash="dot", line_color="#22c55e", opacity=0.8)
    if resistance is not None:
        fig.add_hline(y=resistance, row=1, col=1, line_dash="dot", line_color="#ef4444", opacity=0.8)
    fig.add_trace(go.Bar(x=show_df.index, y=show_df["Volume"], name="成交量", marker_color=vol_colors, opacity=0.8), row=2, col=1)
    if "VOL_MA20" in show_df.columns:
        fig.add_trace(go.Scatter(x=show_df.index, y=show_df["VOL_MA20"], name="VOL_MA20", line=dict(color="#f59e0b", width=1.2)), row=2, col=1)
    fig.add_trace(go.Bar(x=show_df.index, y=show_df["MACD_hist"], name="MACD柱", marker_color=macd_bar_colors, opacity=0.7), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=show_df.index, y=show_df["MACD"], name="DIF", line=dict(color="#2563eb", width=1.4)), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=show_df.index, y=show_df["MACD_signal"], name="DEA", line=dict(color="#9333ea", width=1.2)), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=show_df.index, y=show_df["RSI"], name="RSI", line=dict(color="#0ea5e9", width=1.2, dash="dot"), opacity=0.9), row=3, col=1, secondary_y=True)
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="RSI", row=3, col=1, secondary_y=True, range=[0, 100])
    fig.update_layout(height=860, template="plotly_white", margin=dict(l=8, r=8, t=20, b=8),
                      xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0))
    return fig

def build_hourly_chart(df: pd.DataFrame, ticker: str, support=None, resistance=None):
    if df is None or df.empty:
        return None
    show_df = df.tail(200).copy()
    if show_df.empty:
        return None
    up_color, down_color = market_colors(ticker)
    vol_colors = np.where(show_df["Close"] >= show_df["Open"], up_color, down_color)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=show_df.index, open=show_df["Open"], high=show_df["High"], low=show_df["Low"],
                                 close=show_df["Close"], name="小时K线", increasing_line_color=up_color,
                                 decreasing_line_color=down_color), row=1, col=1)
    for ma, color in [("MA5", "#6366f1"), ("MA20", "#f59e0b")]:
        if ma in show_df.columns:
            fig.add_trace(go.Scatter(x=show_df.index, y=show_df[ma], name=ma, line=dict(color=color, width=1.2)), row=1, col=1)
    fig.add_trace(go.Bar(x=show_df.index, y=show_df["Volume"], name="成交量", marker_color=vol_colors, opacity=0.7), row=2, col=1)
    fig.update_layout(height=500, template="plotly_white", margin=dict(l=8, r=8, t=20, b=8))
    return fig

# 新闻舆情获取函数（保持不变）
@st.cache_data(ttl=3600, show_spinner=False)
def get_news(ticker: str, source: str = "yfinance"):
    news_list = []
    code = ticker.strip().upper()
    if source == "akshare" and is_a_share_code(code):
        try:
            import akshare as ak
            df = ak.stock_news_em(symbol=code)
            if df is None or df.empty:
                return []
            for _, row in df.head(5).iterrows():
                news_list.append({
                    "title": row.get("新闻标题", ""),
                    "time": row.get("发布时间", ""),
                    "source": row.get("文章来源", ""),
                    "url": row.get("新闻链接", "")
                })
        except Exception as e:
            st.warning(f"获取A股新闻失败：{e}")
            return []
    else:
        try:
            ticker_yf = yf.Ticker(normalize_ticker_for_yf(ticker))
            news = ticker_yf.news
            if news is None:
                return []
            for item in news[:5]:
                news_list.append({
                    "title": item.get("title", ""),
                    "time": datetime.fromtimestamp(item.get("providerPublishTime", 0)).strftime("%Y-%m-%d %H:%M") if item.get("providerPublishTime") else "",
                    "source": item.get("publisher", ""),
                    "url": item.get("link", "")
                })
        except Exception as e:
            st.warning(f"获取美股新闻失败：{e}")
            return []
    return news_list

# 基本面趋势分析函数（保持不变）
@st.cache_data(ttl=86400, show_spinner=False)
def get_fundamental_trend(ticker: str, source: str = "yfinance"):
    result = {
        "periods": [], "revenue": [], "net_income": [], "roe": [],
        "gross_margin": [], "debt_ratio": [], "trend_summary": "",
    }
    code = ticker.strip().upper()
    if source == "akshare" and is_a_share_code(code):
        try:
            import akshare as ak
            df = ak.stock_financial_analysis_indicator(symbol=code)
            if df is None or df.empty:
                return result
            df = df.sort_values("日期", ascending=False).head(4)
            for _, row in df.iterrows():
                result["periods"].append(row["日期"].strftime("%Y-%m") if isinstance(row["日期"], pd.Timestamp) else str(row["日期"]))
                result["revenue"].append(row.get("营业总收入", np.nan))
                result["net_income"].append(row.get("净利润", np.nan))
                result["roe"].append(row.get("净资产收益率", np.nan))
                result["gross_margin"].append(row.get("销售毛利率", np.nan))
                result["debt_ratio"].append(row.get("资产负债率", np.nan))
            result["trend_summary"] = _calculate_trend_summary(result)
            return result
        except Exception as e:
            st.warning(f"获取A股财务趋势数据失败：{e}")
            return result
    else:
        try:
            ticker_yf = yf.Ticker(normalize_ticker_for_yf(ticker))
            financials = ticker_yf.quarterly_financials
            balance = ticker_yf.quarterly_balance_sheet
            if financials is None or financials.empty:
                return result
            fin = financials.iloc[:, :4]
            bal = balance.iloc[:, :4] if balance is not None and not balance.empty else None
            revenue_ttm = []
            net_income_ttm = []
            periods = []
            for i in range(4):
                if i + 3 < len(fin.columns):
                    rev = fin.loc["Total Revenue" if "Total Revenue" in fin.index else "Revenue", fin.columns[i:i+4]].sum()
                    net = fin.loc["Net Income" if "Net Income" in fin.index else "NetIncome", fin.columns[i:i+4]].sum()
                    revenue_ttm.append(rev)
                    net_income_ttm.append(net)
                    periods.append(fin.columns[i+3].strftime("%Y-%m"))
            roe_list = []
            gross_margin_list = []
            debt_ratio_list = []
            for i in range(len(periods)):
                if bal is not None and i < len(bal.columns):
                    equity = bal.loc["Total Equity Gross Minority Interest" if "Total Equity Gross Minority Interest" in bal.index else "Stockholders Equity", bal.columns[i]]
                    if equity != 0 and not np.isnan(equity):
                        roe_list.append(net_income_ttm[i] / equity * 100)
                    else:
                        roe_list.append(np.nan)
                else:
                    roe_list.append(np.nan)
                gross_margin_list.append(np.nan)
                if bal is not None and i < len(bal.columns):
                    total_liab = bal.loc["Total Liabilities Net Minority Interest" if "Total Liabilities Net Minority Interest" in bal.index else "Total Liabilities", bal.columns[i]]
                    total_assets = bal.loc["Total Assets", bal.columns[i]]
                    if total_assets != 0 and not np.isnan(total_assets):
                        debt_ratio_list.append(total_liab / total_assets * 100)
                    else:
                        debt_ratio_list.append(np.nan)
                else:
                    debt_ratio_list.append(np.nan)
            result["periods"] = periods
            result["revenue"] = revenue_ttm
            result["net_income"] = net_income_ttm
            result["roe"] = roe_list
            result["gross_margin"] = gross_margin_list
            result["debt_ratio"] = debt_ratio_list
            result["trend_summary"] = _calculate_trend_summary(result)
            return result
        except Exception as e:
            st.warning(f"获取美股财务趋势数据失败：{e}")
            return result

def _calculate_trend_summary(fund_data):
    revs = fund_data["revenue"]
    nets = fund_data["net_income"]
    roes = fund_data["roe"]
    rev_clean = [v for v in revs if not pd.isna(v)]
    net_clean = [v for v in nets if not pd.isna(v)]
    roe_clean = [v for v in roes if not pd.isna(v)]
    summary_parts = []
    if len(rev_clean) >= 2:
        rev_growth = [(rev_clean[i] - rev_clean[i-1]) / rev_clean[i-1] * 100 for i in range(1, len(rev_clean))]
        if all(g > 0 for g in rev_growth):
            summary_parts.append("营收连续增长，增速呈加速/平稳趋势")
        elif any(g > 0 for g in rev_growth) and any(g < 0 for g in rev_growth):
            summary_parts.append("营收增长不稳定，有波动")
        else:
            summary_parts.append("营收连续下滑，基本面承压")
    else:
        summary_parts.append("营收数据不足，无法判断趋势")
    if len(net_clean) >= 2:
        net_growth = [(net_clean[i] - net_clean[i-1]) / net_clean[i-1] * 100 for i in range(1, len(net_clean))]
        if all(g > 0 for g in net_growth):
            summary_parts.append("净利润连续增长，盈利能力改善")
        elif any(g > 0 for g in net_growth) and any(g < 0 for g in net_growth):
            summary_parts.append("净利润波动较大，盈利不稳定")
        else:
            summary_parts.append("净利润连续下滑，盈利恶化")
    else:
        summary_parts.append("净利润数据不足，无法判断趋势")
    if len(roe_clean) >= 2:
        if roe_clean[-1] > roe_clean[0]:
            summary_parts.append("ROE趋势向上，股东回报增强")
        elif roe_clean[-1] < roe_clean[0]:
            summary_parts.append("ROE趋势向下，股东回报减弱")
        else:
            summary_parts.append("ROE保持平稳")
    else:
        summary_parts.append("ROE数据不足")
    return "；".join(summary_parts)

# AI 分析函数（保持不变）
def ai_analysis(analysis: dict, ticker: str, stock_name: str, api_key: str, base_url: str, model: str, view: str,
                extra_data: dict = None, fund_trend: dict = None, news: list = None):
    pe_text = "N/A" if analysis["pe"] is None else f"{analysis['pe']:.2f}"
    pb_text = "N/A" if analysis["pb"] is None else f"{analysis['pb']:.2f}"
    extra_desc = ""
    if extra_data:
        if "hourly" in extra_data:
            h = extra_data["hourly"]
            extra_desc += f"""
【小时线指标（用于极短期/短期）】
- 小时线最新价：{h.get('price', 'N/A')}
- 小时线RSI：{h.get('rsi', 'N/A')}
- 小时线MACD柱：{h.get('macd_hist', 'N/A')}
- 小时线量比：{h.get('vol_ratio', 'N/A')}
"""
        if "weekly" in extra_data:
            w = extra_data["weekly"]
            extra_desc += f"""
【周线指标（用于中长期）】
- 周线最新价：{w.get('price', 'N/A')}
- 周线RSI：{w.get('rsi', 'N/A')}
- 周线MACD柱：{w.get('macd_hist', 'N/A')}
- 周线MA20：{w.get('ma20', 'N/A')}
"""
    fund_desc = ""
    if fund_trend and fund_trend.get("periods"):
        fund_desc = f"""
【基本面趋势（基于最近{len(fund_trend['periods'])}个报告期）】
- 报告期：{', '.join(fund_trend['periods'])}
- 营收TTM（亿元）：{', '.join([f'{x:.2f}' if not pd.isna(x) else 'N/A' for x in fund_trend['revenue']])}
- 净利润TTM（亿元）：{', '.join([f'{x:.2f}' if not pd.isna(x) else 'N/A' for x in fund_trend['net_income']])}
- ROE（%）：{', '.join([f'{x:.2f}' if not pd.isna(x) else 'N/A' for x in fund_trend['roe']])}
- 资产负债率（%）：{', '.join([f'{x:.2f}' if not pd.isna(x) else 'N/A' for x in fund_trend['debt_ratio']])}
- 趋势总结：{fund_trend.get('trend_summary', '无足够数据')}
"""
    news_desc = ""
    if news and len(news) > 0:
        news_items = []
        for n in news[:5]:
            title = n.get("title", "")
            time = n.get("time", "")
            source = n.get("source", "")
            news_items.append(f"- {title}（{time}，{source}）")
        news_desc = f"""
【近期相关新闻（舆情参考）】
{chr(10).join(news_items)}
"""
    prompt = f"""
    你是一位拥有20年实战经验的职业交易员，擅长量价分析、趋势跟踪和资金管理。本次分析请从 **{view}** 的角度出发，根据以下量化技术数据、基本面趋势和新闻舆情，对 {stock_name} ({ticker}) 输出一份专业交易分析报告。

    {view} 的含义：
    - 极短期交易 (1-5天)：侧重捕捉日内/隔夜波动，必须关注小时线动能、日线量比、超买超卖，建议快进快出，严格止损。
    - 短期交易 (1-4周)：侧重捕捉短期波动，关注小时线+日线动量，日线量比，超买超卖，建议持仓灵活。
    - 中长期波段 (1-6个月)：侧重趋势方向和资金流向，关注周线趋势、日线均线排列，建议顺势持有。

    报告必须通俗易懂，避免枯燥的数字堆砌，要用交易员的语言解释每个指标的含义及其背后的市场心理。同时，报告必须完整，不能中断。

【当前市场状态】
- 价格：{analysis['price']:.2f} 元，当日涨跌幅 {analysis['change']:+.2f}%
- 成交量：{analysis['volume']:.0f} 手，20日均量 {analysis['vol_ma20']:.0f} 手，量比 {analysis['vol_ratio']:.2f}
- 估值：PE={pe_text}，PB={pb_text}

【趋势与动能】
- 均线：MA5={analysis['ma5']:.2f}，MA10={analysis['ma10']:.2f}，MA20={analysis['ma20']:.2f}，MA60={analysis['ma60']:.2f}，MA120={analysis['ma120']:.2f}
- 趋势强度：ADX={analysis['adx']:.1f}，+DI={analysis['plus_di']:.1f}，-DI={analysis['minus_di']:.1f}
- MACD：DIF={analysis['macd']:.3f}，DEA={analysis['macd_signal']:.3f}，柱状值={analysis['macd_hist']:.3f}
- RSI={analysis['rsi']:.1f}，乖离率BIAS={analysis['bias']:.2f}%

【资金与结构】
- 资金流：CMF={analysis['cmf']:.3f}，MFI={analysis['mfi']:.1f}，OBV背离={analysis['obv_div']}
- 量价信号：{analysis['vpa_signal']}（{analysis['vpa_detail']}）
- K线形态：{", ".join(analysis['candle_patterns']) if analysis['candle_patterns'] else "无"}
- 价格形态：{", ".join(analysis['price_patterns']) if analysis['price_patterns'] else "无"}
- 背离：MACD={analysis['macd_divergence']}，RSI={analysis['rsi_divergence']}

{extra_desc}
{fund_desc}
{news_desc}

【风险参数】
- 关键支撑/压力：{analysis['support']:.2f} / {analysis['resistance']:.2f}
- 建议止损/目标：{analysis['stop_loss']:.2f} / {analysis['take_profit']:.2f}
- 盈亏比：{analysis['risk_reward']:.2f}
- 当前仓位建议上限：{analysis['position_size']:.0%}

请按以下结构输出，每个部分都要结合数据给出具体分析，语言要像交易员在复盘一样：
1) 趋势与强弱结论（先给出明确的结论：多头、空头、震荡，并说明强度）
2) 关键证据（至少5条，每条证据要解释其市场含义，为什么重要）
3) 舆情与基本面结合（简要分析新闻情绪和基本面趋势对股价的可能影响）
4) 三种情景推演（假设未来走势分为三种情况：突破上涨 / 区间震荡 / 破位下跌，分别给出应对策略）
5) 最终建议（明确给出：买入/卖出/观望，并说明仓位大小和理由） + 风险提示（最大风险点）

注意：回答必须完整，不要中途截断。用简洁、专业、富有洞察力的语言，避免冗余。
"""
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": "你是一位资深交易员，输出要专业、简洁、有洞察力，避免枯燥的数字罗列。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.6,
        }
        if use_max_completion_tokens(model):
            kwargs["max_completion_tokens"] = 3000
        else:
            kwargs["max_tokens"] = 3000
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI分析调用失败：{e}"

def chat_ai(user_message: str, history: list, model: str, api_key: str, base_url: str):
    if not api_key:
        return "请先在左侧输入 API Key。"
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [{"role": "system", "content": "你是一个专业的金融与编程助手，请给出清晰、实用、准确的回答。"}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    try:
        kwargs = {"model": model, "messages": messages, "temperature": 0.5}
        if use_max_completion_tokens(model):
            kwargs["max_completion_tokens"] = 1200
        else:
            kwargs["max_tokens"] = 1200
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content
    except Exception as e1:
        if "codex" in model.lower():
            try:
                input_items = []
                for m in messages:
                    input_items.append({"role": m["role"], "content": [{"type": "input_text", "text": m["content"]}]})
                resp = client.responses.create(model=model, input=input_items, max_output_tokens=1200)
                text = getattr(resp, "output_text", None)
                if text:
                    return text
                outputs = []
                for item in getattr(resp, "output", []):
                    for c in getattr(item, "content", []):
                        ctype = getattr(c, "type", "")
                        if ctype in ("output_text", "text"):
                            outputs.append(getattr(c, "text", ""))
                return "".join(outputs) if outputs else "（模型返回空内容）"
            except Exception as e2:
                return f"⚠️ 聊天调用失败：{e2}"
        return f"⚠️ 聊天调用失败：{e1}"

# 板块热力图函数
SECTOR_LEADERS = {
    "银行": "600036", "证券": "600030", "保险": "601318", "白酒": "600519",
    "新能源车": "002594", "半导体": "688981", "光伏": "300274", "医药": "600276",
    "医疗": "300760", "消费电子": "002475", "计算机": "600588", "通信": "600050",
    "军工": "600893", "有色": "601600", "煤炭": "601088", "钢铁": "600019",
    "地产": "000002", "建材": "600585", "家电": "000333", "农业": "000998",
    "食品": "603288", "旅游": "600138", "电力": "600900", "化工": "600309",
    "机械设备": "600031", "交通运输": "601816", "传媒": "300413", "环保": "300070",
    "人工智能": "002230", "云计算": "000977", "5G": "000063",
}

@st.cache_data(ttl=3600, show_spinner=False)
def get_sector_heatmap(sector_map, period="1mo", source="yfinance"):
    results = []
    progress_bar = st.progress(0, text="正在获取板块数据...")
    total = len(sector_map)
    for i, (sector, code) in enumerate(sector_map.items()):
        df_raw, notes = get_data(code, period, "1d", source)
        if df_raw is not None and len(df_raw) >= 20:
            df = add_advanced_indicators(df_raw)
            score = get_score(df)
            latest = df.iloc[-1]
            if latest["Close"] > latest.get("MA20", 0) and latest.get("MA5", 0) > latest.get("MA10", 0) > latest.get("MA20", 0):
                trend = "🔥 强"
            elif latest["Close"] > latest.get("MA20", 0):
                trend = "📈 偏多"
            elif latest["Close"] < latest.get("MA20", 0):
                trend = "📉 偏空"
            else:
                trend = "⚖️ 震荡"
            data_days = len(df_raw)
        else:
            score = 0
            trend = "❌ 无数据"
            data_days = 0
        results.append({"板块": sector, "代码": code, "评分": score, "趋势": trend, "数据天数": data_days})
        progress_bar.progress((i + 1) / total, text=f"正在获取板块数据... {i+1}/{total}")
    progress_bar.empty()
    df_result = pd.DataFrame(results).sort_values("评分", ascending=False)
    return df_result

# =========================
# 侧边栏配置（重构布局）
# =========================
with st.sidebar:
    # ---------- 板块热力图（顶部） ----------
    st.header("🔥 板块热力图")
    data_source_heat = st.selectbox(
        "板块数据源",
        ["akshare（A股）", "yfinance（美股/港股）"],
        index=0,
        key="heat_source"
    )
    source_heat = "akshare" if data_source_heat.startswith("akshare") else "yfinance"
    if st.button("扫描板块热点", use_container_width=True, key="scan_heat"):
        with st.spinner("正在获取板块龙头数据，请稍候..."):
            df_heat = get_sector_heatmap(SECTOR_LEADERS, period="1mo", source=source_heat)
            st.session_state.heatmap_df = df_heat
    if "heatmap_df" in st.session_state and st.session_state.heatmap_df is not None:
        st.dataframe(
            st.session_state.heatmap_df,
            use_container_width=True,
            height=300,
            column_config={
                "板块": st.column_config.TextColumn("板块", width="small"),
                "代码": st.column_config.TextColumn("代码", width="small"),
                "评分": st.column_config.ProgressColumn("评分", format="%d", min_value=0, max_value=100, width="medium"),
                "趋势": st.column_config.TextColumn("趋势判断", width="small"),
                "数据天数": st.column_config.NumberColumn("数据天数", width="small"),
            },
            hide_index=True,
        )
        st.caption("评分基于日线量价技术指标（最高100），数据天数不足20天可能影响准确性。")
    st.divider()

    # ---------- 个股分析控制台 ----------
    st.header("📌 个股分析")
    ticker_input = st.text_input("标的代码（如 000001 / AAPL）", "000001").strip().upper()

    st.subheader("🗄️ 数据源")
    data_source = st.selectbox(
        "选择数据源",
        ["akshare（A股更稳，需本地已安装 akshare）", "yfinance（免费，部分市场延迟）"],
        index=0,
    )

    st.subheader("🎯 分析视角")
    view = st.radio(
        "选择当前分析的时间维度",
        options=["极短期 (1-5天)", "短期 (1-4周)", "中长期 (1-6个月)"],
        index=1,
        help="极短期：日内/隔夜交易；短期：数周波段；中长期：数月趋势。"
    )

    st.subheader("⚙️ 交易模式")
    trade_mode = st.selectbox(
        "选择你的交易风格",
        options=["趋势", "短线", "超短", "波段"],
        index=1,
        help="不同模式影响ATR倍数、RSI阈值、止盈止损参数。建议与分析视角匹配。"
    )
    mode_config = TRADE_MODES[trade_mode]
    st.caption(f"ATR乘数：{mode_config['atr_mult']} | RSI超买/超卖：{mode_config['rsi_ob']}/{mode_config['rsi_os']}")

    # 根据视角自动推荐权重，但允许用户后续调整（此处简化为固定权重，但增加降级逻辑）
    if view == "极短期 (1-5天)":
        default_period = "1mo"
        w_d, w_w, w_h = 0.3, 0.0, 0.7
        st.info("**极短期模式**：侧重小时线动能，日线辅助，止损严格。")
    elif view == "短期 (1-4周)":
        default_period = "3mo"
        w_d, w_w, w_h = 0.6, 0.0, 0.4
        st.info("**短期模式**：日线趋势+小时线动量，适合数周波段。")
    else:
        default_period = "1y"
        w_d, w_w, w_h = 0.7, 0.3, 0.0
        st.info("**中长期模式**：周线定方向，日线找买点。")

    chart_period = st.selectbox(
        "图表显示周期（K线数量）",
        list(CHART_DAYS.keys()),
        index=list(CHART_DAYS.keys()).index(default_period),
        help="日线图表显示多少条K线"
    )

    st.caption(f"当前权重：日线 {w_d:.0%} / 周线 {w_w:.0%} / 小时线 {w_h:.0%}")

    st.subheader("📊 基本面分析")
    enable_fundamental = st.checkbox("启用基本面趋势分析", value=True, help="获取最近4个报告期的财务数据，判断基本面趋势")
    st.subheader("📰 舆情分析")
    enable_news = st.checkbox("启用新闻舆情分析", value=True, help="获取相关新闻，让AI分析市场情绪")

    st.subheader("🤖 AI 配置")
    default_key = os.getenv("AI_API_KEY")
    ai_api_key = st.text_input("API Key", type="password", value=default_key)
    ai_base_url = st.text_input("Base URL", value="https://aihubmix.com/v1")
    ai_report_model = st.selectbox(
        "AI报告模型",
        ["deepseek-v3.2", "deepseek-v3.2-speciale", "gpt-4.1-free", "gpt-4o-free", "gpt-4.1-mini-free", "gpt-4.1-nano-free"],
        index=0,
    )
    enable_ai_report = st.checkbox("启用 AI 深度报告", value=True)
    chat_model = st.selectbox(
        "聊天模型",
        ["gpt-4.1-free", "gpt-4.1-nano-free", "gpt-5.4-high", "gpt-5.3-codex", "gpt-4.1", "o3-mini", "gpt-5.2"],
        index=0,
        key="chat_model_selector",
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("清空缓存"):
            st.cache_data.clear()
            st.success("缓存已清空")
    with col2:
        if st.button("清空聊天"):
            st.session_state.chat_history = []
            st.success("聊天已清空")

    with st.expander("📖 指标说明"):
        st.markdown("""
        - **均线**：趋势方向（多头排列 MA5>MA10>MA20 为强势）  
        - **ADX/DI**：趋势强度（>25 为强趋势）  
        - **MACD**：动量与拐点（金叉/死叉）  
        - **RSI/MFI**：超买超卖（>70 超买，<30 超卖）  
        - **CMF/OBV**：资金流与背离（正值为资金流入）  
        - **ATR**：波动度（止损基准）  
        - **风险控制**：先定止损，再定仓位
        - **基本面趋势**：基于最近4个报告期的营收、净利润、ROE等数据判断趋势
        - **舆情分析**：获取近期相关新闻，辅助判断市场情绪
        - **板块热力图**：预定义主流板块龙头股，批量评分排序，快速发现强势板块
        """)

# =========================
# 主页面布局
# =========================
left_col, right_col = st.columns([2.2, 1], gap="large")

with left_col:
    run_btn = st.button("🚀 执行深度洞察", use_container_width=True)
    if run_btn:
        with st.spinner("正在解析量价结构与资金行为..."):
            source = "akshare" if data_source.startswith("akshare") else "yfinance"
            notes = []
            stock_name, pe, pb = get_stock_info(ticker_input, source=source)
            fund_trend = None
            if enable_fundamental:
                with st.spinner("正在获取财务趋势数据..."):
                    fund_trend = get_fundamental_trend(ticker_input, source=source)
            news_list = None
            if enable_news:
                with st.spinner("正在获取相关新闻..."):
                    news_list = get_news(ticker_input, source=source)

            # 获取日线
            df_d_raw, n1 = get_data(ticker_input, chart_period, "1d", source=source)
            notes.extend(n1)
            # 周线（仅中长期需要）
            df_w_raw = None
            if view in ["短期 (1-4周)", "中长期 (1-6个月)"]:
                df_w_raw, n2 = get_data(ticker_input, chart_period, "1wk", source=source)
                notes.extend(n2 if n2 else [])
                if (df_w_raw is None or len(df_w_raw) < 20) and df_d_raw is not None and len(df_d_raw) >= 80:
                    df_w_raw = resample_to_weekly(df_d_raw)
                    notes.append("周线使用日线重采样生成。")
            # 小时线（仅极短期和短期需要）
            df_h_raw = None
            hour_available = True
            if view in ["极短期 (1-5天)", "短期 (1-4周)"]:
                df_h_raw, n3 = get_data(ticker_input, "1mo", "60m", source=source)
                notes.extend(n3 if n3 else [])
                if df_h_raw is None or len(df_h_raw) < 20:
                    st.warning("⚠️ 小时线数据不足（少于20根），将自动降级为仅使用日线分析。短线信号可能延迟。")
                    hour_available = False
                    w_h = 0.0
                    w_d = 1.0   # 日线权重提高
            # 添加指标
            df_d = add_advanced_indicators(df_d_raw) if df_d_raw is not None else None
            df_w = add_advanced_indicators(df_w_raw) if df_w_raw is not None else None
            df_h = add_advanced_indicators(df_h_raw) if (hour_available and df_h_raw is not None) else None

            analysis = detailed_analysis(df_d, pe=pe, pb=pb, mode=trade_mode) if df_d is not None else None

            score_d = get_score(df_d) if df_d is not None else 0
            score_w = get_score(df_w) if df_w is not None else 0
            score_h = get_score(df_h) if df_h is not None else 0
            availability = {
                "日线": df_d is not None and len(df_d) >= 20,
                "周线": df_w is not None and len(df_w) >= 20,
                "小时线": df_h is not None and len(df_h) >= 20
            }
            scores = {"日线": score_d, "周线": score_w, "小时线": score_h}
            weights = {"日线": w_d, "周线": w_w, "小时线": w_h}
            valid_keys = [k for k, ok in availability.items() if ok]
            if valid_keys:
                w_sum = sum(weights[k] for k in valid_keys)
                if w_sum > 0:
                    composite_score = sum(scores[k] * weights[k] for k in valid_keys) / w_sum
                else:
                    composite_score = float(np.mean([scores[k] for k in valid_keys]))
            else:
                composite_score = 0.0

            extra_ai = {}
            if df_h is not None and len(df_h) > 0:
                h_last = df_h.iloc[-1]
                extra_ai["hourly"] = {"price": float(h_last["Close"]), "rsi": h_last.get("RSI", np.nan),
                                      "macd_hist": h_last.get("MACD_hist", np.nan),
                                      "vol_ratio": h_last.get("Vol_Ratio", np.nan)}
            if df_w is not None and len(df_w) > 0:
                w_last = df_w.iloc[-1]
                extra_ai["weekly"] = {"price": float(w_last["Close"]), "rsi": w_last.get("RSI", np.nan),
                                      "macd_hist": w_last.get("MACD_hist", np.nan), "ma20": w_last.get("MA20", np.nan)}

            structured_signals = get_structured_signals(df_d) if df_d is not None else []
            trade_triggers = generate_trade_signal(df_d) if df_d is not None else []
            main_behavior = analyze_main_behavior(df_d, analysis) if analysis else {}
            dip_eval = evaluate_dip_entry(df_d, analysis, trade_mode) if analysis else {}

            result = {
                "ticker": ticker_input,
                "stock_name": stock_name,
                "analysis": analysis,
                "df_d": df_d,
                "df_h": df_h,
                "scores": scores,
                "weights": weights,
                "availability": availability,
                "composite_score": composite_score,
                "notes": list(dict.fromkeys(notes)),
                "source": source,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "extra_ai": extra_ai,
                "fund_trend": fund_trend,
                "news": news_list,
                "structured_signals": structured_signals,
                "trade_triggers": trade_triggers,
                "main_behavior": main_behavior,
                "dip_evaluation": dip_eval,
            }
            st.session_state.analysis_result = result

            if enable_ai_report and ai_api_key and analysis is not None:
                with st.spinner("AI 正在自动生成深度报告..."):
                    st.session_state.ai_report_text = ai_analysis(
                        analysis=analysis, ticker=ticker_input, stock_name=stock_name,
                        api_key=ai_api_key, base_url=ai_base_url, model=ai_report_model,
                        view=view, extra_data=extra_ai, fund_trend=fund_trend, news=news_list,
                    )
            else:
                st.session_state.ai_report_text = ""

    result = st.session_state.get("analysis_result")
    if not result:
        st.info("👈 在左侧输入代码后点击 **执行深度洞察**。")
    else:
        analysis = result.get("analysis")
        if analysis is None:
            st.warning("日线数据不足，建议更换标的或拉长周期后重试（至少约20根K线）。")
            if result.get("notes"):
                for note in result["notes"]:
                    st.caption(f"⚠️ {note}")
        else:
            up_color, down_color = market_colors(result["ticker"])
            chg_color = up_color if analysis["change"] >= 0 else down_color
            pe_txt = "N/A" if analysis["pe"] is None else f"{analysis['pe']:.2f}"
            pb_txt = "N/A" if analysis["pb"] is None else f"{analysis['pb']:.2f}"

            hot_tags = []
            if analysis["vpa_signal"] == "放量进攻":
                hot_tags.append('<span class="hot-tag">🔥 放量进攻</span>')
            if analysis["money_status"] == "资金流入":
                hot_tags.append('<span class="hot-tag info-tag">💰 资金流入</span>')
            if analysis["trend"] in ["强势多头", "多头结构"]:
                hot_tags.append('<span class="hot-tag">📈 主升浪</span>')
            if analysis["macd_divergence"] == "顶背离":
                hot_tags.append('<span class="hot-tag" style="background:#ef4444;">⚠️ 顶背离</span>')
            hot_tags_html = " ".join(hot_tags) if hot_tags else ""

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="stock-header">
                <div class="stock-name">{result["stock_name"]} ({result["ticker"]})</div>
                <div class="stock-meta">数据源：{result["source"]} | {result["timestamp"]}</div>
            </div>
            <div class="price-tag">¥{analysis["price"]:.2f} <span style="font-size:1.5rem;color:{chg_color};">{analysis["change"]:+.2f}%</span></div>
            <div style="margin-top: 8px;">成交量：{analysis['volume']:,.0f} | 20日均量：{analysis['vol_ma20']:,.0f} | 量比：{analysis['vol_ratio']:.2f} | PE：{pe_txt} | PB：{pb_txt}</div>
            <div style="margin-top: 4px;">筹码状态：{analysis['chip_concentration']}</div>
            <div style="margin-top: 8px;">{hot_tags_html}</div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if result.get("notes"):
                for note in result["notes"]:
                    st.caption(f"⚠️ {note}")

            risk_messages = []
            if analysis["macd_divergence"] == "顶背离":
                risk_messages.append("⚠️ MACD顶背离：价格新高但动能减弱，警惕回调。")
            if analysis["rsi_divergence"] == "顶背离":
                risk_messages.append("⚠️ RSI顶背离：超买区域背离，可能见顶。")
            if analysis["rsi"] > 80:
                risk_messages.append(f"⚠️ RSI={analysis['rsi']:.1f}进入超买区，追高风险大。")
            if analysis["vpa_signal"] == "放量下杀":
                risk_messages.append("⚠️ 放量下跌，抛压沉重，暂不宜抄底。")
            if risk_messages:
                st.markdown('<div class="risk-box">' + "<br>".join(risk_messages) + '</div>', unsafe_allow_html=True)

            st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">综合评分</div><div class="metric-value">{result["composite_score"]:.1f}</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">趋势</div><div class="metric-value">{analysis["trend"]}</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">量价结构</div><div class="metric-value">{analysis["vpa_signal"]}</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="metric-card"><div class="metric-label">总评分</div><div class="metric-value">{analysis["total_score"]} / 100</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if analysis["action_color"] == "success":
                st.success(f"**{analysis['status']}**｜{analysis['advice']}")
            elif analysis["action_color"] == "warning":
                st.warning(f"**{analysis['status']}**｜{analysis['advice']}")
            elif analysis["action_color"] == "error":
                st.error(f"**{analysis['status']}**｜{analysis['advice']}")
            else:
                st.info(f"**{analysis['status']}**｜{analysis['advice']}")

            cols = st.columns(4)
            cols[0].metric("支撑位", f"{analysis['support']:.2f}")
            cols[1].metric("压力位", f"{analysis['resistance']:.2f}")
            cols[2].metric("止损位", f"{analysis['stop_loss']:.2f}")
            cols[3].metric("目标位", f"{analysis['take_profit']:.2f}")
            st.write(f"盈亏比：**{analysis['risk_reward']:.2f}** ｜ 建议仓位上限：**{analysis['position_size']:.0%}**")

            if result.get("trade_triggers"):
                with st.expander("⚡ 交易触发信号", expanded=False):
                    for trigger in result["trade_triggers"]:
                        st.markdown(f"- ✅ {trigger}")

            main_behavior = result.get("main_behavior", {})
            if main_behavior:
                with st.expander("🧠 主力行为分析", expanded=False):
                    behavior = main_behavior.get("behavior", "未知")
                    conf = main_behavior.get("confidence", 0)
                    desc = main_behavior.get("description", "")
                    st.markdown(f"**当前判断**：{behavior}（置信度 {conf:.0%}）")
                    st.caption(desc)

            dip = result.get("dip_evaluation", {})
            if dip and dip.get("score", 0) > 0:
                with st.expander("📉 短线下跌试仓评估", expanded=False):
                    st.markdown(f"**试仓评分**：{dip['score']} / 100")
                    st.markdown(f"**结论**：{dip['action']}")
                    st.markdown(f"**原因**：{dip['reason']}")
                    if dip['action'] != "不试仓":
                        st.markdown(f"**建议仓位**：{dip['suggested_position']:.0%}")
                        st.markdown(f"**建议止损**：{dip['stop_loss']:.2f}  |  **目标**：{dip['target']:.2f}")

            with st.expander("📊 多维度评分雷达图", expanded=False):
                categories = ['趋势', '量价', '动量', '资金', '综合']
                values = [analysis['trend_score'], analysis['vpa_score'],
                          analysis['rsi_score'] + analysis['bias_score'], analysis['money_score'],
                          analysis['total_score']]
                fig_radar = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself', marker=dict(color='#2563eb')))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=400)
                st.plotly_chart(fig_radar, use_container_width=True)

            with st.expander("🔍 决策推导（技术指标解读）", expanded=False):
                def trend_desc(ma5, ma10, ma20, ma60, ma120, price):
                    if ma5 > ma10 > ma20:
                        short_trend = "短期多头排列，价格处于强势"
                    elif ma5 < ma10 < ma20:
                        short_trend = "短期空头排列，价格承压"
                    else:
                        short_trend = "均线交错，趋势不明朗"
                    if price > ma60:
                        long_trend = "中长期均线上方，长期趋势偏多"
                    else:
                        long_trend = "中长期均线下方，长期趋势偏弱"
                    return f"{short_trend}；{long_trend}"
                st.markdown("**1️⃣ 趋势判断**")
                st.write(f"- {trend_desc(analysis['ma5'], analysis['ma10'], analysis['ma20'], analysis['ma60'], analysis['ma120'], analysis['price'])}")
                st.write(f"- 趋势强度：{'强趋势' if analysis['adx'] >= 25 else '弱趋势/震荡'}，ADX={analysis['adx']:.1f}；多头力量+DI={analysis['plus_di']:.1f}，空头力量-DI={analysis['minus_di']:.1f}。")
                st.write(f"- MACD：{'红柱（多头动能）' if analysis['macd_hist'] > 0 else '绿柱（空头动能）'}，快线DIF={analysis['macd']:.3f}，慢线DEA={analysis['macd_signal']:.3f}。")
                st.markdown("**2️⃣ 量价配合**")
                st.write(f"- {analysis['vpa_detail']}。当前量比{analysis['vol_ratio']:.2f}，{'放量' if analysis['vol_ratio'] > 1.2 else '缩量' if analysis['vol_ratio'] < 0.8 else '正常'}。")
                if analysis["volume_price_signals"]:
                    for sig in analysis["volume_price_signals"]:
                        st.write(f"  - {sig}")
                st.markdown("**3️⃣ 超买超卖**")
                st.write(f"- RSI={analysis['rsi']:.1f}，{'超买区域，注意回调风险' if analysis['rsi'] > 70 else '超卖区域，可能反弹' if analysis['rsi'] < 30 else '中性区域，动能正常'}。")
                st.write(f"- 乖离率BIAS={analysis['bias']:.2f}%，{'偏离过大，有回归需求' if abs(analysis['bias']) > 8 else '偏离正常，趋势延续'}。")
                st.markdown("**4️⃣ 资金流向**")
                st.write(f"- CMF={analysis['cmf']:.3f}，{'资金净流入' if analysis['cmf'] > 0 else '资金净流出' if analysis['cmf'] < 0 else '资金平衡'}。")
                st.write(f"- MFI={analysis['mfi']:.1f}，{'超买过热' if analysis['mfi'] > 80 else '超卖吸筹' if analysis['mfi'] < 20 else '资金健康'}。")
                st.write(f"- OBV背离：{analysis['obv_div']}。")
                st.markdown("**5️⃣ 形态与背离**")
                if analysis["candle_patterns"]:
                    st.write(f"K线形态：{', '.join(analysis['candle_patterns'])}")
                else:
                    st.write("K线形态：无典型反转形态")
                if analysis["price_patterns"]:
                    st.write(f"价格形态：{', '.join(analysis['price_patterns'])}")
                else:
                    st.write("价格形态：无明显结构")
                st.write(f"背离信号：MACD={analysis['macd_divergence']}，RSI={analysis['rsi_divergence']}，{analysis['obv_div']}。")

            if enable_fundamental:
                with st.expander("📊 财务趋势分析（最近4个报告期）", expanded=False):
                    ft = result.get("fund_trend")
                    if ft and ft.get("periods"):
                        periods = ft["periods"]
                        data = {
                            "报告期": periods,
                            "营收TTM（亿元）": [f"{x:.2f}" if not pd.isna(x) else "N/A" for x in ft["revenue"]],
                            "净利润TTM（亿元）": [f"{x:.2f}" if not pd.isna(x) else "N/A" for x in ft["net_income"]],
                            "ROE（%）": [f"{x:.2f}" if not pd.isna(x) else "N/A" for x in ft["roe"]],
                            "资产负债率（%）": [f"{x:.2f}" if not pd.isna(x) else "N/A" for x in ft["debt_ratio"]],
                        }
                        df_fund = pd.DataFrame(data)
                        st.dataframe(df_fund, use_container_width=True)
                        st.write(f"**趋势总结**：{ft['trend_summary']}")
                    else:
                        st.info("暂无财务数据，可能是数据源不支持或获取失败。请检查股票代码或网络，或稍后重试。")

            if enable_news and result.get("news"):
                with st.expander("📰 相关新闻（最新5条）", expanded=False):
                    for n in result["news"][:5]:
                        st.markdown(f"- **{n.get('title', '无标题')}**  \n  {n.get('time', '')} {n.get('source', '')}  \n  [链接]({n.get('url', '#')})")

            with st.expander("📈 量价图谱（日线）", expanded=True):
                fig = build_chart(result["df_d"], chart_period, result["ticker"], support=analysis["support"], resistance=analysis["resistance"])
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("数据已缓存（默认15分钟~1小时，取决于数据源）。")
                else:
                    st.warning("图表数据不足，无法绘制。")

            if (view in ["极短期 (1-5天)", "短期 (1-4周)"]) and result.get("df_h") is not None and len(result["df_h"]) > 0:
                with st.expander("⏱️ 小时线图谱（辅助短线决策）", expanded=False):
                    fig_h = build_hourly_chart(result["df_h"], result["ticker"])
                    if fig_h is not None:
                        st.plotly_chart(fig_h, use_container_width=True)
                    else:
                        st.warning("小时线数据不足，无法绘图。")

            if enable_ai_report:
                with st.expander("🤖 AI 深度报告", expanded=True):
                    if not ai_api_key:
                        st.info("请先在左侧输入 API Key。")
                    else:
                        if st.button("🔄 重新生成 AI 报告", key="btn_regenerate_ai", use_container_width=True):
                            with st.spinner("AI 正在重新生成..."):
                                st.session_state.ai_report_text = ai_analysis(
                                    analysis=analysis, ticker=result["ticker"], stock_name=result["stock_name"],
                                    api_key=ai_api_key, base_url=ai_base_url, model=ai_report_model,
                                    view=view, extra_data=result.get("extra_ai", {}),
                                    fund_trend=result.get("fund_trend"), news=result.get("news"),
                                )
                        ai_text = st.session_state.get("ai_report_text", "")
                        if ai_text:
                            st.markdown(ai_text)
                        else:
                            st.info("AI报告尚未生成，请点击上方按钮手动生成。")
            else:
                st.info("AI深度报告已关闭，如需使用请在左侧开启。")

with right_col:
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
    st.subheader("💬 AI 自由问答助手")
    st.caption("支持金融、编程、日常问题。")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    user_q = st.chat_input("输入你的问题...")
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                resp = chat_ai(user_message=user_q, history=st.session_state.chat_history[:-1], model=chat_model,
                               api_key=ai_api_key, base_url=ai_base_url)
                st.markdown(resp)
                st.session_state.chat_history.append({"role": "assistant", "content": resp})
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("免责声明：本工具仅用于研究与学习，不构成任何投资建议。")
