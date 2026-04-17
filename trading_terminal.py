# ============================================================
# AI TRADING TERMINAL — Complete Fixed Script
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import warnings
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# ── Plotly ────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ── Technical Analysis ────────────────────────────────────────
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

# ── ML / Data ────────────────────────────────────────────────
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Trading Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS STYLING
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700&display=swap');

    /* Base */
    .stApp { background-color: #0a0e1a; color: #e0e6f0; font-family: 'Share Tech Mono', monospace; }
    .main .block-container { padding: 1rem 2rem; max-width: 100%; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1526 0%, #0a0e1a 100%); border-right: 1px solid #1e3a5f; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label { color: #6b8cae; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }

    /* Header */
    .terminal-header {
        background: linear-gradient(135deg, #0d1526 0%, #1a2744 50%, #0d1526 100%);
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 20px 30px;
        margin-bottom: 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .terminal-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00d4ff, #00ff88, #00d4ff, transparent);
    }
    .terminal-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 28px;
        font-weight: 700;
        color: #00d4ff;
        text-shadow: 0 0 20px rgba(0,212,255,0.5);
        letter-spacing: 3px;
        margin: 0;
    }
    .terminal-subtitle { color: #6b8cae; font-size: 12px; letter-spacing: 2px; margin-top: 5px; }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #0d1526 0%, #1a2744 100%);
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    .metric-card:hover { border-color: #00d4ff; box-shadow: 0 0 15px rgba(0,212,255,0.2); transform: translateY(-2px); }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #00d4ff, #00ff88);
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #00d4ff; font-family: 'Orbitron', sans-serif; }
    .metric-label { font-size: 10px; color: #6b8cae; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px; }
    .metric-change { font-size: 12px; margin-top: 3px; }
    .positive { color: #00ff88; }
    .negative { color: #ff4444; }
    .neutral  { color: #6b8cae; }

    /* Signal Box */
    .signal-box {
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        font-family: 'Orbitron', sans-serif;
        font-size: 32px;
        font-weight: 700;
        letter-spacing: 3px;
        position: relative;
        overflow: hidden;
    }
    .signal-buy  { background: linear-gradient(135deg, #0d2b1a, #1a4a2e); border: 2px solid #00ff88; color: #00ff88; text-shadow: 0 0 20px rgba(0,255,136,0.5); }
    .signal-sell { background: linear-gradient(135deg, #2b0d0d, #4a1a1a); border: 2px solid #ff4444; color: #ff4444; text-shadow: 0 0 20px rgba(255,68,68,0.5); }
    .signal-neutral { background: linear-gradient(135deg, #0d1526, #1a2744); border: 2px solid #6b8cae; color: #6b8cae; }

    /* Confidence Bar */
    .confidence-bar { background: #1e3a5f; border-radius: 4px; height: 8px; margin: 10px 0; overflow: hidden; }
    .confidence-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease; }

    /* Trade Table */
    .trade-table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .trade-table th { background: #1a2744; color: #00d4ff; padding: 8px 12px; text-align: left; border-bottom: 1px solid #1e3a5f; font-size: 10px; letter-spacing: 1px; text-transform: uppercase; }
    .trade-table td { padding: 8px 12px; border-bottom: 1px solid #0d1526; color: #e0e6f0; }
    .trade-table tr:hover td { background: #1a2744; }

    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 10px;
        font-weight: bold;
        letter-spacing: 1px;
    }
    .badge-live    { background: rgba(0,255,136,0.2); color: #00ff88; border: 1px solid #00ff88; }
    .badge-paper   { background: rgba(0,212,255,0.2); color: #00d4ff; border: 1px solid #00d4ff; }
    .badge-warning { background: rgba(255,170,0,0.2); color: #ffaa00; border: 1px solid #ffaa00; }

    /* Section Headers */
    .section-header {
        color: #00d4ff;
        font-size: 13px;
        font-weight: bold;
        letter-spacing: 2px;
        text-transform: uppercase;
        border-bottom: 1px solid #1e3a5f;
        padding-bottom: 8px;
        margin-bottom: 15px;
    }

    /* Alert */
    .alert-box {
        border-radius: 6px;
        padding: 12px 15px;
        margin: 8px 0;
        font-size: 12px;
        border-left: 3px solid;
    }
    .alert-success { background: rgba(0,255,136,0.1); border-color: #00ff88; color: #00ff88; }
    .alert-danger  { background: rgba(255,68,68,0.1);  border-color: #ff4444; color: #ff4444; }
    .alert-warning { background: rgba(255,170,0,0.1);  border-color: #ffaa00; color: #ffaa00; }
    .alert-info    { background: rgba(0,212,255,0.1);  border-color: #00d4ff; color: #00d4ff; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #00d4ff; }

    /* Streamlit Overrides */
    .stButton > button {
        background: linear-gradient(135deg, #0d1526, #1a2744);
        color: #00d4ff;
        border: 1px solid #1e3a5f;
        border-radius: 6px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 12px;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover { border-color: #00d4ff; box-shadow: 0 0 10px rgba(0,212,255,0.3); background: #1a2744; }
    div[data-testid="stMetricValue"] { color: #00d4ff !important; font-family: 'Orbitron', sans-serif !important; }
    .stSelectbox > div > div { background: #0d1526; border-color: #1e3a5f; color: #e0e6f0; }
    .stSlider > div > div > div > div { background: #00d4ff; }
    div.stTabs [data-baseweb="tab-list"] { background: #0d1526; border-bottom: 1px solid #1e3a5f; gap: 0; }
    div.stTabs [data-baseweb="tab"] { background: transparent; color: #6b8cae; border: none; padding: 10px 20px; font-family: 'Share Tech Mono', monospace; font-size: 12px; letter-spacing: 1px; }
    div.stTabs [aria-selected="true"] { background: #1a2744; color: #00d4ff; border-bottom: 2px solid #00d4ff; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA CLASSES
# ============================================================
@dataclass
class TradingSignal:
    symbol:      str
    direction:   str         # BUY / SELL / NEUTRAL
    confidence:  float       # 0.0 – 1.0
    entry_price: float
    stop_loss:   float
    take_profit: float
    timestamp:   datetime.datetime
    timeframe:   str
    indicators:  Dict[str, float] = field(default_factory=dict)
    ml_score:    float = 0.0
    risk_reward: float = 0.0
    notes:       str   = ""

@dataclass
class Trade:
    id:          str
    symbol:      str
    direction:   str
    entry_price: float
    stop_loss:   float
    take_profit: float
    position_size: float
    entry_time:  datetime.datetime
    exit_price:  Optional[float]  = None
    exit_time:   Optional[datetime.datetime] = None
    pnl:         float = 0.0
    pnl_pct:     float = 0.0
    status:      str   = "OPEN"   # OPEN / CLOSED / STOPPED
    exit_reason: str   = ""

@dataclass
class Portfolio:
    initial_equity: float = 100_000.0
    total_equity:   float = 100_000.0
    cash:           float = 100_000.0
    open_positions: Dict  = field(default_factory=dict)
    total_pnl:      float = 0.0
    total_pnl_pct:  float = 0.0
    win_rate:       float = 0.0
    total_trades:   int   = 0
    winning_trades: int   = 0
    losing_trades:  int   = 0
    max_drawdown:   float = 0.0
    sharpe_ratio:   float = 0.0
    equity_curve:   List[float] = field(default_factory=list)

# ============================================================
# DATA FETCHER
# ============================================================
class DataFetcher:
    def __init__(self):
        self.cache: Dict[str, pd.DataFrame] = {}
        self.cache_time: Dict[str, datetime.datetime] = {}
        self.cache_ttl = 60  # seconds

    def fetch(self, symbol: str, period: str = "3mo", interval: str = "1h") -> pd.DataFrame:
        key = f"{symbol}_{period}_{interval}"
        now = datetime.datetime.now()

        if (key in self.cache and
                key in self.cache_time and
                (now - self.cache_time[key]).seconds < self.cache_ttl):
            return self.cache[key]

        if YF_AVAILABLE:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                if df is not None and len(df) > 50:
                    df.index = pd.to_datetime(df.index)
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    self.cache[key] = df
                    self.cache_time[key] = now
                    return df
            except Exception as e:
                logging.error(f"Error: {e}")

        return self._generate_synthetic(symbol, period, interval)

    def _generate_synthetic(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        period_map  = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}
        interval_map = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}

        days     = period_map.get(period, 90)
        mins     = interval_map.get(interval, 60)
        n        = min((days * 1440) // mins, 1000)

        np.random.seed(hash(symbol) % 2**31)
        base_prices = {"AAPL": 185, "GOOGL": 175, "MSFT": 415, "AMZN": 195,
                       "TSLA": 250, "NVDA": 875, "META": 550, "SPY": 500,
                       "BTC-USD": 67000, "ETH-USD": 3500}
        start = base_prices.get(symbol, 100)

        returns  = np.random.normal(0.0002, 0.015, n)
        prices   = start * np.exp(np.cumsum(returns))
        noise    = np.random.uniform(0.001, 0.008, n)

        opens    = prices
        closes   = prices * (1 + np.random.normal(0, 0.003, n))
        highs    = np.maximum(opens, closes) * (1 + noise)
        lows     = np.minimum(opens, closes) * (1 - noise)
        volumes  = np.random.lognormal(15, 0.5, n).astype(int)

        end   = datetime.datetime.now()
        start_dt = end - datetime.timedelta(minutes=mins * n)
        idx   = pd.date_range(start=start_dt, periods=n, freq=f"{mins}min")

        return pd.DataFrame({
            "Open": opens, "High": highs, "Low": lows,
            "Close": closes, "Volume": volumes
        }, index=idx)

# ============================================================
# TECHNICAL ANALYSIS ENGINE
# ============================================================
class TechnicalAnalysis:

    @staticmethod
    def compute(df: pd.DataFrame) -> Dict[str, Any]:
        indicators: Dict[str, Any] = {}
        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        volume = df["Volume"] if "Volume" in df.columns else pd.Series(np.ones(len(df)), index=df.index)

        if not TA_AVAILABLE:
            return TechnicalAnalysis._fallback(df)

        try:
            # RSI
            if len(df) >= 14:
                indicators["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]

            # EMAs
            for w in [20, 50, 200]:
                if len(df) >= w:
                    indicators[f"ema_{w}"] = ta.trend.EMAIndicator(close, window=w).ema_indicator().iloc[-1]

            # Bollinger Bands
            if len(df) >= 20:
                bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
                indicators["bb_upper"]  = bb.bollinger_hband().iloc[-1]
                indicators["bb_middle"] = bb.bollinger_mavg().iloc[-1]
                indicators["bb_lower"]  = bb.bollinger_lband().iloc[-1]
                indicators["bb_width"]  = (indicators["bb_upper"] - indicators["bb_lower"]) / indicators["bb_middle"]
                indicators["bb_pct"]    = bb.bollinger_pband().iloc[-1]

            # MACD
            if len(df) >= 26:
                macd_obj = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
                indicators["macd"]          = macd_obj.macd().iloc[-1]
                indicators["macd_signal"]   = macd_obj.macd_signal().iloc[-1]
                indicators["macd_hist"]     = macd_obj.macd_diff().iloc[-1]
                prev_hist = macd_obj.macd_diff().iloc[-2] if len(df) > 26 else 0
                indicators["macd_crossover"] = (
                    1 if (prev_hist < 0 and indicators["macd_hist"] > 0) else
                    -1 if (prev_hist > 0 and indicators["macd_hist"] < 0) else 0
                )

            # ATR
            if len(df) >= 14:
                indicators["atr"] = ta.volatility.AverageTrueRange(
                    high, low, close, window=14).average_true_range().iloc[-1]

            # Stochastic
            if len(df) >= 14:
                stoch = ta.momentum.StochasticOscillator(high, low, close, window=14)
                indicators["stoch_k"] = stoch.stoch().iloc[-1]
                indicators["stoch_d"] = stoch.stoch_signal().iloc[-1]

            # OBV
            if len(df) >= 10:
                obv_series = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
                indicators["obv"]       = obv_series.iloc[-1]
                indicators["obv_trend"] = 1 if obv_series.iloc[-1] > obv_series.iloc[-5] else -1

            # ADX
            if len(df) >= 14:
                adx_obj = ta.trend.ADXIndicator(high, low, close, window=14)
                indicators["adx"]       = adx_obj.adx().iloc[-1]
                indicators["adx_pos"]   = adx_obj.adx_pos().iloc[-1]
                indicators["adx_neg"]   = adx_obj.adx_neg().iloc[-1]

            # Williams %R
            if len(df) >= 14:
                indicators["williams_r"] = ta.momentum.WilliamsRIndicator(
                    high, low, close, lbp=14).williams_r().iloc[-1]

            # CCI
            if len(df) >= 20:
                indicators["cci"] = ta.trend.CCIIndicator(
                    high, low, close, window=20).cci().iloc[-1]

            # Price action
            indicators["price"]   = close.iloc[-1]
            indicators["prev"]    = close.iloc[-2] if len(close) > 1 else close.iloc[-1]
            indicators["change"]  = (indicators["price"] - indicators["prev"]) / indicators["prev"]
            indicators["high_5d"] = high.iloc[-5:].max()  if len(df) >= 5 else high.iloc[-1]
            indicators["low_5d"]  = low.iloc[-5:].min()   if len(df) >= 5 else low.iloc[-1]
            indicators["vol_avg"] = volume.iloc[-20:].mean() if len(df) >= 20 else volume.mean()
            indicators["vol_ratio"] = volume.iloc[-1] / indicators["vol_avg"] if indicators["vol_avg"] > 0 else 1

            # Trend
            p = indicators["price"]
            indicators["trend_short"] = (
                "bullish" if (indicators.get("ema_20", p) > indicators.get("ema_50", p)) else "bearish"
            )
            indicators["trend_long"] = (
                "bullish" if (indicators.get("ema_50", p) > indicators.get("ema_200", p)) else "bearish"
            )
            indicators["above_ema_20"]  = p > indicators.get("ema_20",  p)
            indicators["above_ema_50"]  = p > indicators.get("ema_50",  p)
            indicators["above_ema_200"] = p > indicators.get("ema_200", p)

        except Exception as e:
            logging.error(f"TA error: {e}")
            return TechnicalAnalysis._fallback(df)

        return indicators

    @staticmethod
    def _fallback(df: pd.DataFrame) -> Dict[str, Any]:
        close  = df["Close"]
        price  = close.iloc[-1]
        prev   = close.iloc[-2] if len(close) > 1 else price

        # Manual RSI
        delta  = close.diff().dropna()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rs     = gain / (loss + 1e-9)
        rsi_s  = 100 - 100 / (1 + rs)

        # Manual MACD
        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd_s = ema12 - ema26
        sig_s  = macd_s.ewm(span=9, adjust=False).mean()

        return {
            "price":  price, "prev": prev,
            "change": (price - prev) / prev,
            "rsi":    float(rsi_s.iloc[-1]) if len(rsi_s) else 50.0,
            "macd":        float(macd_s.iloc[-1]),
            "macd_signal": float(sig_s.iloc[-1]),
            "macd_hist":   float((macd_s - sig_s).iloc[-1]),
            "ema_20":  float(close.ewm(span=20,  adjust=False).mean().iloc[-1]),
            "ema_50":  float(close.ewm(span=50,  adjust=False).mean().iloc[-1]),
            "ema_200": float(close.ewm(span=200, adjust=False).mean().iloc[-1]),
            "atr":     float(df["High"].sub(df["Low"]).rolling(14).mean().iloc[-1]),
            "adx":     25.0, "stoch_k": 50.0, "stoch_d": 50.0,
            "trend_short": "bullish" if price > close.ewm(span=20, adjust=False).mean().iloc[-1] else "bearish",
            "trend_long":  "bullish" if price > close.ewm(span=50, adjust=False).mean().iloc[-1] else "bearish",
            "above_ema_20":  price > close.ewm(span=20,  adjust=False).mean().iloc[-1],
            "above_ema_50":  price > close.ewm(span=50,  adjust=False).mean().iloc[-1],
            "above_ema_200": price > close.ewm(span=200, adjust=False).mean().iloc[-1],
            "vol_ratio": 1.0, "bb_pct": 0.5,
        }

# ============================================================
# ML SIGNAL ENGINE
# ============================================================
class MLEngine:
    def __init__(self):
        self.models: Dict = {}
        self.scalers: Dict = {}
        self.trained: Dict[str, bool] = {}
        self.feature_importance: Dict = {}

    def _build_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if len(df) < 50:
            return None

        feats = pd.DataFrame(index=df.index)
        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]
        vol   = df["Volume"] if "Volume" in df.columns else pd.Series(np.ones(len(df)), index=df.index)

        # Returns
        for w in [1, 3, 5, 10, 20]:
            feats[f"ret_{w}"] = close.pct_change(w)

        # Volatility
        for w in [5, 10, 20]:
            feats[f"vol_{w}"] = close.pct_change().rolling(w).std()

        # EMAs & distance
        for w in [10, 20, 50]:
            ema = close.ewm(span=w, adjust=False).mean()
            feats[f"ema_{w}"]      = ema
            feats[f"ema_{w}_dist"] = (close - ema) / (ema + 1e-9)

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        feats["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-9))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        feats["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

        # BB pct
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        feats["bb_pct"] = (close - (sma20 - 2*std20)) / (4*std20 + 1e-9)

        # ATR
        tr = pd.concat([high - low,
                        (high - close.shift()).abs(),
                        (low  - close.shift()).abs()], axis=1).max(axis=1)
        feats["atr_pct"] = tr.rolling(14).mean() / (close + 1e-9)

        # Volume
        vol_ma = vol.rolling(20).mean()
        feats["vol_ratio"] = vol / (vol_ma + 1e-9)

        # Price position (52-bar range)
        feats["price_pos"] = (close - low.rolling(52).min()) / (
            high.rolling(52).max() - low.rolling(52).min() + 1e-9)

        return feats.dropna()

    def train(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        if not SKLEARN_AVAILABLE:
            return {"success": False, "message": "scikit-learn not available"}

        feats = self._build_features(df)
        if feats is None or len(feats) < 50:
            return {"success": False, "message": "Insufficient data for training"}

        try:
            # Target: 1 if price goes up in next 5 bars, else 0
            close = df["Close"].reindex(feats.index)
            future_returns = close.shift(-5) / close - 1
            target = (future_returns > 0).astype(int)

            # Align
            valid_idx = feats.index.intersection(target.dropna().index)
            X = feats.loc[valid_idx].values
            y = target.loc[valid_idx].values

            if len(X) < 50:
                return {"success": False, "message": "Not enough aligned samples"}

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc  = scaler.transform(X_test)

            # Random Forest
            rf = RandomForestClassifier(
                n_estimators=100, max_depth=6,
                min_samples_leaf=5, random_state=42, n_jobs=-1
            )
            rf.fit(X_train_sc, y_train)
            rf_acc = accuracy_score(y_test, rf.predict(X_test_sc))

            # Gradient Boosting
            gb = GradientBoostingClassifier(
                n_estimators=100, max_depth=4,
                learning_rate=0.05, random_state=42
            )
            gb.fit(X_train_sc, y_train)
            gb_acc = accuracy_score(y_test, gb.predict(X_test_sc))

            self.models[symbol]  = {"rf": rf, "gb": gb}
            self.scalers[symbol] = scaler
            self.trained[symbol] = True

            # Feature importance
            feat_names = feats.columns.tolist()
            self.feature_importance[symbol] = dict(
                zip(feat_names, rf.feature_importances_)
            )

            return {
                "success":  True,
                "rf_acc":   rf_acc,
                "gb_acc":   gb_acc,
                "samples":  len(X),
                "features": len(feat_names)
            }

        except Exception as e:
            logging.error(f"ML training error: {e}")
            return {"success": False, "message": str(e)}

    def predict(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        if not SKLEARN_AVAILABLE or symbol not in self.trained:
            return {"score": 0.5, "direction": "NEUTRAL", "confidence": 0.0}

        feats = self._build_features(df)
        if feats is None or len(feats) == 0:
            return {"score": 0.5, "direction": "NEUTRAL", "confidence": 0.0}

        try:
            X_last = feats.iloc[-1:].values
            X_sc   = self.scalers[symbol].transform(X_last)

            rf_prob = self.models[symbol]["rf"].predict_proba(X_sc)[0]
            gb_prob = self.models[symbol]["gb"].predict_proba(X_sc)[0]

            # Ensemble average
            avg_prob = (rf_prob + gb_prob) / 2
            score    = float(avg_prob[1])   # probability of UP

            direction = (
                "BUY"  if score > 0.55 else
                "SELL" if score < 0.45 else
                "NEUTRAL"
            )
            confidence = abs(score - 0.5) * 2   # 0-1 scale

            return {
                "score":      score,
                "direction":  direction,
                "confidence": confidence,
                "rf_prob":    float(rf_prob[1]),
                "gb_prob":    float(gb_prob[1])
            }

        except Exception as e:
            logging.error(f"ML predict error: {e}")
            return {"score": 0.5, "direction": "NEUTRAL", "confidence": 0.0}


# ============================================================
# SIGNAL GENERATOR
# ============================================================
class SignalGenerator:
    def __init__(self, ml_engine: MLEngine):
        self.ml = ml_engine

    def generate(
        self,
        symbol:    str,
        df:        pd.DataFrame,
        indicators: Dict[str, Any],
        risk_pct:  float = 1.0,
        equity:    float = 100_000.0
    ) -> TradingSignal:

        price = indicators.get("price", df["Close"].iloc[-1])
        atr   = indicators.get("atr",   price * 0.01)

        # ── Scoring ───────────────────────────────────────────
        bull_score = 0
        bear_score = 0
        total_signals = 0

        # RSI
        rsi = indicators.get("rsi", 50)
        if rsi < 30:
            bull_score += 2; total_signals += 2
        elif rsi < 45:
            bull_score += 1; total_signals += 1
        elif rsi > 70:
            bear_score += 2; total_signals += 2
        elif rsi > 55:
            bear_score += 1; total_signals += 1
        else:
            total_signals += 1

        # MACD
        macd_hist = indicators.get("macd_hist", 0)
        macd_cross = indicators.get("macd_crossover", 0)
        if macd_cross == 1:
            bull_score += 2; total_signals += 2
        elif macd_cross == -1:
            bear_score += 2; total_signals += 2
        elif macd_hist > 0:
            bull_score += 1; total_signals += 1
        elif macd_hist < 0:
            bear_score += 1; total_signals += 1
        else:
            total_signals += 1

        # EMA trend
        if indicators.get("above_ema_20") and indicators.get("above_ema_50"):
            bull_score += 2; total_signals += 2
        elif not indicators.get("above_ema_20") and not indicators.get("above_ema_50"):
            bear_score += 2; total_signals += 2
        else:
            total_signals += 2

        # Bollinger Bands
        bb_pct = indicators.get("bb_pct", 0.5)
        if bb_pct < 0.1:
            bull_score += 1; total_signals += 1
        elif bb_pct > 0.9:
            bear_score += 1; total_signals += 1
        else:
            total_signals += 1

        # Stochastic
        stoch_k = indicators.get("stoch_k", 50)
        stoch_d = indicators.get("stoch_d", 50)
        if stoch_k < 20 and stoch_d < 20:
            bull_score += 1; total_signals += 1
        elif stoch_k > 80 and stoch_d > 80:
            bear_score += 1; total_signals += 1
        else:
            total_signals += 1

        # ADX trend strength
        adx = indicators.get("adx", 25)
        adx_pos = indicators.get("adx_pos", 0)
        adx_neg = indicators.get("adx_neg", 0)
        if adx > 25:
            if adx_pos > adx_neg:
                bull_score += 1; total_signals += 1
            else:
                bear_score += 1; total_signals += 1
        else:
            total_signals += 1

        # OBV trend
        obv_trend = indicators.get("obv_trend", 0)
        if obv_trend == 1:
            bull_score += 1; total_signals += 1
        elif obv_trend == -1:
            bear_score += 1; total_signals += 1
        else:
            total_signals += 1

        # Volume confirmation
        vol_ratio = indicators.get("vol_ratio", 1.0)
        if vol_ratio > 1.5:
            if bull_score > bear_score:
                bull_score += 1
            else:
                bear_score += 1
            total_signals += 1
        else:
            total_signals += 1

        # Williams %R
        wr = indicators.get("williams_r", -50)
        if wr < -80:
            bull_score += 1; total_signals += 1
        elif wr > -20:
            bear_score += 1; total_signals += 1
        else:
            total_signals += 1

        # CCI
        cci = indicators.get("cci", 0)
        if cci < -100:
            bull_score += 1; total_signals += 1
        elif cci > 100:
            bear_score += 1; total_signals += 1
        else:
            total_signals += 1

        # ── ML Score ─────────────────────────────────────────
        ml_result = self.ml.predict(symbol, df)
        ml_score  = ml_result["score"]

        ml_weight = 3
        total_signals += ml_weight
        bull_score += int(ml_score * ml_weight)
        bear_score += int((1 - ml_score) * ml_weight)

        # ── Determine Direction ───────────────────────────────
        net = bull_score - bear_score
        raw_confidence = max(bull_score, bear_score) / max(total_signals, 1)

        if net >= 3:
            direction  = "BUY"
            confidence = min(raw_confidence + 0.05, 1.0)
        elif net <= -3:
            direction  = "SELL"
            confidence = min(raw_confidence + 0.05, 1.0)
        else:
            direction  = "NEUTRAL"
            confidence = 1 - raw_confidence

        # ── Risk Management ───────────────────────────────────
        atr_mult_sl = 1.5
        atr_mult_tp = 3.0

        if direction == "BUY":
            stop_loss   = price - atr * atr_mult_sl
            take_profit = price + atr * atr_mult_tp
        elif direction == "SELL":
            stop_loss   = price + atr * atr_mult_sl
            take_profit = price - atr * atr_mult_tp
        else:
            stop_loss   = price - atr * atr_mult_sl
            take_profit = price + atr * atr_mult_tp

        risk_per_share = abs(price - stop_loss)
        risk_reward    = (
            abs(take_profit - price) / risk_per_share
            if risk_per_share > 0 else 0
        )

        # Position sizing (fixed % risk)
        risk_amount   = equity * (risk_pct / 100)
        position_size = (
            risk_amount / risk_per_share
            if risk_per_share <= 1e-9
        else
    position_size = risk_amount / risk_per_share
        )

        notes_parts = []
        if rsi < 30:  notes_parts.append("RSI oversold")
        if rsi > 70:  notes_parts.append("RSI overbought")
        if macd_cross ==  1: notes_parts.append("MACD bullish cross")
        if macd_cross == -1: notes_parts.append("MACD bearish cross")
        if adx > 25:  notes_parts.append(f"Strong trend ADX={adx:.0f}")
        if vol_ratio > 1.5: notes_parts.append(f"High volume {vol_ratio:.1f}x")

        return TradingSignal(
            symbol      = symbol,
            direction   = direction,
            confidence  = confidence,
            entry_price = price,
            stop_loss   = stop_loss,
            take_profit = take_profit,
            timestamp   = datetime.datetime.now(),
            timeframe   = "1h",
            indicators  = indicators,
            ml_score    = ml_score,
            risk_reward = risk_reward,
            notes       = " | ".join(notes_parts)
        )


# ============================================================
# PORTFOLIO MANAGER
# ============================================================
class PortfolioManager:
    def __init__(self, initial_equity: float = 100_000.0):
        self.portfolio  = Portfolio(
            initial_equity=initial_equity,
            total_equity=initial_equity,
            cash=initial_equity
        )
        self.trades:    List[Trade] = []
        self.trade_counter = 0

    def execute_trade(self, signal: TradingSignal, position_size: float) -> Optional[Trade]:
        if signal.direction == "NEUTRAL":
            return None

        cost = signal.entry_price * position_size
        if cost > self.portfolio.cash:
            position_size = self.portfolio.cash / signal.entry_price
            cost = signal.entry_price * position_size

        if position_size <= 0:
            return None

        self.trade_counter += 1
        trade = Trade(
            id            = f"T{self.trade_counter:04d}",
            symbol        = signal.symbol,
            direction     = signal.direction,
            entry_price   = signal.entry_price,
            stop_loss     = signal.stop_loss,
            take_profit   = signal.take_profit,
            position_size = position_size,
            entry_time    = signal.timestamp,
            status        = "OPEN"
        )

        self.portfolio.cash -= cost
        self.portfolio.open_positions[trade.id] = trade
        self.trades.append(trade)
        return trade

    def update_positions(self, current_prices: Dict[str, float]):
        closed_ids = []

        for trade_id, trade in self.portfolio.open_positions.items():
            price = current_prices.get(trade.symbol, trade.entry_price)

            # Check stop loss / take profit
            hit_sl = hit_tp = False
            if trade.direction == "BUY":
                hit_sl = price <= trade.stop_loss
                hit_tp = price >= trade.take_profit
            elif trade.direction == "SELL":
                hit_sl = price >= trade.stop_loss
                hit_tp = price <= trade.take_profit

            if hit_sl or hit_tp:
                trade.exit_price  = price
                trade.exit_time   = datetime.datetime.now()
                trade.exit_reason = "TAKE_PROFIT" if hit_tp else "STOP_LOSS"
                trade.status      = "CLOSED"

                if trade.direction == "BUY":
                    trade.pnl = (price - trade.entry_price) * trade.position_size
                else:
                    trade.pnl = (trade.entry_price - price) * trade.position_size

                trade.pnl_pct = trade.pnl / (trade.entry_price * trade.position_size)
                self.portfolio.cash += price * trade.position_size

                # Update stats
                self.portfolio.total_trades  += 1
                if trade.pnl > 0:
                    self.portfolio.winning_trades += 1
                else:
                    self.portfolio.losing_trades += 1

                closed_ids.append(trade_id)

        for tid in closed_ids:
            del self.portfolio.open_positions[tid]

        # Recalculate equity
        open_value = sum(
            t.entry_price * t.position_size
            for t in self.portfolio.open_positions.values()
        )
        self.portfolio.total_equity = self.portfolio.cash + open_value
        self.portfolio.total_pnl    = self.portfolio.total_equity - self.portfolio.initial_equity
        self.portfolio.total_pnl_pct = (
            self.portfolio.total_pnl / self.portfolio.initial_equity * 100
        )
        self.portfolio.win_rate = (
            self.portfolio.winning_trades / max(self.portfolio.total_trades, 1) * 100
        )
        self.portfolio.equity_curve.append(self.portfolio.total_equity)

        # Drawdown
        if self.portfolio.equity_curve:
            peak = max(self.portfolio.equity_curve)
            self.portfolio.max_drawdown = (
                (peak - self.portfolio.total_equity) / peak * 100
            )

    def get_closed_trades(self) -> List[Trade]:
        return [t for t in self.trades if t.status == "CLOSED"]

    def get_open_trades(self) -> List[Trade]:
        return list(self.portfolio.open_positions.values())


# ============================================================
# CHART BUILDER
# ============================================================
class ChartBuilder:

    @staticmethod
    def candlestick(
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        signal: Optional[TradingSignal] = None,
        title: str = "Price Chart"
    ) -> Optional[Any]:

        if not PLOTLY_AVAILABLE or df is None or len(df) < 2:
            return None

        try:
            display_df = df.tail(200)

            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.50, 0.18, 0.16, 0.16],
                subplot_titles=("", "Volume", "RSI", "MACD")
            )

            # ── Candlesticks ──────────────────────────────────
            fig.add_trace(go.Candlestick(
                x=display_df.index,
                open=display_df["Open"], high=display_df["High"],
                low=display_df["Low"],  close=display_df["Close"],
                name="Price",
                increasing_line_color="#00ff88",
                decreasing_line_color="#ff4444",
                increasing_fillcolor="#00ff88",
                decreasing_fillcolor="#ff4444"
            ), row=1, col=1)

            # ── EMAs ─────────────────────────────────────────
            ema_styles = [
                (20,  "#00d4ff", "EMA 20"),
                (50,  "#ffaa00", "EMA 50"),
                (200, "#ff69b4", "EMA 200")
            ]
            for span, color, name in ema_styles:
                if len(df) >= span:
                    ema_vals = df["Close"].ewm(span=span, adjust=False).mean().tail(200)
                    fig.add_trace(go.Scatter(
                        x=display_df.index, y=ema_vals,
                        name=name, line=dict(color=color, width=1.2),
                        opacity=0.8
                    ), row=1, col=1)

            # ── Bollinger Bands ───────────────────────────────
            if len(df) >= 20:
                sma   = df["Close"].rolling(20).mean().tail(200)
                std   = df["Close"].rolling(20).std().tail(200)
                upper = sma + 2 * std
                lower = sma - 2 * std

                fig.add_trace(go.Scatter(
                    x=display_df.index, y=upper,
                    name="BB Upper", line=dict(color="#6b8cae", width=1, dash="dot"),
                    opacity=0.5
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=display_df.index, y=lower,
                    name="BB Lower", line=dict(color="#6b8cae", width=1, dash="dot"),
                    fill="tonexty", fillcolor="rgba(107,140,174,0.05)",
                    opacity=0.5
                ), row=1, col=1)

            # ── Signal Lines ──────────────────────────────────
            if signal and signal.direction != "NEUTRAL":
                color = "#00ff88" if signal.direction == "BUY" else "#ff4444"
                for level, label in [
                    (signal.entry_price, "Entry"),
                    (signal.stop_loss,   "Stop Loss"),
                    (signal.take_profit, "Take Profit")
                ]:
                    line_color = (
                        "#ffaa00" if label == "Entry" else
                        "#ff4444" if label == "Stop Loss" else
                        "#00ff88"
                    )
                    fig.add_hline(
                        y=level, line_dash="dash",
                        line_color=line_color, line_width=1.5,
                        annotation_text=f"{label}: ${level:.2f}",
                        annotation_position="right",
                        annotation_font_color=line_color,
                        row=1, col=1
                    )

            # ── Volume ────────────────────────────────────────
            colors = [
                "#00ff88" if c >= o else "#ff4444"
                for c, o in zip(display_df["Close"], display_df["Open"])
            ]
            fig.add_trace(go.Bar(
                x=display_df.index, y=display_df["Volume"],
                name="Volume", marker_color=colors, opacity=0.7
            ), row=2, col=1)

            # ── RSI ───────────────────────────────────────────
            if len(df) >= 14:
                delta  = df["Close"].diff()
                gain   = delta.clip(lower=0).rolling(14).mean()
                loss   = (-delta.clip(upper=0)).rolling(14).mean()
                rsi_s  = (100 - 100 / (1 + gain / (loss + 1e-9))).tail(200)

                fig.add_trace(go.Scatter(
                    x=display_df.index, y=rsi_s,
                    name="RSI", line=dict(color="#00d4ff", width=1.5)
                ), row=3, col=1)

                for level, color in [(70, "#ff4444"), (30, "#00ff88"), (50, "#6b8cae")]:
                    fig.add_hline(
                        y=level, line_dash="dot",
                        line_color=color, line_width=1, opacity=0.5,
                        row=3, col=1
                    )

            # ── MACD ─────────────────────────────────────────
            if len(df) >= 26:
                ema12  = df["Close"].ewm(span=12, adjust=False).mean()
                ema26  = df["Close"].ewm(span=26, adjust=False).mean()
                macd_l = (ema12 - ema26).tail(200)
                sig_l  = macd_l.ewm(span=9, adjust=False).mean()
                hist_l = macd_l - sig_l

                fig.add_trace(go.Scatter(
                    x=display_df.index, y=macd_l,
                    name="MACD", line=dict(color="#00d4ff", width=1.5)
                ), row=4, col=1)
                fig.add_trace(go.Scatter(
                    x=display_df.index, y=sig_l,
                    name="Signal", line=dict(color="#ffaa00", width=1.5)
                ), row=4, col=1)
                fig.add_trace(go.Bar(
                    x=display_df.index, y=hist_l, name="Histogram",
                    marker_color=["#00ff88" if v >= 0 else "#ff4444" for v in hist_l],
                    opacity=0.7
                ), row=4, col=1)

            # ── Layout ───────────────────────────────────────
            fig.update_layout(
                title=dict(text=title, font=dict(color="#00d4ff", size=14, family="Orbitron")),
                paper_bgcolor="#0a0e1a",
                plot_bgcolor="#0d1526",
                font=dict(color="#e0e6f0", family="Share Tech Mono"),
                xaxis_rangeslider_visible=False,
                height=700,
                showlegend=True,
                legend=dict(
                    bgcolor="rgba(13,21,38,0.8)",
                    bordercolor="#1e3a5f",
                    borderwidth=1,
                    font=dict(size=10)
                ),
                margin=dict(l=60, r=80, t=40, b=40)
            )

            for i in range(1, 5):
                fig.update_xaxes(
                    gridcolor="#1e3a5f", zerolinecolor="#1e3a5f",
                    showgrid=True, row=i, col=1
                )
                fig.update_yaxes(
                    gridcolor="#1e3a5f", zerolinecolor="#1e3a5f",
                    showgrid=True, row=i, col=1
                )

            return fig

        except Exception as e:
            logging.error(f"Chart error: {e}")
            return None

    @staticmethod
    def equity_curve(equity_curve: List[float], initial: float) -> Optional[Any]:
        if not PLOTLY_AVAILABLE or len(equity_curve) < 2:
            return None

        try:
            x   = list(range(len(equity_curve)))
            pnl = [(v - initial) / initial * 100 for v in equity_curve]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=pnl,
                fill="tozeroy",
                fillcolor="rgba(0,212,255,0.1)",
                line=dict(color="#00d4ff", width=2),
                name="Equity %"
            ))

            # Drawdown shading
            peak = pnl[0]
            for i, v in enumerate(pnl):
                if v > peak:
                    peak = v
                if v < peak:
                    fig.add_vrect(
                        x0=i, x1=i + 1,
                        fillcolor="rgba(255,68,68,0.05)",
                        layer="below", line_width=0
                    )

            fig.add_hline(y=0, line_dash="dash", line_color="#6b8cae", line_width=1)

            fig.update_layout(
                title=dict(text="Equity Curve", font=dict(color="#00d4ff", size=12)),
                paper_bgcolor="#0a0e1a",
                plot_bgcolor="#0d1526",
                font=dict(color="#e0e6f0", family="Share Tech Mono"),
                height=300,
                showlegend=False,
                margin=dict(l=60, r=20, t=40, b=40),
                yaxis=dict(ticksuffix="%", gridcolor="#1e3a5f"),
                xaxis=dict(gridcolor="#1e3a5f")
            )
            return fig

        except Exception as e:
            logging.error(f"Equity curve error: {e}")
            return None

    @staticmethod
    def indicator_radar(indicators: Dict[str, Any]) -> Optional[Any]:
        if not PLOTLY_AVAILABLE:
            return None

        try:
            rsi      = indicators.get("rsi", 50)
            macd_h   = indicators.get("macd_hist", 0)
            adx      = indicators.get("adx", 25)
            bb_pct   = indicators.get("bb_pct", 0.5)
            stoch_k  = indicators.get("stoch_k", 50)
            vol_r    = indicators.get("vol_ratio", 1)

            # Normalize to 0-100
            scores = {
                "RSI\nMomentum":   rsi,
                "MACD\nStrength":  max(0, min(100, (macd_h + 1) * 50)),
                "Trend\nStrength": min(100, adx * 2),
                "BB\nPosition":    bb_pct * 100,
                "Stoch\nK":        stoch_k,"Volume\nRatio":   min(100, vol_r * 50)
            }

            categories = list(scores.keys())
            values     = list(scores.values())
            values    += values[:1]  # close the polygon
            categories += categories[:1]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                fillcolor="rgba(0,212,255,0.15)",
                line=dict(color="#00d4ff", width=2),
                name="Indicators"
            ))

            fig.update_layout(
                polar=dict(
                    bgcolor="#0d1526",
                    radialaxis=dict(
                        visible=True, range=[0, 100],
                        gridcolor="#1e3a5f",
                        tickfont=dict(color="#6b8cae", size=8)
                    ),
                    angularaxis=dict(
                        gridcolor="#1e3a5f",
                        tickfont=dict(color="#e0e6f0", size=9)
                    )
                ),
                paper_bgcolor="#0a0e1a",
                font=dict(color="#e0e6f0", family="Share Tech Mono"),
                title=dict(
                    text="Indicator Radar",
                    font=dict(color="#00d4ff", size=12)
                ),
                height=300,
                margin=dict(l=40, r=40, t=50, b=40),
                showlegend=False
            )
            return fig

        except Exception as e:
            logging.error(f"Radar chart error: {e}")
            return None


# ============================================================
# TECHNICAL INDICATOR ENGINE
# ============================================================

# ============================================================
# STREAMLIT UI
# ============================================================
def apply_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Share Tech Mono', monospace;
        background-color: #0a0e1a;
        color: #e0e6f0;
    }
    .stApp { background-color: #0a0e1a; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1526 0%, #0a0e1a 100%);
        border-right: 1px solid #1e3a5f;
    }

    /* ── Cards ── */
    .metric-card {
        background: linear-gradient(135deg, #0d1526 0%, #111827 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,212,255,0.15);
    }
    .metric-label {
        color: #6b8cae;
        font-size: 11px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 22px;
        font-weight: 700;
        color: #00d4ff;
    }
    .metric-value.positive { color: #00ff88; }
    .metric-value.negative { color: #ff4444; }

    /* ── Signal Badge ── */
    .signal-badge {
        display: inline-block;
        padding: 8px 24px;
        border-radius: 24px;
        font-family: 'Orbitron', sans-serif;
        font-size: 20px;
        font-weight: 900;
        letter-spacing: 3px;
        text-align: center;
    }
    .signal-BUY  {
        background: rgba(0,255,136,0.15);
        border: 2px solid #00ff88;
        color: #00ff88;
        box-shadow: 0 0 20px rgba(0,255,136,0.3);
    }
    .signal-SELL {
        background: rgba(255,68,68,0.15);
        border: 2px solid #ff4444;
        color: #ff4444;
        box-shadow: 0 0 20px rgba(255,68,68,0.3);
    }
    .signal-NEUTRAL {
        background: rgba(107,140,174,0.15);
        border: 2px solid #6b8cae;
        color: #6b8cae;
    }

    /* ── Headers ── */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #00d4ff;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f, #0d1526);
        color: #00d4ff;
        border: 1px solid #00d4ff;
        border-radius: 8px;
        font-family: 'Share Tech Mono', monospace;
        letter-spacing: 1px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: rgba(0,212,255,0.2);
        box-shadow: 0 0 12px rgba(0,212,255,0.4);
        transform: translateY(-1px);
    }

    /* ── Inputs ── */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background-color: #0d1526 !important;
        border-color: #1e3a5f !important;
        color: #e0e6f0 !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0d1526;
        border-bottom: 1px solid #1e3a5f;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #6b8cae;
        font-family: 'Share Tech Mono', monospace;
        font-size: 12px;
        letter-spacing: 1px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        color: #00d4ff !important;
        border-bottom: 2px solid #00d4ff !important;
        background-color: rgba(0,212,255,0.05) !important;
    }

    /* ── Divider ── */
    hr { border-color: #1e3a5f; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar       { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #00d4ff; }
    </style>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str, css_class: str = "") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {css_class}">{value}</div>
    </div>
    """


def signal_badge(direction: str) -> str:
    return f'<div class="signal-badge signal-{direction}">{direction}</div>'


# ── Session-state init ────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "ml_engine":       MLEngine(),
        "portfolio_mgr":   PortfolioManager(initial_equity=100_000.0),
        "signal_gen":      None,
        "data_cache":      {},
        "last_analysis":   {},
        "alerts":          [],
        "auto_refresh":    False,
        "refresh_interval": 60,
        "trained_symbols": set(),
        "analysis_history": []
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if st.session_state["signal_gen"] is None:
        st.session_state["signal_gen"] = SignalGenerator(st.session_state["ml_engine"])


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.markdown(
            '<h2 style="font-family:Orbitron;color:#00d4ff;font-size:18px;">'
            '⚡ NEURAL TRADE AI</h2>',
            unsafe_allow_html=True
        )
        st.markdown("---")

        # Symbol selection
        st.markdown(
            '<div style="color:#6b8cae;font-size:11px;letter-spacing:1px;">WATCHLIST</div>',
            unsafe_allow_html=True
        )
        preset_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
            "TSLA", "META", "SPY",   "QQQ",  "BTC-USD",
            "ETH-USD", "GLD", "EURUSD=X", "JPY=X"
        ]
        symbol = st.selectbox(
            "Symbol", preset_symbols, index=0,
            label_visibility="collapsed"
        )
        custom = st.text_input(
            "Custom Symbol", placeholder="e.g. NFLX, AMD",
            label_visibility="collapsed"
        )
        if custom.strip():
            symbol = custom.strip().upper()

        st.markdown("---")

        # Timeframe
        st.markdown(
            '<div style="color:#6b8cae;font-size:11px;letter-spacing:1px;">TIMEFRAME</div>',
            unsafe_allow_html=True
        )
        timeframe_map = {
            "1 Hour":  ("6mo",  "1h"),
            "4 Hours": ("1y",   "1h"),
            "Daily":   ("2y",   "1d"),
            "Weekly":  ("5y",   "1wk")
        }
        tf_label = st.selectbox(
            "Timeframe", list(timeframe_map.keys()),
            label_visibility="collapsed"
        )
        period, interval = timeframe_map[tf_label]

        st.markdown("---")

        # Risk settings
        st.markdown(
            '<div style="color:#6b8cae;font-size:11px;letter-spacing:1px;">RISK MANAGEMENT</div>',
            unsafe_allow_html=True
        )
        risk_pct = st.slider(
            "Risk per Trade (%)", 0.5, 5.0, 1.0, 0.25,
            help="Percentage of equity risked per trade"
        )
        equity = st.number_input(
            "Account Equity ($)", 1000, 10_000_000,
            value=100_000, step=1000
        )

        st.markdown("---")

        # ML options
        st.markdown(
            '<div style="color:#6b8cae;font-size:11px;letter-spacing:1px;">ML ENGINE</div>',
            unsafe_allow_html=True
        )
        auto_train = st.checkbox("Auto-train on load", value=True)
        show_importance = st.checkbox("Show feature importance", value=True)

        st.markdown("---")

        # Auto-refresh
        st.markdown(
            '<div style="color:#6b8cae;font-size:11px;letter-spacing:1px;">AUTO REFRESH</div>',
            unsafe_allow_html=True
        )
        auto_refresh = st.checkbox(
            "Enable auto-refresh",
            value=st.session_state.get("auto_refresh", False)
        )
        st.session_state["auto_refresh"] = auto_refresh
        if auto_refresh:
            refresh_interval = st.slider("Interval (seconds)", 30, 300, 60, 10)
            st.session_state["refresh_interval"] = refresh_interval

        st.markdown("---")

        # Info
        st.markdown("""
        <div style="color:#6b8cae;font-size:10px;line-height:1.8;padding:8px;
                    background:rgba(0,212,255,0.03);border-radius:8px;
                    border:1px solid #1e3a5f;">
        ⚠️ <b>Disclaimer</b><br>
        For educational purposes only.<br>
        Not financial advice.<br>
        Trade at your own risk.
        </div>
        """, unsafe_allow_html=True)

    return {
        "symbol":         symbol,
        "period":         period,
        "interval":       interval,
        "tf_label":       tf_label,
        "risk_pct":       risk_pct,
        "equity":         equity,
        "auto_train":     auto_train,
        "show_importance": show_importance
    }


# ── Main Analysis ─────────────────────────────────────────────────────────────
def run_analysis(cfg: Dict[str, Any]) -> Dict[str, Any]:
    symbol   = cfg["symbol"]
    period   = cfg["period"]
    interval = cfg["interval"]
    risk_pct = cfg["risk_pct"]
    equity   = cfg["equity"]

    with st.spinner(f"🔄 Fetching data for {symbol}..."):
        df = DataFetcher.fetch(symbol, period, interval)

    if df is None or len(df) < 30:
        st.error(f"❌ Unable to fetch data for {symbol}")
        return {}

    with st.spinner("⚙️ Computing indicators..."):
        indicators = IndicatorEngine.compute(df)

    # ML Training
    ml_engine = st.session_state["ml_engine"]
    if (cfg["auto_train"] and
            symbol not in st.session_state["trained_symbols"] and
            SKLEARN_AVAILABLE):
        with st.spinner("🧠 Training ML models..."):
            result = ml_engine.train(symbol, df)
            if result.get("success"):
                st.session_state["trained_symbols"].add(symbol)
                st.success(
                    f"✅ ML trained — RF: {result['rf_acc']:.1%} | "
                    f"GB: {result['gb_acc']:.1%} | "
                    f"Samples: {result['samples']}"
                )
            else:
                st.warning(f"⚠️ ML training: {result.get('message', 'unknown error')}")

    # Generate signal
    signal_gen = st.session_state["signal_gen"]
    with st.spinner("📊 Generating signal..."):
        signal = signal_gen.generate(
            symbol, df, indicators, risk_pct, equity
        )

    result = {
        "df":         df,
        "indicators": indicators,
        "signal":     signal,
        "symbol":     symbol
    }

    # Cache result and history
    st.session_state["last_analysis"][symbol] = result
    st.session_state["analysis_history"].append({
        "time":      datetime.datetime.now().strftime("%H:%M:%S"),
        "symbol":    symbol,
        "direction": signal.direction,
        "confidence": signal.confidence,
        "price":     signal.entry_price,
        "ml_score":  signal.ml_score
    })
    # Keep last 50 entries
    st.session_state["analysis_history"] = \
        st.session_state["analysis_history"][-50:]

    return result


# ── Tab: Overview ─────────────────────────────────────────────────────────────
def tab_overview(result: Dict[str, Any], cfg: Dict[str, Any]):
    signal     = result["signal"]
    indicators = result["indicators"]
    df         = result["df"]

    # ── Signal Banner ─────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_sig, col_conf, col_rr, col_ml = st.columns(4)

    with col_sig:
        st.markdown(signal_badge(signal.direction), unsafe_allow_html=True)

    with col_conf:
        st.markdown(
            metric_card(
                "CONFIDENCE",
                f"{signal.confidence:.1%}",
                "positive" if signal.confidence > 0.65 else ""
            ),
            unsafe_allow_html=True
        )

    with col_rr:
        st.markdown(
            metric_card(
                "RISK / REWARD",
                f"{signal.risk_reward:.2f}x",
                "positive" if signal.risk_reward >= 2 else ""
            ),
            unsafe_allow_html=True
        )

    with col_ml:
        ml_dir = "positive" if signal.ml_score > 0.55 else \
                 "negative" if signal.ml_score < 0.45 else ""
        st.markdown(
            metric_card("ML SCORE", f"{signal.ml_score:.3f}", ml_dir),
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Price Levels ──────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            metric_card("ENTRY PRICE", f"${signal.entry_price:.2f}"),
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            metric_card("STOP LOSS", f"${signal.stop_loss:.2f}", "negative"),
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            metric_card("TAKE PROFIT", f"${signal.take_profit:.2f}", "positive"),
            unsafe_allow_html=True
        )
    with col4:
        atr = indicators.get("atr", 0)
        st.markdown(
            metric_card("ATR", f"${atr:.4f}"),
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart ─────────────────────────────────────────────────
    chart = ChartBuilder.candlestick(
        df, indicators, signal,
        title=f"{cfg['symbol']} — {cfg['tf_label']}"
    )
    if chart:
        st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("📊 Install plotly for interactive charts")

    # ── Notes ─────────────────────────────────────────────────
    if signal.notes:
        st.markdown(
            f'<div style="background:rgba(0,212,255,0.05);border:1px solid #1e3a5f;'
            f'border-radius:8px;padding:12px;color:#6b8cae;font-size:12px;">'
            f'📝 {signal.notes}</div>',
            unsafe_allow_html=True
        )


# ── Tab: Indicators ───────────────────────────────────────────────────────────
def tab_indicators(result: Dict[str, Any], cfg: Dict[str, Any]):
    indicators = result["indicators"]

    col_radar, col_vals = st.columns([1, 1])

    with col_radar:
        radar = ChartBuilder.indicator_radar(indicators)
        if radar:
            st.plotly_chart(radar, use_container_width=True,
                            config={"displayModeBar": False})

    with col_vals:
        st.markdown("#### Indicator Values")
        rows = [
            ("RSI (14)",        f"{indicators.get('rsi', 0):.2f}",
             "🟢" if indicators.get('rsi', 50) < 40 else
             "🔴" if indicators.get('rsi', 50) > 60 else "⚪"),

            ("MACD",            f"{indicators.get('macd', 0):.4f}",
             "🟢" if indicators.get('macd_hist', 0) > 0 else "🔴"),

            ("MACD Histogram",  f"{indicators.get('macd_hist', 0):.4f}",
             "🟢" if indicators.get('macd_hist', 0) > 0 else "🔴"),

            ("ADX",             f"{indicators.get('adx', 0):.2f}",
             "🟢" if indicators.get('adx', 0) > 25 else "⚪"),

            ("BB %",            f"{indicators.get('bb_pct', 0.5):.3f}",
             "🟢" if indicators.get('bb_pct', 0.5) < 0.2 else
             "🔴" if indicators.get('bb_pct', 0.5) > 0.8 else "⚪"),

            ("Stoch K",         f"{indicators.get('stoch_k', 50):.2f}",
             "🟢" if indicators.get('stoch_k', 50) < 20 else
             "🔴" if indicators.get('stoch_k', 50) > 80 else "⚪"),

            ("Williams %R",     f"{indicators.get('williams_r', -50):.2f}",
             "🟢" if indicators.get('williams_r', -50) < -80 else
             "🔴" if indicators.get('williams_r', -50) > -20 else "⚪"),

            ("CCI",             f"{indicators.get('cci', 0):.2f}",
             "🟢" if indicators.get('cci', 0) < -100 else
             "🔴" if indicators.get('cci', 0) > 100 else "⚪"),

            ("OBV Trend",
             "BULLISH" if indicators.get('obv_trend', 0) == 1 else
             "BEARISH" if indicators.get('obv_trend', 0) == -1 else "NEUTRAL",
             "🟢" if indicators.get('obv_trend', 0) == 1 else
             "🔴" if indicators.get('obv_trend', 0) == -1 else "⚪"),

            ("Volume Ratio",    f"{indicators.get('vol_ratio', 1):.2f}x",
             "🟢" if indicators.get('vol_ratio', 1) > 1.5 else "⚪"),

            ("EMA 20",
             f"{'▲' if indicators.get('above_ema_20') else '▼'} "
             f"${indicators.get('ema_20', 0):.2f}",
             "🟢" if indicators.get('above_ema_20') else "🔴"),

            ("EMA 50",
             f"{'▲' if indicators.get('above_ema_50') else '▼'} "
             f"${indicators.get('ema_50', 0):.2f}",
             "🟢" if indicators.get('above_ema_50') else "🔴"),

            ("EMA 200",
             f"{'▲' if indicators.get('above_ema_200') else '▼'} "
             f"${indicators.get('ema_200', 0):.2f}",
             "🟢" if indicators.get('above_ema_200') else "🔴"),
        ]

        tbl_html = """
        <table style="width:100%;border-collapse:collapse;font-size:12px;">
        <tr style="color:#6b8cae;border-bottom:1px solid #1e3a5f;">
            <th style="text-align:left;padding:6px;">Indicator</th>
            <th style="text-align:right;padding:6px;">Value</th>
            <th style="text-align:center;padding:6px;">Signal</th>
        </tr>
        """
        for name, value, icon in rows:
            tbl_html += (
                f'<tr style="border-bottom:1px solid rgba(30,58,95,0.5);">'
                f'<td style="padding:5px 6px;color:#e0e6f0;">{name}</td>'
                f'<td style="padding:5px 6px;color:#00d4ff;text-align:right;">{value}</td>'
                f'<td style="padding:5px 6px;text-align:center;">{icon}</td>'
                f'</tr>'
            )
        tbl_html += "</table>"
        st.markdown(tbl_html, unsafe_allow_html=True)


# ── Tab: ML Analysis ──────────────────────────────────────────────────────────
def tab_ml(result: Dict[str, Any], cfg: Dict[str, Any]):
    signal    = result["signal"]
    df        = result["df"]
    symbol    = cfg["symbol"]
    ml_engine = st.session_state["ml_engine"]

    st.markdown("#### 🧠 Machine Learning Analysis")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            metric_card("ML Score", f"{signal.ml_score:.4f}",
                        "positive" if signal.ml_score > 0.55 else
                        "negative" if signal.ml_score < 0.45 else ""),
            unsafe_allow_html=True
        )
    with col2:
        ml_direction = (
            "BUY"  if signal.ml_score > 0.55 else
            "SELL" if signal.ml_score < 0.45 else "NEUTRAL"
        )
        st.markdown(
            metric_card("ML Direction", ml_direction,
                        "positive" if ml_direction == "BUY" else
                        "negative" if ml_direction == "SELL" else ""),
            unsafe_allow_html=True
        )
    with col3:
        ml_conf = abs(signal.ml_score - 0.5) * 2
        st.markdown(
            metric_card("ML Confidence", f"{ml_conf:.1%}"),
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature importance chart
    if (cfg.get("show_importance") and
            symbol in ml_engine.feature_importance and
            PLOTLY_AVAILABLE):
        fi     = ml_engine.feature_importance[symbol]
        sorted_fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15])

        fig = go.Figure(go.Bar(
            x=list(sorted_fi.values()),
            y=list(sorted_fi.keys()),
            orientation="h",
            marker=dict(
                color=list(sorted_fi.values()),
                colorscale=[[0, "#1e3a5f"], [1, "#00d4ff"]],
                line=dict(color="#0a0e1a", width=0.5)
            )
        ))
        fig.update_layout(
            title=dict(text="Feature Importance (Random Forest)",
                       font=dict(color="#00d4ff", size=12)),
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#0d1526",
            font=dict(color="#e0e6f0", family="Share Tech Mono", size=10),
            height=400,
            margin=dict(l=120, r=20, t=50, b=40),
            xaxis=dict(gridcolor="#1e3a5f", title="Importance"),
            yaxis=dict(gridcolor="#1e3a5f")
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

    # Manual retrain button
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("🔄 Retrain Models", use_container_width=True):
            with st.spinner("Training..."):
                train_result = ml_engine.train(symbol, df)
            if train_result.get("success"):
                st.session_state["trained_symbols"].add(symbol)
                st.success(
                    f"✅ Retrained — RF: {train_result['rf_acc']:.1%} | "
                    f"GB: {train_result['gb_acc']:.1%}"
                )
            else:
                st.error(f"❌ {train_result.get('message')}")


# ── Tab: Portfolio ────────────────────────────────────────────────────────────
def tab_portfolio(result: Dict[str, Any], cfg: Dict[str, Any]):
    pm        = st.session_state["portfolio_mgr"]
    portfolio = pm.portfolio
    signal    = result["signal"]

    st.markdown("#### 📊 Portfolio Overview")

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("TOTAL EQUITY", f"${portfolio.total_equity:,.2f}", ""),
        ("CASH",         f"${portfolio.cash:,.2f}",         ""),
        ("TOTAL P&L",
         f"${portfolio.total_pnl:,.2f}",
         "positive" if portfolio.total_pnl >= 0 else "negative"),
        ("WIN RATE",
         f"{portfolio.win_rate:.1f}%",
         "positive" if portfolio.win_rate >= 50 else "negative"),
    ]
    for col, (lbl, val, cls) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(metric_card(lbl, val, cls), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col5, col6, col7, col8 = st.columns(4)
    metrics2 = [
        ("OPEN POSITIONS", str(len(portfolio.open_positions)), ""),
        ("TOTAL TRADES",   str(portfolio.total_trades),        ""),
        ("MAX DRAWDOWN",   f"{portfolio.max_drawdown:.2f}%",
         "negative" if portfolio.max_drawdown > 5 else ""),
        ("P&L %",
         f"{portfolio.total_pnl_pct:.2f}%",
         "positive" if portfolio.total_pnl_pct >= 0 else "negative"),
    ]
    for col, (lbl, val, cls) in zip([col5, col6, col7, col8], metrics2):
        with col:
            st.markdown(metric_card(lbl, val, cls), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Equity curve
    if len(portfolio.equity_curve) > 1:
        eq_chart = ChartBuilder.equity_curve(
            portfolio.equity_curve, portfolio.initial_equity
        )
        if eq_chart:
            st.plotly_chart(eq_chart, use_container_width=True,
                            config={"displayModeBar": False})

    # Execute trade button
    st.markdown("#### 💼 Execute Trade")
    col_ex, col_sz, _ = st.columns([1, 1, 2])
    with col_sz:
        manual_size = st.number_input(
            "Position Size", 1, 10000, 100, 1,
            help="Number of shares / units"
        )
    with col_ex:
        if st.button("⚡ Execute Signal", use_container_width=True):
            if signal.direction != "NEUTRAL":
                trade = pm.execute_trade(signal, float(manual_size))
                if trade:
                    st.success(
                        f"✅ Trade executed: {trade.id} — "
                        f"{trade.direction} {manual_size} {signal.symbol} @ "
                        f"${trade.entry_price:.2f}"
                    )
                else:
                    st.warning("⚠️ Could not execute trade (check equity)")
            else:
                st.info("ℹ️ No actionable signal to execute")

    # Open positions table
    open_trades = pm.get_open_trades()
    if open_trades:
        st.markdown("#### 📋 Open Positions")
        rows = []
        for t in open_trades:
            curr_price = result["indicators"].get("price", t.entry_price)
            unr_pnl = (
                (curr_price - t.entry_price) * t.position_size
                if t.direction == "BUY"
                else (t.entry_price - curr_price) * t.position_size
            )
            rows.append({
                "ID":        t.id,
                "Symbol":    t.symbol,
                "Direction": t.direction,
                "Entry":     f"${t.entry_price:.2f}",
                "SL":        f"${t.stop_loss:.2f}",
                "TP":        f"${t.take_profit:.2f}",
                "Size":      f"{t.position_size:.0f}",
                "Unr. P&L":  f"${unr_pnl:+.2f}"
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )

    # Closed trades table
    closed_trades = pm.get_closed_trades()
    if closed_trades:
        st.markdown("#### 📜 Trade History")
        rows = []
        for t in closed_trades[-20:]:   # last 20
            rows.append({
                "ID":        t.id,
                "Symbol":    t.symbol,
                "Direction": t.direction,
                "Entry":     f"${t.entry_price:.2f}",
                "Exit":      f"${t.exit_price:.2f}" if t.exit_price else "—",
                "P&L":       f"${t.pnl:+.2f}" if t.pnl is not None else "—",
                "P&L %":     f"{t.pnl_pct:+.2%}" if t.pnl_pct is not None else "—",
                "Reason":    t.exit_reason or "—"
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )


# ── Tab: History ──────────────────────────────────────────────────────────────
def tab_history():
    history = st.session_state.get("analysis_history", [])
    if not history:
        st.info("No analysis history yet. Run an analysis to populate this tab.")
        return

    st.markdown("#### 📈 Analysis History")
    df_hist = pd.DataFrame(history[::-1])  # newest first

    if PLOTLY_AVAILABLE and len(df_hist) > 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(df_hist))),
            y=df_hist["price"],
            mode="lines+markers",
            line=dict(color="#00d4ff", width=2),
            marker=dict(
                color=[
                    "#00ff88" if d == "BUY" else
                    "#ff4444" if d == "SELL" else
                    "#6b8cae"
                    for d in df_hist["direction"]
                ],
                size=8
            ),
            name="Price"
        ))
        fig.update_layout(
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#0d1526",
            font=dict(color="#e0e6f0", family="Share Tech Mono"),
            height=250,
            margin=dict(l=60, r=20, t=30, b=40),
            title=dict(text="Signal History", font=dict(color="#00d4ff", size=12)),
            yaxis=dict(gridcolor="#1e3a5f"),
            xaxis=dict(gridcolor="#1e3a5f")
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

    st.dataframe(df_hist, use_container_width=True, hide_index=True)


# ============================================================
# MAIN APP ENTRY POINT
# ============================================================
def main():
    st.set_page_config(
        page_title="Neural Trade AI",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    apply_css()
    init_session_state()

    cfg = render_sidebar()

    # Header
    st.markdown(
        '<h1 style="font-family:Orbitron;color:#00d4ff;font-size:28px;'
        'letter-spacing:4px;margin-bottom:0;">⚡ NEURAL TRADE AI</h1>'
        '<p style="color:#6b8cae;font-size:12px;letter-spacing:2px;margin-top:4px;">'
        'ADVANCED ALGORITHMIC TRADING SYSTEM</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Analyse button / auto-refresh
    col_btn, col_sym, col_status = st.columns([1, 2, 3])
    with col_btn:
        analyse_clicked = st.button(
            "🔍 ANALYSE", use_container_width=True, type="primary"
        )
    with col_sym:
        st.markdown(
            f'<div style="color:#00d4ff;font-family:Orbitron;font-size:20px;'
            f'font-weight:700;padding-top:4px;">{cfg["symbol"]}</div>',
            unsafe_allow_html=True
        )
    with col_status:
        trained_count = len(st.session_state["trained_symbols"])
        open_count    = len(st.session_state["portfolio_mgr"].portfolio.open_positions)
        st.markdown(
            f'<div style="color:#6b8cae;font-size:11px;padding-top:8px;">'
            f'Trained: {trained_count} symbols &nbsp;|&nbsp; '
            f'Open positions: {open_count} &nbsp;|&nbsp; '
            f'sklearn: {"✅" if SKLEARN_AVAILABLE else "❌"} &nbsp;|&nbsp; '
            f'yfinance: {"✅" if YFINANCE_AVAILABLE else "❌"} &nbsp;|&nbsp; '
            f'plotly: {"✅" if PLOTLY_AVAILABLE else "❌"}'
            f'</div>',
            unsafe_allow_html=True
        )

    # Auto-refresh trigger
    if st.session_state.get("auto_refresh"):
        time.sleep(st.session_state.get("refresh_interval", 60))
        st.rerun()

    # Run analysis
    if analyse_clicked or (
        st.session_state.get("auto_refresh") and
        cfg["symbol"] not in st.session_state["last_analysis"]
    ):
        result = run_analysis(cfg)
        if result:
            st.session_state["last_analysis"][cfg["symbol"]] = result
    else:
        result = st.session_state["last_analysis"].get(cfg["symbol"])

    if not result:
        st.markdown("""
        <div style="text-align:center;padding:60px;color:#6b8cae;">
            <div style="font-family:Orbitron;font-size:48px;margin-bottom:16px;">⚡</div>
            <div style="font-size:18px;letter-spacing:2px;">Click ANALYSE to begin</div>
            <div style="font-size:12px;margin-top:8px;">
                Select a symbol and timeframe, then press ANALYSE
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview",
        "📊 Indicators",
        "🧠 ML Analysis",
        "💼 Portfolio",
        "📜 History"
    ])

    with tab1:
        tab_overview(result, cfg)
    with tab2:
        tab_indicators(result, cfg)
    with tab3:
        tab_ml(result, cfg)
    with tab4:
        tab_portfolio(result, cfg)
    with tab5:
        tab_history()

    # Update portfolio positions with latest price
    current_price = result["indicators"].get("price", 0)
    if current_price > 0:
        st.session_state["portfolio_mgr"].update_positions(
            {cfg["symbol"]: current_price}
        )


if __name__ == "__main__":
    main()