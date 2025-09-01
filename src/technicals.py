# src/technicals.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict

def _require_pandas_ta():
    # Safety net: ensure aliases exist even if sitecustomize didn't run
    import numpy as _np
    if not hasattr(_np, "NaN"):
        _np.NaN = _np.nan
    if not hasattr(_np, "Inf"):
        _np.Inf = _np.inf

    try:
        import pandas_ta as ta  # noqa: F401
    except Exception as e:
        raise ImportError(
            "pandas_ta failed to import. On NumPy 2.x, older pandas_ta expects "
            "np.NaN/np.Inf aliases. We shimmed them, but the import still failed.\n"
            "Try upgrading/reinstalling:\n"
            "  python -m pip install --upgrade pip setuptools wheel pandas_ta\n\n"
            f"Original error: {type(e).__name__}: {e}"
        )

def load_daily_for_technicals(path: str, cfg: Dict) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"date", "ticker", cfg["monthly"]["close_col"]}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Found: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    if cfg.get("parsing", {}).get("ticker_uppercase", True):
        df["ticker"] = df["ticker"].astype(str).str.upper()

    # coerce numerics
    for c in [x for x in df.columns if x not in ("date", "ticker")]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # de-dup & sort
    return df.drop_duplicates(subset=["date", "ticker"]).sort_values(["ticker", "date"]).reset_index(drop=True)

def build_monthly_frame(df_daily: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    close_col = cfg["monthly"]["close_col"]
    high_col  = cfg["monthly"].get("high_col")
    low_col   = cfg["monthly"].get("low_col")
    have_hl = bool(high_col) and bool(low_col) and (high_col in df_daily.columns) and (low_col in df_daily.columns)

    df = df_daily.copy()
    df["_ym"] = df["date"].dt.to_period("M")

    # close (last trading day of month per ticker)
    last_rows = (
        df.sort_values(["ticker", "date"])
          .groupby(["ticker", "_ym"], as_index=False)
          .tail(1)
          .rename(columns={"date": "date_month_end", close_col: "monthly_close"})
          .loc[:, ["ticker", "_ym", "date_month_end", "monthly_close"]]
    )

    if have_hl:
        hl_agg = (
            df.groupby(["ticker", "_ym"], as_index=False)
              .agg(monthly_high=(high_col, "max"), monthly_low=(low_col, "min"))
        )
        monthly = pd.merge(last_rows, hl_agg, on=["ticker", "_ym"], how="left")
    else:
        monthly = last_rows.copy()
        monthly["monthly_high"] = np.nan
        monthly["monthly_low"] = np.nan

    monthly = monthly.rename(columns={"date_month_end": "date"}).sort_values(["ticker", "date"]).reset_index(drop=True)
    return monthly

def compute_monthly_indicators_pandasta(monthly: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    _require_pandas_ta()
    import pandas_ta as ta

    out = monthly.copy()
    # pre-create all output columns with full names
    out["trend_price_to_simple_moving_average_12_ratio"] = np.nan
    out["oscillator_relative_strength_index_14"] = np.nan
    out["volatility_average_true_range_14"] = np.nan
    out["volatility_average_true_range_14_normalized"] = np.nan
    out["market_strength_average_directional_index_14"] = np.nan
    out["market_strength_moving_average_convergence_divergence_histogram"] = np.nan

    emit_hl = cfg.get("high_low_features", {}).get("emit", True)
    range_zero_policy = cfg.get("high_low_features", {}).get("range_zero_policy", "half")

    sma_len  = cfg["indicators"]["trend"]["sma_months"]
    rsi_len  = cfg["indicators"]["oscillator"]["period_months"]
    atr_len  = cfg["indicators"]["volatility"]["period_months"]
    adx_len  = cfg["indicators"]["market_strength"]["primary"]["period_months"]
    macd_f   = cfg["indicators"]["market_strength"]["fallback"]["fast"]
    macd_s   = cfg["indicators"]["market_strength"]["fallback"]["slow"]
    macd_sig = cfg["indicators"]["market_strength"]["fallback"]["signal"]

    def _per_ticker(g: pd.DataFrame) -> pd.DataFrame:
        # Trend: price to SMA(12)
        sma = ta.sma(g["monthly_close"], length=sma_len)
        g["trend_price_to_simple_moving_average_12_ratio"] = g["monthly_close"] / sma

        # Oscillator: RSI(14)
        g["oscillator_relative_strength_index_14"] = ta.rsi(g["monthly_close"], length=rsi_len)

        have_hl = g["monthly_high"].notna().any() and g["monthly_low"].notna().any()

        # Volatility: ATR(14) if H/L available
        if have_hl:
            atr = ta.atr(high=g["monthly_high"], low=g["monthly_low"], close=g["monthly_close"], length=atr_len)
            g["volatility_average_true_range_14"] = atr
            g["volatility_average_true_range_14_normalized"] = atr / g["monthly_close"]
        else:
            g["volatility_average_true_range_14"] = np.nan
            g["volatility_average_true_range_14_normalized"] = np.nan

        # Market strength: ADX(14) if H/L available, else MACD histogram (close-only) as fallback
        if have_hl:
            adx = ta.adx(high=g["monthly_high"], low=g["monthly_low"], close=g["monthly_close"], length=adx_len)
            adx_col = next((c for c in adx.columns if c.startswith("ADX_")), None)
            g["market_strength_average_directional_index_14"] = adx[adx_col] if adx_col else np.nan
        else:
            macd = ta.macd(g["monthly_close"], fast=macd_f, slow=macd_s, signal=macd_sig)
            # pandas_ta columns usually: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            hist_col = next((c for c in macd.columns if c.startswith("MACDh_")), None)
            g["market_strength_moving_average_convergence_divergence_histogram"] = macd[hist_col] if hist_col else np.nan

        # Monthly HL features (not indicators; derived from OHLC)
        if emit_hl and have_hl:
            rng = g["monthly_high"] - g["monthly_low"]
            pos = (g["monthly_close"] - g["monthly_low"]) / rng
            if range_zero_policy == "half":
                pos = np.where(rng == 0, 0.5, pos)
            else:
                pos = np.where(rng == 0, np.nan, pos)
            g["position_within_monthly_range"] = np.clip(pos, 0.0, 1.0)
            g["percent_distance_from_monthly_high"] = (g["monthly_close"] - g["monthly_high"]) / g["monthly_high"]
            g["percent_distance_from_monthly_low"]  = (g["monthly_close"] - g["monthly_low"]) / g["monthly_low"]
        elif emit_hl:
            g["position_within_monthly_range"] = np.nan
            g["percent_distance_from_monthly_high"] = np.nan
            g["percent_distance_from_monthly_low"] = np.nan

        return g

    out = out.groupby("ticker", group_keys=False).apply(_per_ticker)

    cols = [
        "date", "ticker",
        "trend_price_to_simple_moving_average_12_ratio",
        "oscillator_relative_strength_index_14",
        "volatility_average_true_range_14",
        "volatility_average_true_range_14_normalized",
        "market_strength_average_directional_index_14",
        "market_strength_moving_average_convergence_divergence_histogram",
        "monthly_high", "monthly_low",
        "position_within_monthly_range",
        "percent_distance_from_monthly_high",
        "percent_distance_from_monthly_low",
    ]
    # Only keep columns that exist (e.g., HL-related may be absent if H/L missing)
    cols = [c for c in cols if c in out.columns]
    return out.loc[:, cols].sort_values(["date", "ticker"]).reset_index(drop=True)
