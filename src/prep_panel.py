# src/prep_panel.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List

def month_end_price(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Take the last available trading day per (ticker, month) for PRICE ONLY.
    Returns a DataFrame with ['date','ticker','price'] at month-end rows.
    """
    if not {'date','ticker','price'}.issubset(df_daily.columns):
        raise ValueError("Expected columns 'date','ticker','price' in df_daily")

    # last trading day in that month for each (ticker, month)
    me = (
        df_daily.assign(_ym=df_daily['date'].dt.to_period('M'))
                .sort_values(['ticker', 'date'])
                .groupby(['ticker', '_ym'], as_index=False)
                .tail(1)
                .drop(columns=['_ym'])
                .loc[:, ['date','ticker','price']]
                .sort_values(['ticker','date'])
                .reset_index(drop=True)
    )
    return me

def _ffill_daily_column_by_ticker(df_daily: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    For a single fundamental column:
      - Sort by (ticker, date)
      - Track the observation date where 'col' is non-null
      - Forward-fill both the value and the obs_date within each ticker
    Returns a daily DataFrame with ['ticker','date', col, 'obs_date__{col}'].
    """
    if col not in df_daily.columns:
        # Column not present at all: return empty frame with needed schema
        out = df_daily.loc[:, ['ticker','date']].copy()
        out[col] = np.nan
        out[f'obs_date__{col}'] = pd.NaT
        return out

    tmp = df_daily.loc[:, ['ticker', 'date', col]].copy()
    # Ensure proper ordering
    tmp = tmp.sort_values(['ticker', 'date']).reset_index(drop=True)

    # Record obs date only where we actually have a value
    obs_col = f'obs_date__{col}'
    tmp[obs_col] = tmp['date'].where(tmp[col].notna())

    # Forward-fill value and obs_date within each ticker
    tmp[[col, obs_col]] = tmp.groupby('ticker', group_keys=False)[[col, obs_col]].ffill()

    return tmp

def forward_fill_fundamentals_monthend(
    df_daily: pd.DataFrame,
    df_me_price: pd.DataFrame,
    cfg: Dict
) -> pd.DataFrame:
    """
    Forward-fill fundamentals to month-end (per-ticker), then apply staleness:
      if (month_end - last_obs_date) > max_age_days => NaN.
    Keeps original column names; returns month-end panel with price + fundamentals.
    """
    # Identify fundamental columns as all numeric columns except 'price'
    numeric_cols = df_daily.select_dtypes(include=[np.number]).columns.tolist()
    fund_cols = [c for c in numeric_cols if c != 'price']

    # Start from the month-end price frame
    me = df_me_price.sort_values(['ticker', 'date']).reset_index(drop=True).copy()

    max_age = int(cfg.get('staleness', {}).get('max_fundamental_age_days', 365))

    # Merge each fundamental via exact join on (ticker, date) AFTER daily ffill
    for col in fund_cols:
        daily_ff = _ffill_daily_column_by_ticker(df_daily, col)
        # Left-join onto month-end dates
        merged = me[['ticker', 'date']].merge(daily_ff, on=['ticker', 'date'], how='left')

        obs_col = f'obs_date__{col}'
        # Staleness in days
        age_days = (merged['date'] - merged[obs_col]).dt.days
        # Apply staleness threshold
        valid = age_days.notna() & (age_days <= max_age)
        me[col] = np.where(valid, merged[col], np.nan)
        # Keep helper for summary; will drop at end
        me[obs_col] = merged[obs_col]

    # ---- Logging summary (coverage & staleness) ----
    if cfg.get('logging', {}).get('enabled', True):
        _print_summary(me, fund_cols)

    # Drop helper obs_date__* columns from final output
    drop_helpers = [c for c in me.columns if c.startswith('obs_date__')]
    out = me.drop(columns=drop_helpers).sort_values(['date','ticker']).reset_index(drop=True)
    return out

def _print_summary(me: pd.DataFrame, fund_cols: List[str]) -> None:
    n_rows = len(me)
    n_tickers = me['ticker'].nunique()
    n_months = me['date'].dt.to_period('M').nunique()
    print(f"[panel] rows={n_rows:,}  tickers={n_tickers}  months={n_months}")

    for col in fund_cols:
        cov = me[col].notna().mean() * 100.0 if n_rows else 0.0
        helper = f'obs_date__{col}'
        stale_est = None
        if helper in me.columns:
            stale_est = int(((me[helper].notna()) & (me[col].isna())).sum())
        if stale_est is not None:
            print(f"  - {col}: coverage={cov:.1f}%  stale_dropped≈{stale_est:,}")
        else:
            print(f"  - {col}: coverage={cov:.1f}%")
