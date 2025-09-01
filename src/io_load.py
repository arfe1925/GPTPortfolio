# src/io_load.py
from __future__ import annotations
import pandas as pd
from typing import Dict, Optional

def load_raw_csv(path: str, cfg: Dict) -> pd.DataFrame:
    """
    Load the raw daily CSV. Assumptions:
      - MUST contain columns: 'date', 'ticker', 'price'
      - All other columns are treated as fundamentals (kept as-is; no renaming)
    Returns a DataFrame with:
      date: pandas.Timestamp (tz-naive), sorted
      ticker: string (uppercased if configured)
      price: float
      ...fundamental columns unchanged...
    """
    df = pd.read_csv(path)

    # Basic presence checks
    required = {'date', 'ticker', 'price'}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    # Parse date
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_convert(None)
    if cfg.get('parsing', {}).get('ticker_uppercase', True):
        df['ticker'] = df['ticker'].astype(str).str.upper()

    # Coerce numerics (except date/ticker)
    num_cols = [c for c in df.columns if c not in ('date', 'ticker')]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Optional range filter
    start = cfg.get('range', {}).get('start_date')
    end = cfg.get('range', {}).get('end_date')
    if start:
        df = df[df['date'] >= pd.Timestamp(start)]
    if end:
        df = df[df['date'] <= pd.Timestamp(end)]

    # Drop exact duplicates, sort
    df = df.drop_duplicates(subset=['date', 'ticker']).sort_values(['ticker', 'date']).reset_index(drop=True)
    return df
