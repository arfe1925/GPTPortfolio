# src/backtest/run.py
from __future__ import annotations
import os, json, math
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt


# ------------------------- IO / Utils -------------------------

def _load_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _to_dt(x):
    """
    Robust datetime parser that returns tz-naive datetimes for both Series and scalars.
    Works whether the input has tz info or not.
    """
    v = pd.to_datetime(x, errors="coerce", utc=True)
    # Series
    if isinstance(v, pd.Series):
        try:
            return v.dt.tz_localize(None)   # strip tz if present
        except (TypeError, AttributeError):
            # If already tz-naive or not a datetime Series, just return as-is
            return pd.to_datetime(v, errors="coerce")
    # Scalar Timestamp
    if isinstance(v, pd.Timestamp):
        return v.tz_localize(None) if v.tzinfo else v
    # Fallback
    return pd.to_datetime(v, errors="coerce")

def _spearman(x: pd.Series, y: pd.Series) -> float:
    """
    Spearman rank correlation with robust guards:
    - Align and drop NaNs
    - If either side has <3 obs or zero variance, return NaN (no IC that month)
    - Use np.errstate to silence harmless divide-by-zero in edge cases
    """
    xr = x.rank(method="average")
    yr = y.rank(method="average")
    df = pd.concat([xr, yr], axis=1).dropna()
    if len(df) < 3:
        return np.nan
    xr = df.iloc[:, 0]
    yr = df.iloc[:, 1]
    # zero-variance or effectively constant ranks → undefined Spearman
    if xr.nunique() < 2 or yr.nunique() < 2:
        return np.nan
    xstd = xr.std(ddof=0)
    ystd = yr.std(ddof=0)
    if not np.isfinite(xstd) or not np.isfinite(ystd) or xstd == 0 or ystd == 0:
        return np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.corrcoef((xr - xr.mean()) / xstd, (yr - yr.mean()) / ystd)[0, 1]
    return float(r) if np.isfinite(r) else np.nan

# ------------------------- Data Prep -------------------------

def _load_panel(merged_llm_csv: str, start: str | None, end: str | None) -> pd.DataFrame:
    df = pd.read_csv(merged_llm_csv)
    df["date"] = _to_dt(df["date"])
    if start: df = df[df["date"] >= pd.Timestamp(start)]
    if end:   df = df[df["date"] <= pd.Timestamp(end)]
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)

def _load_daily_prices(daily_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(daily_csv)
    d["date"] = _to_dt(d["date"])
    d["ticker"] = d["ticker"].astype(str).str.upper()
    d = d.sort_values(["ticker", "date"]).reset_index(drop=True)
    # pivot to returns matrix
    d["ret"] = d.groupby("ticker")["price"].pct_change()
    r = d.pivot(index="date", columns="ticker", values="ret").sort_index()
    px = d.pivot(index="date", columns="ticker", values="price").sort_index()
    return r, px

def _month_ends(panel: pd.DataFrame) -> List[pd.Timestamp]:
    # month-end dates come from panel
    return sorted(panel["date"].unique().tolist())

def _next_trade_days(month_ends: List[pd.Timestamp], trading_days: pd.Index) -> List[pd.Timestamp]:
    trade_days = []
    for me in month_ends:
        i = trading_days.searchsorted(me, side="right")
        if i < len(trading_days):
            trade_days.append(trading_days[i])   # next trading day close
    # drop last if we can’t trade out (no next period)
    return trade_days

# ------------------------- Selection & Scoring -------------------------

def _select_topk(df_me: pd.DataFrame, cfg: Dict, strat: Dict) -> Tuple[List[str], pd.Series]:
    """Return (tickers, score_series) used for ranking."""
    df = df_me.copy()
    if cfg["universe"].get("require_eligible", True) and "is_eligible" in df.columns:
        df = df[df["is_eligible"] == True]  # noqa: E712
    if df.empty:
        return [], pd.Series(dtype=float)

    rank_col = strat["rank_by"]
    scores = df[rank_col].astype(float)

    # sort descending (higher is better). break ties by ticker
    df["_score"] = scores
    df = df.sort_values(["_score", "ticker"], ascending=[False, True])
    K = int(strat.get("K", 20))
    if K > len(df): K = len(df)
    sel = df.head(K)
    return sel["ticker"].tolist(), sel.set_index("ticker")["_score"]

def _select_ensemble(df_me: pd.DataFrame, cfg: Dict, strat: Dict) -> Tuple[List[str], pd.Series]:
    df = df_me.copy()
    if cfg["universe"].get("require_eligible", True) and "is_eligible" in df.columns:
        df = df[df["is_eligible"] == True]
    if df.empty:
        return [], pd.Series(dtype=float)

    comps = strat["components"]
    weights = strat.get("weights")
    combine = strat.get("combine", "mean")
    m = len(comps)
    if weights is None or len(weights) != m:
        weights = [1.0] * m
    w_sum = sum(weights)
    weights = [w / w_sum if w_sum != 0 else 1.0/m for w in weights]

    score = pd.Series(0.0, index=df.index)
    for w, comp in zip(weights, comps):
        col = comp["column"]
        typ = comp.get("type", "identity")
        s = df[col].astype(float)
        if typ == "zscore":
            # z within this eligible cross-section
            mu = s.mean(skipna=True); sd = s.std(skipna=True, ddof=0)
            s = (s - mu) / sd if sd and sd != 0 else s*0.0
        score = score.add(w * s, fill_value=0.0)

    df["_score"] = score
    df = df.sort_values(["_score", "ticker"], ascending=[False, True])
    K = int(strat.get("K", 20))
    if K > len(df): K = len(df)
    sel = df.head(K)
    return sel["ticker"].tolist(), sel.set_index("ticker")["_score"]

def _select_rerank(df_me: pd.DataFrame, cfg: Dict, strat: Dict) -> Tuple[List[str], pd.Series]:
    df = df_me.copy()
    if cfg["universe"].get("require_eligible", True) and "is_eligible" in df.columns:
        df = df[df["is_eligible"] == True]
    if df.empty:
        return [], pd.Series(dtype=float)
    # shortlist
    shortlist_col = strat["shortlist_by"]
    K_pre = int(strat.get("K_pre", 40))
    df["_pri"] = df[shortlist_col].astype(float)
    df = df.sort_values(["_pri", "ticker"], ascending=[False, True]).head(K_pre)
    # rerank
    rerank_col = strat["rerank_by"]
    df["_score"] = df[rerank_col].astype(float)
    df = df.sort_values(["_score", "ticker"], ascending=[False, True])
    K = int(strat.get("K", 20))
    if K > len(df): K = len(df)
    sel = df.head(K)
    return sel["ticker"].tolist(), sel.set_index("ticker")["_score"]

def _select_bundle_weighted(df_me: pd.DataFrame, cfg: Dict, strat: Dict) -> Tuple[List[str], pd.Series]:
    """
    Build a combined signal from bundle-weighted LLM (z_adj) and Quant bundles, then overall-weighted blend.
    We z-score each side (LLM, Quant) within the eligible cross-section before the overall mix
    to avoid scale mismatch. Returns (tickers, score_series).
    """
    df = df_me.copy()
    if cfg["universe"].get("require_eligible", True) and "is_eligible" in df.columns:
        df = df[df["is_eligible"] == True]  # noqa: E712
    if df.empty:
        return [], pd.Series(dtype=float)

    bw = strat["bundle_weights"]
    # 5 bundles
    bundles = ["valuation", "quality", "income", "balance", "technical"]

    # LLM bundle scores (use z_adj for cross-section comparability)
    llm_cols = [f"llm_score_{b}_z_adj" for b in bundles]
    llm_w = np.array([bw["llm"].get(b, 0.0) for b in bundles], dtype=float)
    if llm_w.sum() != 0:
        llm_w = llm_w / llm_w.sum()
    llm_mat = df[llm_cols].astype(float)
    llm_side = (llm_mat * llm_w).sum(axis=1)

    # Quant bundle scores (preproc bundle scores, already “z-ish” but we still normalize within pool)
    quant_cols = [f"score_{b}" for b in bundles]
    quant_w = np.array([bw["quant"].get(b, 0.0) for b in bundles], dtype=float)
    if quant_w.sum() != 0:
        quant_w = quant_w / quant_w.sum()
    quant_mat = df[quant_cols].astype(float)
    quant_side = (quant_mat * quant_w).sum(axis=1)

    # Z within eligible pool for each side
    def _z(s: pd.Series) -> pd.Series:
        mu = s.mean(skipna=True); sd = s.std(skipna=True, ddof=0)
        return (s - mu) / sd if sd and sd != 0 else s * 0.0

    llm_z = _z(llm_side)
    quant_z = _z(quant_side)

    # Overall blend (LLM vs Quant)
    ow = bw.get("overall", {})
    w_llm = float(ow.get("llm", 0.5))
    w_quant = float(ow.get("quant", 0.5))
    wsum = (w_llm + w_quant) if (w_llm + w_quant) != 0 else 1.0
    w_llm /= wsum; w_quant /= wsum

    combined = w_llm * llm_z + w_quant * quant_z

    # Rank descending by combined; tie-break by ticker
    df["_score"] = combined
    df = df.sort_values(["_score", "ticker"], ascending=[False, True])
    K = int(strat.get("K", 20))
    if K > len(df): K = len(df)
    sel = df.head(K)
    tickers = sel["ticker"].tolist()
    score_series = sel.set_index("ticker")["_score"]
    return tickers, score_series


def _select_for_strategy(df_me: pd.DataFrame, cfg: Dict, strat: Dict) -> Tuple[List[str], pd.Series]:
    kind = strat["kind"]
    if kind == "topk":
        return _select_topk(df_me, cfg, strat)
    if kind == "ensemble_topk":
        return _select_ensemble(df_me, cfg, strat)
    if kind == "rerank":
        return _select_rerank(df_me, cfg, strat)
    if kind == "bundle_weighted":
        return _select_bundle_weighted(df_me, cfg, strat)
    raise ValueError(f"Unknown strategy kind: {kind}")

# ------------------------- Weighting -------------------------

def _weights_equal(tickers: List[str]) -> pd.Series:
    if len(tickers) == 0:
        return pd.Series(dtype=float)
    w = 1.0 / len(tickers)
    return pd.Series({t: w for t in tickers}, dtype=float)

def _weights_score_proportional(tickers: List[str], score_series: pd.Series) -> pd.Series:
    """
    w_i ∝ max(score_i - min(score), 0). If all zero/NaN → fall back to equal.
    """
    if len(tickers) == 0:
        return pd.Series(dtype=float)
    s = (score_series or pd.Series(dtype=float)).reindex(tickers).astype(float)
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() == 0:
        return _weights_equal(tickers)
    s = s.fillna(s.min(skipna=True))
    s = s - s.min(skipna=True)
    total = s.sum()
    if not np.isfinite(total) or total <= 0:
        return _weights_equal(tickers)
    return (s / total).astype(float)

def _form_weights(tickers: List[str], score_series: pd.Series, strat: Dict) -> pd.Series:
    weighting = strat.get("weighting", "equal").lower()
    if weighting == "equal":
        return _weights_equal(tickers)
    if weighting == "score_proportional":
        return _weights_score_proportional(tickers, score_series)
    # default fallback
    return _weights_equal(tickers)


# ------------------------- Simulation -------------------------

def _period_dates(trade_days: List[pd.Timestamp], all_days: pd.Index, idx: int) -> pd.DatetimeIndex:
    """Dates where weights[idx] are applied: (trade_day_i + 1) .. (trade_day_{i+1}) inclusive."""
    td = trade_days[idx]
    td_next = trade_days[idx+1] if idx + 1 < len(trade_days) else None
    start = all_days[all_days.get_indexer([td], method="bfill")[0]]  # td itself (trade close)
    # weights effective AFTER trade close -> start from next day
    start_pos = all_days.get_loc(start)
    if start_pos + 1 >= len(all_days):
        return all_days[0:0]
    start_eff = all_days[start_pos + 1]
    if td_next is None:
        return all_days[all_days >= start_eff]
    stop = all_days[all_days.get_indexer([td_next], method="bfill")[0]]
    return all_days[(all_days >= start_eff) & (all_days <= stop)]

def _renorm_weight_for_day(w: pd.Series, day_ret_row: pd.Series) -> pd.Series:
    mask = day_ret_row.notna()
    if not mask.any():
        return pd.Series(dtype=float)
    w_active = w.reindex(day_ret_row.index).fillna(0.0) * mask.astype(float)
    s = w_active.sum()
    if s <= 0:
        return pd.Series(dtype=float)
    return w_active / s

def _simulate_strategy(
    panel: pd.DataFrame,
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: Dict,
    strat: Dict,
    out_dir: str
) -> Dict:
    me_dates = _month_ends(panel)
    trade_days = _next_trade_days(me_dates, returns.index)
    if len(trade_days) < 1:
        raise ValueError("No trade days available after month ends.")

    nav_list = []
    ret_rows = []
    hold_rows = []
    trade_rows = []

    nav = 1.0
    w_prev = pd.Series(dtype=float)  # weights in effect BEFORE rebalancing
    cost_bps = float(cfg["frictions"]["cost_bps"])

    # pre-compute ME -> df_me view
    me_groups = {me: g.copy() for me, g in panel.groupby("date")}

    for i, td in enumerate(trade_days):
        me = me_dates[i]  # formation for this rebalance
        df_me = me_groups.get(me)
        if df_me is None:
            continue

        # selection (no min_names gating)
        tickers, score_series = _select_for_strategy(df_me, cfg, strat)

        # if selection ended up empty, just hold previous; otherwise rebalance into selected set
        pass_tickers = tickers if len(tickers) > 0 else w_prev.index.tolist()

        # target weights (supports equal or score_proportional)
        w_tgt = _form_weights(pass_tickers, score_series, strat)

        # turnover & cost (applied at trade close)
        union = sorted(set(w_prev.index) | set(w_tgt.index))
        prev = w_prev.reindex(union).fillna(0.0)
        tgt  = w_tgt.reindex(union).fillna(0.0)
        turnover = 0.5 * np.abs(tgt - prev).sum()
        day_cost = turnover * (cost_bps / 10000.0)

        # record trades
        if len(union) > 0:
            for tk in union:
                trade_rows.append({
                    "date": td, "ticker": tk,
                    "weight_prev": float(prev.get(tk, 0.0)),
                    "weight_target": float(tgt.get(tk, 0.0)),
                    "abs_change": float(abs(tgt.get(tk, 0.0) - prev.get(tk, 0.0)))
                })

        # apply cost at trade day (after that day's PnL with old weights has been already accounted previously)
        # here we ensure we log a day row for the trade day cost impact
        ret_rows.append({"date": td, "ret_gross": 0.0, "ret_net": -day_cost, "turnover": turnover, "cost_bps_applied": cost_bps})
        nav *= (1.0 - day_cost)
        nav_list.append({"date": td, "nav": nav})

        # update holdings effective AFTER trade close
        w_prev = w_tgt.copy()

        # record holdings snapshot (post-trade)
        for tk, wi in w_prev.items():
            hold_rows.append({"date": td, "ticker": tk, "weight": float(wi)})

        # hold through daily period until next rebalance effective date
        period_days = _period_dates(trade_days, returns.index, i)
        for day in period_days:
            r_row = returns.loc[day]
            # renormalize if missing daily returns
            w_day = _renorm_weight_for_day(w_prev, r_row) if cfg["frictions"].get("renormalize_on_missing", True) else w_prev
            if w_day.empty:
                g = 0.0
            else:
                r_vec = r_row.reindex(w_day.index)
                g = float(np.nansum(w_day.values * r_vec.values))
            ret_rows.append({"date": day, "ret_gross": g, "ret_net": g, "turnover": 0.0, "cost_bps_applied": 0.0})
            nav *= (1.0 + g)
            nav_list.append({"date": day, "nav": nav})

    # build DataFrames
    nav_df = pd.DataFrame(nav_list).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    rets_df = pd.DataFrame(ret_rows).drop_duplicates(subset=["date", "turnover"]).sort_values("date").reset_index(drop=True)
    holds_df = pd.DataFrame(hold_rows).sort_values(["date","ticker"]).reset_index(drop=True)
    trades_df = pd.DataFrame(trade_rows).sort_values(["date","ticker"]).reset_index(drop=True)

    # save CSVs
    _ensure_dir(out_dir)
    nav_df.to_csv(os.path.join(out_dir, "nav.csv"), index=False)
    rets_df.to_csv(os.path.join(out_dir, "returns.csv"), index=False)
    holds_df.to_csv(os.path.join(out_dir, "holdings.csv"), index=False)
    trades_df.to_csv(os.path.join(out_dir, "trades.csv"), index=False)

    # PnL chart
    plt.figure()
    plt.plot(nav_df["date"], nav_df["nav"])
    plt.title("Portfolio NAV")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "nav.png"), dpi=160)
    plt.close()

    # metrics
    daily = rets_df.set_index("date")["ret_net"].astype(float).sort_index()
    metrics = _compute_metrics(daily)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # diagnostics (IC & quantiles per ME)
    diag = _diagnostics(panel, returns, _select_for_strategy, cfg, strat)
    if diag.get("ic") is not None:
        diag["ic"].to_csv(os.path.join(out_dir, "ic.csv"), index=False)
    if diag.get("quantiles") is not None:
        diag["quantiles"].to_csv(os.path.join(out_dir, "quantiles.csv"), index=False)

    return {
        "nav": nav_df, "returns": rets_df, "holdings": holds_df, "trades": trades_df,
        "summary": metrics, "diag": diag
    }

# ------------------------- Metrics & Diagnostics -------------------------

def _compute_metrics(daily: pd.Series) -> Dict:
    daily = daily.dropna()
    if len(daily) == 0:
        return {}
    nav = (1.0 + daily).cumprod()
    total_days = len(daily)
    ann = 252.0
    cagr = nav.iloc[-1] ** (ann / total_days) - 1.0
    vol = daily.std(ddof=0) * math.sqrt(ann)
    sharpe = (daily.mean() / (daily.std(ddof=0) + 1e-12)) * math.sqrt(ann)
    downside = daily.clip(upper=0.0)
    sortino = (daily.mean() / (downside.std(ddof=0) + 1e-12)) * math.sqrt(ann)
    dd = (nav / nav.cummax() - 1.0).min()
    calmar = (cagr / abs(dd)) if dd < 0 else np.nan
    win_rate = float((daily > 0).mean())
    avg = float(daily.mean())

    return {
        "cagr": float(cagr),
        "ann_vol": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(dd),
        "calmar": float(calmar),
        "win_rate": win_rate,
        "avg_daily_return": avg,
        "days": int(total_days)
    }

def _next_month_period_returns(returns: pd.DataFrame, trade_days: List[pd.Timestamp], tickers: List[str], idx: int) -> pd.Series:
    """Return next-month cumulative returns for each ticker over the holding window of period idx."""
    if idx >= len(trade_days):
        return pd.Series(index=tickers, dtype=float)
    start_td = trade_days[idx]     # trade day at ME_t+1d close
    end_td = trade_days[idx+1] if idx + 1 < len(trade_days) else None
    all_days = returns.index
    # hold from next day after trade to end_td inclusive
    start = all_days.get_loc(all_days[all_days.get_indexer([start_td], method="bfill")[0]])
    if start + 1 >= len(all_days):
        return pd.Series(index=tickers, dtype=float)
    start_eff = all_days[start + 1]
    if end_td is None:
        idxs = (all_days >= start_eff)
    else:
        stop = all_days[all_days.get_indexer([end_td], method="bfill")[0]]
        idxs = (all_days >= start_eff) & (all_days <= stop)
    sub = returns.loc[idxs, tickers]
    cum = (1.0 + sub).prod() - 1.0
    return cum

def _diagnostics(panel: pd.DataFrame, returns: pd.DataFrame, selector, cfg: Dict, strat: Dict) -> Dict:
    if not cfg["reporting"].get("compute_ic", True) and not cfg["reporting"].get("compute_quantiles", True):
        return {}
    me_dates = _month_ends(panel)
    trade_days = _next_trade_days(me_dates, returns.index)
    bins = int(cfg["reporting"].get("quantile_bins", 10))
    ic_rows = []
    q_rows = []

    me_groups = {me: g.copy() for me, g in panel.groupby("date")}
    for i, me in enumerate(me_dates):
        if i >= len(trade_days) - 1:
            break
        df_me = me_groups.get(me)
        if df_me is None or df_me.empty:
            continue
        # eligible pool for diagnostics (not only selected names)
        if cfg["universe"].get("require_eligible", True) and "is_eligible" in df_me.columns:
            pool = df_me[df_me["is_eligible"] == True].copy()  # noqa: E712
        else:
            pool = df_me.copy()
        if pool.empty:
            continue

        # strategy score used for ranking at ME_t
        # reuse selection logic to get the score vector where possible
        kind = strat["kind"]
        if kind == "topk":
            rank_col = strat["rank_by"]
            score = pool[rank_col].astype(float)
        elif kind == "ensemble_topk":
            comps = strat["components"]
            weights = strat.get("weights")
            m = len(comps)
            if weights is None or len(weights) != m:
                weights = [1.0]*m
            w_sum = sum(weights); weights = [w/sum(weights) for w in weights] if w_sum != 0 else [1.0/m]*m
            score = pd.Series(0.0, index=pool.index)
            for w, comp in zip(weights, comps):
                s = pool[comp["column"]].astype(float)
                if comp.get("type","identity") == "zscore":
                    mu = s.mean(skipna=True); sd = s.std(skipna=True, ddof=0)
                    s = (s - mu) / sd if sd and sd != 0 else s*0.0
                score = score.add(w * s, fill_value=0.0)
        elif kind == "rerank":
            # diagnostics score = rerank signal
            score = pool[strat["rerank_by"]].astype(float)
        else:
            continue

        tickers = pool["ticker"].astype(str).tolist()
        fwd = _next_month_period_returns(returns, trade_days, tickers, i)
        # align
        sc = pd.Series(score.values, index=tickers).replace([np.inf, -np.inf], np.nan)
        fwd = fwd.replace([np.inf, -np.inf], np.nan)

        # IC
        if cfg["reporting"].get("compute_ic", True):
            ic = _spearman(sc.reindex(fwd.index), fwd)
            ic_rows.append({"date": me, "ic": float(ic) if pd.notna(ic) else np.nan})

        # quantiles
        if cfg["reporting"].get("compute_quantiles", True):
            try:
                qcats = pd.qcut(sc, bins, labels=False, duplicates="drop")
                dfq = pd.DataFrame({"score_q": qcats, "fwd": fwd.values})
                qmeans = dfq.groupby("score_q")["fwd"].mean()
                row = {"date": me}
                for q, val in qmeans.items():
                    row[f"q{int(q)+1}"] = float(val)
                q_rows.append(row)
            except Exception:
                pass

    out = {}
    if ic_rows:
        out["ic"] = pd.DataFrame(ic_rows).sort_values("date").reset_index(drop=True)
    if q_rows:
        out["quantiles"] = pd.DataFrame(q_rows).sort_values("date").reset_index(drop=True)
    return out


# ------------------------- Orchestrator -------------------------

def run_backtests(backtest_cfg_path: str) -> Dict[str, Dict]:
    cfg = _load_yaml(backtest_cfg_path)
    out_root = cfg["io"]["out_dir"]; _ensure_dir(out_root)

    panel = _load_panel(cfg["io"]["merged_llm_csv"], cfg["rebalancing"]["start_date"], cfg["rebalancing"]["end_date"])
    rets, px = _load_daily_prices(cfg["io"]["daily_prices_csv"])

    results = {}
    for strat in cfg["strategies"]:
        sdir = os.path.join(out_root, strat["name"]); _ensure_dir(sdir)
        print(f"[bt] running strategy: {strat['name']}")
        res = _simulate_strategy(panel, rets, px, cfg, strat, sdir)
        results[strat["name"]] = res

    return results
