# src/preproc.py
from __future__ import annotations
from typing import Dict, List, Tuple, Set
import numpy as np
import pandas as pd


def _collect_feature_set(cfg: Dict) -> List[str]:
    feats: Set[str] = set()
    for bundle in cfg["bundles"].values():
        feats.update(bundle["features"])
    return sorted(feats)


def _load_inputs(cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fpath = cfg["io"]["fundamentals_csv"]
    tpath = cfg["io"]["technicals_csv"]

    df_f = pd.read_csv(fpath)
    df_t = pd.read_csv(tpath)

    # Parse dates & normalize ticker case (inputs came from our pipeline so should already be fine)
    df_f["date"] = pd.to_datetime(df_f["date"], utc=True).dt.tz_convert(None)
    df_t["date"] = pd.to_datetime(df_t["date"], utc=True).dt.tz_convert(None)
    df_f["ticker"] = df_f["ticker"].astype(str).str.upper()
    df_t["ticker"] = df_t["ticker"].astype(str).str.upper()

    # Drop columns we never want (monthly_high/low only)
    drop_cols = set(cfg.get("drop_columns_from_inputs", []))
    df_f = df_f.drop(columns=[c for c in drop_cols if c in df_f.columns], errors="ignore")
    df_t = df_t.drop(columns=[c for c in drop_cols if c in df_t.columns], errors="ignore")

    return df_f, df_t


def _merge_frames(df_f: pd.DataFrame, df_t: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    # Inner-join on (date, ticker) for clean alignment
    df = pd.merge(df_f, df_t, on=["date", "ticker"], how="inner")
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    if cfg["eligibility"].get("drop_all_na_rows", True):
        # Drop rows where absolutely all non-key columns are NA
        non_keys = [c for c in df.columns if c not in ("date", "ticker")]
        df = df.dropna(subset=non_keys, how="all")
    return df


def _winsorize_cross_section(df: pd.DataFrame, features: List[str], lower_q: float, upper_q: float) -> pd.DataFrame:
    """
    Clip per-month cross-sections to [q_lower, q_upper] using groupby.transform.
    Raw columns remain untouched; we create a working table for z-scoring.
    """
    work = df[["date", "ticker"] + features].copy()
    if lower_q is None or upper_q is None:
        return work

    # For each feature, compute per-month quantiles and clip
    for col in features:
        s = work[col]
        if col not in work.columns:
            continue
        if s.notna().sum() == 0:
            continue
        ql = work.groupby("date")[col].transform(lambda x: x.quantile(lower_q))
        qu = work.groupby("date")[col].transform(lambda x: x.quantile(upper_q))
        work[col] = s.clip(lower=ql, upper=qu)
    return work



def _zscore_cross_section(work: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Compute per-month z-scores via transform (no FutureWarning). Returns a DataFrame with *_z columns.
    If a month's std == 0 (constant feature), z is NaN for that month (by design).
    """
    out = work.copy()
    for col in features:
        if col not in out.columns:
            continue
        s = out[col]
        m = out.groupby("date")[col].transform("mean")
        sd = out.groupby("date")[col].transform(lambda x: x.std(ddof=0))
        z = (s - m) / sd
        z = z.where(sd.notna() & (sd != 0))
        out[f"{col}__z"] = z
    return out



def _apply_direction(out: pd.DataFrame, features: List[str], lower_is_better: List[str]) -> pd.DataFrame:
    """
    Produce <feature>_z_adj = +/- <feature>__z so that higher is always better.
    """
    for col in features:
        zcol = f"{col}__z"
        zadj = f"{col}_z_adj"
        if col in lower_is_better:
            out[zadj] = -out[zcol]
        else:
            out[zadj] = out[zcol]
    return out


def _bundle_scores(df: pd.DataFrame, cfg: Dict, features: List[str]) -> pd.DataFrame:
    out = df.copy()
    # For each bundle, average available _z_adj columns equally
    for bname, bcfg in cfg["bundles"].items():
        cols = [f"{c}_z_adj" for c in bcfg["features"] if f"{c}_z_adj" in out.columns]
        if len(cols) == 0:
            out[f"score_{bname}"] = np.nan
        else:
            out[f"score_{bname}"] = out[cols].mean(axis=1, skipna=True)
    return out


def _composite_score(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Weighted average of available bundle scores with per-row weight renormalization.
    """
    out = df.copy()
    bundle_names = list(cfg["bundles"].keys())
    weights = np.array([cfg["bundles"][b]["weight"] for b in bundle_names], dtype=float)

    # Compose a matrix of bundle scores
    score_cols = [f"score_{b}" for b in bundle_names]
    S = out[score_cols].to_numpy(dtype=float)

    # For each row, mask missing bundles and renormalize weights
    W = np.tile(weights, (S.shape[0], 1))
    mask = ~np.isnan(S)
    W_masked = np.where(mask, W, 0.0)
    row_sums = W_masked.sum(axis=1, keepdims=True)
    # Avoid division by zero: rows with all-NaN bundles stay NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        W_norm = np.where(row_sums > 0, W_masked / row_sums, 0.0)
    composite = np.nansum(W_norm * np.nan_to_num(S, nan=0.0), axis=1)
    composite = np.where(row_sums.squeeze() > 0, composite, np.nan)

    out["score_quant_composite"] = composite
    return out


def _features_available(df: pd.DataFrame, features: List[str], min_feats: int) -> pd.DataFrame:
    out = df.copy()
    zcols = [f"{c}_z_adj" for c in features if f"{c}_z_adj" in out.columns]
    avail = out[zcols].notna().sum(axis=1)
    out["features_available"] = avail
    out["is_eligible"] = avail >= int(min_feats)
    return out


def preprocess_and_score(cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main driver: load, merge, winsorize, z-adjust, bundle scores, composite, eligibility.
    Returns:
      df_out: final panel with raw + _z_adj + scores
      df_stats: per-month diagnostics
    """
    all_feats = _collect_feature_set(cfg)                  # <-- consistent name
    lower_better = set(cfg.get("lower_is_better", []))

    # 1) Load & merge
    df_f, df_t = _load_inputs(cfg)
    df = _merge_frames(df_f, df_t, cfg)

    # 2) Restrict to features that exist and have at least one non-null
    feats = [c for c in all_feats if (c in df.columns) and df[c].notna().any()]
    skipped = sorted(set(all_feats) - set(feats))
    if skipped:
        print("[preproc] skipped features (missing or all-NaN in merged data):", skipped)

    # 3) Working copy for winsor & z only on the features we actually score
    if cfg["winsorization"]["method"] == "quantile":
        work = _winsorize_cross_section(
            df, feats, cfg["winsorization"]["lower"], cfg["winsorization"]["upper"]
        )
    else:
        work = df[["date", "ticker"] + feats].copy()

    # 4) Z-score by month and direction-adjust
    zed = _zscore_cross_section(work, feats)              # adds <feat>__z
    zed = _apply_direction(zed, feats, list(lower_better))# adds <feat>_z_adj

    # 5) Attach _z_adj back to the main df; keep raw columns intact
    keep_cols = ["date", "ticker"]
    raw_cols  = [c for c in feats if c in df.columns] if cfg["persist"].get("keep_raw", True) else []
    zadj_cols = [f"{c}_z_adj" for c in feats if f"{c}_z_adj" in zed.columns] if cfg["persist"].get("emit_z_adj", True) else []

    df_out = pd.merge(df[keep_cols + raw_cols], zed[keep_cols + zadj_cols],
                      on=["date", "ticker"], how="left").sort_values(["date", "ticker"]).reset_index(drop=True)

    # 6) Bundle scores (equal feature weights)
    df_out = _bundle_scores(df_out, cfg, feats)

    # 7) Composite with per-row weight renormalization
    df_out = _composite_score(df_out, cfg)

    # 8) Eligibility
    df_out = _features_available(df_out, feats, cfg["eligibility"]["min_features_per_name"])

    # 9) Diagnostics
    df_stats = _build_stats(df_out, cfg, feats)

    return df_out, df_stats


def _build_stats(df: pd.DataFrame, cfg: Dict, feats: List[str]) -> pd.DataFrame:
    """
    Per-month coverage and bundle summary.
    """
    bundle_names = list(cfg["bundles"].keys())
    score_cols = [f"score_{b}" for b in bundle_names]

    rows = []
    for dt, g in df.groupby("date"):
        row = {"date": dt, "eligible_count": int(g["is_eligible"].sum())}
        # coverage on RAW feature columns (for LLM usefulness)
        for c in feats:
            if c in g.columns:
                row[f"coverage__{c}"] = float(g[c].notna().mean())
        # bundle means/std
        for sc in score_cols:
            if sc in g.columns:
                row[f"mean__{sc}"] = float(g[sc].mean(skipna=True))
                row[f"std__{sc}"] = float(g[sc].std(skipna=True, ddof=0))
        rows.append(row)
    stats = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return stats
