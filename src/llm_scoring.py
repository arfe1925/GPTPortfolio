# src/llm_scoring.py
from __future__ import annotations
from typing import Dict, List, Tuple
import json, os, time, hashlib
import numpy as np
import pandas as pd
import yaml

from .llm_payloads import build_month_payloads, _ensure_dirs
from .llm_clients import build_static_messages, mock_llm_score, openai_llm_score

def _load_cfg(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _filter_panel(df: pd.DataFrame, llm_cfg: Dict) -> pd.DataFrame:
    df = df.copy()
    if llm_cfg["selection"].get("require_eligible", True) and "is_eligible" in df.columns:
        df = df[df["is_eligible"] == True]  # noqa: E712
    s, e = llm_cfg["selection"].get("start_date"), llm_cfg["selection"].get("end_date")
    if s: df = df[df["date"] >= pd.Timestamp(s)]
    if e: df = df[df["date"] <= pd.Timestamp(e)]
    return df.sort_values(["date","ticker"]).reset_index(drop=True)

def _cache_paths(llm_cfg: Dict, month: str) -> Tuple[str,str]:
    # JSONL: one JSON object per line (audit/monitoring friendly)
    payload_path = os.path.join(llm_cfg["io"]["payload_dir"], f"{month}.json")
    scores_path  = os.path.join(llm_cfg["io"]["scores_dir"],  f"{month}.json")
    return payload_path, scores_path

def _local_cache_get(cache_dir: str, cache_key: str) -> dict | None:
    path = os.path.join(cache_dir, f"{cache_key}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def _local_cache_put(cache_dir: str, cache_key: str, obj: dict) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, f"{cache_key}.json"), "w") as f:
        json.dump(obj, f, ensure_ascii=False)

def _normalize_month_z(df_scores: pd.DataFrame, cols_0_100: List[str]) -> pd.DataFrame:
    out = df_scores.copy()
    for c in cols_0_100:
        zc = c.replace("_0_100", "_z_adj")
        m = out.groupby("date")[c].transform("mean")
        sd = out.groupby("date")[c].transform(lambda x: x.std(ddof=0))
        z = (out[c] - m) / sd
        z = z.where(sd.notna() & (sd != 0))
        out[zc] = z
    return out

def run_llm_scoring(preproc_csv: str, preproc_cfg_path: str, llm_cfg_path: str) -> pd.DataFrame:
    llm_cfg = _load_cfg(llm_cfg_path)
    cache_only = bool(llm_cfg.get("caching", {}).get("cache_only", False) or os.getenv("LLM_CACHE_ONLY") == "1")
    if cache_only:
        print("[llm] cache-only mode: will skip API calls and use cache where available")
    pre_df = pd.read_csv(preproc_csv)
    pre_df["date"] = pd.to_datetime(pre_df["date"], utc=True).dt.tz_convert(None)

    # 1) filter rows (eligible + date range)
    df = _filter_panel(pre_df, llm_cfg)

    # 2) per-month payloads (anonymized)
    month_payloads, month_idmap = build_month_payloads(df, preproc_cfg_path, llm_cfg)

    # 3) prepare dirs & static messages
    _ensure_dirs(llm_cfg["io"]["payload_dir"], llm_cfg["io"]["scores_dir"], llm_cfg["io"]["cache_dir"])
    static_msgs = build_static_messages(llm_cfg)

    lo, hi = llm_cfg["postprocess"]["clamp_scores"]
    provider = llm_cfg["model"]["provider"].lower()

    # 4) iterate months, cache payloads & responses
    all_rows = []
    for month, items in month_payloads.items():
        payload_path, scores_path = _cache_paths(llm_cfg, month)
        # write the payloads we intend to send (for audit/repro)
        with open(payload_path, "w") as f:
            for it in items:
                f.write(json.dumps(it["payload"], ensure_ascii=False) + "\n")

        # gather scores, using local response cache
        month_scores = []
        for it in items:
            cached = _local_cache_get(llm_cfg["io"]["cache_dir"], it["cache_key"]) if llm_cfg["caching"]["local_response_cache"] else None
            # Skip API call entirely in cache-only mode (row remains unscored)
            if cached is None and cache_only:
                continue
            # call provider
            if cached is None:
                if provider == "mock":
                    # deterministic seed from cache_key for reproducibility
                    seed = int(hashlib.sha256(it["cache_key"].encode()).hexdigest()[:8], 16)
                    result = mock_llm_score(it["payload"], seed=seed, clamp=(lo,hi))
                    meta = {"provider":"mock"}
                else:
                    result = openai_llm_score(it["payload"], static_msgs, llm_cfg)
                    meta = {"provider":"openai", "model": llm_cfg["model"]["openai"]["model"]}
                # cache
                if llm_cfg["caching"]["local_response_cache"]:
                    _local_cache_put(llm_cfg["io"]["cache_dir"], it["cache_key"], result)
            else:
                result = cached
                meta = {"provider": provider, "cached": True}

            # store (without identifiers)
            month_scores.append({"local_id": it["local_id"], "result": result, "meta": meta})

        # write month raw scores jsonl (still without ids; side-map is separate)
        with open(scores_path, "w") as f:
            for r in month_scores:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # merge month scores with id map and expand to columns
        id_map = month_idmap[month]
        for r in month_scores:
            lid = r["local_id"]
            dt = id_map[lid]["date"]
            tk = id_map[lid]["ticker"]
            sc = r["result"]["scores"]
            row = {
                "date": pd.Timestamp(dt),
                "ticker": tk,
                "llm_score_valuation_0_100": sc.get("valuation", 50),
                "llm_score_quality_0_100":   sc.get("quality", 50),
                "llm_score_income_0_100":    sc.get("income", 50),
                "llm_score_balance_0_100":   sc.get("balance", 50),
                "llm_score_technical_0_100": sc.get("technical", 50),
                "llm_score_overall_0_100":   sc.get("overall", 50),
                "llm_reasoning": (r["result"].get("reasoning") or "")[: llm_cfg["prompt"]["rationale_char_limit"]],
                "llm_provider": r["meta"].get("provider"),
            }
            all_rows.append(row)

    # 5) to DataFrame and normalize per-month to *_z_adj
    llm_df = pd.DataFrame(all_rows)
    if not llm_df.empty and llm_cfg["postprocess"].get("normalize_llm_scores", True):
        cols = [c for c in llm_df.columns if c.endswith("_0_100")]
        llm_df = _normalize_month_z(llm_df, cols)

    # 6) merge back to the preprocessed panel and save
    out = pd.merge(pre_df, llm_df, on=["date","ticker"], how="left").sort_values(["date","ticker"]).reset_index(drop=True)
    out.to_csv(llm_cfg["io"]["merged_out_csv"], index=False)
    return out
