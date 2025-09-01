# src/gridsearch/run_grid.py
from __future__ import annotations
import os, json, hashlib, itertools
from typing import Dict, List, Tuple
import pandas as pd
import yaml

# reuse backtester internals to avoid re-IO per candidate
from src.backtest.run import (
    _load_yaml, _ensure_dir, _load_panel, _load_daily_prices,
    _simulate_strategy
)

def _hash_obj(o: dict) -> str:
    m = hashlib.sha256()
    m.update(json.dumps(o, sort_keys=True, default=str).encode("utf-8"))
    return m.hexdigest()[:12]

def _normalize(w: List[float]) -> List[float]:
    s = sum(w)
    if s == 0: return [1.0/len(w)]*len(w)
    return [x/s for x in w]

def _make_strategy_name(K, wt_mode, llm_w, q_w, over_w, cost):
    llm_tag = "_".join(f"{int(round(x*100)):02d}" for x in _normalize(llm_w))
    q_tag   = "_".join(f"{int(round(x*100)):02d}" for x in _normalize(q_w))
    o_tag   = "_".join(f"{int(round(x*100)):02d}" for x in _normalize(over_w))
    return f"bw_K{K}_{wt_mode}_LLM{llm_tag}_Q{q_tag}_OV{o_tag}_C{int(cost)}"

def _strategy_dict(name: str, K: int, wt_mode: str,
                   llm_w: List[float], q_w: List[float], over_w: List[float]) -> Dict:
    bundles = ["valuation","quality","income","balance","technical"]
    return {
        "name": name,
        "kind": "bundle_weighted",
        "K": int(K),
        "weighting": wt_mode,
        "bundle_weights": {
            "llm": {b: float(w) for b, w in zip(bundles, _normalize(llm_w))},
            "quant": {b: float(w) for b, w in zip(bundles, _normalize(q_w))},
            "overall": {"llm": float(_normalize(over_w)[0]), "quant": float(_normalize(over_w)[1])}
        },
    }

def _score_row(summ: dict) -> dict:
    # flatten the summary.json into leaderboard fields
    return {
        "cagr": summ.get("cagr"),
        "ann_vol": summ.get("ann_vol"),
        "sharpe": summ.get("sharpe"),
        "sortino": summ.get("sortino"),
        "max_drawdown": summ.get("max_drawdown"),
        "calmar": summ.get("calmar"),
        "win_rate": summ.get("win_rate"),
        "avg_daily_return": summ.get("avg_daily_return"),
        "days": summ.get("days")
    }

def run_gridsearch(grid_cfg_path: str):
    with open(grid_cfg_path, "r") as f:
        gcfg = yaml.safe_load(f)

    # base backtest defaults
    base = _load_yaml(gcfg["io"]["backtest_base_config"])

    # force diagnostics from grid settings
    base["reporting"]["compute_ic"] = bool(gcfg["evaluation"].get("compute_ic", False))
    base["reporting"]["compute_quantiles"] = bool(gcfg["evaluation"].get("compute_quantiles", False))
    base["reporting"]["rolling_windows_days"] = gcfg["evaluation"].get("rolling_windows_days", [63, 252])

    # override universe, frictions basics once
    base["universe"]["require_eligible"] = gcfg["universe"].get("require_eligible", True)
    base["universe"]["min_names"] = int(gcfg.get("universe", {}).get("min_names", 1))
    base["frictions"]["renormalize_on_missing"] = gcfg["frictions"].get("renormalize_on_missing", True)

    out_root = gcfg["io"]["out_dir"]; _ensure_dir(out_root)
    cache_dir = gcfg["caching"]["cache_dir"]; _ensure_dir(cache_dir)

    # Load data once
    panel = _load_panel(base["io"]["merged_llm_csv"],
                        base["rebalancing"].get("start_date"),
                        base["rebalancing"].get("end_date"))
    rets, px = _load_daily_prices(base["io"]["daily_prices_csv"])

    # Build candidate grid
    K_list = gcfg["K_list"]
    wt_list = gcfg["weighting_list"]
    llm_ws  = gcfg["llm_bundle_weights"]
    q_ws    = gcfg["quant_bundle_weights"]
    ov_ws   = gcfg["overall_weights"]
    costs   = gcfg["frictions"]["cost_bps_list"]

    candidates = []
    for K, wt, llm_w, qw, ow, c in itertools.product(K_list, wt_list, llm_ws, q_ws, ov_ws, costs):
        name = _make_strategy_name(K, wt, llm_w, qw, ow, c)
        strat = _strategy_dict(name, K, wt, llm_w, qw, ow)
        candidates.append((name, strat, float(c)))

    leaderboard = []

    for name, strat, cost_bps in candidates:
        # shallow copy base and inject per-candidate friction + dirs
        cfg = json.loads(json.dumps(base))
        cfg["frictions"]["cost_bps"] = float(cost_bps)
        sdir = os.path.join(gcfg["io"]["out_dir"], name); _ensure_dir(sdir)

        # cache key
        ck = _hash_obj({
            "panel_mtime": os.path.getmtime(cfg["io"]["merged_llm_csv"]),
            "prices_mtime": os.path.getmtime(cfg["io"]["daily_prices_csv"]),
            "cfg": cfg, "strat": strat
        })
        summ_path = os.path.join(sdir, "summary.json")
        if gcfg["caching"]["enable"] and os.path.exists(summ_path):
            with open(summ_path, "r") as f:
                summ = json.load(f)
            row = {"name": name, "cost_bps": cost_bps, "K": strat["K"], "weighting": strat["weighting"], **_score_row(summ)}
            leaderboard.append(row)
            continue

        # run the simulation once per candidate
        print(f"[grid] running {name}")
        res = _simulate_strategy(panel, rets, px, cfg, strat, sdir)
        summ = res.get("summary", {})
        row = {"name": name, "cost_bps": cost_bps, "K": strat["K"], "weighting": strat["weighting"], **_score_row(summ)}
        leaderboard.append(row)

    # Rank & write leaderboard
    df = pd.DataFrame(leaderboard).sort_values(
        by=["sharpe","cagr","ann_vol","turnover"] if "turnover" in leaderboard[0].keys() else ["sharpe","cagr","ann_vol"],
        ascending=[False, False, True, True] if "turnover" in leaderboard[0].keys() else [False, False, True]
    ).reset_index(drop=True)

    lb_path = os.path.join(out_root, "leaderboard.csv")
    df.to_csv(lb_path, index=False)
    print(f"[grid] leaderboard saved: {lb_path}")

    # Save best candidate's full strategy for drop-in use
    if not df.empty:
        best_name = df.iloc[0]["name"]
        best_dir = os.path.join(out_root, best_name)
        # Reconstruct the strategy dict we used
        parts = best_name  # name already encoded; but we also have the strat object in loop
        # Simpler: find the matching candidate again
        best_row = next((c for c in candidates if c[0] == best_name), None)
        if best_row:
            _, best_strat, best_cost = best_row
            best_full = {
                "base_backtest_config": base,  # optional snapshot, can omit if large
                "strategy": best_strat,
                "frictions": {"cost_bps": float(best_cost)}
            }
            with open(os.path.join(out_root, "best_full.yaml"), "w") as f:
                yaml.safe_dump(best_full, f, sort_keys=False)
        # Keep the small pointer too
        with open(os.path.join(out_root, "best_config.yaml"), "w") as f:
            yaml.safe_dump({"name": best_name}, f, sort_keys=False)
        print(f"[grid] best strategy: {best_name}")

