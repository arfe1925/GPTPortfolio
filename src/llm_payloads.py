# src/llm_payloads.py
from __future__ import annotations
from typing import Dict, List, Tuple
import hashlib
import json
import os
import pandas as pd
import yaml

def _load_preproc_cfg(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)

def _row_bundles(row: pd.Series, preproc_cfg: Dict) -> Dict:
    """
    Build a compact dict of bundles -> { raw: {...}, z_adj: {...} } for only features present.
    """
    bundles = {}
    for bname, bcfg in preproc_cfg["bundles"].items():
        feats = bcfg["features"]
        raw = {}
        zadj = {}
        for c in feats:
            if c in row and pd.notna(row[c]):
                raw[c] = row[c]
            zc = f"{c}_z_adj"
            if zc in row and pd.notna(row[zc]):
                zadj[zc] = row[zc]
        bundles[bname] = {"raw": raw, "z_adj": zadj}
    return bundles

def _payload_cache_key(payload: Dict, model_sig: str) -> str:
    m = hashlib.sha256()
    m.update(model_sig.encode("utf-8"))
    m.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return m.hexdigest()

def build_month_payloads(
    df_pre: pd.DataFrame,
    preproc_cfg_path: str,
    llm_cfg: Dict
) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict]]:
    """
    Returns:
      month_to_payloads: { 'YYYY-MM': [ {payload_no_ids, cache_key, local_id, ...}, ... ] }
      local_map: { 'YYYY-MM': { local_id: {'date':..., 'ticker':...} } }
    """
    preproc_cfg = _load_preproc_cfg(preproc_cfg_path)
    month_to_payloads: Dict[str, List[Dict]] = {}
    local_map: Dict[str, Dict] = {}

    # Build a model signature for cache keys (provider+model+prefix_version+rationale limit)
    model = llm_cfg["model"]
    model_sig = f"{model['provider']}|{model.get('openai', {}).get('model')}|v={llm_cfg['prompt'].get('prefix_version','v1')}|limit={llm_cfg['prompt']['rationale_char_limit']}"

    for (month, grp) in df_pre.groupby(df_pre["date"].dt.to_period("M")):
        key = str(month)
        payloads = []
        id_map = {}
        for _, row in grp.iterrows():
            # anonymized payload (no date/ticker)
            bundles = _row_bundles(row, preproc_cfg)
            payload = { "bundles": bundles }

            # local handle
            local_id = hashlib.md5(f"{row['ticker']}|{row['date']}".encode("utf-8")).hexdigest()
            cache_key = _payload_cache_key(payload, model_sig)

            payloads.append({
                "payload": payload,
                "cache_key": cache_key,
                "local_id": local_id,
            })
            id_map[local_id] = {"date": row["date"], "ticker": row["ticker"]}
        month_to_payloads[key] = payloads
        local_map[key] = id_map
    return month_to_payloads, local_map
