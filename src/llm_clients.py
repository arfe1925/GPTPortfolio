# src/llm_clients.py
from __future__ import annotations
from typing import Dict, Any
import json
import os
import time
import random
import math

def build_static_messages(llm_cfg: Dict) -> list[dict]:
    """
    Build the static, cacheable prefix messages (system + instructions).
    Provider-side prompt caching benefits from these being identical across calls.
    """
    sys = llm_cfg["prompt"]["system"].strip()
    instr = llm_cfg["prompt"]["instructions"].replace("{{rationale_char_limit}}", str(llm_cfg["prompt"]["rationale_char_limit"])).strip()
    # Include a short version tag so you can bump and intentionally miss the provider cache
    version_line = f"[PREFIX_VERSION:{llm_cfg['prompt'].get('prefix_version','v1')}]"
    return [
        {"role": "system", "content": sys + "\n" + version_line},
        {"role": "user", "content": instr},
    ]

def _clamp(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(v))))

def mock_llm_score(payload: Dict, seed: int, clamp: tuple[int,int]) -> Dict[str, Any]:
    """
    Deterministic mock: derive bundle scores from available z_adj means; reasoning empty.
    """
    rng = random.Random(seed)
    bundles = payload.get("bundles", {})
    scores = {}
    per_bundle_vals = []
    for bname, bd in bundles.items():
        zvals = list((bd.get("z_adj") or {}).values())
        if len(zvals) == 0:
            s = 50  # conservative default
        else:
            # center 50, scale 15 per stdev unit, plus small deterministic jitter
            mu = sum(zvals) / len(zvals)
            s = 50 + 15 * mu + rng.uniform(-2.0, 2.0)
        scores[bname] = _clamp(s, clamp[0], clamp[1])
        per_bundle_vals.append(scores[bname])

    # If some categories missing, average what we have; else average all 5
    if per_bundle_vals:
        overall = sum(per_bundle_vals) / len(per_bundle_vals)
    else:
        overall = 50
    scores["overall"] = _clamp(overall, clamp[0], clamp[1])

    return {"scores": scores, "reasoning": ""}

def openai_llm_score(payload: Dict, static_msgs: list[dict], llm_cfg: Dict) -> Dict[str, Any]:
    """
    Real LLM call using OpenAI. Requires `openai` package.
    - Sends static messages (system+instructions) first, then a dynamic data message.
    - Expects STRICT JSON; retries once with a minimal fixer if needed.
    """
    try:
        from openai import OpenAI
    except Exception as e:
        raise ImportError("OpenAI SDK not installed. Install with `pip install openai`") from e

    cfg = llm_cfg["model"]["openai"]
    api_key = os.getenv("OPENAI_API_KEY", cfg.get("api_key") or "")
    if not api_key:
        raise RuntimeError("OpenAI API key missing. Set in config.model.openai.api_key or OPENAI_API_KEY env.")

    client = OpenAI(api_key=api_key, base_url=cfg.get("base_url") or None)

    data_msg = {
        "role": "user",
        "content": llm_cfg["prompt"]["data_preamble"].strip() + "\n" +
                   json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    }

    messages = static_msgs + [data_msg]

    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=messages,
        temperature=cfg.get("temperature", 0.1),
        max_tokens=cfg.get("max_tokens", 400),
        timeout=cfg.get("timeout_s", 60),
        response_format={"type": "json_object"},
    )

    text = resp.choices[0].message.content
    try:
        out = json.loads(text)
    except Exception:
        # minimal retry: tighten instruction
        fix_messages = static_msgs + [{"role":"user","content":"Return STRICT JSON only with the exact required keys. No text. Use integers for all scores."}, data_msg]
        resp2 = client.chat.completions.create(
            model=cfg["model"],
            messages=fix_messages,
            temperature=0.0,
            max_tokens=cfg.get("max_tokens", 400),
            timeout=cfg.get("timeout_s", 60),
            response_format={"type": "json_object"},
        )
        out = json.loads(resp2.choices[0].message.content)

    # Clamp scores and enforce rationale limit
    lo, hi = llm_cfg["postprocess"]["clamp_scores"]
    limit = llm_cfg["prompt"]["rationale_char_limit"]
    scores = out.get("scores", {})
    for k in ["valuation","quality","income","balance","technical","overall"]:
        if k in scores:
            scores[k] = _clamp(scores[k], lo, hi)
        else:
            scores[k] = 50
    reasoning = (out.get("reasoning") or "")[:limit]
    return {"scores": scores, "reasoning": reasoning}
