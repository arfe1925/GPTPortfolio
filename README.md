# BIST100 GPT Portfolio — Proof of Concept

> End‑to‑end pipeline that replicates the whitepaper on the BIST100 universe using **fundamentals + monthly technicals**, cross‑sectional preprocessing, **LLM scoring**, portfolio backtesting, and parameter grid search.

---

## Quick start

```bash
# 1) Create & activate a virtual env (example)
python -m venv .venv && source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Set secrets (prefer env over YAML)
export OPENAI_API_KEY=sk-...

# 4) Run the pipeline (only LLM stage while iterating)
python main.py --main-config config/main.yaml
```

**Pro tip:** Keep `prompt.prefix_version` unchanged to reuse the LLM cache. Bump it only when you intentionally change instructions.

---

## Project structure

```
├── main.py
├── config/
│   ├── main.yaml              # stage switches + global IO + orchestration
│   ├── data.yaml              # raw data IO + panel construction
│   ├── technicals.yaml        # monthly technicals config (pandas_ta)
│   ├── preproc.yaml           # winsorize/z-score per-date + bundle composites
│   ├── llm.yaml               # prompts, model, batching, caching, postprocess
│   ├── backtest.yaml          # selection rules + portfolio mechanics
│   └── gridsearch.yaml        # parameter search space
├── src/
│   ├── io_load.py             # CSV load + schema checks
│   ├── prep_panel.py          # month-end price; ffill fundamentals
│   ├── technicals.py          # daily→monthly; pandas_ta indicators
│   ├── preproc.py             # winsorize→z→z_adj; bundle composites
│   ├── llm_scoring.py         # prompt render; batching; caching; merge
│   ├── llm_clients.py         # mock + OpenAI client wrappers
│   ├── backtest/
│   │   └── run.py             # top-K selection; daily P&L; plots
│   └── gridsearch/
│       └── run_grid.py        # grid simulation; pick best by metric
└── outputs/
    ├── panel_monthend_raw.csv
    ├── technicals_monthly_raw.csv
    ├── panel_monthend_preprocessed.csv
    ├── panel_monthend_with_llm.csv
    ├── llm_payloads/   # JSONL audit of sent prompts (1 object/line)
    ├── llm_scores/     # per-month scored JSONL (optional)
    ├── llm_cache/      # <cache_key>.json responses (source of truth)
    ├── backtests/      # equity curve PNG/CSV by strategy
    └── gridsearch/     # runs, leaderboards, best_full.yaml
```

---

## Data & panel construction

**Goal:** month‑end fundamentals aligned to trading calendar, forward‑filled until next report **with staleness caps**; daily prices aggregated to month‑end for technicals.

1. **Fundamentals** (`prep_panel.py`)

   * `month_end_price(df_daily)` computes last trading day each month per ticker.
   * `forward_fill_fundamentals_monthend()` as‑of merges each fundamental to month‑end with optional `stale_max_months` per field.
   * Output → `outputs/panel_monthend_raw.csv`.

2. **Technicals** (`technicals.py`)

   * `load_daily_for_technicals()` reads daily OHLC.
   * `build_monthly_frame()` aggregates to month‑end frame.
   * `compute_monthly_indicators_pandasta()` computes chosen **monthly** indicators via `pandas_ta` (trend/momentum/volatility/market strength; volume‑based skipped if no volume).
   * Output → `outputs/technicals_monthly_raw.csv`.

---

## Preprocessing & composite scores (`preproc.py`)

* **Per‑date (cross‑sectional)** winsorization → z‑score → direction fix → `_z_adj` (higher is better).
* Retain **raw** and **`_z_adj`** versions.
* **Bundles (equal‑weighted features)**:

  * **valuation**, **quality**, **income**, **balance**, **technical**.
* Output:

  * `panel_monthend_preprocessed.csv`
  * `preproc_stats.csv` (coverage, clipping stats, skipped fields etc.)

---

## LLM scoring (`llm_scoring.py` + `llm.yaml`)

* **Privacy:** `include_identifiers_in_prompt: false` (no ticker/date leakage).
* **Prompt** (system + instructions + data preamble) instructs the model to output **strict JSON**:

  ```json
  {"scores":{"valuation":int,"quality":int,"income":int,"balance":int,"technical":int,"overall":int},"reasoning":"string"}
  ```
* **Batching:** `tickers_per_batch=1`, `max_concurrent` tuned by tier (start 6; increase if no 429s).
* **Caching:** Local **response cache** keyed by prompt version + features; safe to stop/resume.
* **Outputs:**

  * Payloads: `outputs/llm_payloads/*.jsonl` (audit; 1 JSON per line)
  * Cache: `outputs/llm_cache/<cache_key>.json` (source of truth)
  * Merged panel: `outputs/panel_monthend_with_llm.csv`

### Performance tips

* Keep `temperature=0.0`, `max_tokens≈320`, `rationale_char_limit≈300`.
* Raise `max_concurrent` gradually (6 → 8 → 10) while watching 429s.
* Optionally **shard by date** (start/end in `llm.yaml`) and run 2–3 processes in parallel; cache dedupes.

### Debug/QA (optional)

* Set `OPENAI_LOG=info` to surface 429s.
* Use the provided QA notebooks/scripts to inspect **coverage**, **length**, **leakage flags**, and **boilerplate openings**.

---

## Backtesting (`backtest/run.py`)

* **Signal timing:** rank at month‑end; **enter next day’s close**; rebalance monthly; exit on next rebalance.
* **Selection:** Top‑K by **ensemble** of quant composite `score_quant_composite_z` and LLM `llm_score_overall_z_adj` (weights configurable).
* **Weights:** equal‑weight by default; score‑based also available.
* **Returns:** daily P\&L from daily close data; equity curves and summary metrics saved under `outputs/backtests/` (PNG + CSV).

---

## Grid search (`gridsearch/run_grid.py`)

* Search over:

  * **LLM bundle weights** (within LLM composite)
  * **Quant bundle weights**
  * **Overall ensemble weights** (LLM vs Quant)
  * **Top‑K**
  * **Portfolio weights** (equal vs score‑based)
* Saves leaderboard and `best_full.yaml`, which can drive backtests.

---

## Configuration cheat‑sheet

* **Run only LLM stage** while iterating:

  ```yaml
  # config/main.yaml
  stages:
    fundamentals: false
    technicals:  false
    preproc:     false
    llm_scoring: true
    backtest:    false
    gridsearch:  false
  resume:
    skip_existing_outputs: true
  ```
* **LLM concurrency** (Tier‑1 safe):

  ```yaml
  # config/llm.yaml
  model:
    openai:
      temperature: 0.0
      max_tokens: 320
    batching:
      tickers_per_batch: 1
      max_concurrent: 6   # bump cautiously if no 429s
      retry: { tries: 3, backoff_s: 3 }
  prompt:
    rationale_char_limit: 300
  ```
* **Secrets:** set `OPENAI_API_KEY` via environment. Leave YAML key empty.

---

## Reproducibility

* Keep `prompt.prefix_version` fixed to reuse cache. Bump when changing prompt text.
* With `temperature=0.0`, identical inputs → identical outputs (subject to provider determinism).
* All payloads are archived in JSONL for audit.

---

## Troubleshooting

* **Missing API key:** ensure `OPENAI_API_KEY` is defined in your IDE run config (PyCharm → Edit Configurations → Environment), not just in a shell.
* **pandas\_ta import issues:** upgrade `numpy`/`setuptools` and install `pandas_ta`; restart the venv if `pkg_resources`/`NaN` import errors occur.
* **merge\_asof “keys must be sorted”:** the panel merge requires sorted `date` per ticker; see `_asof_merge_per_column` for the canonical sort order.
* **Payload/JSONL counts look wrong:** keep payloads as `.jsonl` (1 object/line). Renaming to `.json` breaks counters.
* **Too many 429s:** reduce `max_concurrent`, or shard by date ranges.

---

## Contributing

* Style: Black + isort recommended.
* Type hints where practical.
* Keep configs declarative; avoid hard‑coding paths or weights in code.

---

## License

TBD (add your preferred license).

---

## Acknowledgements

* Original whitepaper provided by the user (see `/mnt/data/The GPT Portfolio White Paper.docx`).
* `pandas_ta` for technical indicators.
* OpenAI API for LLM scoring.
