# main.py
from __future__ import annotations
import argparse, os, yaml

from src.io_load import load_raw_csv
from src.prep_panel import month_end_price, forward_fill_fundamentals_monthend
from src.technicals import load_daily_for_technicals, build_monthly_frame, compute_monthly_indicators_pandasta
from src.preproc import preprocess_and_score
from src.llm_scoring import run_llm_scoring
from src.backtest.run import run_backtests
from src.gridsearch.run_grid import run_gridsearch

def _yload(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser(description="Orchestrator: run any subset of stages via main.yaml")
    ap.add_argument("--main-config", type=str, default="config/main.yaml")
    args = ap.parse_args()
    main_cfg = _yload(args.main_config)

    stages = main_cfg["stages"]
    inp = main_cfg["inputs"]
    ovr = main_cfg.get("overrides", {})
    resume = main_cfg.get("resume", {})
    skip_existing = bool(resume.get("skip_existing_outputs", True))

    # -------- Fundamentals --------
    if stages.get("fundamentals", False):
        data_cfg = _yload(inp["data_config"])
        out_csv = data_cfg["io"]["output_panel_csv"]
        if skip_existing and os.path.exists(out_csv):
            print(f"[panel] skip (exists): {out_csv}")
        else:
            df_daily = load_raw_csv(data_cfg["io"]["input_csv"], data_cfg)
            df_me_price = month_end_price(df_daily)
            panel_me = forward_fill_fundamentals_monthend(df_daily, df_me_price, data_cfg)
            panel_me.to_csv(out_csv, index=False)
            print(f"[panel] saved: {out_csv}")

    # -------- Technicals --------
    if stages.get("technicals", False):
        tech_cfg = _yload(inp["technicals_config"])
        tech_out = tech_cfg["io"]["output_csv"]
        if skip_existing and os.path.exists(tech_out):
            print(f"[technicals] skip (exists): {tech_out}")
        else:
            df_daily_tech = load_daily_for_technicals(tech_cfg["io"]["input_csv"], tech_cfg)
            monthly = build_monthly_frame(df_daily_tech, tech_cfg)
            df_me_ind = compute_monthly_indicators_pandasta(monthly, tech_cfg)
            df_me_ind.to_csv(tech_out, index=False)
            print(f"[technicals] saved: {tech_out}")

    # -------- Preproc --------
    if stages.get("preproc", False):
        pre_cfg = _yload(inp["preproc_config"])
        pre_out = pre_cfg["io"]["output_csv"]
        if skip_existing and os.path.exists(pre_out):
            print(f"[preproc] skip (exists): {pre_out}")
        else:
            df_out, df_stats = preprocess_and_score(pre_cfg)
            df_out.to_csv(pre_out, index=False)
            df_stats.to_csv(pre_cfg["io"]["stats_csv"], index=False)
            print(f"[preproc] saved: {pre_out}")
            print(f"[preproc] stats saved: {pre_cfg['io']['stats_csv']}")

    # -------- LLM scoring --------
    if stages.get("llm_scoring", False):
        llm_cfg = _yload(inp["llm_config"])
        merged_out = llm_cfg["io"]["merged_out_csv"]
        if ovr.get("llm_scoring", {}).get("force_provider"):
            llm_cfg["model"]["provider"] = ovr["llm_scoring"]["force_provider"]
        if skip_existing and os.path.exists(merged_out) and ovr.get("llm_scoring", {}).get("skip_if_exists", True):
            print(f"[llm] skip (exists): {merged_out}")
        else:
            print(f"[llm] scoring using: {inp['llm_config']}")
            # ensure we hand the (maybe updated) cfg path content to runner
            # quick temp write: not strictly necessary if you only override in-memory
            run_llm_scoring(
                preproc_csv=_yload(inp["preproc_config"])["io"]["output_csv"],
                preproc_cfg_path=inp["preproc_config"],
                llm_cfg_path=inp["llm_config"]
            )
            print(f"[llm] merged panel saved: {merged_out}")

    # -------- Gridsearch (optimizer placeholder) --------
    if stages.get("gridsearch", False):
        gcfg_path = inp["grid_config"]
        # optional: apply materialize_top_n override on the fly
        if "gridsearch" in ovr and "materialize_top_n" in ovr["gridsearch"]:
            gcfg = _yload(gcfg_path); gcfg["io"]["out_dir"] = gcfg["io"]["out_dir"]  # touch
            # You can write override back if needed; or just rely on gcfg file as-is.
        print(f"[grid] starting: {gcfg_path}")
        run_gridsearch(gcfg_path)

    # -------- Backtest --------
    if stages.get("backtest", False):
        bt_cfg_path = inp["backtest_config"]
        if ovr.get("backtest", {}).get("strategy_source", "config") == "grid_best":
            # Load a full strategy dump from gridsearch (see patch below)
            best_full = _yload(ovr["backtest"]["grid_best_path"])
            # Inject the best strategy (single) into the backtest config in-memory
            bt_cfg = _yload(bt_cfg_path)
            bt_cfg["strategies"] = [best_full["strategy"]]
            # Write a temp merged cfg to run
            tmp_bt = "config/_backtest_autogen.yaml"
            os.makedirs("config", exist_ok=True)
            with open(tmp_bt, "w") as f:
                yaml.safe_dump(bt_cfg, f, sort_keys=False)
            print(f"[bt] running best from grid: {ovr['backtest']['grid_best_path']}")
            _ = run_backtests(tmp_bt)
        else:
            print(f"[bt] running config: {bt_cfg_path}")
            _ = run_backtests(bt_cfg_path)

if __name__ == "__main__":
    main()
