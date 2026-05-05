"""
run_experiment.py
=================
Main experiment script.  Runs the full four-step framework for multiple
PMU treatment groups and exports all results.

Treatment groups (paper Section VI):
  G0  — no PMUs         (blind baseline)
  GA  — random-10 PMUs
  GB  — N-1 observable  (default)
  GC  — full coverage

Usage
-----
  python run_experiment.py                    # synthetic profiles, N=1000
  python run_experiment.py --N 500            # smaller Monte Carlo
  python run_experiment.py --ercot ./ercot_profiles   # real ERCOT data
  python run_experiment.py --groups G0 GB GC  # only run selected groups

Parameter philosophy
--------------------
relay_cfg and sev_w use RelayConfig() / SeverityWeights() defaults defined
in cpps_cascade.py — the single source of truth.  Override here only when
running sensitivity sweeps (see commented example in main()).
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from datetime import datetime

from cpps_cascade_428 import (
    # configs
    RelayConfig, SeverityWeights, PMUConfig, ScenarioConfig, PMUPlacement,
    PMUNetwork,
    # network
    build_rts24, add_pv_as_sgen, dispatch_pv, cap_slack,
    # profiles
    classify_load_buses, build_load_shape_matrix,
    # scenario generation
    generate_scenarios,
    # operating point
    apply_load_scaling, apply_initial_outages,
    # cascade
    run_cascade,
    # metrics
    aggregate, aggregate_table, treatment_group_table,
    # export
    results_to_dataframe, stage_records_to_dataframe,
    # types
    CascadeResult, ERCOTProfileLibrary,
)


# ══════════════════════════════════════════════════════════════════════════════
# Treatment group definitions
# ══════════════════════════════════════════════════════════════════════════════

TREATMENT_GROUPS: Dict[str, PMUConfig] = {
    "G0": PMUConfig(placement=PMUPlacement.NONE,      seed=42),
    "GA": PMUConfig(placement=PMUPlacement.RANDOM,    n_pmu_random=10, seed=42),
    "GB": PMUConfig(placement=PMUPlacement.N_MINUS_1, seed=42),
    "GC": PMUConfig(placement=PMUPlacement.FULL,      seed=42),
}


# ══════════════════════════════════════════════════════════════════════════════
# Single-group simulation
# ══════════════════════════════════════════════════════════════════════════════

def run_group(
    group_label:       str,
    pmu_cfg:           PMUConfig,
    scenarios,
    net_template,
    pv_sgen_ids:       List[int],
    base_load_p,
    base_load_q,
    load_shape_matrix: np.ndarray,
    relay_cfg:         RelayConfig,
    sev_w:             SeverityWeights,
    slack_cap:         float,
) -> List[CascadeResult]:
    """Run all scenarios for one PMU treatment group."""
    pmu     = PMUNetwork(net_template, pmu_cfg)
    results: List[CascadeResult] = []

    t0 = time.time()
    for scen in scenarios:
        net = copy.deepcopy(net_template)
        apply_load_scaling(net, base_load_p, base_load_q, load_shape_matrix, scen.t_index)
        dispatch_pv(net, pv_sgen_ids, scen.pv_cf, scen.t_index)
        apply_initial_outages(net, scen.init_outages)

        r             = run_cascade(net, relay_cfg, sev_w, slack_cap, pmu, scen.cyber_event)
        r.scenario_id = scen.scenario_id
        r.t_index     = scen.t_index
        r.source_date = scen.source_date
        results.append(r)

    elapsed = time.time() - t0
    m       = aggregate(results)
    print(f"  [{group_label}] done  {elapsed:.1f}s  "
          f"CFR={m['CFR']:.3f}  ACS={m['ACS']:.3f}  CRI={m['CRI']:.3f}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Print helpers
# ══════════════════════════════════════════════════════════════════════════════

def _print_stage_stats(label: str, df: pd.DataFrame) -> None:
    s = df["n_stages"]
    print(f"\nStage distribution ({label}):")
    print(s.describe().to_string())
    print(f"  n_stages == 0 : {(s == 0).sum():>5d}  ({(s == 0).mean()*100:.1f}%)")
    print(f"  n_stages == 1 : {(s == 1).sum():>5d}  ({(s == 1).mean()*100:.1f}%)")
    print(f"  n_stages >= 2 : {(s >= 2).sum():>5d}  ({(s >= 2).mean()*100:.1f}%)")
    print(f"  n_stages >= 5 : {(s >= 5).sum():>5d}  ({(s >= 5).mean()*100:.1f}%)")


def _print_cross_group_stages(all_results: Dict[str, List[CascadeResult]]) -> None:
    print("\nMean n_stages by group:")
    for label, res in all_results.items():
        df = results_to_dataframe(res)
        s  = df["n_stages"]
        print(f"  {label}: mean={s.mean():.2f}  max={s.max()}  "
              f"zero_frac={(s == 0).mean():.3f}  "
              f"multi_frac={(s >= 2).mean():.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="CPPS Resilience Experiment")
    parser.add_argument("--N",      type=int,   default=1000,
                        help="Number of Monte Carlo scenarios (default: 1000)")
    parser.add_argument("--seed",   type=int,   default=7)
    parser.add_argument("--season", default="equinox",
                        choices=["summer", "equinox", "winter"])
    parser.add_argument("--cyber",  type=float, default=0.35,
                        help="Fraction of scenarios with cyber attack")
    parser.add_argument("--ercot",  default=None,
                        help="Path to ercot_profiles/ directory")
    parser.add_argument("--groups", nargs="+", default=None,
                        help="Treatment groups to run (default: all)")
    parser.add_argument("--outdir", default="results",
                        help="Output directory")
    args = parser.parse_args(argv)

    script_dir = Path(__file__).parent.resolve()
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir     = script_dir / args.outdir / f"run_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[Output] saving to: {outdir.resolve()}")

    # ── Optional ERCOT library ────────────────────────────────────────────────
    ercot_lib = None
    if args.ercot:
        p = Path(args.ercot)
        ercot_lib = ERCOTProfileLibrary(
            solar_cf_path  = str(p / "ercot_solar_cf_shapes.npy"),
            load_norm_path = str(p / "ercot_load_norm_shapes.npy"),
            metadata_path  = str(p / "ercot_metadata.csv"),
        )

    # ── Configs ───────────────────────────────────────────────────────────────
    # Single source of truth: all relay and severity defaults live in
    # cpps_cascade.RelayConfig / SeverityWeights.  Do NOT repeat them here.
    # To run a sensitivity sweep, override only the fields you want, e.g.:
    #   relay_cfg = RelayConfig(tau=110.0, tau_hi=160.0, K_line=2)
    relay_cfg = RelayConfig()
    sev_w     = SeverityWeights()

    scen_cfg = ScenarioConfig(
        N=args.N, mode="MC", p_n2=0.20, p_gen_outage=0.15,
        seed=args.seed, T=24, season=args.season,
        cyber_fraction=args.cyber,
        use_real_profiles=(ercot_lib is not None),
        real_profile_lib=ercot_lib,
    )

    # ── Network ───────────────────────────────────────────────────────────────
    print("\n── Building network ─────────────────────────────────────────────")
    net_template = build_rts24(load_scale=1.25)
    pv_sgen_ids  = add_pv_as_sgen(net_template, n_pv=5, pv_pen_frac=0.15,
                                   seed=scen_cfg.seed)
    slack_cap    = cap_slack(net_template, relay_cfg)
    print(f"[Slack] ext_grid capped at ±{slack_cap:.1f} MW  "
          f"(slack_cap_factor={relay_cfg.slack_cap_factor})")
    print(f"[Relay] tau={relay_cfg.tau}%  tau_hi={relay_cfg.tau_hi}%  "
          f"K_line={relay_cfg.K_line}  K_uvls={relay_cfg.K_uvls}")

    bus_type_map      = classify_load_buses(net_template)
    load_shape_matrix = build_load_shape_matrix(net_template, scen_cfg.T, bus_type_map)
    base_load_p       = net_template.load["p_mw"].copy()
    base_load_q       = net_template.load["q_mvar"].copy()

    # ── Generate scenario library (shared across all groups) ──────────────────
    print("\n── Generating scenario library ──────────────────────────────────")
    ref_pmu   = PMUNetwork(net_template, TREATMENT_GROUPS["GB"])
    scenarios = generate_scenarios(net_template, scen_cfg, ref_pmu, load_shape_matrix)

    # ── Run selected treatment groups ─────────────────────────────────────────
    selected = args.groups if args.groups else list(TREATMENT_GROUPS.keys())
    print(f"\n── Running treatment groups: {selected} ─────────────────────────")

    all_results: Dict[str, List[CascadeResult]] = {}
    for label in selected:
        if label not in TREATMENT_GROUPS:
            print(f"  Warning: unknown group '{label}', skipping")
            continue
        print(f"\n  Group {label}:")
        all_results[label] = run_group(
            label, TREATMENT_GROUPS[label],
            scenarios, net_template, pv_sgen_ids,
            base_load_p, base_load_q, load_shape_matrix,
            relay_cfg, sev_w, slack_cap,
        )

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n══ RESULTS ══════════════════════════════════════════════════════")

    # Treatment group comparison
    tg_table = treatment_group_table(all_results)
    print("\nTreatment group comparison:")
    print(tg_table.to_string(index=False, float_format="{:.4f}".format))
    tg_table.to_csv(outdir / "treatment_groups.csv", index=False)

    # Per-attack-type breakdown (GB)
    if "GB" in all_results:
        at_table = aggregate_table(all_results["GB"])
        print("\nAttack-type breakdown (GB group):")
        print(at_table.to_string(index=False, float_format="{:.4f}".format))
        at_table.to_csv(outdir / "attack_type_breakdown_GB.csv", index=False)

    # Per-scenario + stage CSVs
    for label, res in all_results.items():
        df = results_to_dataframe(res)
        df.to_csv(outdir / f"scenarios_{label}.csv", index=False)
        sf = stage_records_to_dataframe(res)
        if not sf.empty:
            sf.to_csv(outdir / f"stages_{label}.csv", index=False)

    # Severity distribution (GB)
    if "GB" in all_results:
        df_gb = results_to_dataframe(all_results["GB"])

        print("\nSeverity distribution (GB):")
        print(df_gb["severity"].describe().to_string())

        sev_cls = pd.cut(
            df_gb["severity"],
            bins=[-0.001, 0.0, 0.05, 0.20, 1.001],
            labels=["None", "Small", "Medium", "Large"],
        )
        print("\nSeverity class counts (GB):")
        print(sev_cls.value_counts().sort_index().to_string())

        # Stage stats (GB)
        _print_stage_stats("GB", df_gb)

    # Cross-group stage comparison
    _print_cross_group_stages(all_results)

    print(f"\nAll outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
