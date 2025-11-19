"""
CLI wrapper to train a new model inside/outside Docker and promote it
only if it beats the current production baseline.

Usage:
    python -m src.models.train_and_promote --config configs/pipeline_config.yaml

Arguments:
- --config: path to base pipeline config.
- --model-dir: production artefact dir (default: models).
- --registry-dir: where run histories are stored (default: models/registry).
- --promote-metric: metric key in baseline_stats.json to compare (default: test_rmse).
- --run-id: optional custom run id (otherwise timestamp is used).
"""

from __future__ import annotations

import argparse
import copy
import os
from datetime import datetime

from src.models.registry import (
    RunInfo,
    build_output_paths,
    candidate_beats_baseline,
    load_production_stats,
    make_run_dir,
    promote_run,
    save_run_info,
    update_production_pointer,
)
from src.models.train_pipeline import load_config, run_training_from_config


def _override_output_paths(cfg: dict, run_dir: str) -> dict:
    cfg_copy = copy.deepcopy(cfg)
    out_paths = build_output_paths(run_dir)
    cfg_copy.setdefault("output", {})
    cfg_copy["output"].update(out_paths)
    return cfg_copy


def main():
    parser = argparse.ArgumentParser(
        description="Train model and promote if it beats current production."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Base pipeline config path.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Production model artefact directory.",
    )
    parser.add_argument(
        "--registry-dir",
        type=str,
        default="models/registry",
        help="Run history / candidate storage directory.",
    )
    parser.add_argument(
        "--promote-metric",
        type=str,
        default="test_rmse",
        help="Metric key in baseline_stats.json used for promotion.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id (default: timestamp).",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    run_dir = make_run_dir(args.registry_dir, args.run_id)
    cfg = _override_output_paths(base_cfg, run_dir)

    print(f"[train] Starting training run in {run_dir}")
    result = run_training_from_config(cfg)
    candidate_stats = result["baseline_stats"]

    baseline = load_production_stats(args.model_dir)
    metric = args.promote_metric
    candidate_metric = candidate_stats.get(metric)
    if candidate_metric is None:
        raise ValueError(
            f"Metric '{metric}' not found in candidate stats: {candidate_stats}"
        )
    baseline_metric = baseline.get(metric) if baseline else None

    is_better = candidate_beats_baseline(candidate_stats, baseline, metric=metric)
    if is_better:
        promote_run(run_dir, args.model_dir)
        print(
            f"[promote] Candidate {candidate_metric:.4f} beat "
            f"baseline {baseline_metric} on {metric}; promoted to {args.model_dir}."
        )
    else:
        print(
            f"[skip] Candidate {candidate_metric} did not beat "
            f"{baseline_metric} on {metric}; keeping production as-is."
        )

    run_info = RunInfo(
        run_id=os.path.basename(run_dir),
        run_dir=run_dir,
        promoted=is_better,
        metric_used=metric,
        candidate_metric=float(candidate_metric),
        baseline_metric=float(baseline_metric) if baseline_metric is not None else None,
        source_config=os.path.abspath(args.config),
        created_at=datetime.utcnow().isoformat(),
        output_paths=result["output_paths"],
    )
    save_run_info(run_dir, run_info)
    if is_better:
        update_production_pointer(args.registry_dir, run_info)


if __name__ == "__main__":
    main()
