from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from CompleteExperimentalReproduction import run_complete_experiments


def _parse_seeds(seeds_arg: str | None) -> list[int]:
    if not seeds_arg:
        return list(range(40, 48))
    return [int(x.strip()) for x in seeds_arg.split(",") if x.strip()]


def _summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "median": float("nan"),
            "q1": float("nan"),
            "q3": float("nan"),
            "iqr": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
        }
    median = float(np.median(values))
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = float(q3 - q1)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return {"median": median, "q1": q1, "q3": q3, "iqr": iqr, "mean": mean, "std": std}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cross-validation over multiple seeds (default 8).")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed list (default 40-47).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cv_8seeds"),
        help="Output directory for per-seed results and summaries.",
    )
    parser.add_argument("--backend", type=str, default="torch", choices=("torch", "tf"), help="Backend to use.")
    parser.add_argument("--epochs", type=int, default=800, help="Training epochs.")
    parser.add_argument(
        "--max-physics-weight",
        type=float,
        default=0.1,
        help="Max physics weight (lambda_physics,max).",
    )
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict[str, float | int]] = []
    pinn_rrmse_vals: list[float] = []
    vteam_rrmse_vals: list[float] = []
    improvement_vals: list[float] = []

    for seed in seeds:
        run_dir = output_dir / f"seed_{seed}"
        results = run_complete_experiments(
            backend=args.backend,
            seed=seed,
            epochs=args.epochs,
            max_physics_weight=args.max_physics_weight,
            results_dir=run_dir,
            show_plots=False,
        )
        metrics = results.get("metrics_summary", {})
        pinn_rrmse = float(metrics.get("pinn_rrmse", float("nan")))
        vteam_rrmse = float(metrics.get("vteam_rrmse", float("nan")))
        improvement = float(
            metrics.get("vteam_improvement", vteam_rrmse / pinn_rrmse if pinn_rrmse else float("nan"))
        )

        per_seed_rows.append(
            {
                "seed": int(seed),
                "pinn_rrmse": pinn_rrmse,
                "vteam_rrmse": vteam_rrmse,
                "improvement": improvement,
                "training_time_minutes": float(metrics.get("training_time_minutes", float("nan"))),
            }
        )

        pinn_rrmse_vals.append(pinn_rrmse)
        vteam_rrmse_vals.append(vteam_rrmse)
        improvement_vals.append(improvement)

    pinn_rrmse_arr = np.array(pinn_rrmse_vals, dtype=float)
    vteam_rrmse_arr = np.array(vteam_rrmse_vals, dtype=float)
    improvement_arr = np.array(improvement_vals, dtype=float)

    aggregate = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "pinn_rrmse": _summary(pinn_rrmse_arr),
        "vteam_rrmse": _summary(vteam_rrmse_arr),
        "improvement": _summary(improvement_arr),
        "best_seed": int(seeds[int(np.nanargmin(pinn_rrmse_arr))]) if len(seeds) else None,
        "worst_seed": int(seeds[int(np.nanargmax(pinn_rrmse_arr))]) if len(seeds) else None,
    }

    per_seed_path = output_dir / "per_seed_metrics.csv"
    with per_seed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_seed_rows)

    (output_dir / "metrics_aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    agg_csv_path = output_dir / "metrics_aggregate.csv"
    with agg_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["seeds", json.dumps(seeds)])
        writer.writerow(["n_seeds", len(seeds)])
        for key in ("pinn_rrmse", "vteam_rrmse", "improvement"):
            for subkey, val in aggregate[key].items():
                writer.writerow([f"{key}_{subkey}", val])
        writer.writerow(["best_seed", aggregate["best_seed"]])
        writer.writerow(["worst_seed", aggregate["worst_seed"]])

    print("Cross-validation completed.")
    print(f"Seeds: {seeds}")
    print(
        f"PRNS median RRMSE: {aggregate['pinn_rrmse']['median']:.4f} "
        f"(IQR {aggregate['pinn_rrmse']['q1']:.4f}–{aggregate['pinn_rrmse']['q3']:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
