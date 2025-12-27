from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATASET = Path(__file__).with_name("printed_memristor_training_data.csv")


def _train_test_split(
    *arrays: np.ndarray,
    ratio: float = 0.8,
    split_seed: int = 12345,
) -> tuple[np.ndarray, ...]:
    if not arrays:
        raise ValueError("At least one array required for train/test split.")
    length = len(arrays[0])
    for arr in arrays:
        if len(arr) != length:
            raise ValueError("All arrays must share the same length for splitting.")
    rng = np.random.default_rng(split_seed)
    indices = np.arange(length)
    rng.shuffle(indices)
    split_idx = int(ratio * length)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    splitted: list[np.ndarray] = []
    for arr in arrays:
        splitted.append(arr[train_idx])
        splitted.append(arr[test_idx])
    return tuple(splitted)


def _to_float(x) -> float:
    try:
        return float(x.detach().cpu().item())  # torch
    except Exception:
        return float(x)


def _save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Loss-weight sensitivity sweep (lambda_physics) for printed memristor physics-regularized surrogate."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="torch",
        choices=("torch", "tf"),
        help="Backend for the neural surrogate. Use 'torch' on Python 3.13/3.14; TensorFlow is typically unavailable there.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--concentration-label", type=str, default="10_percent_PMMA")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--noise-std", type=float, default=0.002)
    parser.add_argument("--variability-bound", type=float, default=0.05)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--neurons", type=int, default=128)
    parser.add_argument("--state-mixing", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-physics-weights",
        type=str,
        default="0.0,0.1,0.5,1.0",
        help="Comma-separated sweep values for max physics weight (lambda_physics).",
    )
    parser.add_argument("--split-seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path, default=Path("results/lambda_sensitivity"))
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    parser.add_argument("--benchmark-repeats", type=int, default=200)
    args = parser.parse_args()

    weights = [float(x.strip()) for x in args.max_physics_weights.split(",") if x.strip() != ""]
    if not weights:
        raise ValueError("No --max-physics-weights provided.")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset": str(args.dataset),
        "concentration_label": args.concentration_label,
        "device_id": args.device_id,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "noise_std": args.noise_std,
        "variability_bound": args.variability_bound,
        "hidden_layers": args.hidden_layers,
        "neurons_per_layer": args.neurons,
        "state_mixing": args.state_mixing,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "max_physics_weights": weights,
    }
    _save_json(output_dir / "run_config.json", run_meta)

    # Load once, split once, reuse for all sweeps
    if args.backend == "tf":
        from TrainingFrameworkwithNoiseInjection import PINNTrainer
        from mainPINNmodel import PrintedMemristorPINN

        trainer_for_data = PINNTrainer(
            PrintedMemristorPINN(hidden_layers=1, neurons_per_layer=8, random_seed=args.seed),
            learning_rate=args.learning_rate,
            seed=args.seed,
            state_mixing=args.state_mixing,
        )
        voltage, current, state, concentration = trainer_for_data.load_experimental_data(
            args.dataset,
            concentration_label=args.concentration_label,
            device_id=args.device_id,
            use_noisy_columns=True,
        )
    else:
        from torch_surrogate import PrintedMemristorSurrogateTorch
        from torch_training import SurrogateTrainerTorch

        trainer_for_data = SurrogateTrainerTorch(
            PrintedMemristorSurrogateTorch(hidden_layers=1, neurons_per_layer=8, seed=args.seed),
            learning_rate=args.learning_rate,
            seed=args.seed,
            state_mixing=args.state_mixing,
        )
        voltage, current, state, concentration = trainer_for_data.load_experimental_data(
            args.dataset,
            concentration_label=args.concentration_label,
            device_id=args.device_id,
            use_noisy_columns=True,
        )
    (V_train, V_test, I_train, I_test, x_train, x_test, c_train, c_test) = _train_test_split(
        voltage,
        current,
        state,
        concentration,
        ratio=0.8,
        split_seed=args.split_seed,
    )

    rows: list[dict[str, float | int | str]] = []
    for max_physics_weight in weights:
        print(f"\n=== Sweep: max_physics_weight={max_physics_weight:g} ===")

        if args.backend == "tf":
            import tensorflow as tf

            from ExperimentalValidationFramework import ExperimentalValidator
            from TrainingFrameworkwithNoiseInjection import PINNTrainer
            from mainPINNmodel import PrintedMemristorPINN

            pinn = PrintedMemristorPINN(
                hidden_layers=args.hidden_layers,
                neurons_per_layer=args.neurons,
                input_features=("voltage", "state"),
                random_seed=args.seed,
                trainable_params=("ohmic_conductance",),
            )
            trainer = PINNTrainer(
                pinn,
                learning_rate=args.learning_rate,
                seed=args.seed,
                state_mixing=args.state_mixing,
            )
            loss_history = trainer.train(
                epochs=args.epochs,
                voltage=V_train,
                current=I_train,
                state=x_train,
                concentration=None,
                noise_std=args.noise_std,
                variability_bound=args.variability_bound,
                verbose_every=50,
                max_physics_weight=max_physics_weight,
            )
            validator = ExperimentalValidator(pinn, seed=args.seed)
            I_pred, _ = validator.predict_current(V_test, x_test)
            rrmse = validator.calculate_rrmse(I_pred, I_test)
            mape = validator.calculate_mape(I_pred, I_test)
            inputs = validator._build_inputs(V_test, x_test, None)
            I_pred_tf, x_deriv_pred_tf = pinn.model(inputs, training=False)
            components = pinn.physics_loss_components(
                tf.convert_to_tensor(V_test, dtype=pinn.dtype),
                I_pred_tf,
                tf.convert_to_tensor(x_test, dtype=pinn.dtype),
                x_deriv_pred=x_deriv_pred_tf,
                params=pinn.physical_params,
            )
            waveform_bench = validator.benchmark_predict_current(
                V_test,
                x_test,
                warmup=args.benchmark_warmup,
                repeats=args.benchmark_repeats,
                include_input_packing=True,
            )
            sample_bench = validator.benchmark_predict_current(
                np.asarray([float(V_test[0])], dtype=np.float32),
                np.asarray([float(x_test[0])], dtype=np.float32),
                warmup=args.benchmark_warmup,
                repeats=args.benchmark_repeats,
                include_input_packing=True,
            )
        else:
            import torch

            from torch_surrogate import PrintedMemristorSurrogateTorch
            from torch_training import SurrogateTrainerTorch
            from torch_validation import ExperimentalValidatorTorch

            pinn = PrintedMemristorSurrogateTorch(
                hidden_layers=args.hidden_layers,
                neurons_per_layer=args.neurons,
                input_features=("voltage", "state"),
                seed=args.seed,
                trainable_params=("ohmic_conductance",),
                device=torch.device("cpu"),
            )
            trainer = SurrogateTrainerTorch(
                pinn,
                learning_rate=args.learning_rate,
                seed=args.seed,
                state_mixing=args.state_mixing,
            )
            loss_history = trainer.train(
                epochs=args.epochs,
                voltage=V_train,
                current=I_train,
                state=x_train,
                concentration=None,
                noise_std=args.noise_std,
                variability_bound=args.variability_bound,
                verbose_every=50,
                max_physics_weight=max_physics_weight,
            )
            validator = ExperimentalValidatorTorch(pinn, seed=args.seed)
            I_pred, _ = validator.predict_current(V_test, x_test)
            rrmse = validator.calculate_rrmse(I_pred, I_test)
            mape = validator.calculate_mape(I_pred, I_test)
            with torch.no_grad():
                inputs = validator._build_inputs(V_test, x_test, None)
                I_pred_t, xdot_pred_t = pinn.forward(inputs)
                comps = pinn.physics_loss_components(
                    torch.tensor(V_test, dtype=pinn.dtype, device=pinn.device).reshape(-1),
                    I_pred_t.reshape(-1),
                    torch.tensor(x_test, dtype=pinn.dtype, device=pinn.device).reshape(-1),
                    xdot_pred_t.reshape(-1),
                    params_override=pinn.physical_params,
                )
            components = comps
            waveform_bench = validator.benchmark_predict_current(
                V_test,
                x_test,
                warmup=args.benchmark_warmup,
                repeats=args.benchmark_repeats,
                include_input_packing=True,
            )
            sample_bench = validator.benchmark_predict_current(
                np.asarray([float(V_test[0])], dtype=np.float32),
                np.asarray([float(x_test[0])], dtype=np.float32),
                warmup=args.benchmark_warmup,
                repeats=args.benchmark_repeats,
                include_input_packing=True,
            )

        final = loss_history[-1] if loss_history else {}
        has_nan = any((not np.isfinite(entry.get("total_loss", np.inf))) for entry in loss_history)

        row: dict[str, float | int | str] = {
            "max_physics_weight": float(max_physics_weight),
            "epochs": int(args.epochs),
            "seed": int(args.seed),
            "train_samples": int(V_train.size),
            "test_samples": int(V_test.size),
            "rrmse": float(rrmse),
            "mape": float(mape),
            "final_total_loss": float(final.get("total_loss", float("nan"))),
            "final_data_loss": float(final.get("data_loss", float("nan"))),
            "final_physics_loss": float(final.get("physics_loss", float("nan"))),
            "final_lambda_physics": float(final.get("lambda_physics", float("nan"))),
            "training_time_seconds": float(final.get("training_time_seconds", float("nan"))),
            "diverged_nan": str(bool(has_nan)),
            "res_ohmic": _to_float(components["ohmic"]),
            "res_sclc": _to_float(components["sclc"]),
            "res_ode": _to_float(components["ode"]),
            "res_total": _to_float(components["total"]),
            "waveform_latency_mean_ms": float(waveform_bench["latency_mean_ms"]),
            "waveform_latency_p95_ms": float(waveform_bench["latency_p95_ms"]),
            "waveform_throughput_sps": float(waveform_bench["throughput_samples_per_s"]),
            "sample_latency_mean_ms": float(sample_bench["latency_mean_ms"]),
            "sample_latency_p95_ms": float(sample_bench["latency_p95_ms"]),
            "sample_throughput_sps": float(sample_bench["throughput_samples_per_s"]),
        }
        rows.append(row)
        _save_json(output_dir / f"metrics_wphys_{max_physics_weight:g}.json", row)

    # Write CSV summary
    csv_path = output_dir / "sweep_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Plots
    weights_arr = np.array([r["max_physics_weight"] for r in rows], dtype=float)
    rrmse_arr = np.array([r["rrmse"] for r in rows], dtype=float)
    mape_arr = np.array([r["mape"] for r in rows], dtype=float)
    res_ohm = np.array([r["res_ohmic"] for r in rows], dtype=float)
    res_sclc = np.array([r["res_sclc"] for r in rows], dtype=float)
    res_ode = np.array([r["res_ode"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(weights_arr, rrmse_arr, "-o", label="RRMSE", linewidth=2)
    ax.plot(weights_arr, mape_arr, "-s", label="MAPE", linewidth=2)
    ax.set_xlabel(r"max $\lambda_{physics}$ (with $\lambda_{data}=1$)")
    ax.set_ylabel("Error")
    ax.set_title("Loss-weight sensitivity: accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_weight_sensitivity_accuracy.png", dpi=600)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(weights_arr, res_ohm, "-o", label="Ohmic residual", linewidth=2)
    ax.plot(weights_arr, res_sclc, "-s", label="SCLC residual", linewidth=2)
    ax.plot(weights_arr, res_ode, "-^", label="ODE residual", linewidth=2)
    ax.set_xlabel(r"max $\lambda_{physics}$ (with $\lambda_{data}=1$)")
    ax.set_ylabel("Residual (MSE)")
    ax.set_title("Loss-weight sensitivity: physics residuals")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_weight_sensitivity_residuals.png", dpi=600)
    plt.close(fig)

    print(f"\nSaved sweep results to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
