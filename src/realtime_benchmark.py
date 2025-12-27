from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from VTEAMModelComparison import VTEAMModel


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


def _percentile_ms(times_ns: list[int], q: float) -> float:
    arr = np.asarray(times_ns, dtype=np.float64) / 1e6
    return float(np.percentile(arr, q))


def _mean_ms(times_ns: list[int]) -> float:
    arr = np.asarray(times_ns, dtype=np.float64) / 1e6
    return float(np.mean(arr))


def _benchmark_vteam(voltage: np.ndarray, repeats: int, warmup: int) -> dict[str, float]:
    model = VTEAMModel()

    for _ in range(int(warmup)):
        _ = model.simulate_iv(voltage)

    times_ns: list[int] = []
    for _ in range(int(repeats)):
        t0 = perf_counter_ns()
        _ = model.simulate_iv(voltage)
        t1 = perf_counter_ns()
        times_ns.append(t1 - t0)

    n_samples = int(np.asarray(voltage).size)
    mean_ms = _mean_ms(times_ns)
    p95_ms = _percentile_ms(times_ns, 95)

    # Also benchmark a true single-sample call (captures Python overhead)
    v1 = np.asarray(voltage[:1], dtype=np.float32)
    for _ in range(int(warmup)):
        _ = model.simulate_iv(v1)
    times1_ns: list[int] = []
    for _ in range(int(repeats)):
        t0 = perf_counter_ns()
        _ = model.simulate_iv(v1)
        t1 = perf_counter_ns()
        times1_ns.append(t1 - t0)
    sample_mean_ms = _mean_ms(times1_ns)
    sample_p95_ms = _percentile_ms(times1_ns, 95)

    return {
        "waveform_latency_mean_ms": float(mean_ms),
        "waveform_latency_p95_ms": float(p95_ms),
        "per_sample_latency_mean_ms": float(sample_mean_ms),
        "per_sample_latency_p95_ms": float(sample_p95_ms),
        "waveform_divided_per_sample_mean_ms": float(mean_ms / max(n_samples, 1)),
        "waveform_divided_per_sample_p95_ms": float(p95_ms / max(n_samples, 1)),
    }


class YakopcicModel:
    """Lightweight Yakopcic generalized memristor model (for baseline timing)."""

    def __init__(self):
        self.a1 = 0.17
        self.a2 = 0.17
        self.b = 0.05
        self.vp = 0.16
        self.vn = 0.15
        self.ap = 4000
        self.an = 4000
        self.xp = 0.3
        self.xn = 0.5
        self.alphap = 1
        self.alphan = 5
        self.eta = 1

    def current(self, V: float, x: float) -> float:
        g = self.a1 * x * np.sinh(self.b * V) if V >= 0 else self.a2 * x * np.sinh(self.b * V)
        return g * V

    def state_derivative(self, V: float, x: float) -> float:
        def f(V_app):
            if V_app >= self.vp:
                return self.ap * (np.exp(V_app) - np.exp(self.vp))
            if V_app <= -self.vn:
                return -self.an * (np.exp(-V_app) - np.exp(self.vn))
            return 0.0

        def window(x_val: float, p: float) -> float:
            return 1.0 - (2.0 * x_val - 1.0) ** (2.0 * p)

        if V >= 0:
            return self.eta * f(V) * window(x, self.alphap) if x < self.xp else 0.0
        return self.eta * f(V) * window(x, self.alphan) if x > self.xn else 0.0

    def simulate_iv(self, voltage_sweep: np.ndarray) -> np.ndarray:
        x = 0.5
        current = np.zeros_like(voltage_sweep, dtype=float)
        dt = 0.001
        for i, V in enumerate(voltage_sweep):
            x = float(np.clip(x + self.state_derivative(float(V), x) * dt, 0.0, 1.0))
            current[i] = self.current(float(V), x)
        return current


def _benchmark_yakopcic(voltage: np.ndarray, repeats: int, warmup: int) -> dict[str, float]:
    model = YakopcicModel()

    for _ in range(int(warmup)):
        _ = model.simulate_iv(voltage)

    times_ns: list[int] = []
    for _ in range(int(repeats)):
        t0 = perf_counter_ns()
        _ = model.simulate_iv(voltage)
        t1 = perf_counter_ns()
        times_ns.append(t1 - t0)

    n_samples = int(np.asarray(voltage).size)
    mean_ms = _mean_ms(times_ns)
    p95_ms = _percentile_ms(times_ns, 95)

    v1 = np.asarray(voltage[:1], dtype=np.float32)
    for _ in range(int(warmup)):
        _ = model.simulate_iv(v1)
    times1_ns: list[int] = []
    for _ in range(int(repeats)):
        t0 = perf_counter_ns()
        _ = model.simulate_iv(v1)
        t1 = perf_counter_ns()
        times1_ns.append(t1 - t0)
    sample_mean_ms = _mean_ms(times1_ns)
    sample_p95_ms = _percentile_ms(times1_ns, 95)

    return {
        "waveform_latency_mean_ms": float(mean_ms),
        "waveform_latency_p95_ms": float(p95_ms),
        "per_sample_latency_mean_ms": float(sample_mean_ms),
        "per_sample_latency_p95_ms": float(sample_p95_ms),
        "waveform_divided_per_sample_mean_ms": float(mean_ms / max(n_samples, 1)),
        "waveform_divided_per_sample_p95_ms": float(p95_ms / max(n_samples, 1)),
    }


def _model_size_bytes_torch(total_params: int) -> dict[str, float]:
    total_params_f = float(total_params)
    return {
        "total_params": total_params_f,
        "float32_param_bytes": float(total_params_f * 4.0),
        "float32_param_kib": float(total_params_f * 4.0 / 1024.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-time inference benchmarks (mean/P95) for surrogate vs VTEAM.")
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
    parser.add_argument("--split-seed", type=int, default=12345)
    parser.add_argument("--max-physics-weight", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("results/realtime_benchmark"))
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "tf":
        from mainPINNmodel import PrintedMemristorPINN
        from TrainingFrameworkwithNoiseInjection import PINNTrainer
        from ExperimentalValidationFramework import ExperimentalValidator

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
        import torch

        from torch_surrogate import PrintedMemristorSurrogateTorch
        from torch_training import SurrogateTrainerTorch
        from torch_validation import ExperimentalValidatorTorch

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

    (V_train, V_test, I_train, I_test, x_train, x_test, _, _) = _train_test_split(
        voltage,
        current,
        state,
        concentration,
        ratio=0.8,
        split_seed=args.split_seed,
    )

    if args.backend == "tf":
        # Train physics-regularized surrogate
        pinn_phys = PrintedMemristorPINN(
            hidden_layers=args.hidden_layers,
            neurons_per_layer=args.neurons,
            input_features=("voltage", "state"),
            random_seed=args.seed,
            trainable_params=("ohmic_conductance",),
        )
        trainer_phys = PINNTrainer(
            pinn_phys,
            learning_rate=args.learning_rate,
            seed=args.seed,
            state_mixing=args.state_mixing,
        )
        _ = trainer_phys.train(
            epochs=args.epochs,
            voltage=V_train,
            current=I_train,
            state=x_train,
            concentration=None,
            noise_std=args.noise_std,
            variability_bound=args.variability_bound,
            verbose_every=0,
            max_physics_weight=args.max_physics_weight,
        )
        validator_phys = ExperimentalValidator(pinn_phys, seed=args.seed)

        # Train data-only baseline (same architecture, max_physics_weight=0)
        pinn_data = PrintedMemristorPINN(
            hidden_layers=args.hidden_layers,
            neurons_per_layer=args.neurons,
            input_features=("voltage", "state"),
            random_seed=args.seed,
            trainable_params=("ohmic_conductance",),
        )
        trainer_data = PINNTrainer(
            pinn_data,
            learning_rate=args.learning_rate,
            seed=args.seed,
            state_mixing=args.state_mixing,
        )
        _ = trainer_data.train(
            epochs=args.epochs,
            voltage=V_train,
            current=I_train,
            state=x_train,
            concentration=None,
            noise_std=args.noise_std,
            variability_bound=args.variability_bound,
            verbose_every=0,
            max_physics_weight=0.0,
        )
        validator_data = ExperimentalValidator(pinn_data, seed=args.seed)

        model_size_phys = {
            "total_params": float(pinn_phys.model.count_params() + len(pinn_phys.get_trainable_param_values())),
            "float32_param_kib": float((pinn_phys.model.count_params() + len(pinn_phys.get_trainable_param_values())) * 4.0 / 1024.0),
        }
        model_size_data = {
            "total_params": float(pinn_data.model.count_params() + len(pinn_data.get_trainable_param_values())),
            "float32_param_kib": float((pinn_data.model.count_params() + len(pinn_data.get_trainable_param_values())) * 4.0 / 1024.0),
        }
    else:
        # Train physics-regularized surrogate
        pinn_phys = PrintedMemristorSurrogateTorch(
            hidden_layers=args.hidden_layers,
            neurons_per_layer=args.neurons,
            input_features=("voltage", "state"),
            seed=args.seed,
            trainable_params=("ohmic_conductance",),
            device=torch.device("cpu"),
        )
        trainer_phys = SurrogateTrainerTorch(
            pinn_phys,
            learning_rate=args.learning_rate,
            seed=args.seed,
            state_mixing=args.state_mixing,
        )
        _ = trainer_phys.train(
            epochs=args.epochs,
            voltage=V_train,
            current=I_train,
            state=x_train,
            concentration=None,
            noise_std=args.noise_std,
            variability_bound=args.variability_bound,
            verbose_every=0,
            max_physics_weight=args.max_physics_weight,
        )
        validator_phys = ExperimentalValidatorTorch(pinn_phys, seed=args.seed)

        # Train data-only baseline (same architecture, max_physics_weight=0)
        pinn_data = PrintedMemristorSurrogateTorch(
            hidden_layers=args.hidden_layers,
            neurons_per_layer=args.neurons,
            input_features=("voltage", "state"),
            seed=args.seed,
            trainable_params=("ohmic_conductance",),
            device=torch.device("cpu"),
        )
        trainer_data = SurrogateTrainerTorch(
            pinn_data,
            learning_rate=args.learning_rate,
            seed=args.seed,
            state_mixing=args.state_mixing,
        )
        _ = trainer_data.train(
            epochs=args.epochs,
            voltage=V_train,
            current=I_train,
            state=x_train,
            concentration=None,
            noise_std=args.noise_std,
            variability_bound=args.variability_bound,
            verbose_every=0,
            max_physics_weight=0.0,
        )
        validator_data = ExperimentalValidatorTorch(pinn_data, seed=args.seed)

        model_size_phys = _model_size_bytes_torch(sum(p.numel() for p in pinn_phys.trainable_parameters()))
        model_size_data = _model_size_bytes_torch(sum(p.numel() for p in pinn_data.trainable_parameters()))

    # Benchmarks
    phys_waveform = validator_phys.benchmark_predict_current(
        V_test,
        x_test,
        warmup=args.warmup,
        repeats=args.repeats,
        include_input_packing=True,
    )
    phys_sample = validator_phys.benchmark_predict_current(
        np.asarray([float(V_test[0])], dtype=np.float32),
        np.asarray([float(x_test[0])], dtype=np.float32),
        warmup=args.warmup,
        repeats=args.repeats,
        include_input_packing=True,
    )
    data_waveform = validator_data.benchmark_predict_current(
        V_test,
        x_test,
        warmup=args.warmup,
        repeats=args.repeats,
        include_input_packing=True,
    )
    data_sample = validator_data.benchmark_predict_current(
        np.asarray([float(V_test[0])], dtype=np.float32),
        np.asarray([float(x_test[0])], dtype=np.float32),
        warmup=args.warmup,
        repeats=args.repeats,
        include_input_packing=True,
    )
    vteam_bench = _benchmark_vteam(V_test, repeats=args.repeats, warmup=args.warmup)
    yakopcic_bench = _benchmark_yakopcic(V_test, repeats=args.repeats, warmup=args.warmup)

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "settings": {
            "backend": args.backend,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "noise_std": args.noise_std,
            "variability_bound": args.variability_bound,
            "hidden_layers": args.hidden_layers,
            "neurons_per_layer": args.neurons,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "max_physics_weight": args.max_physics_weight,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "test_samples": int(np.asarray(V_test).size),
        "physics_regularized": {
            "model_size": model_size_phys,
            "waveform": phys_waveform,
            "per_sample": phys_sample,
        },
        "data_only": {
            "model_size": model_size_data,
            "waveform": data_waveform,
            "per_sample": data_sample,
        },
        "vteam": vteam_bench,
        "yakopcic": yakopcic_bench,
    }

    (output_dir / "realtime_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Flat CSV (nice for paper tables)
    rows = [
        {
            "model": "physics_regularized_surrogate",
            "waveform_latency_mean_ms": phys_waveform["latency_mean_ms"],
            "waveform_latency_p95_ms": phys_waveform["latency_p95_ms"],
            "waveform_throughput_sps": phys_waveform["throughput_samples_per_s"],
            "per_sample_latency_mean_ms": phys_sample["latency_mean_ms"],
            "per_sample_latency_p95_ms": phys_sample["latency_p95_ms"],
            "per_sample_throughput_sps": phys_sample["throughput_samples_per_s"],
            "params_total": payload["physics_regularized"]["model_size"]["total_params"],
            "float32_param_kib": payload["physics_regularized"]["model_size"]["float32_param_kib"],
        },
        {
            "model": "data_only_surrogate",
            "waveform_latency_mean_ms": data_waveform["latency_mean_ms"],
            "waveform_latency_p95_ms": data_waveform["latency_p95_ms"],
            "waveform_throughput_sps": data_waveform["throughput_samples_per_s"],
            "per_sample_latency_mean_ms": data_sample["latency_mean_ms"],
            "per_sample_latency_p95_ms": data_sample["latency_p95_ms"],
            "per_sample_throughput_sps": data_sample["throughput_samples_per_s"],
            "params_total": payload["data_only"]["model_size"]["total_params"],
            "float32_param_kib": payload["data_only"]["model_size"]["float32_param_kib"],
        },
        {
            "model": "vteam",
            "waveform_latency_mean_ms": vteam_bench["waveform_latency_mean_ms"],
            "waveform_latency_p95_ms": vteam_bench["waveform_latency_p95_ms"],
            "waveform_throughput_sps": float(int(np.asarray(V_test).size) / (vteam_bench["waveform_latency_mean_ms"] / 1e3)),
            "per_sample_latency_mean_ms": vteam_bench["per_sample_latency_mean_ms"],
            "per_sample_latency_p95_ms": vteam_bench["per_sample_latency_p95_ms"],
            "per_sample_throughput_sps": float(1.0 / (vteam_bench["per_sample_latency_mean_ms"] / 1e3)),
            "params_total": "",
            "float32_param_kib": "",
        },
        {
            "model": "yakopcic",
            "waveform_latency_mean_ms": yakopcic_bench["waveform_latency_mean_ms"],
            "waveform_latency_p95_ms": yakopcic_bench["waveform_latency_p95_ms"],
            "waveform_throughput_sps": float(int(np.asarray(V_test).size) / (yakopcic_bench["waveform_latency_mean_ms"] / 1e3)),
            "per_sample_latency_mean_ms": yakopcic_bench["per_sample_latency_mean_ms"],
            "per_sample_latency_p95_ms": yakopcic_bench["per_sample_latency_p95_ms"],
            "per_sample_throughput_sps": float(1.0 / (yakopcic_bench["per_sample_latency_mean_ms"] / 1e3)),
            "params_total": "",
            "float32_param_kib": "",
        },
    ]

    with (output_dir / "realtime_benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved benchmarks to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
