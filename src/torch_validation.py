from __future__ import annotations

from time import perf_counter_ns
import time

import numpy as np
import torch

from torch_surrogate import PrintedMemristorSurrogateTorch


class ExperimentalValidatorTorch:
    def __init__(self, surrogate: PrintedMemristorSurrogateTorch, seed: int | None = None) -> None:
        self.surrogate = surrogate
        self._rng = np.random.default_rng(seed)

    def _build_inputs(self, voltage: np.ndarray, state: np.ndarray | None, concentration: np.ndarray | None) -> torch.Tensor:
        voltage = np.asarray(voltage, dtype=np.float32)
        if state is None:
            state = np.zeros_like(voltage, dtype=np.float32)
        else:
            state = np.asarray(state, dtype=np.float32)
        cols = []
        for feature in self.surrogate.input_features:
            if feature == "voltage":
                cols.append(torch.tensor(voltage, dtype=self.surrogate.dtype, device=self.surrogate.device))
            elif feature == "state":
                cols.append(torch.tensor(state, dtype=self.surrogate.dtype, device=self.surrogate.device))
            elif feature == "concentration":
                if concentration is None:
                    raise ValueError("Concentration feature required but not provided.")
                conc = np.asarray(concentration, dtype=np.float32)
                cols.append(torch.tensor(conc, dtype=self.surrogate.dtype, device=self.surrogate.device))
            else:
                raise ValueError(f"Unsupported input feature '{feature}'.")
        return torch.stack(cols, dim=1)

    def predict_current(
        self,
        voltage: np.ndarray,
        state: np.ndarray | None = None,
        concentration: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        self.surrogate.model.eval()
        start = time.time()
        with torch.no_grad():
            inputs = self._build_inputs(voltage, state, concentration)
            current_pred, _ = self.surrogate.forward(inputs)
            out = current_pred.detach().cpu().numpy().reshape(-1)
        return out, float(time.time() - start)

    def calculate_noise_robustness(
        self,
        V_clean: np.ndarray,
        baseline_prediction: np.ndarray,
        noise_levels: list[float],
        state: np.ndarray | None = None,
        concentration: np.ndarray | None = None,
    ) -> dict[float, float]:
        results: dict[float, float] = {}
        V_clean = np.asarray(V_clean, dtype=np.float32)
        baseline_prediction = np.asarray(baseline_prediction, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32) if state is not None else None
        concentration = np.asarray(concentration, dtype=np.float32) if concentration is not None else None

        for noise_std in noise_levels:
            V_noisy = V_clean + self._rng.normal(0.0, noise_std * (np.std(V_clean) + 1e-12), size=V_clean.shape)
            noisy_pred, _ = self.predict_current(V_noisy, state, concentration)
            results[float(noise_std)] = self.calculate_rrmse(noisy_pred, baseline_prediction)
        return results

    def statistical_validation(self, n_cycles: int = 50) -> dict[str, object]:
        set_voltages = []
        on_resistances = []
        for _ in range(int(n_cycles)):
            V_cycle, I_cycle, _ = self.generate_cycle_with_variability()
            set_voltages.append(self.detect_set_voltage(V_cycle, I_cycle))
            on_resistances.append(self.calculate_on_resistance(V_cycle, I_cycle))
        set_voltages = np.array(set_voltages, dtype=float)
        on_resistances = np.array(on_resistances, dtype=float)
        return {
            "set_voltage_distribution": (float(np.mean(set_voltages)), float(np.std(set_voltages))),
            "on_resistance_cv": float(np.std(on_resistances) / (np.mean(on_resistances) + 1e-12)),
            "set_voltages": set_voltages.tolist(),
            "on_resistances": on_resistances.tolist(),
        }

    def generate_cycle_with_variability(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        V = np.linspace(-2.0, 2.0, 500)
        set_threshold = 1.0 + self._rng.normal(0.0, 0.1)
        reset_threshold = -1.0 + self._rng.normal(0.0, 0.1)
        on_resistance_var = 1e3 * (1.0 + self._rng.normal(0.0, 0.05))
        I = np.zeros_like(V)
        for i, v in enumerate(V):
            if v >= set_threshold and v > 0:
                I[i] = v / on_resistance_var
            elif v <= reset_threshold and v < 0:
                I[i] = v / (on_resistance_var * 100.0)
            else:
                I[i] = v / (on_resistance_var * 10.0)
        state = np.abs(I) / (np.max(np.abs(I)) + 1e-12)
        return V.astype(np.float32), I.astype(np.float32), state.astype(np.float32)

    @staticmethod
    def detect_set_voltage(V: np.ndarray, I: np.ndarray) -> float:
        dI_dV = np.gradient(I, V)
        positive_indices = np.where(V > 0)[0]
        if positive_indices.size == 0:
            return float(0.0)
        set_idx = positive_indices[np.argmax(dI_dV[positive_indices])]
        return float(V[set_idx])

    @staticmethod
    def calculate_on_resistance(V: np.ndarray, I: np.ndarray) -> float:
        on_region = (V > 1.0) & (V < 1.5)
        if np.any(on_region):
            return float(np.mean(V[on_region] / (I[on_region] + 1e-12)))
        return float(1e6)

    def benchmark_predict_current(
        self,
        voltage: np.ndarray,
        state: np.ndarray | None = None,
        concentration: np.ndarray | None = None,
        warmup: int = 20,
        repeats: int = 200,
        include_input_packing: bool = True,
    ) -> dict[str, float]:
        if warmup < 0 or repeats <= 0:
            raise ValueError("warmup must be >= 0 and repeats must be > 0.")
        voltage = np.asarray(voltage, dtype=np.float32)
        n_samples = int(voltage.size)
        if n_samples == 0:
            raise ValueError("voltage must contain at least one sample.")

        self.surrogate.model.eval()

        if include_input_packing:

            def run_once() -> None:
                with torch.no_grad():
                    inputs = self._build_inputs(voltage, state, concentration)
                    current_pred, _ = self.surrogate.forward(inputs)
                    _ = current_pred.detach().cpu().numpy()

        else:
            inputs = self._build_inputs(voltage, state, concentration)

            def run_once() -> None:
                with torch.no_grad():
                    current_pred, _ = self.surrogate.forward(inputs)
                    _ = current_pred.detach().cpu().numpy()

        for _ in range(int(warmup)):
            run_once()

        times_ns: list[int] = []
        for _ in range(int(repeats)):
            t0 = perf_counter_ns()
            run_once()
            t1 = perf_counter_ns()
            times_ns.append(t1 - t0)

        times_ms = np.asarray(times_ns, dtype=np.float64) / 1e6
        mean_ms = float(np.mean(times_ms))
        p95_ms = float(np.percentile(times_ms, 95))
        throughput_sps = float((n_samples / (mean_ms / 1e3)) if mean_ms > 0 else float("inf"))
        return {
            "n_samples": float(n_samples),
            "warmup": float(warmup),
            "repeats": float(repeats),
            "latency_mean_ms": mean_ms,
            "latency_p95_ms": p95_ms,
            "throughput_samples_per_s": throughput_sps,
        }

    @staticmethod
    def calculate_rrmse(I_pred: np.ndarray, I_meas: np.ndarray) -> float:
        I_pred = np.asarray(I_pred, dtype=np.float64)
        I_meas = np.asarray(I_meas, dtype=np.float64)
        rmse = float(np.sqrt(np.mean((I_pred - I_meas) ** 2)))
        rng = float(np.max(I_meas) - np.min(I_meas) + 1e-12)
        return float(rmse / rng)

    @staticmethod
    def calculate_mape(
        I_pred: np.ndarray,
        I_meas: np.ndarray,
        eps_abs: float = 1e-12,
        eps_fraction: float = 1e-3,
    ) -> float:
        I_pred = np.asarray(I_pred, dtype=np.float64)
        I_meas = np.asarray(I_meas, dtype=np.float64)
        eps_floor = max(float(eps_abs), float(eps_fraction) * float(np.max(np.abs(I_meas)) + 0.0))
        denom = np.maximum(np.abs(I_meas), eps_floor)
        return float(np.mean(np.abs((I_pred - I_meas) / denom)))
