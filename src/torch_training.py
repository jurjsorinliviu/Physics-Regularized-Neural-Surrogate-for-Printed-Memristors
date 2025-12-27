from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

from torch_surrogate import PrintedMemristorSurrogateTorch


class SurrogateTrainerTorch:
    def __init__(
        self,
        surrogate: PrintedMemristorSurrogateTorch,
        learning_rate: float = 1e-3,
        seed: int | None = None,
        state_mixing: float = 0.2,
    ) -> None:
        self.surrogate = surrogate
        self.seed = seed
        self.state_mixing = float(np.clip(state_mixing, 0.0, 1.0))
        self.loss_history: list[dict[str, float]] = []
        self._rng = np.random.default_rng(seed)
        self.optimizer = torch.optim.Adam(self.surrogate.trainable_parameters(), lr=float(learning_rate))

    def _compute_state_variable(
        self,
        current: np.ndarray,
        concentration: np.ndarray | None = None,
    ) -> np.ndarray:
        norm_current = np.abs(current) / (np.max(np.abs(current)) + 1e-12)
        if concentration is None:
            return np.clip(norm_current, 0.0, 1.0)
        conc = (concentration - np.min(concentration)) / (np.max(concentration) - np.min(concentration) + 1e-12)
        mixed = (1.0 - self.state_mixing) * norm_current + self.state_mixing * conc
        return np.clip(mixed, 0.0, 1.0)

    def load_experimental_data(
        self,
        csv_path: str | Path,
        concentration_label: Optional[str] = None,
        device_id: Optional[int] = None,
        use_noisy_columns: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        df = pd.read_csv(csv_path)
        if concentration_label is not None:
            df = df[df["concentration_label"] == concentration_label]
        if device_id is not None:
            df = df[df["device_id"] == device_id]
        if df.empty:
            raise ValueError("No data matches the provided filters.")

        voltage_col = "voltage_noisy" if use_noisy_columns and "voltage_noisy" in df else "voltage"
        current_col = "current_noisy" if use_noisy_columns and "current_noisy" in df else "current"

        voltage = df[voltage_col].to_numpy(dtype=np.float32)
        current = df[current_col].to_numpy(dtype=np.float32)
        concentration = df.get("pmma_concentration", pd.Series(np.zeros_like(current))).to_numpy(dtype=np.float32)
        state = self._compute_state_variable(current, concentration)
        return voltage, current, state.astype(np.float32), concentration.astype(np.float32)

    def add_synthetic_noise(self, voltage: np.ndarray, current: np.ndarray, noise_std: float) -> tuple[np.ndarray, np.ndarray]:
        v_noise = self._rng.normal(0.0, noise_std * (np.std(voltage) + 1e-12), size=voltage.shape)
        i_noise = self._rng.normal(0.0, noise_std * (np.std(current) + 1e-12), size=current.shape)
        return voltage + v_noise, current + i_noise

    def parameter_variability(self, params: dict[str, float], variation_bound: float) -> dict[str, float]:
        varied: dict[str, float] = {}
        for key, value in params.items():
            if key in getattr(self.surrogate, "trainable_param_keys", ()):
                continue
            delta = self._rng.uniform(-variation_bound, variation_bound)
            varied[key] = float(value) * (1.0 + float(delta))
        return varied

    def _build_inputs(self, voltage: np.ndarray, state: np.ndarray, concentration: np.ndarray | None) -> torch.Tensor:
        voltage = np.asarray(voltage, dtype=np.float32)
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

    def train(
        self,
        epochs: int = 500,
        voltage: Optional[np.ndarray] = None,
        current: Optional[np.ndarray] = None,
        state: Optional[np.ndarray] = None,
        concentration: Optional[np.ndarray] = None,
        noise_std: float = 0.02,
        variability_bound: float = 0.1,
        verbose_every: int = 50,
        max_physics_weight: float = 0.4,
        lambda_data: float = 1.0,
    ) -> list[dict[str, float]]:
        if voltage is None or current is None or state is None:
            raise ValueError("This trainer expects explicit arrays (voltage/current/state).")

        voltage = np.asarray(voltage, dtype=np.float32)
        current = np.asarray(current, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        concentration = np.asarray(concentration, dtype=np.float32) if concentration is not None else None

        self.loss_history = []
        min_physics_weight = 0.0
        max_physics_weight = max(min_physics_weight, float(max_physics_weight))

        self.surrogate.model.train()
        training_start = time.time()

        for epoch in range(int(epochs)):
            progress = epoch / max(int(epochs) - 1, 1)
            lambda_physics = min_physics_weight + (max_physics_weight - min_physics_weight) * float(
                np.clip(progress, 0.0, 1.0)
            )

            V_noisy, I_noisy = self.add_synthetic_noise(voltage, current, noise_std)
            varied_params = self.parameter_variability(self.surrogate.physical_params, variability_bound)

            inputs = self._build_inputs(V_noisy, state, concentration)
            I_target = torch.tensor(I_noisy.reshape(-1, 1), dtype=self.surrogate.dtype, device=self.surrogate.device)
            x_tensor = torch.tensor(state.reshape(-1, 1), dtype=self.surrogate.dtype, device=self.surrogate.device)
            V_tensor = torch.tensor(V_noisy.reshape(-1, 1), dtype=self.surrogate.dtype, device=self.surrogate.device).reshape(-1)

            self.optimizer.zero_grad(set_to_none=True)
            I_pred, xdot_pred = self.surrogate.forward(inputs)
            data_loss = torch.mean((I_pred - I_target) ** 2)
            components = self.surrogate.physics_loss_components(
                V_tensor,
                I_pred.reshape(-1),
                x_tensor.reshape(-1),
                xdot_pred.reshape(-1),
                params_override=varied_params,
            )
            physics_loss = components["total"]
            total_loss = float(lambda_data) * data_loss + float(lambda_physics) * physics_loss
            total_loss.backward()
            self.optimizer.step()

            entry: dict[str, float] = {
                "epoch": float(epoch),
                "total_loss": float(total_loss.detach().cpu().item()),
                "data_loss": float(data_loss.detach().cpu().item()),
                "physics_loss": float(physics_loss.detach().cpu().item()),
                "lambda_physics": float(lambda_physics),
            }
            for key, value in self.surrogate.get_trainable_param_values().items():
                entry[f"param_{key}"] = float(value)
            self.loss_history.append(entry)

            if verbose_every and epoch % int(verbose_every) == 0:
                print(
                    f"Epoch {epoch:04d} | Total {entry['total_loss']:.3e} | Data {entry['data_loss']:.3e} "
                    f"| Physics {entry['physics_loss']:.3e} | w_phys {lambda_physics:.2f}"
                )

        training_end = time.time()
        duration_s = training_end - training_start
        if self.loss_history:
            self.loss_history[-1]["training_time_seconds"] = float(duration_s)
            self.loss_history[-1]["training_time_minutes"] = float(duration_s / 60.0)
        return self.loss_history

