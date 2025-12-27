from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class VoltageMasks:
    low_max_abs_v: float = 0.1
    mid_min_abs_v: float = 0.1
    mid_max_abs_v: float = 0.5


class PrintedMemristorSurrogateTorch:
    """
    PyTorch implementation of the physics-regularized neural surrogate used in this repository.

    The network maps (V, x [, c]) -> (I_pred, dx/dt_pred). Physics enters via
    regularization terms (masked Ohmic + masked SCLC + ODE consistency).
    """

    def __init__(
        self,
        hidden_layers: int = 3,
        neurons_per_layer: int = 64,
        input_features: Tuple[str, ...] = ("voltage", "state"),
        seed: int | None = None,
        trainable_params: Tuple[str, ...] | None = ("ohmic_conductance",),
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        masks: VoltageMasks = VoltageMasks(),
    ) -> None:
        if "voltage" not in input_features or "state" not in input_features:
            raise ValueError("input_features must include both 'voltage' and 'state'.")
        self.input_features = tuple(input_features)
        self.input_dim = len(self.input_features)
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self.masks = masks

        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        self._base_physical_params = self._initialize_physical_parameters()
        self.trainable_param_keys: Tuple[str, ...] = tuple(trainable_params or ())
        self._trainable_params = self._build_trainable_params()
        self.model = self._build_model(hidden_layers, neurons_per_layer).to(self.device, dtype=self.dtype)

    def _build_model(self, hidden_layers: int, neurons_per_layer: int) -> nn.Module:
        layers: list[nn.Module] = []
        in_dim = self.input_dim
        for _ in range(int(hidden_layers)):
            layers.append(nn.Linear(in_dim, int(neurons_per_layer)))
            layers.append(nn.Tanh())
            in_dim = int(neurons_per_layer)
        layers.append(nn.Linear(in_dim, 2))  # [I_pred, xdot_pred]
        return nn.Sequential(*layers)

    def _initialize_physical_parameters(self) -> Dict[str, float]:
        return {
            "epsilon_r": 3.5,
            "mu": 1e-10,
            "d": 100e-9,
            "area": 1e-8,
            "richardson": 1.2e6,
            "temperature": 300.0,
            "barrier_height": 0.8,
            "alpha": 0.1,
            "beta": 0.05,
            "ohmic_conductance": 1e-6,
        }

    def _build_trainable_params(self) -> Dict[str, nn.Parameter]:
        trainable: Dict[str, nn.Parameter] = {}
        for key in self.trainable_param_keys:
            if key not in self._base_physical_params:
                raise KeyError(f"Unknown physical parameter '{key}' for trainable configuration.")
            initial = float(self._base_physical_params[key])
            param = nn.Parameter(torch.tensor(np.log(initial), dtype=self.dtype, device=self.device))
            trainable[key] = param
        return trainable

    @property
    def physical_params(self) -> Dict[str, float]:
        return dict(self._base_physical_params)

    def get_trainable_param_values(self) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for key, param in self._trainable_params.items():
            values[key] = float(torch.exp(param).detach().cpu().item())
        return values

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.model.parameters()
        yield from self._trainable_params.values()

    def _merge_params(self, overrides: Dict[str, float] | None) -> Dict[str, torch.Tensor]:
        combined: Dict[str, float] = dict(self._base_physical_params)
        if overrides:
            combined.update(overrides)
        # overwrite trainable keys
        for key, param in self._trainable_params.items():
            combined[key] = float(torch.exp(param).detach().cpu().item())
        return {k: torch.tensor(v, dtype=self.dtype, device=self.device) for k, v in combined.items()}

    def ohmic_conduction(self, V: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return params["ohmic_conductance"] * V

    def sclc_conduction(self, V: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        epsilon_0 = torch.tensor(8.854187817e-12, dtype=self.dtype, device=self.device)
        epsilon = params["epsilon_r"] * epsilon_0
        mu = params["mu"]
        thickness = params["d"]
        area = params["area"]
        current_density = (9.0 / 8.0) * epsilon * mu * (V**2) / (thickness**3)
        return current_density * area

    def state_variable_ode(self, V: torch.Tensor, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        return params["alpha"] * V - params["beta"] * x

    @staticmethod
    def _masked_mse(target: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.any().item() is False:
            return torch.zeros((), dtype=target.dtype, device=target.device)
        diff = target[mask] - reference[mask]
        return torch.mean(diff**2)

    def physics_loss_components(
        self,
        V: torch.Tensor,
        I_pred: torch.Tensor,
        x: torch.Tensor,
        x_deriv_pred: torch.Tensor,
        params_override: Dict[str, float] | None = None,
    ) -> Dict[str, torch.Tensor]:
        params = self._merge_params(params_override)
        V = V.reshape(-1).to(self.device, dtype=self.dtype)
        I_pred = I_pred.reshape(-1).to(self.device, dtype=self.dtype)
        x = x.reshape(-1).to(self.device, dtype=self.dtype)
        x_deriv_pred = x_deriv_pred.reshape(-1).to(self.device, dtype=self.dtype)

        abs_v = torch.abs(V)
        mask_low = abs_v < self.masks.low_max_abs_v
        mask_mid = (abs_v >= self.masks.mid_min_abs_v) & (abs_v < self.masks.mid_max_abs_v)

        ohmic_expected = self.ohmic_conduction(V, params)
        sclc_expected = self.sclc_conduction(V, params)
        ode_target = self.state_variable_ode(V, x, params)

        ohmic = self._masked_mse(I_pred, ohmic_expected, mask_low)
        sclc = self._masked_mse(I_pred, sclc_expected, mask_mid)
        ode = torch.mean((x_deriv_pred - ode_target) ** 2)
        total = ohmic + sclc + ode
        return {"ohmic": ohmic, "sclc": sclc, "ode": ode, "total": total}

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(inputs.to(self.device, dtype=self.dtype))
        current = outputs[:, 0:1]
        xdot = outputs[:, 1:2]
        return current, xdot

