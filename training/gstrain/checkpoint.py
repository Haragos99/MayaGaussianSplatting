from pathlib import Path

import torch
import torch.nn as nn

from .model_torch import PARAMETER_ATTRIBUTES, TorchGaussianModel


def save_checkpoint(
    path: str | Path,
    model: TorchGaussianModel,
    optimizer: torch.optim.Optimizer,
    iteration: int,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": int(iteration),
            "active_sh_degree": model.active_sh_degree,
            "max_sh_degree": model.max_sh_degree,
            "params": {
                name: getattr(model, attribute).detach().cpu()
                for name, attribute in PARAMETER_ATTRIBUTES.items()
            },
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    return path


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[TorchGaussianModel, dict]:
    """-> (model with the saved parameters, checkpoint dict for train(resume=...))."""
    state = torch.load(path, map_location=device, weights_only=False)
    params = state["params"]
    count = params["xyz"].shape[0]

    model = TorchGaussianModel(
        xyz=torch.zeros(count, 3, device=device),
        scale=torch.ones(count, 3, device=device),
        rotation=torch.zeros(count, 4, device=device),
        opacity=torch.full((count,), 0.5, device=device),
        color=torch.zeros(count, 3, device=device),
        sh_degree=int(state["max_sh_degree"]),
    )
    for name, attribute in PARAMETER_ATTRIBUTES.items():
        setattr(model, attribute, nn.Parameter(params[name].to(device)))
    model.active_sh_degree = int(state["active_sh_degree"])
    return model, state
