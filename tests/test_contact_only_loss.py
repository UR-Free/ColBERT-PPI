from __future__ import annotations

import torch

from colbert_ppi.trainer import compute_batch_losses
from colbert_ppi.model import ColBERTPPIModel


class TinyContactModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(4, 4, bias=False)

    def forward(self, repr1, repr2, mask1, mask2):
        h1 = torch.nn.functional.normalize(self.projection(repr1), dim=-1)
        h2 = torch.nn.functional.normalize(self.projection(repr2), dim=-1)
        return h1, h2, h1.new_tensor(1.0)


def test_model_has_only_query_and_candidate_projection_heads() -> None:
    model = ColBERTPPIModel(input_dim=4, hidden_dim=4, dropout=0.0)
    module_names = dict(model.named_modules())
    assert "query_projector" in module_names
    assert "candidate_projector" in module_names
    assert "residue_attn" not in module_names
    assert not any("transformer" in name.lower() for name in module_names)

    mask = torch.ones(2, 3, dtype=torch.bool)
    outputs = model(torch.randn(2, 3, 4), torch.randn(2, 3, 4), mask, mask)
    assert len(outputs) == 3


def test_contact_loss_is_the_only_optimization_output() -> None:
    model = TinyContactModel()
    labels = torch.tensor(
        [
            [[1, -1], [-1, 1]],
            [[1, -1], [-1, 1]],
        ],
        dtype=torch.int8,
    )
    batch = {
        "repr1": torch.randn(2, 2, 4),
        "repr2": torch.randn(2, 2, 4),
        "mask1": torch.ones(2, 2, dtype=torch.bool),
        "mask2": torch.ones(2, 2, dtype=torch.bool),
        "contact_matrix": labels,
        "contact_mask": torch.ones_like(labels, dtype=torch.bool),
    }

    losses = compute_batch_losses(model, batch, device=torch.device("cpu"))

    assert set(losses) == {"loss", "contact_loss"}
    assert torch.equal(losses["loss"].detach(), losses["contact_loss"])
    losses["loss"].backward()
    assert model.projection.weight.grad is not None


def test_contact_labels_are_required() -> None:
    model = TinyContactModel()
    batch = {
        "repr1": torch.randn(1, 2, 4),
        "repr2": torch.randn(1, 2, 4),
        "mask1": torch.ones(1, 2, dtype=torch.bool),
        "mask2": torch.ones(1, 2, dtype=torch.bool),
    }

    try:
        compute_batch_losses(model, batch, device=torch.device("cpu"))
    except ValueError as error:
        assert "requires contact_matrix and contact_mask" in str(error)
    else:
        raise AssertionError("A batch without contact supervision was accepted")
