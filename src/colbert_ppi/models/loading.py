"""Construct a neural model, restore learned components, or transfer its protein branch."""

import torch


def build_model(task, saprot_dir, ernie_checkpoint=None, ernie_code=None):
    if task == "ppi":
        from .ppi import ProteinPairModel

        return ProteinPairModel(saprot_dir)
    from .pri import ProteinNucleicColBERT

    if not ernie_checkpoint or not ernie_code:
        raise ValueError("PRI requires the ERNIE-RNA checkpoint and code directory")
    return ProteinNucleicColBERT(
        saprot_dir,
        "",
        512,
        16,
        3,
        0.1,
        nucleic_model_type="ernie_rna",
        ernie_rna_checkpoint=ernie_checkpoint,
        ernie_rna_code_dir=ernie_code,
        rna_structure_weight=0.0,
        rna_relation_adapter_rank=0,
        saprot_lora_rank=8,
        saprot_lora_alpha=8,
        saprot_lora_dropout=0.1,
        nucleic_lora_rank=0,
    )


def component_state(model):
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    return {
        name: t.detach().cpu().clone()
        for name, t in model.state_dict().items()
        if name in names
    }


def restore_components(model, path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    missing, extra = model.load_state_dict(saved["model"], strict=False)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    if extra or trainable.intersection(missing):
        raise ValueError("Checkpoint does not cover the trainable model parameters")
    return saved


def initialise_protein_branch(model, path):
    """Map the PPI contextual encoder and SaProt LoRA into the PRI protein branch."""
    state = torch.load(path, map_location="cpu", weights_only=True)["model"]
    target = model.state_dict()
    mapped = {}
    for name, value in state.items():
        if name.startswith("encoder."):
            key = "protein_encoder." + name[len("encoder.") :]
        elif name.startswith("residue_attn."):
            key = "protein_attention." + name[len("residue_attn.") :]
        elif name.startswith("saprot_backbone.esm.") and "lora_" in name:
            key = name.replace("saprot_backbone.esm.", "saprot.esm.", 1)
        else:
            continue
        if key not in target or target[key].shape != value.shape:
            raise ValueError("Incompatible transfer component: " + key)
        mapped[key] = value
    if not any(k.startswith("protein_encoder.") for k in mapped) or not any(
        "lora_" in k for k in mapped
    ):
        raise ValueError("No complete protein encoder/LoRA transfer found")
    model.load_state_dict(mapped, strict=False)
    return len(mapped)
