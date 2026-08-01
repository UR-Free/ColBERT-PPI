"""
Default ColBERT-PPI training configuration.

SaProt model: westlake-repl/SaProt_650M_AF2 on HuggingFace.
Feature dim: 1280 (ESM-2 650M embedding dim).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    # --- SaProt backbone ---
    saprot_model_name: str = "westlake-repl/SaProt_650M_AF2"
    saprot_input_dim: int = 1280  # ESM-2 650M embedding dim
    saprot_dir: str = "SaProt/weights/PLMs/SaProt_650M_AF2"  # local checkpoint dir

    # --- Query/candidate residue projection heads ---
    hidden_dim: int = 512
    dropout: float = 0.1

    # --- Scoring ---
    ppi_temperature: float = 0.07

    # --- Ablations ---
    # Replace each amino-acid/3Di token (for example ``Ap``) with the
    # sequence-only token for the same residue (``A#``).
    sequence_only: bool = False
    # --- Training ---
    epochs: int = 100
    batch_size: int = 12
    lr: float = 1e-4
    weight_decay: float = 1e-6
    warmup_epochs: int = 0  # No warmup — use pure cosine annealing
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    save_epoch_components: bool = False

    # --- Contact contrastive loss ---
    pos_per_sample: int = 5   # positive residue pairs per complex (contact < 8A)
    neg_per_sample: int = 5   # negative residue pairs per complex (CB > 12A)
    contact_threshold: float = 8.0   # contact positive threshold
    noncontact_threshold: float = 24.0  # contact negative threshold

    # --- Evaluation ---
    eval_freq: int = 1  # evaluate every N epochs
    sync_eval: bool = False
    eval_selection_metric: str = "val_auprc"
    val_label_protocol: str = ""
    test_label_protocol: str = ""
    # Historical runs monitored test every epoch. New leakage-controlled runs
    # disable this and score test once after validation-only checkpoint choice.
    eval_test_each_epoch: bool = True
    final_test_after_training: bool = False

    # --- Data ---
    max_seq_len: Optional[int] = None   # eval (None = no truncation)
    train_max_seq_len: int = 256          # training truncation
    num_workers: int = 4
    train_subset_ratio: float = 1.0

    # --- Paths ---
    data_dir: str = "data"
    output_dir: str = "outputs"

    # --- Data mode ---
    # "ddi"    : train on DDI only (v3 per-pair, PAE-filtered contacts)
    # "pinder" : train on PINDER train split
    # "mix"    : train on DDI + PINDER train split at mix_ratio:1 (DDI:PINDER),
    #            each epoch fully traverses PINDER once.
    # "pinder_structure_mix": mix PINDER PDB-complex and AFDB-monomer views.
    data_mode: str = "pinder"

    # --- DDI v3 paths (per-pair, PAE-filtered) ---
    # PINDER split paths (pre-extracted)
    # DDI paths
    train_csv: str = "data/processed_ddi_v2/pair_mapping_v3.csv"
    train_cb: str = "data/processed_ddi_v2/ddi_v3_cb.npz"
    train_saprot_inputs: str = "data/processed_ddi_v2/ddi_v3_saprot_inputs.pt"
    # DDI v3 PAE-filtered contact indices (for loss weighting)
    ddi_v3_contact_pae: str = "data/processed_ddi_v2/ddi_v3_contact_pae_lt1.pt"
    # PINDER eval splits — heterodimer only (hetero suffix)
    val_csv: str = "data/processed_all_v2/pinder_val_hetero.csv"
    val_cb: str = "data/processed_all_v2/pinder_val_hetero_cb.npz"
    val_saprot_inputs: str = "data/processed_all_v2/pinder_val_hetero_saprot_inputs.pt"
    test_csv: str = "data/processed_all_v2/pinder_test_hetero.csv"
    test_cb: str = "data/processed_all_v2/pinder_test_hetero_cb.npz"
    test_saprot_inputs: str = "data/processed_all_v2/pinder_test_hetero_saprot_inputs.pt"
    # PINDER train split (used when data_mode == "pinder" or "mix")
    pinder_train_csv: str = "data/processed_all_v2/pinder_train.csv"
    pinder_train_cb: str = "data/processed_all_v2/pinder_train_cb.npz"
    pinder_train_saprot_inputs: str = "data/processed_all_v2/pinder_train_saprot_inputs.pt"
    # PINDER train SaProt features (non-LoRA path, kept for backward compat)
    pinder_train_saprot: str = "data/processed_all_v2/pinder_train_saprot.pt"

    # --- Mix ratio (DDI : PINDER) ---
    mix_ratio: float = 2.0  # DDI samples per PINDER sample in each epoch
    pdb_afdb_mix_ratio: float = 1.0  # PDB samples per AFDB sample
    entity_unique_batches: bool = False

    # --- GPU ---
    gpu: str = "0,1,2,3,4,5"
    bf16: bool = True

    # --- LoRA fine-tuning (SaProt backbone) ---
    use_lora: bool = True       # Enable LoRA fine-tuning of SaProt backbone
    lora_r: int = 8              # LoRA rank
    lora_alpha: int = 8          # LoRA scaling factor
    lora_dropout: float = 0.1    # LoRA dropout
    gradient_checkpointing: bool = False

    # --- Compile ---
    compile_model: bool = False  # torch.compile (disabled by default — CUDA graph hangs on A100)

    # --- Misc ---
    seed: int = 42
    log_interval: int = 10


DEFAULT_CONFIG = TrainingConfig()
