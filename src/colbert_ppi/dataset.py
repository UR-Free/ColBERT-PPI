"""
SaProtContactDataset — loads pre-extracted SaProt features + CB coords + contact maps.

Data format (from pre_extract.py):
  - saprot_features.pt : dict {protein_id: (L, 1280) float16 tensor}
  - cb_coords.npz       : NPZ with arrays 'labels', 'offsets', 'coords'
  - pairs.csv           : "protein_a:protein_b" rows (no header)

Contact maps are built on-the-fly from CB coordinates during collation
(positive: CB < 8Å, negative: CB > 12Å).

The dataset mirrors v7's PPIContactDataset with SaProt replacing ESM.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler


def _normalize_label(label: str) -> str:
    """Strip whitespace and .pdb suffix."""
    label = label.strip()
    if label.startswith(">"):
        label = label[1:]
    if label.endswith(".pdb"):
        label = label[:-4]
    return label


def _label_aliases(label: str) -> List[str]:
    """Generate possible label aliases for fuzzy matching."""
    label = _normalize_label(label)
    aliases = [label]
    if "_" in label:
        aliases.append(label.split("_", 1)[1])
    if "__" in label:
        aliases.append(label.split("__", 1)[1])
    return list(dict.fromkeys(a for a in aliases if a))


_PINDER_UNIPROT_RE = re.compile(r"_([A-Za-z0-9]+(?:-\d+)?)-[RL]$")


def _extract_uniprot_accession(label: str) -> Optional[str]:
    """Extract the UniProt accession encoded in a PINDER-style label."""
    label = _normalize_label(label)
    match = _PINDER_UNIPROT_RE.search(label)
    if not match:
        return None
    accession = match.group(1)
    # AFDB files are keyed by canonical accession. PINDER labels may include
    # isoform suffixes, e.g. P12345-2-R.
    if "-" in accession:
        accession = accession.split("-", 1)[0]
    return accession


def _label_aliases_with_uniprot(label: str) -> List[str]:
    aliases = _label_aliases(label)
    accession = _extract_uniprot_accession(label)
    if accession:
        aliases.append(accession)
    return list(dict.fromkeys(a for a in aliases if a))


def _read_csv_pairs(csv_file: Path) -> List[Tuple[str, str]]:
    """Read protein pair list from CSV (first column: protein_a:protein_b)."""
    pairs = []
    with csv_file.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            pair_str = row[0].strip()
            if not pair_str or ":" not in pair_str:
                continue
            left, right = pair_str.split(":", 1)
            pairs.append((_normalize_label(left), _normalize_label(right)))
    return pairs


def _load_saprot_features(pt_file: Path) -> Dict[str, torch.Tensor]:
    """Load pre-extracted SaProt features from .pt file."""
    data = torch.load(pt_file, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"SaProt features must be a dict: {pt_file}")

    features = {}
    for raw_label, value in data.items():
        label = _normalize_label(str(raw_label))
        if isinstance(value, dict):
            tensor = value.get("value") or value.get("representation")
        else:
            tensor = value
        if tensor is None:
            raise ValueError(f"Missing tensor for '{label}' in {pt_file}")
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        features[label] = tensor.detach().cpu().float()
    return features


def _load_cb_npz(npz_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load CB coordinates from NPZ file. Returns (labels, offsets, coords)."""
    data = np.load(npz_file, allow_pickle=False)
    labels = np.asarray(
        [str(item) if isinstance(item, bytes) else str(item) for item in data["labels"]],
        dtype=np.str_,
    )
    offsets = np.asarray(data["offsets"], dtype=np.int64)
    coords = np.array(data["coords"], dtype=np.float32, copy=True)
    return labels, offsets, coords


def _build_contact_matrix(
    coords_a: torch.Tensor,
    coords_b: torch.Tensor,
    positive_threshold: float = 8.0,
    negative_threshold: float = 12.0,
    pae_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute contact label matrix (1/0/-1) and pair mask.

    Labels: 1 = contact (< pos_thresh Å), -1 = non-contact (> neg_thresh Å), 0 = neutral.

    If ``pae_mask`` is provided, a contact is only considered positive when
    both CB distance < positive_threshold AND pae_mask[i,j] == True.
    Contacts with CB < threshold but PAE mask False are demoted to neutral (0).
    """
    valid_a = torch.isfinite(coords_a).all(dim=-1)
    valid_b = torch.isfinite(coords_b).all(dim=-1)
    pair_mask = valid_a[:, None] & valid_b[None, :]

    if coords_a.numel() == 0 or coords_b.numel() == 0:
        shape = (coords_a.size(0), coords_b.size(0))
        return torch.zeros(shape, dtype=torch.int8), pair_mask

    dist2 = (coords_a[:, None, :] - coords_b[None, :, :]).pow(2).sum(dim=-1)
    labels = torch.zeros(dist2.shape, dtype=torch.int8)
    pos_mask = (dist2 < positive_threshold**2) & pair_mask
    neg_mask = (dist2 > negative_threshold**2) & pair_mask

    if pae_mask is not None:
        # PAE filter: only keep contacts where PAE ≤ threshold
        pae_mask = pae_mask.to(labels.device)
        labels[(pos_mask & pae_mask)] = 1
        labels[(pos_mask & ~pae_mask)] = 0  # demote to neutral
    else:
        labels[pos_mask] = 1
    labels[neg_mask] = -1
    return labels, pair_mask


class SaProtContactDataset(Dataset):
    """Load paired proteins from CSV + pre-extracted SaProt features + CB coords.

    Parameters
    ----------
    csv_file : str or Path
        CSV with first column "protein_a:protein_b".
    saprot_pt : str or Path
        Pre-extracted SaProt features .pt file (dict: id → (L, D) tensor).
    cb_npz : str or Path
        CB coordinates .npz file (arrays: labels, offsets, coords).
    max_seq_len : int, optional
        Truncation length.
    build_contacts : bool
        If True, build contact matrices during collation (for training).
    """

    def __init__(
        self,
        csv_file: str | Path,
        saprot_pt: str | Path | None = None,
        cb_npz: str | Path | None = None,
        processed_dir: str | Path | None = None,
        processed_prefix: str | None = None,
        max_seq_len: Optional[int] = None,
        build_contacts: bool = True,
    ):
        self.csv_file = Path(csv_file)
        base_dir = Path(processed_dir) if processed_dir else self.csv_file.parent
        prefix = processed_prefix if processed_prefix else self.csv_file.stem

        self.saprot_pt = (
            Path(saprot_pt) if saprot_pt else base_dir / f"{prefix}_saprot.pt"
        )
        self.cb_npz = (
            Path(cb_npz) if cb_npz else base_dir / f"{prefix}_cb.npz"
        )
        self.max_seq_len = max_seq_len
        self.build_contacts = build_contacts

        # Read pairs
        self.samples = _read_csv_pairs(self.csv_file)
        if not self.samples:
            raise ValueError(f"No valid pairs found in {self.csv_file}")

        # Load SaProt features
        if not self.saprot_pt.exists():
            raise FileNotFoundError(
                f"SaProt features not found: {self.saprot_pt}. "
                "Run pre_extract.py first."
            )

        requested_labels = {a for a, b in self.samples} | {b for a, b in self.samples}

        self._features = {}
        self._feature_aliases: Dict[str, str] = {}
        for label, feat in _load_saprot_features(self.saprot_pt).items():
            if not self._matches_requested(label, requested_labels):
                continue
            self._features[label] = feat
            for alias in _label_aliases(label):
                self._feature_aliases.setdefault(alias, label)

        if not self._features:
            raise ValueError(f"No requested features found in {self.saprot_pt}")
        self.feat_dim = int(next(iter(self._features.values())).shape[-1])

        # Load CB coordinates
        if not self.cb_npz.exists():
            raise FileNotFoundError(f"CB coords not found: {self.cb_npz}")

        labels_np, offsets, coords = _load_cb_npz(self.cb_npz)
        self._coords: Dict[str, np.ndarray] = {}
        self._coord_aliases: Dict[str, str] = {}
        for i, label in enumerate(labels_np.tolist()):
            label = _normalize_label(label)
            if not self._matches_requested(label, requested_labels):
                continue
            start = int(offsets[i])
            end = int(offsets[i + 1])
            self._coords[label] = np.array(coords[start:end], dtype=np.float32)
            for alias in _label_aliases(label):
                self._coord_aliases.setdefault(alias, label)

        # Filter valid samples
        self._resolve_cache: Dict[str, str] = {}
        missing = []
        valid_samples = []
        for left, right in self.samples:
            try:
                key_a = self._resolve(left)
                key_b = self._resolve(right)

                # Length sanity check
                len_a = self._features[key_a].shape[0]
                len_b = self._features[key_b].shape[0]
                coord_len_a = self._coords[key_a].shape[0]
                coord_len_b = self._coords[key_b].shape[0]

                if coord_len_a > len_a:
                    self._coords[key_a] = self._coords[key_a][:len_a]
                if coord_len_b > len_b:
                    self._coords[key_b] = self._coords[key_b][:len_b]

                if self._features[key_a].shape[0] != self._coords[key_a].shape[0]:
                    raise ValueError(
                        f"Feature/CB length mismatch for '{key_a}'"
                    )
                if self._features[key_b].shape[0] != self._coords[key_b].shape[0]:
                    raise ValueError(
                        f"Feature/CB length mismatch for '{key_b}'"
                    )
                valid_samples.append((left, right))
            except KeyError:
                missing.append(f"{left}:{right}")

        if missing:
            import logging
            _ds_logger = logging.getLogger(__name__)
            _ds_logger.warning(
                "Skipped %d unresolvable pairs, kept %d.",
                len(missing), len(valid_samples),
            )

        self.samples = valid_samples
        if not self.samples:
            raise ValueError("All pairs were skipped — check data consistency.")

    @staticmethod
    def _matches_requested(label: str, requested: set[str]) -> bool:
        if label in requested:
            return True
        return any(a in requested for a in _label_aliases(label))

    def _resolve(self, label: str) -> str:
        """Resolve a label to the canonical key in feature/coord dicts."""
        normalized = _normalize_label(label)
        if normalized in self._resolve_cache:
            return self._resolve_cache[normalized]

        for alias in _label_aliases(normalized):
            feat_key = self._feature_aliases.get(alias)
            coord_key = self._coord_aliases.get(alias)
            if feat_key and coord_key and feat_key == coord_key:
                self._resolve_cache[normalized] = feat_key
                return feat_key
        raise KeyError(f"Cannot resolve label '{label}'")

    def _prepare(self, idx: int) -> dict:
        left, right = self.samples[idx]
        key_a = self._resolve(left)
        key_b = self._resolve(right)

        repr_a = self._features[key_a]
        repr_b = self._features[key_b]
        coord_a = self._coords[key_a]
        coord_b = self._coords[key_b]

        if self.max_seq_len:
            repr_a = repr_a[: self.max_seq_len]
            repr_b = repr_b[: self.max_seq_len]
            coord_a = coord_a[: self.max_seq_len]
            coord_b = coord_b[: self.max_seq_len]

        return {
            "pair_id": f"{key_a}:{key_b}",
            "label_a": key_a,
            "label_b": key_b,
            "repr1": repr_a,
            "repr2": repr_b,
            "cb_coords1": torch.tensor(
                np.nan_to_num(np.asarray(coord_a), nan=0.0, posinf=0.0, neginf=0.0),
                dtype=torch.float32,
            ),
            "cb_coords2": torch.tensor(
                np.nan_to_num(np.asarray(coord_b), nan=0.0, posinf=0.0, neginf=0.0),
                dtype=torch.float32,
            ),
            "len1": int(repr_a.shape[0]),
            "len2": int(repr_b.shape[0]),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self._prepare(idx)

    # ------------------------------------------------------------------
    # Collation
    # ------------------------------------------------------------------
    @staticmethod
    def _pad_side(
        items: Sequence[torch.Tensor], padding_value: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([t.size(0) for t in items], dtype=torch.long)
        padded = pad_sequence(items, batch_first=True, padding_value=padding_value)
        max_len = padded.size(1) if padded.ndim >= 2 else 0
        mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
        return padded, mask

    @staticmethod
    def _pad_coords(items: Sequence[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(items, batch_first=True, padding_value=0.0)

    @staticmethod
    def _pad_contacts(
        contacts: Sequence[torch.Tensor],
        pair_masks: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_a = max(c.size(0) for c in contacts)
        max_b = max(c.size(1) for c in contacts)
        contact_batch = torch.zeros((len(contacts), max_a, max_b), dtype=torch.int8)
        mask_batch = torch.zeros((len(contacts), max_a, max_b), dtype=torch.bool)
        for i, (c, pm) in enumerate(zip(contacts, pair_masks)):
            a, b = c.shape
            contact_batch[i, :a, :b] = c
            mask_batch[i, :a, :b] = pm
        return contact_batch, mask_batch

    def collate_fn(self, batch: Sequence[dict]) -> dict:
        batch = [b for b in batch if b is not None]
        if not batch:
            raise ValueError("Empty batch")

        repr1_list = [b["repr1"] for b in batch]
        repr2_list = [b["repr2"] for b in batch]
        cb1_list = [b["cb_coords1"] for b in batch]
        cb2_list = [b["cb_coords2"] for b in batch]

        repr1, mask1 = self._pad_side(repr1_list, padding_value=0.0)
        repr2, mask2 = self._pad_side(repr2_list, padding_value=0.0)
        cb1 = self._pad_coords(cb1_list)
        cb2 = self._pad_coords(cb2_list)

        result = {
            "pair_ids": [b["pair_id"] for b in batch],
            "label_a": [b["label_a"] for b in batch],
            "label_b": [b["label_b"] for b in batch],
            "repr1": repr1,
            "repr2": repr2,
            "mask1": mask1,
            "mask2": mask2,
            "cb_coords1": cb1,
            "cb_coords2": cb2,
            "len1": torch.tensor([b["len1"] for b in batch], dtype=torch.long),
            "len2": torch.tensor([b["len2"] for b in batch], dtype=torch.long),
        }

        if self.build_contacts:
            contact_list = []
            pair_mask_list = []
            for b_item in batch:
                c, pm = _build_contact_matrix(
                    b_item["cb_coords1"], b_item["cb_coords2"],
                )
                contact_list.append(c)
                pair_mask_list.append(pm)
            contact_matrix, contact_mask = self._pad_contacts(contact_list, pair_mask_list)
            result["contact_matrix"] = contact_matrix
            result["contact_mask"] = contact_mask

        return result


# ===========================================================================
# LoRA Dataset — loads tokenized inputs instead of pre-extracted features
# ===========================================================================

def _load_saprot_inputs(pt_file: Path) -> Dict[str, Dict[str, torch.Tensor]]:
    """Load pre-tokenized SaProt inputs from .pt file.

    Returns dict: canonical_id → {"input_ids": (1, L), "attention_mask": (1, L)}
    """
    data = torch.load(pt_file, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"Tokenized inputs must be a dict: {pt_file}")
    inputs = {}
    for raw_label, value in data.items():
        label = _normalize_label(str(raw_label))
        if isinstance(value, dict) and "input_ids" in value and "attention_mask" in value:
            inputs[label] = {
                "input_ids": value["input_ids"].long(),
                "attention_mask": value["attention_mask"].bool(),
            }
    return inputs


def _load_explicit_contact_maps(pt_file: Path) -> Dict[str, dict]:
    data = torch.load(pt_file, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "contacts" in data and isinstance(data["contacts"], dict):
        data = data["contacts"]
    if not isinstance(data, dict):
        raise ValueError(f"Explicit contact maps must be a dict: {pt_file}")
    return {str(k): v for k, v in data.items()}


def _contact_entry_size(entry: dict, prefix: str) -> int:
    for key in (f"{prefix}_i", f"{prefix}_idx1", f"{prefix}_res_idx1"):
        if key in entry:
            value = entry[key]
            return int(value.numel() if torch.is_tensor(value) else len(value))
    return 0


def _entry_indices(entry: dict, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    aliases_i = (f"{prefix}_i", f"{prefix}_idx1", f"{prefix}_res_idx1")
    aliases_j = (f"{prefix}_j", f"{prefix}_idx2", f"{prefix}_res_idx2")
    key_i = next((k for k in aliases_i if k in entry), None)
    key_j = next((k for k in aliases_j if k in entry), None)
    if key_i is None or key_j is None:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    idx_i = entry[key_i]
    idx_j = entry[key_j]
    if not torch.is_tensor(idx_i):
        idx_i = torch.as_tensor(idx_i)
    if not torch.is_tensor(idx_j):
        idx_j = torch.as_tensor(idx_j)
    return idx_i.long(), idx_j.long()


class SaProtLoRAContactDataset(Dataset):
    """Load paired proteins from CSV + pre-tokenized SaProt inputs + CB coords.

    Mirrors SaProtContactDataset but loads tokenized input_ids/attention_mask
    instead of pre-extracted features.  The SaProt backbone is run live during
    training (with optional LoRA adapters), producing per-residue embeddings.

    Parameters
    ----------
    csv_file : str or Path
    saprot_inputs_pt : str or Path
        Pre-tokenized SaProt inputs .pt file.
    cb_npz : str or Path
    max_seq_len : int, optional
    build_contacts : bool
    """

    def __init__(
        self,
        csv_file: str | Path,
        saprot_inputs_pt: str | Path | None = None,
        cb_npz: str | Path | None = None,
        processed_dir: str | Path | None = None,
        processed_prefix: str | None = None,
        max_seq_len: Optional[int] = None,
        build_contacts: bool = True,
    ):
        self.csv_file = Path(csv_file)
        base_dir = Path(processed_dir) if processed_dir else self.csv_file.parent
        prefix = processed_prefix if processed_prefix else self.csv_file.stem

        self.saprot_inputs_pt = (
            Path(saprot_inputs_pt) if saprot_inputs_pt
            else base_dir / f"{prefix}_saprot_inputs.pt"
        )
        self.cb_npz = (
            Path(cb_npz) if cb_npz else base_dir / f"{prefix}_cb.npz"
        )
        self.max_seq_len = max_seq_len
        self.build_contacts = build_contacts

        # Read pairs
        self.samples = _read_csv_pairs(self.csv_file)
        if not self.samples:
            raise ValueError(f"No valid pairs found in {self.csv_file}")

        # Load tokenized inputs
        if not self.saprot_inputs_pt.exists():
            raise FileNotFoundError(
                f"Tokenized SaProt inputs not found: {self.saprot_inputs_pt}. "
                "Prepare the tokenized inputs described in docs/DATA_FORMAT.md."
            )

        requested_labels = {a for a, b in self.samples} | {b for a, b in self.samples}

        self._inputs: Dict[str, Dict[str, torch.Tensor]] = {}
        self._input_aliases: Dict[str, str] = {}
        for label, inp in _load_saprot_inputs(self.saprot_inputs_pt).items():
            if not self._matches_requested(label, requested_labels):
                continue
            self._inputs[label] = inp
            for alias in _label_aliases(label):
                self._input_aliases.setdefault(alias, label)

        if not self._inputs:
            raise ValueError(f"No requested tokenized inputs found in {self.saprot_inputs_pt}")

        # SaProt 650M hidden dim (used for logging; features are produced live)
        self.feat_dim = 1280

        # Load CB coordinates (same as SaProtContactDataset)
        if not self.cb_npz.exists():
            raise FileNotFoundError(f"CB coords not found: {self.cb_npz}")

        labels_np, offsets, coords = _load_cb_npz(self.cb_npz)
        self._coords: Dict[str, np.ndarray] = {}
        self._coord_aliases: Dict[str, str] = {}
        for i, label in enumerate(labels_np.tolist()):
            label = _normalize_label(label)
            if not self._matches_requested(label, requested_labels):
                continue
            start = int(offsets[i])
            end = int(offsets[i + 1])
            self._coords[label] = np.array(coords[start:end], dtype=np.float32)
            for alias in _label_aliases(label):
                self._coord_aliases.setdefault(alias, label)

        # Filter valid samples
        self._resolve_cache: Dict[str, str] = {}
        missing = []
        valid_samples = []
        for left, right in self.samples:
            try:
                key_a = self._resolve(left)
                key_b = self._resolve(right)

                # Tokenized length includes <cls> + residues + <eos> → N+2
                tok_len_a = int(self._inputs[key_a]["input_ids"].shape[-1])
                tok_len_b = int(self._inputs[key_b]["input_ids"].shape[-1])
                res_len_a = max(tok_len_a - 2, 0)  # actual residues
                res_len_b = max(tok_len_b - 2, 0)

                coord_len_a = self._coords[key_a].shape[0]
                coord_len_b = self._coords[key_b].shape[0]

                # Align CB coords to residue count
                if coord_len_a > res_len_a:
                    self._coords[key_a] = self._coords[key_a][:res_len_a]
                if coord_len_b > res_len_b:
                    self._coords[key_b] = self._coords[key_b][:res_len_b]

                if res_len_a < 1 or res_len_b < 1:
                    missing.append(f"{left}:{right} (empty sequence)")
                    continue

                valid_samples.append((left, right))
            except KeyError:
                missing.append(f"{left}:{right}")

        if missing:
            import logging
            _ds_logger = logging.getLogger(__name__)
            _ds_logger.warning(
                "Skipped %d unresolvable pairs, kept %d.",
                len(missing), len(valid_samples),
            )

        self.samples = valid_samples
        if not self.samples:
            raise ValueError("All pairs were skipped — check data consistency.")

    @staticmethod
    def _matches_requested(label: str, requested: set[str]) -> bool:
        if label in requested:
            return True
        return any(a in requested for a in _label_aliases(label))

    def _resolve(self, label: str) -> str:
        normalized = _normalize_label(label)
        if normalized in self._resolve_cache:
            return self._resolve_cache[normalized]
        for alias in _label_aliases(normalized):
            inp_key = self._input_aliases.get(alias)
            coord_key = self._coord_aliases.get(alias)
            if inp_key and coord_key and inp_key == coord_key:
                self._resolve_cache[normalized] = inp_key
                return inp_key
        raise KeyError(f"Cannot resolve label '{label}'")

    def _prepare(self, idx: int) -> dict:
        left, right = self.samples[idx]
        key_a = self._resolve(left)
        key_b = self._resolve(right)

        inp_a = self._inputs[key_a]
        inp_b = self._inputs[key_b]
        coord_a = self._coords[key_a]
        coord_b = self._coords[key_b]

        # Tokenized length: <cls> + residues + <eos>
        tok_len_a = int(inp_a["input_ids"].shape[-1])
        tok_len_b = int(inp_b["input_ids"].shape[-1])

        # Truncate tokenized inputs to max_seq_len (in token space)
        if self.max_seq_len:
            inp_a = {
                "input_ids": inp_a["input_ids"][:, :self.max_seq_len],
                "attention_mask": inp_a["attention_mask"][:, :self.max_seq_len],
            }
            inp_b = {
                "input_ids": inp_b["input_ids"][:, :self.max_seq_len],
                "attention_mask": inp_b["attention_mask"][:, :self.max_seq_len],
            }
            tok_len_a = min(tok_len_a, self.max_seq_len)
            tok_len_b = min(tok_len_b, self.max_seq_len)

        # CB coords correspond to residues only (no <cls>/<eos>).
        # Truncate to residue count = token_len - 2.
        res_len_a = max(tok_len_a - 2, 0)
        res_len_b = max(tok_len_b - 2, 0)
        coord_a = coord_a[:res_len_a]
        coord_b = coord_b[:res_len_b]

        return {
            "pair_id": f"{key_a}:{key_b}",
            "label_a": key_a,
            "label_b": key_b,
            "input_ids1": inp_a["input_ids"].squeeze(0),      # (L_tok,)
            "attention_mask1": inp_a["attention_mask"].squeeze(0),  # (L_tok,)
            "input_ids2": inp_b["input_ids"].squeeze(0),
            "attention_mask2": inp_b["attention_mask"].squeeze(0),
            "cb_coords1": torch.tensor(
                np.nan_to_num(np.asarray(coord_a), nan=0.0, posinf=0.0, neginf=0.0),
                dtype=torch.float32,
            ),
            "cb_coords2": torch.tensor(
                np.nan_to_num(np.asarray(coord_b), nan=0.0, posinf=0.0, neginf=0.0),
                dtype=torch.float32,
            ),
            "len1": tok_len_a,
            "len2": tok_len_b,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self._prepare(idx)

    # ------------------------------------------------------------------
    # Collation
    # ------------------------------------------------------------------
    @staticmethod
    def _pad_sequence_1d(
        items: Sequence[torch.Tensor], padding_value: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([t.size(0) for t in items], dtype=torch.long)
        padded = pad_sequence(items, batch_first=True, padding_value=padding_value)
        max_len = padded.size(1)
        mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
        return padded, mask

    @staticmethod
    def _pad_coords(items: Sequence[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(items, batch_first=True, padding_value=0.0)

    @staticmethod
    def _pad_contacts(
        contacts: Sequence[torch.Tensor],
        pair_masks: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_a = max(c.size(0) for c in contacts)
        max_b = max(c.size(1) for c in contacts)
        contact_batch = torch.zeros((len(contacts), max_a, max_b), dtype=torch.int8)
        mask_batch = torch.zeros((len(contacts), max_a, max_b), dtype=torch.bool)
        for i, (c, pm) in enumerate(zip(contacts, pair_masks)):
            a, b = c.shape
            contact_batch[i, :a, :b] = c
            mask_batch[i, :a, :b] = pm
        return contact_batch, mask_batch

    def collate_fn(self, batch: Sequence[dict]) -> dict:
        batch = [b for b in batch if b is not None]
        if not batch:
            raise ValueError("Empty batch")

        input_ids1_list = [b["input_ids1"] for b in batch]
        input_ids2_list = [b["input_ids2"] for b in batch]
        cb1_list = [b["cb_coords1"] for b in batch]
        cb2_list = [b["cb_coords2"] for b in batch]

        input_ids1, mask1 = self._pad_sequence_1d(input_ids1_list, padding_value=1)  # pad_token_id=1
        input_ids2, mask2 = self._pad_sequence_1d(input_ids2_list, padding_value=1)
        cb1 = self._pad_coords(cb1_list)
        cb2 = self._pad_coords(cb2_list)

        result = {
            "pair_ids": [b["pair_id"] for b in batch],
            "label_a": [b["label_a"] for b in batch],
            "label_b": [b["label_b"] for b in batch],
            "input_ids1": input_ids1,
            "input_ids2": input_ids2,
            "mask1": mask1,
            "mask2": mask2,
            "cb_coords1": cb1,
            "cb_coords2": cb2,
            "len1": torch.tensor([b["len1"] for b in batch], dtype=torch.long),
            "len2": torch.tensor([b["len2"] for b in batch], dtype=torch.long),
        }

        if self.build_contacts:
            contact_list = []
            pair_mask_list = []
            for b_item in batch:
                c, pm = _build_contact_matrix(
                    b_item["cb_coords1"], b_item["cb_coords2"],
                )
                contact_list.append(c)
                pair_mask_list.append(pm)
            contact_matrix, contact_mask = self._pad_contacts(contact_list, pair_mask_list)
            result["contact_matrix"] = contact_matrix
            result["contact_mask"] = contact_mask

        return result


# ===========================================================================
# LoRA Dataset with explicit contact labels
# ===========================================================================

class SaProtLoRAExplicitContactDataset(Dataset):
    """Load tokenized SaProt inputs plus sparse, pre-mapped contact labels.

    This dataset is used for AFDB monomer replacement. The SaProt/CB inputs are
    keyed by UniProt accession (AFDB monomer), while the sparse contact labels
    are keyed by the original PINDER pair id after projecting complex-derived
    residue contacts onto AFDB residue indices.
    """

    def __init__(
        self,
        csv_file: str | Path,
        saprot_inputs_pt: str | Path,
        cb_npz: str | Path,
        contact_map_pt: str | Path,
        max_seq_len: Optional[int] = None,
        build_contacts: bool = True,
    ):
        self.csv_file = Path(csv_file)
        self.saprot_inputs_pt = Path(saprot_inputs_pt)
        self.cb_npz = Path(cb_npz)
        self.contact_map_pt = Path(contact_map_pt)
        self.max_seq_len = max_seq_len
        self.build_contacts = build_contacts

        self.samples = _read_csv_pairs(self.csv_file)
        if not self.samples:
            raise ValueError(f"No valid pairs found in {self.csv_file}")

        requested_labels = {a for a, b in self.samples} | {b for a, b in self.samples}
        requested_accessions = {
            acc for label in requested_labels
            for acc in [_extract_uniprot_accession(label)]
            if acc
        }

        if not self.saprot_inputs_pt.exists():
            raise FileNotFoundError(f"Tokenized SaProt inputs not found: {self.saprot_inputs_pt}")
        raw_inputs = _load_saprot_inputs(self.saprot_inputs_pt)
        self._inputs: Dict[str, Dict[str, torch.Tensor]] = {}
        self._input_aliases: Dict[str, str] = {}
        for label, inp in raw_inputs.items():
            if label not in requested_accessions and not self._matches_requested(label, requested_labels):
                continue
            self._inputs[label] = inp
            for alias in _label_aliases_with_uniprot(label):
                self._input_aliases.setdefault(alias, label)

        if not self._inputs:
            raise ValueError(f"No requested AFDB tokenized inputs found in {self.saprot_inputs_pt}")
        self.feat_dim = 1280

        if not self.cb_npz.exists():
            raise FileNotFoundError(f"CB coords not found: {self.cb_npz}")
        labels_np, offsets, coords = _load_cb_npz(self.cb_npz)
        self._coords: Dict[str, np.ndarray] = {}
        self._coord_aliases: Dict[str, str] = {}
        for i, label in enumerate(labels_np.tolist()):
            label = _normalize_label(label)
            if label not in requested_accessions and not self._matches_requested(label, requested_labels):
                continue
            start = int(offsets[i])
            end = int(offsets[i + 1])
            self._coords[label] = np.array(coords[start:end], dtype=np.float32)
            for alias in _label_aliases_with_uniprot(label):
                self._coord_aliases.setdefault(alias, label)

        if not self._coords:
            raise ValueError(f"No requested AFDB CB coordinates found in {self.cb_npz}")

        if not self.contact_map_pt.exists():
            raise FileNotFoundError(f"Explicit contact labels not found: {self.contact_map_pt}")
        self._contact_maps = _load_explicit_contact_maps(self.contact_map_pt)

        self._resolve_cache: Dict[str, str] = {}
        valid_samples = []
        missing = []
        missing_contacts = 0
        for left, right in self.samples:
            try:
                self._resolve(left)
                self._resolve(right)
                pair_id = f"{left}:{right}"
                entry = self._contact_maps.get(pair_id)
                if entry is None:
                    missing_contacts += 1
                    continue
                if self.build_contacts and (
                    _contact_entry_size(entry, "pos") < 1
                    or _contact_entry_size(entry, "neg") < 1
                ):
                    missing_contacts += 1
                    continue
                valid_samples.append((left, right))
            except KeyError:
                missing.append(f"{left}:{right}")

        if missing or missing_contacts:
            import logging
            _ds_logger = logging.getLogger(__name__)
            _ds_logger.warning(
                "Skipped %d unresolvable pairs and %d pairs without explicit contacts; kept %d.",
                len(missing), missing_contacts, len(valid_samples),
            )

        self.samples = valid_samples
        if not self.samples:
            raise ValueError("All pairs were skipped — check AFDB/contact-map consistency.")

    @staticmethod
    def _matches_requested(label: str, requested: set[str]) -> bool:
        if label in requested:
            return True
        aliases = _label_aliases_with_uniprot(label)
        return any(alias in requested for alias in aliases)

    def _resolve(self, label: str) -> str:
        normalized = _normalize_label(label)
        if normalized in self._resolve_cache:
            return self._resolve_cache[normalized]
        for alias in _label_aliases_with_uniprot(normalized):
            inp_key = self._input_aliases.get(alias)
            coord_key = self._coord_aliases.get(alias)
            if inp_key and coord_key and inp_key == coord_key:
                self._resolve_cache[normalized] = inp_key
                return inp_key
        raise KeyError(f"Cannot resolve label '{label}'")

    @staticmethod
    def _make_explicit_contact_matrix(entry: dict, len1: int, len2: int) -> tuple[torch.Tensor, torch.Tensor]:
        labels = torch.zeros((len1, len2), dtype=torch.int8)
        mask = torch.zeros((len1, len2), dtype=torch.bool)

        for prefix, value in (("pos", 1), ("neg", -1)):
            idx_i, idx_j = _entry_indices(entry, prefix)
            if idx_i.numel() == 0:
                continue
            valid = (idx_i >= 0) & (idx_i < len1) & (idx_j >= 0) & (idx_j < len2)
            if valid.any():
                ii = idx_i[valid]
                jj = idx_j[valid]
                labels[ii, jj] = value
                mask[ii, jj] = True

        return labels, mask

    def _prepare(self, idx: int) -> dict:
        left, right = self.samples[idx]
        key_a = self._resolve(left)
        key_b = self._resolve(right)

        inp_a = self._inputs[key_a]
        inp_b = self._inputs[key_b]
        coord_a = self._coords[key_a]
        coord_b = self._coords[key_b]

        tok_a = {
            "input_ids": inp_a["input_ids"],
            "attention_mask": inp_a["attention_mask"],
        }
        tok_b = {
            "input_ids": inp_b["input_ids"],
            "attention_mask": inp_b["attention_mask"],
        }
        tok_len_a = int(tok_a["input_ids"].shape[-1])
        tok_len_b = int(tok_b["input_ids"].shape[-1])

        if self.max_seq_len:
            tok_a = {
                "input_ids": tok_a["input_ids"][:, :self.max_seq_len],
                "attention_mask": tok_a["attention_mask"][:, :self.max_seq_len],
            }
            tok_b = {
                "input_ids": tok_b["input_ids"][:, :self.max_seq_len],
                "attention_mask": tok_b["attention_mask"][:, :self.max_seq_len],
            }
            tok_len_a = min(tok_len_a, self.max_seq_len)
            tok_len_b = min(tok_len_b, self.max_seq_len)

        res_len_a = min(max(tok_len_a - 2, 0), int(coord_a.shape[0]))
        res_len_b = min(max(tok_len_b - 2, 0), int(coord_b.shape[0]))
        tok_len_a = min(tok_len_a, res_len_a + 2)
        tok_len_b = min(tok_len_b, res_len_b + 2)
        tok_a = {
            "input_ids": tok_a["input_ids"][:, :tok_len_a],
            "attention_mask": tok_a["attention_mask"][:, :tok_len_a],
        }
        tok_b = {
            "input_ids": tok_b["input_ids"][:, :tok_len_b],
            "attention_mask": tok_b["attention_mask"][:, :tok_len_b],
        }

        pair_id = f"{left}:{right}"
        entry = self._contact_maps[pair_id]
        contact_matrix, contact_mask = self._make_explicit_contact_matrix(entry, res_len_a, res_len_b)

        return {
            "pair_id": pair_id,
            "label_a": left,
            "label_b": right,
            "afdb_label_a": key_a,
            "afdb_label_b": key_b,
            "input_ids1": tok_a["input_ids"].squeeze(0),
            "attention_mask1": tok_a["attention_mask"].squeeze(0),
            "input_ids2": tok_b["input_ids"].squeeze(0),
            "attention_mask2": tok_b["attention_mask"].squeeze(0),
            "cb_coords1": torch.tensor(
                np.nan_to_num(np.asarray(coord_a[:res_len_a]), nan=0.0, posinf=0.0, neginf=0.0),
                dtype=torch.float32,
            ),
            "cb_coords2": torch.tensor(
                np.nan_to_num(np.asarray(coord_b[:res_len_b]), nan=0.0, posinf=0.0, neginf=0.0),
                dtype=torch.float32,
            ),
            "contact_matrix": contact_matrix,
            "contact_mask": contact_mask,
            "len1": tok_len_a,
            "len2": tok_len_b,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self._prepare(idx)

    @staticmethod
    def _pad_sequence_1d(
        items: Sequence[torch.Tensor], padding_value: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([t.size(0) for t in items], dtype=torch.long)
        padded = pad_sequence(items, batch_first=True, padding_value=padding_value)
        max_len = padded.size(1)
        mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
        return padded, mask

    @staticmethod
    def _pad_coords(items: Sequence[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(items, batch_first=True, padding_value=0.0)

    @staticmethod
    def _pad_contacts(
        contacts: Sequence[torch.Tensor],
        pair_masks: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_a = max(c.size(0) for c in contacts)
        max_b = max(c.size(1) for c in contacts)
        contact_batch = torch.zeros((len(contacts), max_a, max_b), dtype=torch.int8)
        mask_batch = torch.zeros((len(contacts), max_a, max_b), dtype=torch.bool)
        for i, (c, pm) in enumerate(zip(contacts, pair_masks)):
            a, b = c.shape
            contact_batch[i, :a, :b] = c
            mask_batch[i, :a, :b] = pm
        return contact_batch, mask_batch

    def collate_fn(self, batch: Sequence[dict]) -> dict:
        batch = [b for b in batch if b is not None]
        if not batch:
            raise ValueError("Empty batch")

        input_ids1, mask1 = self._pad_sequence_1d(
            [b["input_ids1"] for b in batch], padding_value=1,
        )
        input_ids2, mask2 = self._pad_sequence_1d(
            [b["input_ids2"] for b in batch], padding_value=1,
        )
        cb1 = self._pad_coords([b["cb_coords1"] for b in batch])
        cb2 = self._pad_coords([b["cb_coords2"] for b in batch])

        result = {
            "pair_ids": [b["pair_id"] for b in batch],
            "label_a": [b["label_a"] for b in batch],
            "label_b": [b["label_b"] for b in batch],
            "afdb_label_a": [b["afdb_label_a"] for b in batch],
            "afdb_label_b": [b["afdb_label_b"] for b in batch],
            "input_ids1": input_ids1,
            "input_ids2": input_ids2,
            "mask1": mask1,
            "mask2": mask2,
            "cb_coords1": cb1,
            "cb_coords2": cb2,
            "len1": torch.tensor([b["len1"] for b in batch], dtype=torch.long),
            "len2": torch.tensor([b["len2"] for b in batch], dtype=torch.long),
        }

        if self.build_contacts:
            contact_matrix, contact_mask = self._pad_contacts(
                [b["contact_matrix"] for b in batch],
                [b["contact_mask"] for b in batch],
            )
            result["contact_matrix"] = contact_matrix
            result["contact_mask"] = contact_mask

        return result


# ===========================================================================
# Mixed dataset — interleaves two sub-datasets at a fixed ratio.
# Each epoch fully traverses the *secondary* dataset once.
# ===========================================================================


def _collate_with_ddi_flag(dataset_a_collate, batch):
    """Wrap sub-dataset collate_fn: tag each sample with is_ddi boolean."""
    is_ddi_flags = [b.pop("is_ddi", False) for b in batch if b is not None]
    result = dataset_a_collate(batch)
    result["is_ddi"] = torch.tensor(is_ddi_flags, dtype=torch.bool)
    return result


class MixedContactDataset(Dataset):
    """Interleave two SaProtContactDataset / SaProtLoRAContactDataset instances.

    Samples are drawn in repeating blocks: *ratio* samples from `dataset_a`
    followed by 1 sample from `dataset_b`.  This guarantees that each epoch
    traverses all of `dataset_b` exactly once.

    .. note::
       When ``num_workers > 0``, each DataLoader worker forks a copy of the
       dataset with its own ``_a_indices`` buffer.  Within a single epoch,
       different workers may therefore serve the same DDI sample.  This is
       acceptable for contrastive training and keeps implementation simple.

    Parameters
    ----------
    dataset_a : Dataset
        Primary dataset (e.g. DDI).  Sampled `ratio` times per block.
    dataset_b : Dataset
        Secondary dataset (e.g. PINDER train).  Sampled once per block;
        its length defines the epoch length.
    ratio : float
        Number of `dataset_a` samples per `dataset_b` sample  (default 2.0).
    shuffle_a : bool
        If True, shuffle `dataset_a` indices whenever they are exhausted
        within an epoch.
    seed : int
        RNG seed for reproducible shuffles.
    """

    def __init__(
        self,
        dataset_a: Dataset,
        dataset_b: Dataset,
        ratio: float = 2.0,
        shuffle_a: bool = True,
        seed: int = 42,
    ):
        if ratio <= 0:
            raise ValueError(f"ratio must be positive, got {ratio}")

        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.ratio = float(ratio)
        self.shuffle_a = shuffle_a
        self._rng = np.random.RandomState(seed)

        self._len_a = len(dataset_a)
        self._len_b = len(dataset_b)

        if self._len_a == 0 or self._len_b == 0:
            raise ValueError(
                f"Both sub-datasets must be non-empty "
                f"(got len_a={self._len_a}, len_b={self._len_b})"
            )

        # Per-epoch DDI index buffer (refilled on exhaustion)
        self._a_indices: np.ndarray = self._fresh_a_indices()
        self._a_ptr: int = 0

        # Ensure sub-datasets are compatible for collation — use dataset_a's
        # collate_fn as the canonical one, wrapped to propagate is_ddi flags.
        if not hasattr(dataset_a, "collate_fn"):
            raise TypeError("dataset_a must expose a collate_fn method")
        from functools import partial
        self.collate_fn = partial(_collate_with_ddi_flag, dataset_a.collate_fn)

        # Expose feat_dim for compatibility with downstream code
        self.feat_dim = getattr(dataset_a, "feat_dim", None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fresh_a_indices(self) -> np.ndarray:
        """Return a freshly shuffled copy of [0 .. len_a-1]."""
        indices = np.arange(self._len_a, dtype=np.int64)
        if self.shuffle_a:
            self._rng.shuffle(indices)
        return indices

    def _next_a_idx(self) -> int:
        """Pop the next dataset_a index, refilling + reshuffling on exhaustion."""
        if self._a_ptr >= len(self._a_indices):
            self._a_indices = self._fresh_a_indices()
            self._a_ptr = 0
        idx = int(self._a_indices[self._a_ptr])
        self._a_ptr += 1
        return idx

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Total samples per epoch = (ratio+1) * len(dataset_b)."""
        a_per_block = int(self.ratio)
        return (a_per_block + 1) * self._len_b

    def __getitem__(self, idx: int) -> dict:
        a_per_block = int(self.ratio)
        block_size = a_per_block + 1  # e.g. 3 when ratio=2
        block_idx = idx // block_size
        pos_in_block = idx % block_size

        if pos_in_block < a_per_block:
            # dataset_a slot (DDI)
            a_idx = self._next_a_idx()
            sample = self.dataset_a[a_idx]
            sample["is_ddi"] = True
            return sample
        else:
            # dataset_b slot (PINDER) — linear scan through all of b
            sample = self.dataset_b[block_idx]
            sample["is_ddi"] = False
            return sample

    # ------------------------------------------------------------------
    # Epoch lifecycle
    # ------------------------------------------------------------------
    def on_epoch_begin(self) -> None:
        """Reset the dataset_a index buffer (called at start of each epoch)."""
        self._a_indices = self._fresh_a_indices()
        self._a_ptr = 0


# ===========================================================================
# PINDER PDB + AFDB structure-view mixing with entity-safe global batches.
# ===========================================================================


class MixedStructureContactDataset(Dataset):
    """Combine PDB-complex and AFDB-monomer views without losing contacts.

    PDB samples derive contact labels from complex coordinates. AFDB samples
    carry contacts projected from the original PINDER complex. The collator
    preserves the explicit AFDB labels instead of recomputing physically
    meaningless inter-chain distances between independently predicted monomers.
    """

    def __init__(self, pdb_dataset: Dataset, afdb_dataset: Dataset):
        if not hasattr(pdb_dataset, "samples") or not hasattr(afdb_dataset, "samples"):
            raise TypeError("Both structure datasets must expose a samples list")
        self.pdb_dataset = pdb_dataset
        self.afdb_dataset = afdb_dataset
        self._pdb_len = len(pdb_dataset)
        self._afdb_len = len(afdb_dataset)
        self.samples = list(pdb_dataset.samples) + list(afdb_dataset.samples)
        self.source_ids = ["pdb"] * self._pdb_len + ["afdb"] * self._afdb_len
        self.entity_pairs = [
            (
                _extract_uniprot_accession(left) or f"RECORD::{_normalize_label(left)}",
                _extract_uniprot_accession(right) or f"RECORD::{_normalize_label(right)}",
            )
            for left, right in self.samples
        ]
        self.feat_dim = getattr(pdb_dataset, "feat_dim", 1280)

    def __len__(self) -> int:
        return self._pdb_len + self._afdb_len

    def __getitem__(self, idx: int) -> dict:
        if idx < self._pdb_len:
            sample = dict(self.pdb_dataset[idx])
            source = "pdb"
        else:
            sample = dict(self.afdb_dataset[idx - self._pdb_len])
            source = "afdb"
        sample["structure_source"] = source
        return sample

    @staticmethod
    def _pad_sequence_1d(
        items: Sequence[torch.Tensor], padding_value: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([item.size(0) for item in items], dtype=torch.long)
        padded = pad_sequence(items, batch_first=True, padding_value=padding_value)
        mask = torch.arange(padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
        return padded, mask

    @staticmethod
    def _pad_coords(items: Sequence[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(items, batch_first=True, padding_value=0.0)

    @staticmethod
    def _pad_contacts(
        contacts: Sequence[torch.Tensor],
        pair_masks: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_a = max(contact.size(0) for contact in contacts)
        max_b = max(contact.size(1) for contact in contacts)
        contact_batch = torch.zeros((len(contacts), max_a, max_b), dtype=torch.int8)
        mask_batch = torch.zeros((len(contacts), max_a, max_b), dtype=torch.bool)
        for index, (contact, pair_mask) in enumerate(zip(contacts, pair_masks)):
            len_a, len_b = contact.shape
            contact_batch[index, :len_a, :len_b] = contact
            mask_batch[index, :len_a, :len_b] = pair_mask
        return contact_batch, mask_batch

    def collate_fn(self, batch: Sequence[dict]) -> dict:
        batch = [sample for sample in batch if sample is not None]
        if not batch:
            raise ValueError("Empty mixed-structure batch")

        input_ids1, mask1 = self._pad_sequence_1d(
            [sample["input_ids1"] for sample in batch], padding_value=1,
        )
        input_ids2, mask2 = self._pad_sequence_1d(
            [sample["input_ids2"] for sample in batch], padding_value=1,
        )
        result = {
            "pair_ids": [sample["pair_id"] for sample in batch],
            "label_a": [sample["label_a"] for sample in batch],
            "label_b": [sample["label_b"] for sample in batch],
            "structure_source": [sample["structure_source"] for sample in batch],
            "input_ids1": input_ids1,
            "input_ids2": input_ids2,
            "mask1": mask1,
            "mask2": mask2,
            "cb_coords1": self._pad_coords([sample["cb_coords1"] for sample in batch]),
            "cb_coords2": self._pad_coords([sample["cb_coords2"] for sample in batch]),
            "len1": torch.tensor([sample["len1"] for sample in batch], dtype=torch.long),
            "len2": torch.tensor([sample["len2"] for sample in batch], dtype=torch.long),
        }

        contacts: list[torch.Tensor] = []
        pair_masks: list[torch.Tensor] = []
        for sample in batch:
            if "contact_matrix" in sample and "contact_mask" in sample:
                contact = sample["contact_matrix"]
                pair_mask = sample["contact_mask"]
            else:
                contact, pair_mask = _build_contact_matrix(
                    sample["cb_coords1"], sample["cb_coords2"]
                )
            contacts.append(contact)
            pair_masks.append(pair_mask)
        result["contact_matrix"], result["contact_mask"] = self._pad_contacts(
            contacts, pair_masks
        )
        return result


class EntityUniqueDistributedBatchSampler(Sampler[list[int]]):
    """Build globally entity-unique batches and shard each one across ranks.

    An entity may occur only once in the union of all per-rank mini-batches.
    A homodimer is valid because the two endpoints are reduced to one entity
    before checking conflicts. The same entity still cannot occur in another
    pair in that global batch.
    """

    def __init__(
        self,
        entity_pairs: Sequence[Tuple[str, str]],
        *,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
        drop_last: bool = True,
        source_ids: Optional[Sequence[str]] = None,
        pdb_to_afdb_ratio: float = 1.0,
    ):
        if batch_size < 1 or num_replicas < 1:
            raise ValueError("batch_size and num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} outside [0, {num_replicas})")
        if not drop_last:
            raise ValueError("EntityUniqueDistributedBatchSampler requires drop_last=True")
        if source_ids is not None and len(source_ids) != len(entity_pairs):
            raise ValueError("source_ids and entity_pairs must have the same length")
        if pdb_to_afdb_ratio <= 0:
            raise ValueError("pdb_to_afdb_ratio must be positive")
        self.entity_pairs = list(entity_pairs)
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.source_ids = list(source_ids) if source_ids is not None else None
        self.pdb_to_afdb_ratio = float(pdb_to_afdb_ratio)
        self.epoch = 0
        self._cached_epoch: Optional[int] = None
        self._cached_global_batches: list[list[int]] = []

    @property
    def global_batch_size(self) -> int:
        return self.batch_size * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._cached_epoch = None
        self._cached_global_batches = []

    def _epoch_candidates(self, rng: np.random.RandomState) -> list[int]:
        if self.source_ids is None or not {"pdb", "afdb"}.issubset(set(self.source_ids)):
            candidates = list(range(len(self.entity_pairs)))
            rng.shuffle(candidates)
            return candidates

        pdb_indices = np.asarray(
            [i for i, source in enumerate(self.source_ids) if source == "pdb"],
            dtype=np.int64,
        )
        afdb_indices = np.asarray(
            [i for i, source in enumerate(self.source_ids) if source == "afdb"],
            dtype=np.int64,
        )
        rng.shuffle(pdb_indices)
        rng.shuffle(afdb_indices)
        target_pdb = max(1, int(round(len(afdb_indices) * self.pdb_to_afdb_ratio)))
        if target_pdb <= len(pdb_indices):
            selected_pdb = pdb_indices[:target_pdb]
        else:
            selected_pdb = np.resize(pdb_indices, target_pdb)
            rng.shuffle(selected_pdb)
        candidates = np.concatenate([selected_pdb, afdb_indices]).tolist()
        rng.shuffle(candidates)
        return [int(index) for index in candidates]

    def _build_global_batches(self) -> list[list[int]]:
        if self._cached_epoch == self.epoch:
            return self._cached_global_batches
        rng = np.random.RandomState(self.seed + self.epoch)
        remaining = self._epoch_candidates(rng)
        batches: list[list[int]] = []
        global_size = self.global_batch_size

        while len(remaining) >= global_size:
            accepted: list[int] = []
            deferred: list[int] = []
            used_entities: set[str] = set()
            stop_at = len(remaining)
            for position, index in enumerate(remaining):
                entities = set(self.entity_pairs[index])
                if used_entities.isdisjoint(entities):
                    accepted.append(index)
                    used_entities.update(entities)
                    if len(accepted) == global_size:
                        stop_at = position + 1
                        break
                else:
                    deferred.append(index)
            if len(accepted) != global_size:
                break
            deferred.extend(remaining[stop_at:])
            batches.append(accepted)
            remaining = deferred

        self._cached_epoch = self.epoch
        self._cached_global_batches = batches
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        start = self.rank * self.batch_size
        end = start + self.batch_size
        for global_batch in self._build_global_batches():
            yield global_batch[start:end]

    def __len__(self) -> int:
        return len(self._build_global_batches())

    def audit(self) -> dict[str, int]:
        batches = self._build_global_batches()
        violations = 0
        homodimers = 0
        pdb_samples = 0
        afdb_samples = 0
        for batch in batches:
            used: set[str] = set()
            for index in batch:
                left, right = self.entity_pairs[index]
                if left == right:
                    homodimers += 1
                entities = {left, right}
                if not used.isdisjoint(entities):
                    violations += 1
                used.update(entities)
                if self.source_ids is not None:
                    pdb_samples += int(self.source_ids[index] == "pdb")
                    afdb_samples += int(self.source_ids[index] == "afdb")
        return {
            "global_batches": len(batches),
            "global_batch_size": self.global_batch_size,
            "samples": len(batches) * self.global_batch_size,
            "pdb_samples": pdb_samples,
            "afdb_samples": afdb_samples,
            "homodimer_pairs": homodimers,
            "entity_reuse_violations": violations,
        }


# ===========================================================================
# V3 DDI Dataset — per-pair storage with PAE-filtered contacts
# ===========================================================================

class SaProtLoRAContactDatasetV3(Dataset):
    """DDI v3 dataset: per-pair trimmed coords + SaProt tokens + PAE contact filter.

    Key differences from v2 (SaProtLoRAContactDataset):
      - Per-pair storage (not per-domain) — no label alias resolution needed.
      - Coords pre-trimmed to domain boundaries (no gap regions).
      - PAE-filtered contact labels: positive = CB < 8Å AND PAE ≤ 1.

    Batch output format is identical to SaProtLoRAContactDataset for
    seamless use with MixedContactDataset and the standard training pipeline.

    Parameters
    ----------
    pair_csv : str or Path
        pair_mapping_v3.csv (PAIRID column for pair identifiers).
    saprot_inputs_pt : str or Path
        ddi_v3_saprot_inputs.pt — dict[pair_id → {dom1, dom2}].
    cb_npz : str or Path
        ddi_v3_cb.npz — flat arrays (pair_ids, offsets1/2, len1/2, coords1/2).
    contact_pae_pt : str or Path, optional
        ddi_v3_contact_pae_lt1.pt — sparse PAE≤1 contact indices.
        If provided, positive contacts require PAE≤1 in addition to CB<8Å.
    max_seq_len : int, optional
    build_contacts : bool
    """

    def __init__(
        self,
        pair_csv: str | Path,
        saprot_inputs_pt: str | Path,
        cb_npz: str | Path,
        contact_pae_pt: str | Path | None = None,
        max_seq_len: Optional[int] = None,
        build_contacts: bool = True,
    ):
        import logging
        self._log = logging.getLogger(__name__)

        self.pair_csv = Path(pair_csv)
        self.saprot_inputs_pt = Path(saprot_inputs_pt)
        self.cb_npz = Path(cb_npz)
        self.contact_pae_pt = Path(contact_pae_pt) if contact_pae_pt else None
        self.max_seq_len = max_seq_len
        self.build_contacts = build_contacts

        # --- Read pair list ---
        pair_ids = []
        with self.pair_csv.open("r", newline="") as fh:
            reader = csv.reader(fh)
            next(reader, None)  # skip header
            for row in reader:
                if not row:
                    continue
                pid = row[0].strip()
                if pid:
                    pair_ids.append(pid)
        self.pair_ids = pair_ids
        if not self.pair_ids:
            raise ValueError(f"No pairs found in {self.pair_csv}")

        # --- Load CB coordinates (flat arrays, kept as numpy on CPU) ---
        if not self.cb_npz.exists():
            raise FileNotFoundError(f"CB file not found: {self.cb_npz}")
        cb_data = np.load(self.cb_npz, allow_pickle=True)
        self._cb_pair_ids = np.asarray(cb_data["pair_ids"], dtype=np.str_)
        self._cb_offsets1 = np.asarray(cb_data["offsets1"], dtype=np.int64)
        self._cb_offsets2 = np.asarray(cb_data["offsets2"], dtype=np.int64)
        self._cb_len1 = np.asarray(cb_data["len1"], dtype=np.int32)
        self._cb_len2 = np.asarray(cb_data["len2"], dtype=np.int32)
        self._cb_coords1 = np.asarray(cb_data["coords1"], dtype=np.float32)
        self._cb_coords2 = np.asarray(cb_data["coords2"], dtype=np.float32)
        # Build pair_id → index mapping
        self._cb_idx_map = {str(pid): i for i, pid in enumerate(self._cb_pair_ids)}

        # --- Load SaProt tokenized inputs ---
        if not self.saprot_inputs_pt.exists():
            raise FileNotFoundError(f"SaProt inputs not found: {self.saprot_inputs_pt}")
        self._inputs = torch.load(self.saprot_inputs_pt, map_location="cpu", weights_only=False)
        # self._inputs: dict[pair_id → {"dom1": {input_ids, attention_mask}, "dom2": {...}}]

        # --- Load PAE contact indices (optional) ---
        self._pae_data = None
        if self.contact_pae_pt and self.contact_pae_pt.exists():
            self._pae_data = torch.load(self.contact_pae_pt, map_location="cpu", weights_only=False)
            # dict[pair_id → {"res_idx1": tensor[int32], "res_idx2": tensor[int32], "pae": tensor[float16]}]

        # --- Filter to pairs present in all data sources ---
        valid = []
        missing_cb = 0
        missing_inputs = 0
        for pid in self.pair_ids:
            if pid not in self._cb_idx_map:
                missing_cb += 1
                continue
            if pid not in self._inputs:
                missing_inputs += 1
                continue
            valid.append(pid)

        if missing_cb or missing_inputs:
            self._log.warning(
                "Skipped %d pairs missing CB, %d missing inputs; kept %d.",
                missing_cb, missing_inputs, len(valid),
            )

        self.pair_ids = valid
        if not self.pair_ids:
            raise ValueError("All pairs filtered — check data consistency.")

        # SaProt 650M hidden dim (for logging; features produced live by backbone)
        self.feat_dim = 1280

        # --- Precompute PAE dense masks for all pairs ---
        self._pae_masks: dict[str, torch.Tensor] = {}
        if self._pae_data is not None:
            import logging
            _log = logging.getLogger(__name__)
            _log.info("Precomputing PAE dense masks for %d pairs...", len(self.pair_ids))
            for pid in self.pair_ids:
                cb_i = self._cb_idx_map[pid]
                cb_l1 = int(self._cb_len1[cb_i])
                cb_l2 = int(self._cb_len2[cb_i])
                inp = self._inputs[pid]
                tok_l1 = int(inp["dom1"]["input_ids"].shape[-1])
                tok_l2 = int(inp["dom2"]["input_ids"].shape[-1])
                if self.max_seq_len:
                    tok_l1 = min(tok_l1, self.max_seq_len)
                    tok_l2 = min(tok_l2, self.max_seq_len)
                res1 = min(cb_l1, max(tok_l1 - 2, 0))
                res2 = min(cb_l2, max(tok_l2 - 2, 0))
                pae_mask = torch.zeros((res1, res2), dtype=torch.bool)
                if pid in self._pae_data:
                    entry = self._pae_data[pid]
                    r1 = entry["res_idx1"]
                    r2 = entry["res_idx2"]
                    if not isinstance(r1, torch.Tensor):
                        r1 = torch.as_tensor(r1)
                    if not isinstance(r2, torch.Tensor):
                        r2 = torch.as_tensor(r2)
                    r1, r2 = r1.long(), r2.long()
                    valid = (r1 < res1) & (r2 < res2)
                    if valid.any():
                        pae_mask[r1[valid], r2[valid]] = True
                self._pae_masks[pid] = pae_mask
            _log.info("  Done: %d PAE masks precomputed", len(self._pae_masks))

    def __len__(self) -> int:
        return len(self.pair_ids)

    def __getitem__(self, idx: int) -> dict:
        pair_id = self.pair_ids[idx]

        # --- CB coords ---
        cb_i = self._cb_idx_map[pair_id]
        o1 = int(self._cb_offsets1[cb_i])
        l1 = int(self._cb_len1[cb_i])
        o2 = int(self._cb_offsets2[cb_i])
        l2 = int(self._cb_len2[cb_i])
        coord1 = np.asarray(self._cb_coords1[o1:o1 + l1], dtype=np.float32)
        coord2 = np.asarray(self._cb_coords2[o2:o2 + l2], dtype=np.float32)

        # --- SaProt tokens ---
        inp = self._inputs[pair_id]
        tok1 = inp["dom1"]
        tok2 = inp["dom2"]
        tok_len1 = int(tok1["input_ids"].shape[-1])
        tok_len2 = int(tok2["input_ids"].shape[-1])

        # --- Truncate tokens ---
        if self.max_seq_len:
            tok1 = {
                "input_ids": tok1["input_ids"][:, :self.max_seq_len],
                "attention_mask": tok1["attention_mask"][:, :self.max_seq_len],
            }
            tok2 = {
                "input_ids": tok2["input_ids"][:, :self.max_seq_len],
                "attention_mask": tok2["attention_mask"][:, :self.max_seq_len],
            }
            tok_len1 = min(tok_len1, self.max_seq_len)
            tok_len2 = min(tok_len2, self.max_seq_len)

        # Truncate coords to residue count.
        # In the ideal case tok_len == cb_len + 2 (BOS + residues + EOS),
        # but ~6% of domains have small mismatches (tok==cb or tok==cb±1).
        # Use the minimum to guarantee we never read past coord bounds.
        max_res1 = max(tok_len1 - 2, 0)
        max_res2 = max(tok_len2 - 2, 0)
        res_len1 = min(l1, max_res1)
        res_len2 = min(l2, max_res2)
        coord1 = coord1[:res_len1]
        coord2 = coord2[:res_len2]

        item = {
            "pair_id": pair_id,
            "label_a": pair_id.split(":")[0] if ":" in pair_id else pair_id,
            "label_b": pair_id.split(":")[1] if ":" in pair_id else pair_id,
            "input_ids1": tok1["input_ids"].squeeze(0),
            "attention_mask1": tok1["attention_mask"].squeeze(0),
            "input_ids2": tok2["input_ids"].squeeze(0),
            "attention_mask2": tok2["attention_mask"].squeeze(0),
            "cb_coords1": torch.tensor(
                np.nan_to_num(coord1, nan=0.0, posinf=0.0, neginf=0.0),
                dtype=torch.float32,
            ),
            "cb_coords2": torch.tensor(
                np.nan_to_num(coord2, nan=0.0, posinf=0.0, neginf=0.0),
                dtype=torch.float32,
            ),
            "len1": tok_len1,
            "len2": tok_len2,
        }

        # --- PAE dense mask (precomputed in __init__) ---
        item["pae_mask"] = self._pae_masks.get(pair_id)

        return item

    # ------------------------------------------------------------------
    # Collation — identical output format to SaProtLoRAContactDataset
    # ------------------------------------------------------------------
    @staticmethod
    def _pad_sequence_1d(
        items: Sequence[torch.Tensor], padding_value: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([t.size(0) for t in items], dtype=torch.long)
        padded = pad_sequence(items, batch_first=True, padding_value=padding_value)
        max_len = padded.size(1)
        mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
        return padded, mask

    @staticmethod
    def _pad_coords(items: Sequence[torch.Tensor]) -> torch.Tensor:
        return pad_sequence(items, batch_first=True, padding_value=0.0)

    @staticmethod
    def _pad_contacts(
        contacts: Sequence[torch.Tensor],
        pair_masks: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_a = max(c.size(0) for c in contacts)
        max_b = max(c.size(1) for c in contacts)
        contact_batch = torch.zeros((len(contacts), max_a, max_b), dtype=torch.int8)
        mask_batch = torch.zeros((len(contacts), max_a, max_b), dtype=torch.bool)
        for i, (c, pm) in enumerate(zip(contacts, pair_masks)):
            a, b = c.shape
            contact_batch[i, :a, :b] = c
            mask_batch[i, :a, :b] = pm
        return contact_batch, mask_batch

    def collate_fn(self, batch: Sequence[dict]) -> dict:
        batch = [b for b in batch if b is not None]
        if not batch:
            raise ValueError("Empty batch")

        input_ids1_list = [b["input_ids1"] for b in batch]
        input_ids2_list = [b["input_ids2"] for b in batch]
        cb1_list = [b["cb_coords1"] for b in batch]
        cb2_list = [b["cb_coords2"] for b in batch]

        input_ids1, mask1 = self._pad_sequence_1d(input_ids1_list, padding_value=1)
        input_ids2, mask2 = self._pad_sequence_1d(input_ids2_list, padding_value=1)
        cb1 = self._pad_coords(cb1_list)
        cb2 = self._pad_coords(cb2_list)

        result = {
            "pair_ids": [b["pair_id"] for b in batch],
            "label_a": [b["label_a"] for b in batch],
            "label_b": [b["label_b"] for b in batch],
            "input_ids1": input_ids1,
            "input_ids2": input_ids2,
            "mask1": mask1,
            "mask2": mask2,
            "cb_coords1": cb1,
            "cb_coords2": cb2,
            "len1": torch.tensor([b["len1"] for b in batch], dtype=torch.long),
            "len2": torch.tensor([b["len2"] for b in batch], dtype=torch.long),
        }

        if self.build_contacts:
            contact_list = []
            pair_mask_list = []
            for b_item in batch:
                c, pm = _build_contact_matrix(
                    b_item["cb_coords1"], b_item["cb_coords2"],
                    pae_mask=b_item.get("pae_mask"),
                )
                contact_list.append(c)
                pair_mask_list.append(pm)
            contact_matrix, contact_mask = self._pad_contacts(contact_list, pair_mask_list)
            result["contact_matrix"] = contact_matrix
            result["contact_mask"] = contact_mask

        return result
