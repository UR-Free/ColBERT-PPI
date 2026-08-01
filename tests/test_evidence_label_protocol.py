from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from colbert_ppi.eval_labels import (
    RetrievalLabelProtocol,
    collapse_retrieval_protocol_by_uniprot,
    load_retrieval_label_protocol,
)
from colbert_ppi.retrieval import compute_retrieval_metrics


class EvidenceProtocolTest(unittest.TestCase):
    def protocol(self) -> RetrievalLabelProtocol:
        positive = torch.tensor(
            [[True, False, False], [False, True, False], [False, False, True]]
        )
        negative = torch.tensor(
            [[False, True, False], [True, False, False], [False, False, False]]
        )
        candidate = torch.tensor(
            [[True, True, False], [True, True, True], [True, True, True]]
        )
        operational = candidate & ~positive
        operational[2, 0] = False  # known association, censored from binary AP
        observed = positive | operational
        unknown = candidate & ~positive & ~operational
        result = RetrievalLabelProtocol(
            positive_mask=positive,
            operational_negative_mask=operational,
            verified_negative_mask=negative,
            candidate_mask=candidate,
            observed_label_mask=observed,
            unknown_mask=unknown,
            source="synthetic",
            protocol_version="test-v1",
        )
        result.validate()
        return result

    def test_metrics_mask_ineligible_cells(self) -> None:
        protocol = self.protocol()
        # The largest score is ineligible and must not affect rank metrics.
        scores = torch.tensor(
            [[0.8, 0.2, 100.0], [0.1, 0.9, 0.3], [0.2, 0.1, 0.7]],
            dtype=torch.float32,
        )
        metrics = compute_retrieval_metrics(
            scores,
            positive_mask=protocol.positive_mask,
            candidate_mask=protocol.candidate_mask,
            observed_label_mask=protocol.observed_label_mask,
            verified_negative_mask=protocol.verified_negative_mask,
        )
        self.assertEqual(metrics["acc"], 1.0)
        self.assertEqual(metrics["hit_at_1"], 1.0)
        self.assertEqual(metrics["hit_at_5"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)
        self.assertEqual(metrics["verified_negative_pairs"], 2.0)
        self.assertEqual(metrics["operational_negative_pairs"], 4.0)
        self.assertIsNotNone(metrics["operational_binary_auprc"])
        self.assertIsNotNone(metrics["operational_binary_auroc"])
        self.assertIsNotNone(metrics["strict_binary_auprc"])
        self.assertIsNotNone(metrics["strict_binary_auroc"])

    def test_missing_negative_class_is_null(self) -> None:
        protocol = self.protocol()
        no_negatives = RetrievalLabelProtocol(
            positive_mask=protocol.positive_mask,
            operational_negative_mask=protocol.operational_negative_mask,
            verified_negative_mask=torch.zeros_like(protocol.positive_mask),
            candidate_mask=protocol.candidate_mask,
            observed_label_mask=protocol.observed_label_mask,
            unknown_mask=(
                protocol.candidate_mask
                & ~protocol.positive_mask
                & ~protocol.operational_negative_mask
            ),
            source="synthetic",
            protocol_version="test-v1",
        )
        no_negatives.validate()
        scores = torch.eye(3)
        metrics = compute_retrieval_metrics(
            scores,
            positive_mask=no_negatives.positive_mask,
            candidate_mask=no_negatives.candidate_mask,
            observed_label_mask=no_negatives.observed_label_mask,
            verified_negative_mask=no_negatives.verified_negative_mask,
        )
        self.assertIsNone(metrics["strict_binary_auprc"])
        self.assertIsNone(metrics["strict_binary_auroc"])

    def test_npz_order_is_verified(self) -> None:
        protocol = self.protocol()
        samples = [
            ("1abc__A1_P00001-R", "1abc__B1_Q00001-L"),
            ("2abc__A1_P00002-R", "2abc__B1_Q00002-L"),
            ("3abc__A1_P00003-R", "3abc__B1_Q00003-L"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.npz"
            np.savez_compressed(
                path,
                protocol_version=np.asarray(protocol.protocol_version),
                receptor_labels=np.asarray([item[0] for item in samples]),
                ligand_labels=np.asarray([item[1] for item in samples]),
                positive_mask=protocol.positive_mask.numpy(),
                operational_negative_mask=protocol.operational_negative_mask.numpy(),
                verified_negative_mask=protocol.verified_negative_mask.numpy(),
                unknown_mask=protocol.unknown_mask.numpy(),
                candidate_mask=protocol.candidate_mask.numpy(),
                observed_label_mask=protocol.observed_label_mask.numpy(),
            )
            loaded = load_retrieval_label_protocol(path, samples)
            self.assertTrue(torch.equal(loaded.positive_mask, protocol.positive_mask))
            with self.assertRaisesRegex(ValueError, "order mismatch"):
                load_retrieval_label_protocol(path, list(reversed(samples)))

    def test_entity_collapse_preserves_evidence(self) -> None:
        positive = torch.tensor([[True, True], [True, True]])
        candidate = torch.ones((2, 2), dtype=torch.bool)
        protocol = RetrievalLabelProtocol(
            positive_mask=positive,
            operational_negative_mask=torch.zeros_like(positive),
            verified_negative_mask=torch.zeros_like(positive),
            candidate_mask=candidate,
            observed_label_mask=candidate,
            unknown_mask=torch.zeros_like(positive),
            source="synthetic",
            protocol_version="test-v1",
        )
        samples = [
            ("1abc__A1_P00001-R", "1abc__B1_Q00001-L"),
            ("2abc__A1_P00001-R", "2abc__B1_Q00001-L"),
        ]
        scores = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        collapsed_scores, collapsed, metadata = collapse_retrieval_protocol_by_uniprot(
            scores, protocol, samples, reduction="max"
        )
        self.assertEqual(collapsed_scores.shape, (1, 1))
        self.assertAlmostEqual(float(collapsed_scores[0, 0]), 0.4)
        self.assertEqual(int(collapsed.positive_mask.sum()), 1)
        self.assertEqual(int(collapsed.operational_negative_mask.sum()), 0)
        self.assertEqual(metadata["unique_positive_edges"], 1)

    def test_rectangular_entity_metrics(self) -> None:
        scores = torch.tensor([[0.9, 0.2, 0.1], [0.1, 0.3, 0.8]])
        positive = torch.tensor(
            [[True, False, False], [False, False, True]]
        )
        candidate = torch.ones_like(positive)
        metrics = compute_retrieval_metrics(
            scores,
            positive_mask=positive,
            candidate_mask=candidate,
            observed_label_mask=candidate,
            verified_negative_mask=torch.zeros_like(positive),
        )
        self.assertEqual(metrics["hit_at_1"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)
        self.assertIsNotNone(metrics["operational_binary_auprc"])
        self.assertIsNone(metrics["strict_binary_auprc"])


if __name__ == "__main__":
    unittest.main()
