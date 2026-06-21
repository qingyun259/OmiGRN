"""
Tiny synthetic smoke test for OmiGRN.

Trains the genomic branch end-to-end on random data (no real data, no DGL) so you
can verify the install and the training loop work before pointing it at a dataset.

    python smoke_test.py
"""

from __future__ import annotations

import numpy as np
import torch

from omigrn import TrainingConfig, run_cross_validation


def main() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n_samples, n_snp = 160, 256

    # Random 0/1/2 SNP genotypes; a subset of loci drive the target.
    geno = rng.integers(0, 3, size=(n_samples, n_snp)).astype("float32")
    weights = np.zeros(n_snp, dtype="float32")
    weights[:24] = rng.normal(size=24)
    targets = (geno @ weights + rng.normal(scale=0.3, size=n_samples)).astype("float32")

    modalities = {"genotype": geno}
    sample_ids = [f"sample_{i:03d}" for i in range(n_samples)]

    config = TrainingConfig(
        epochs=40,
        batch_size=16,
        folds=2,
        patience=40,
        device="cpu",
        embed_dim=32,
        mlp_hidden=32,
        dropout=0.1,
    )

    results = run_cross_validation(modalities, targets, sample_ids, config)
    mean_train = float(np.mean([r.train_pearson for r in results]))
    mean_val = float(np.mean([r.val_pearson for r in results]))
    print("=" * 60)
    print(f"Smoke test OK: {len(results)} folds trained on synthetic data.")
    print(f"Mean train Pearson r = {mean_train:.3f}  (sanity check: should be clearly > 0)")
    print(f"Mean val   Pearson r = {mean_val:.3f}  (noisy on tiny synthetic data)")


if __name__ == "__main__":
    main()
