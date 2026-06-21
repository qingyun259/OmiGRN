"""
OmiGRN — a minimal multi-omics genomic-selection codebase.

OmiGRN is a dual-branch deep-learning framework that fuses SNP genotype and
transcriptome data to predict complex crop traits:

* genomic branch        -> multi-scale genomic context mixer (MGCM); see ``blocks``
* transcriptomic branch -> GRN message-passing network;            see ``grn_encoder``
* fusion + output       -> concatenation feature-fusion + MLP head; see ``model``

Public API
----------
    OmiGRN, ModalityConfig, ModalityEncoder            (model)
    GRNModalityEncoder, normalize_gene_name            (transcriptomic GRN branch)
    prepare_dataset, DatasetBundle                     (data loading)
    TrainingConfig, CheckpointConfig, FoldResult,
    run_cross_validation                               (k-fold training)

Note: importing this package only requires ``torch`` / ``numpy`` / ``pandas`` /
``scikit-learn``. The GRN-MPNN branch additionally needs ``dgl``, which is only
imported when a GRN encoder is actually constructed.
"""

from .model import OmiGRN, ModalityConfig, ModalityEncoder
from .grn_encoder import GRNModalityEncoder, normalize_gene_name
from .data import prepare_dataset, DatasetBundle
from .trainer import (
    TrainingConfig,
    CheckpointConfig,
    FoldResult,
    run_cross_validation,
)

__all__ = [
    "OmiGRN",
    "ModalityConfig",
    "ModalityEncoder",
    "GRNModalityEncoder",
    "normalize_gene_name",
    "prepare_dataset",
    "DatasetBundle",
    "TrainingConfig",
    "CheckpointConfig",
    "FoldResult",
    "run_cross_validation",
]
