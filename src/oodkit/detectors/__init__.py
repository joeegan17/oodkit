"""Concrete OOD detectors (sklearn-style ``fit`` / ``score`` / ``predict``)."""

from oodkit.detectors.base import BaseDetector
from oodkit.detectors.energy import Energy
from oodkit.detectors.knn import KNN
from oodkit.detectors.mahalanobis import Mahalanobis
from oodkit.detectors.msp import MSP
from oodkit.detectors.pca import PCA
from oodkit.detectors.pca_fusion import PCAFusion
from oodkit.detectors.vim import ViM
from oodkit.detectors.wdiscood import WDiscOOD

__all__ = [
    "BaseDetector",
    "Energy",
    "KNN",
    "Mahalanobis",
    "MSP",
    "PCA",
    "PCAFusion",
    "ViM",
    "WDiscOOD",
]
