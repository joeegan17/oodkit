"""
OODKit — Out-of-distribution detection library.

MVP package for OOD detection with sklearn-style API.
"""

from oodkit.detectors.base import BaseDetector
from oodkit.detectors.vim import ViM
from oodkit.data.features import Features

__all__ = ["BaseDetector", "Features", "ViM"]
