"""
OOD detectors with sklearn-style fit / score / predict API.
"""

from oodkit.detectors.base import BaseDetector
from oodkit.detectors.vim import ViM

__all__ = ["BaseDetector", "ViM"]
