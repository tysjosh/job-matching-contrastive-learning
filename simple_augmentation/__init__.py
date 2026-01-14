"""
Simple Resume Augmentation System

A lightweight data augmentation tool that generates augmented versions of resume records
using SentenceTransformer-based paraphrasing and strategic field masking.
"""

__version__ = "1.0.0"
__author__ = "Simple Augmentation System"

from .pipeline import SimpleAugmentationPipeline
from .paraphraser import SentenceTransformerParaphraser
from .masker import FieldMasker
from .output_manager import OutputManager

__all__ = [
    "SimpleAugmentationPipeline",
    "SentenceTransformerParaphraser", 
    "FieldMasker",
    "OutputManager"
]