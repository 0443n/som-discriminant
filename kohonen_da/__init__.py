"""Discriminant analysis based on Kohonen statistics (Komarov & Kuravsky, MDA 2023)."""

from .classifier import KohonenDA, sturges_bins
from .v2 import KohonenDA2

__all__ = ["KohonenDA", "KohonenDA2", "sturges_bins"]
